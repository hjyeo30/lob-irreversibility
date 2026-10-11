#!/usr/bin/env python3
"""Algebraic directional reflection and frozen Phase 10 score inference.

``J`` is a feature-space side/price-direction relabeling, not a unique
physical reflection of raw prices.  Reflection is applied to the unscaled
37-channel physical arrays before the immutable transform and scaler.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PHASE10_CODE = PROJECT / "phases/10_predictive_bridge/codebook.py"
SPEC = importlib.util.spec_from_file_location("phase12_phase10", PHASE10_CODE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Phase 10 code: {PHASE10_CODE}")
PHASE10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PHASE10
SPEC.loader.exec_module(PHASE10)
UPSTREAM = PHASE10.UPSTREAM

SYMBOLS = PHASE10.SYMBOLS
DATES = PHASE10.ALL_DATES
MODELS = PHASE10.MODELS
SEEDS = PHASE10.SEEDS
K = PHASE10.K
STRIDE = PHASE10.STRIDE
BOUND = PHASE10.BOUND
DEFAULT_ARTIFACTS = HERE / "artifacts"
DEFAULT_SCORE_ROOT = PROJECT / "phases/10_predictive_bridge/artifacts/data/scores"
CONTRACT = "phase12_reflected_scores_v1"

KEY_FIELDS = (
    "symbol",
    "source_date",
    "segment_id",
    "path_start_block_index",
    "path_end_block_index",
)
SCORE_FIELDS = tuple(
    f"reflected_{model}_seed_{seed}" for model in MODELS for seed in SEEDS
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("source_date", pa.string()),
        ("segment_id", pa.string()),
        ("path_start_block_index", pa.int64()),
        ("path_end_block_index", pa.int64()),
        *((name, pa.float64()) for name in SCORE_FIELDS),
    ]
)

# Exact indices in continuous_path_37_v1's raw boundary/interval families.
_BOUNDARY_SWAPS = ((1, 2), (3, 4))
_BOUNDARY_NEGATE = (5, 6)
_INTERVAL_SWAPS = ((0, 1), (2, 3), (9, 10), (11, 12), (19, 21), (20, 22))
_INTERVAL_NEGATED_SWAPS = ((7, 8),)
_INTERVAL_NEGATE = (5, 6)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps({"utc": utc_now(), "event": event, **values}, sort_keys=True),
        flush=True,
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=131_072)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _swap(values: np.ndarray, left: int, right: int) -> None:
    saved = values[..., left].copy()
    values[..., left] = values[..., right]
    values[..., right] = saved


def reflect_raw(
    boundaries: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the approved algebraic reflection J to unscaled path arrays."""
    if boundaries.ndim != 3 or intervals.ndim != 3:
        raise ValueError("reflection expects rank-three path arrays")
    if boundaries.shape[:2] != (intervals.shape[0], intervals.shape[1] + 1):
        raise ValueError("boundary/interval path shape mismatch")
    if boundaries.shape[2] != len(UPSTREAM.BOUNDARY_FIELDS):
        raise ValueError("unexpected boundary channel count")
    if intervals.shape[2] != len(UPSTREAM.INTERVAL_FIELDS):
        raise ValueError("unexpected interval channel count")

    reflected_boundaries = boundaries.copy()
    reflected_intervals = intervals.copy()
    for left, right in _BOUNDARY_SWAPS:
        _swap(reflected_boundaries, left, right)
    reflected_boundaries[..., _BOUNDARY_NEGATE] *= -1.0
    for left, right in _INTERVAL_SWAPS:
        _swap(reflected_intervals, left, right)
    for left, right in _INTERVAL_NEGATED_SWAPS:
        _swap(reflected_intervals, left, right)
        reflected_intervals[..., (left, right)] *= -1.0
    reflected_intervals[..., _INTERVAL_NEGATE] *= -1.0
    return reflected_boundaries, reflected_intervals


def physical_path_arrays(
    day: Any, starts: np.ndarray, k: int = K
) -> tuple[np.ndarray, np.ndarray]:
    offsets = starts[:, None] + np.arange(k, dtype=np.int64)[None, :]
    intervals = day.intervals[offsets]
    boundaries = np.empty(
        (len(starts), k + 1, len(UPSTREAM.BOUNDARY_FIELDS)), dtype=np.float64
    )
    boundaries[:, 0, :] = day.starts[starts]
    boundaries[:, 1:, :] = day.ends[offsets]
    return boundaries, intervals


def reflection_audit(boundaries: np.ndarray, intervals: np.ndarray) -> dict[str, Any]:
    reflected_boundaries, reflected_intervals = reflect_raw(boundaries, intervals)
    twice_boundaries, twice_intervals = reflect_raw(
        reflected_boundaries, reflected_intervals
    )
    reverse_boundaries, reverse_intervals = UPSTREAM.reverse_continuous(
        boundaries, intervals
    )
    jr_boundaries, jr_intervals = reflect_raw(reverse_boundaries, reverse_intervals)
    j_boundaries, j_intervals = reflect_raw(boundaries, intervals)
    rj_boundaries, rj_intervals = UPSTREAM.reverse_continuous(j_boundaries, j_intervals)
    result = {
        "j_squared_exact": bool(
            np.array_equal(twice_boundaries, boundaries)
            and np.array_equal(twice_intervals, intervals)
        ),
        "jr_equals_rj_exact": bool(
            np.array_equal(jr_boundaries, rj_boundaries)
            and np.array_equal(jr_intervals, rj_intervals)
        ),
        "reflected_finite": bool(
            np.isfinite(reflected_boundaries).all()
            and np.isfinite(reflected_intervals).all()
        ),
    }
    if not all(result.values()):
        raise RuntimeError(f"directional reflection audit failed: {result}")
    return result


def reflected_tokens(
    day: Any, starts: np.ndarray, k: int = K
) -> tuple[np.ndarray, np.ndarray]:
    boundaries, intervals = physical_path_arrays(day, starts, k)
    reflected_boundaries, reflected_intervals = reflect_raw(boundaries, intervals)
    reverse_boundaries, reverse_intervals = UPSTREAM.reverse_continuous(
        reflected_boundaries, reflected_intervals
    )
    return (
        UPSTREAM.make_tokens(reflected_boundaries, reflected_intervals),
        UPSTREAM.make_tokens(reverse_boundaries, reverse_intervals),
    )


def original_score_path(score_root: Path, symbol: str, source_date: str) -> Path:
    return score_root / f"symbol={symbol}" / f"source_date={source_date}.parquet"


def default_output(artifacts: Path, symbol: str, source_date: str) -> Path:
    return (
        artifacts
        / "data/reflected_scores"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def _column_numpy(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name])


def key_table(table: pa.Table) -> pa.Table:
    return table.select(KEY_FIELDS)


def keys_equal(left: pa.Table, right: pa.Table) -> bool:
    if left.num_rows != right.num_rows:
        return False
    return all(pc.all(pc.equal(left[name], right[name])).as_py() for name in KEY_FIELDS)


def validate_output(output: Path, completion: Path, config: dict[str, Any]) -> bool:
    if not output.is_file() or not completion.is_file():
        return False
    try:
        record = json.loads(completion.read_text(encoding="utf-8"))
        if (
            record.get("status") != "complete"
            or record.get("contract") != CONTRACT
            or record.get("config") != config
            or record.get("bytes") != output.stat().st_size
        ):
            return False
        parquet = pq.ParquetFile(output)
        if (
            parquet.schema_arrow != OUTPUT_SCHEMA
            or parquet.metadata.num_rows != record.get("rows")
        ):
            return False
        original = pq.read_table(
            config["original_score_path"], columns=list(KEY_FIELDS)
        )
        result = parquet.read()
        if not keys_equal(key_table(result), original):
            return False
        scores = result.select(SCORE_FIELDS)
        for name in SCORE_FIELDS:
            if pc.any(pc.invert(pc.is_finite(scores[name]))).as_py():
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pa.ArrowException):
        return False


def unit_config(args: argparse.Namespace, original: Path) -> dict[str, Any]:
    block_root = (
        Path(args.block_root)
        if args.block_root
        else PHASE10.DEFAULT_BLOCK_ROOTS[args.symbol]
    )
    model_base = Path(args.model_base) if args.model_base else None
    return {
        "symbol": args.symbol,
        "source_date": args.source_date,
        "k": K,
        "stride": STRIDE,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "bound": BOUND,
        "batch_size": args.batch_size,
        "device": args.device,
        "threads": args.threads,
        "block_root": str(block_root),
        "model_base": None if model_base is None else str(model_base),
        "original_score_path": str(original),
        "original_score_rows": pq.ParquetFile(original).metadata.num_rows,
        "original_score_bytes": original.stat().st_size,
        "input_block_files": len(PHASE10.block_paths(block_root, args.source_date)),
    }


def run_score_unit(args: argparse.Namespace, device: torch.device) -> Path:
    if args.symbol not in SYMBOLS or args.source_date not in DATES:
        raise ValueError(
            "symbol/date outside the approved Phase 12 reflection universe"
        )
    score_root = Path(args.score_root)
    original_path = original_score_path(score_root, args.symbol, args.source_date)
    if not original_path.is_file():
        raise FileNotFoundError(f"missing Phase 10 score input: {original_path}")
    output = (
        Path(args.output)
        if args.output
        else default_output(Path(args.artifacts), args.symbol, args.source_date)
    )
    completion = output.with_suffix(".complete.json")
    config = unit_config(args, original_path)
    if not args.replace and validate_output(output, completion, config):
        emit(
            "reflection_skip",
            symbol=args.symbol,
            source_date=args.source_date,
            output=str(output),
        )
        return output

    original = pq.read_table(original_path, columns=list(KEY_FIELDS))
    block_root = (
        Path(args.block_root)
        if args.block_root
        else PHASE10.DEFAULT_BLOCK_ROOTS[args.symbol]
    )
    model_base = Path(args.model_base) if args.model_base else None
    day = PHASE10.read_model_day(block_root, args.source_date)
    starts = UPSTREAM.day_path_starts(day, K)
    generated_keys = pa.table(
        {
            "symbol": np.repeat(args.symbol, len(starts)),
            "source_date": np.repeat(args.source_date, len(starts)),
            "segment_id": np.repeat("0000", len(starts)),
            "path_start_block_index": day.block_index[starts],
            "path_end_block_index": day.block_index[starts + K - 1],
        }
    )
    if not keys_equal(generated_keys, original):
        raise RuntimeError("Phase 10 score keys do not match current block path keys")
    probe_count = min(32, len(starts))
    if probe_count == 0:
        raise RuntimeError("empty reflected-score unit")
    probe_boundaries, probe_intervals = physical_path_arrays(day, starts[:probe_count])
    audit = reflection_audit(probe_boundaries, probe_intervals)

    columns: dict[str, Any] = {
        name: _column_numpy(generated_keys, name) for name in KEY_FIELDS
    }
    for model_kind in MODELS:
        for seed in SEEDS:
            root = PHASE10.model_unit_root(args.symbol, model_kind, seed, model_base)
            model, scaler, alpha = PHASE10.load_model(root, model_kind, device)
            pieces: list[np.ndarray] = []
            for first in range(0, len(starts), args.batch_size):
                selected = starts[first : first + args.batch_size]
                reflected, reflected_reverse = reflected_tokens(day, selected)
                transformed = scaler.transform(reflected)
                transformed_reverse = scaler.transform(reflected_reverse)
                if (
                    not np.isfinite(transformed).all()
                    or not np.isfinite(transformed_reverse).all()
                ):
                    raise RuntimeError("nonfinite reflected standardized input")
                with torch.inference_mode():
                    raw = UPSTREAM.antisymmetric_score(
                        model,
                        torch.from_numpy(transformed).to(device),
                        torch.from_numpy(transformed_reverse).to(device),
                    )
                bounded = np.clip(
                    alpha * raw.detach().cpu().numpy().astype(np.float64), -BOUND, BOUND
                )
                if not np.isfinite(bounded).all():
                    raise RuntimeError("nonfinite reflected frozen score")
                pieces.append(bounded)
            columns[f"reflected_{model_kind}_seed_{seed}"] = np.concatenate(pieces)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            emit(
                "reflected_model_scored",
                symbol=args.symbol,
                source_date=args.source_date,
                model=model_kind,
                seed=seed,
                rows=len(starts),
            )

    table = pa.Table.from_pydict(columns, schema=OUTPUT_SCHEMA)
    if table.num_rows != original.num_rows:
        raise RuntimeError("reflected output row count mismatch")
    atomic_parquet(output, table)
    record = {
        "status": "complete",
        "contract": CONTRACT,
        "config": config,
        "rows": table.num_rows,
        "bytes": output.stat().st_size,
        "audits": audit,
        "completed_utc": utc_now(),
    }
    atomic_json(completion, record)
    if not validate_output(output, completion, config):
        raise RuntimeError("published reflected-score validation failed")
    emit(
        "reflection_complete",
        symbol=args.symbol,
        source_date=args.source_date,
        rows=table.num_rows,
        output=str(output),
    )
    return output


def configure_device(args: argparse.Namespace) -> torch.device:
    return PHASE10.configure_device(args.device, args.threads)


def run_queue(args: argparse.Namespace) -> None:
    device = configure_device(args)
    symbols = tuple(args.symbols) if args.symbols else SYMBOLS
    dates = tuple(args.dates) if args.dates else DATES
    total = len(symbols) * len(dates)
    done = 0
    for symbol in symbols:
        for source_date in dates:
            child = argparse.Namespace(**vars(args))
            child.symbol = symbol
            child.source_date = source_date
            child.output = None
            run_score_unit(child, device)
            done += 1
            emit("reflection_queue_progress", completed=done, total=total)


def run_audit() -> None:
    rng = np.random.default_rng(20260801)
    boundaries = rng.integers(
        -20, 21, size=(8, K + 1, len(UPSTREAM.BOUNDARY_FIELDS))
    ).astype(np.float64)
    intervals = rng.integers(
        -20, 21, size=(8, K, len(UPSTREAM.INTERVAL_FIELDS))
    ).astype(np.float64)
    result = reflection_audit(boundaries, intervals)
    print(json.dumps(result, sort_keys=True))


def run_support_audit(args: argparse.Namespace) -> Path:
    """Summarize reflected standardized tails under the development-only scaler."""
    rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for source_date in DATES:
            block_root = PHASE10.DEFAULT_BLOCK_ROOTS[symbol]
            day = PHASE10.read_model_day(block_root, source_date)
            starts = UPSTREAM.day_path_starts(day, K)
            if len(starts) == 0:
                raise RuntimeError(f"empty support audit unit: {symbol} {source_date}")
            take = min(args.sample_paths, len(starts))
            positions = np.linspace(0, len(starts) - 1, num=take, dtype=np.int64)
            selected = starts[np.unique(positions)]
            boundaries, intervals = physical_path_arrays(day, selected)
            reflected_boundaries, reflected_intervals = reflect_raw(
                boundaries, intervals
            )
            original_tokens = UPSTREAM.make_tokens(boundaries, intervals)
            reflected = UPSTREAM.make_tokens(reflected_boundaries, reflected_intervals)

            scaler_payloads = []
            for model_kind in MODELS:
                for seed in SEEDS:
                    root = PHASE10.model_unit_root(symbol, model_kind, seed, None)
                    scaler_payloads.append(
                        json.loads((root / "scaler.json").read_text())
                    )
            if any(payload != scaler_payloads[0] for payload in scaler_payloads[1:]):
                raise RuntimeError(f"frozen scaler mismatch: {symbol}")
            scaler = UPSTREAM.ChannelScaler.from_dict(scaler_payloads[0])
            original_z = np.abs(scaler.transform(original_tokens)).ravel()
            reflected_z = np.abs(scaler.transform(reflected)).ravel()
            if not np.isfinite(original_z).all() or not np.isfinite(reflected_z).all():
                raise RuntimeError("nonfinite standardized support audit")
            rows.append(
                {
                    "symbol": symbol,
                    "source_date": source_date,
                    "sample_paths": len(selected),
                    "standardized_values": len(reflected_z),
                    "scaler_fit": scaler_payloads[0].get("method"),
                    "tail_threshold_abs_z": float(args.tail_threshold),
                    "original_abs_z_q999": float(np.quantile(original_z, 0.999)),
                    "reflected_abs_z_q999": float(np.quantile(reflected_z, 0.999)),
                    "original_abs_z_max": float(original_z.max()),
                    "reflected_abs_z_max": float(reflected_z.max()),
                    "original_tail_share": float(
                        np.mean(original_z > args.tail_threshold)
                    ),
                    "reflected_tail_share": float(
                        np.mean(reflected_z > args.tail_threshold)
                    ),
                }
            )
    output = Path(args.output)
    atomic_parquet(output, pa.Table.from_pylist(rows))
    emit("reflection_support_audit_complete", rows=len(rows), output=str(output))
    return output


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--block-root")
    parser.add_argument("--model-base")
    parser.add_argument("--replace", action="store_true")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=lambda args: run_audit())
    support = subparsers.add_parser("support-audit")
    support.add_argument("--output", required=True)
    support.add_argument("--sample-paths", type=int, default=1024)
    support.add_argument("--tail-threshold", type=float, default=8.0)
    support.set_defaults(func=run_support_audit)
    score = subparsers.add_parser("score-unit")
    score.add_argument("--symbol", required=True, choices=SYMBOLS)
    score.add_argument("--source-date", required=True, choices=DATES)
    score.add_argument("--output")
    common_arguments(score)
    score.set_defaults(func=lambda args: run_score_unit(args, configure_device(args)))
    queue = subparsers.add_parser("run-queue")
    queue.add_argument("--symbols", nargs="*", choices=SYMBOLS)
    queue.add_argument("--dates", nargs="*", choices=DATES)
    common_arguments(queue)
    queue.set_defaults(func=run_queue)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
