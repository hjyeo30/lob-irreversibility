#!/usr/bin/env python3
"""Phase 10 frozen-sigma scoring and causal forecast-row construction.

The script deliberately separates expensive frozen-model inference from block
feature extraction.  Every command publishes one immutable symbol-date unit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
UPSTREAM_CODE = PROJECT / "upstream/phases/05_antisymmetric_sequence_models/codebook.py"
SPEC = importlib.util.spec_from_file_location("phase10_upstream_models", UPSTREAM_CODE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load immutable upstream code: {UPSTREAM_CODE}")
UPSTREAM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UPSTREAM
SPEC.loader.exec_module(UPSTREAM)

SYMBOLS = ("NVDA", "QQQ", "SOFI", "SAIA")
CALIBRATION_DATES = ("2026-07-16", "2026-07-17", "2026-07-20")
TEST_DATES = ("2026-07-21", "2026-07-22", "2026-07-23")
ALL_DATES = CALIBRATION_DATES + TEST_DATES
MODELS = ("pooled_mlp", "ordered_tcn")
SEEDS = (20260730, 20260731, 20260732)
K = 6
HORIZON = 6
STRIDE = 3
BOUND = math.log(99.0)

LOCAL_DATA = Path("/data")
DEFAULT_BLOCK_ROOTS = {
    "NVDA": LOCAL_DATA
    / "lob_liquidity_time_irreversibility/derived/continuous_blocks_micro_v1/symbol=NVDA",
    "QQQ": LOCAL_DATA
    / "lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=QQQ",
    "SOFI": LOCAL_DATA
    / "lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=SOFI",
    "SAIA": LOCAL_DATA
    / "lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=SAIA",
}
NVDA_MODEL_ROOT = (
    PROJECT / "upstream/phases/05_antisymmetric_sequence_models/"
    "artifacts/data/primary_v1/B10_K6"
)
CROSS_MODEL_ROOT = (
    LOCAL_DATA / "lob_irreversibility_interpretation_and_execution/"
    "artifacts/phase07_cross_symbol/units"
)
DEFAULT_ARTIFACTS = HERE / "artifacts"

BLOCK_COLUMNS = (
    "block_index",
    "block_start_ns",
    "block_end_ns",
    "realized_source_event_count",
    "completion_overshoot",
    "end_best_bid_nano",
    "end_best_ask_nano",
    "end_midprice_x2_nano",
    "end_spread_ticks",
    "end_bid_depth_l1",
    "end_ask_depth_l1",
    "end_bid_depth_l5",
    "end_ask_depth_l5",
    "end_imbalance_l1",
    "end_imbalance_l5",
    "bid_depth_l1_net_change",
    "ask_depth_l1_net_change",
    "bid_depth_l5_net_change",
    "ask_depth_l5_net_change",
    "bid_depth_l1_gross_variation",
    "ask_depth_l1_gross_variation",
    "bid_depth_l5_gross_variation",
    "ask_depth_l5_gross_variation",
    "bid_best_depletion_count",
    "ask_best_depletion_count",
    "bid_best_appearance_count",
    "ask_best_appearance_count",
    "depletion_plus_appearance_count",
    "appearance_minus_depletion_count",
    "mid_log_return",
    "mid_change_half_ticks",
    "bid_quote_change_ticks",
    "ask_quote_change_ticks",
    "mid_abs_variation_half_ticks",
    "spread_abs_variation_ticks",
    "same_price_refill_count",
    "cause_add_count",
    "cause_cancel_count",
    "cause_replace_count",
    "cause_displayed_execution_count",
    "cause_other_count",
    "elapsed_ns",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=131_072)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps({"utc": utc_now(), "event": event, **values}, sort_keys=True),
        flush=True,
    )


def block_paths(root: Path, source_date: str) -> list[Path]:
    direct = sorted(
        (root / f"source_date={source_date}" / "target_b=10").glob("*.parquet")
    )
    nested = sorted(
        (root / f"source_date={source_date}" / "regime=daytime").glob(
            "segment=*/target_b=10/*.parquet"
        )
    )
    paths = direct or nested
    if not paths:
        raise FileNotFoundError(f"no B10 blocks: {root} {source_date}")
    segments = {
        next(
            (part for part in path.parts if part.startswith("segment=")), "segment=0000"
        )
        for path in paths
    }
    if len(segments) != 1:
        raise RuntimeError(
            f"expected one continuous segment, observed={sorted(segments)}"
        )
    return paths


def read_columns(
    root: Path, source_date: str, columns: Iterable[str]
) -> dict[str, np.ndarray]:
    paths = block_paths(root, source_date)
    tables = [pq.ParquetFile(path).read(columns=list(columns)) for path in paths]
    table = pa.concat_tables(tables, promote_options="default")
    order = np.asarray(table["block_index"], dtype=np.int64)
    position = np.argsort(order, kind="stable")
    order = order[position]
    if len(order) > 1 and np.any(np.diff(order) != 1):
        raise RuntimeError(f"non-contiguous block_index: {source_date}")
    result: dict[str, np.ndarray] = {"block_index": order}
    for name in columns:
        if name != "block_index":
            result[name] = np.asarray(table[name])[position]
    return result


def read_model_day(root: Path, source_date: str) -> Any:
    names = ("block_index", "block_start_ns", "block_end_ns") + tuple(
        dict.fromkeys(
            [f"start_{field}" for field in UPSTREAM.BOUNDARY_FIELDS]
            + [f"end_{field}" for field in UPSTREAM.BOUNDARY_FIELDS]
            + list(UPSTREAM.INTERVAL_FIELDS)
        )
    )
    values = read_columns(root, source_date, names)
    starts = np.column_stack(
        [
            values[f"start_{field}"].astype(np.float64)
            for field in UPSTREAM.BOUNDARY_FIELDS
        ]
    )
    ends = np.column_stack(
        [
            values[f"end_{field}"].astype(np.float64)
            for field in UPSTREAM.BOUNDARY_FIELDS
        ]
    )
    intervals = np.column_stack(
        [values[field].astype(np.float64) for field in UPSTREAM.INTERVAL_FIELDS]
    )
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or not np.isfinite(intervals).all()
    ):
        raise RuntimeError("nonfinite model inputs")
    if len(starts) > 1 and not np.allclose(ends[:-1], starts[1:], rtol=0.0, atol=1e-6):
        raise RuntimeError("adjacent boundary mismatch")
    return UPSTREAM.DayBlocks(
        source_date=source_date,
        block_index=values["block_index"],
        block_start_ns=values["block_start_ns"].astype(np.int64),
        block_end_ns=values["block_end_ns"].astype(np.int64),
        starts=starts,
        ends=ends,
        intervals=intervals,
    )


def model_unit_root(
    symbol: str, model: str, seed: int, model_base: Path | None
) -> Path:
    if model_base is not None:
        base = model_base
    elif symbol == "NVDA":
        base = NVDA_MODEL_ROOT
    else:
        base = CROSS_MODEL_ROOT / f"symbol={symbol}"
    return base / model / f"seed={seed}"


def load_model(
    root: Path, model_kind: str, device: torch.device
) -> tuple[Any, Any, float]:
    required = ("model.pt", "scaler.json", "calibration.json")
    if not all((root / name).is_file() for name in required):
        raise FileNotFoundError(f"incomplete frozen model: {root}")
    scaler = UPSTREAM.ChannelScaler.from_dict(
        json.loads((root / "scaler.json").read_text())
    )
    alpha = float(json.loads((root / "calibration.json").read_text())["alpha"])
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    model = UPSTREAM.build_model(model_kind)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, scaler, alpha


def configure_device(name: str, threads: int) -> torch.device:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if name == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(name)


def score_unit(args: argparse.Namespace) -> None:
    if args.symbol not in SYMBOLS or args.source_date not in ALL_DATES:
        raise ValueError("symbol/date outside Phase 10 contract")
    device = configure_device(args.device, args.threads)
    block_root = (
        Path(args.block_root) if args.block_root else DEFAULT_BLOCK_ROOTS[args.symbol]
    )
    model_base = Path(args.model_base) if args.model_base else None
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_ARTIFACTS
        / "data/scores"
        / f"symbol={args.symbol}"
        / f"source_date={args.source_date}.parquet"
    )
    if output.is_file() and not args.replace:
        emit("score_skip", output=str(output))
        return
    started = time.monotonic()
    day = read_model_day(block_root, args.source_date)
    starts = UPSTREAM.day_path_starts(day, K)
    columns: dict[str, Any] = {
        "symbol": np.repeat(args.symbol, len(starts)),
        "source_date": np.repeat(args.source_date, len(starts)),
        "segment_id": np.repeat("0000", len(starts)),
        "path_start_block_index": day.block_index[starts],
        "path_end_block_index": day.block_index[starts + K - 1],
    }
    path_events: np.ndarray | None = None
    duration = (day.block_end_ns[starts + K - 1] - day.block_start_ns[starts]) / 1e9
    columns["path_duration_seconds"] = duration
    for model_kind in MODELS:
        for seed in SEEDS:
            root = model_unit_root(args.symbol, model_kind, seed, model_base)
            model, scaler, alpha = load_model(root, model_kind, device)
            pieces: list[np.ndarray] = []
            for first in range(0, len(starts), args.batch_size):
                selected = starts[first : first + args.batch_size]
                forward, reverse, events = UPSTREAM.path_arrays(day, selected, K)
                if path_events is None:
                    path_events = np.empty(len(starts), dtype=np.int64)
                path_events[first : first + len(selected)] = events.astype(np.int64)
                with torch.inference_mode():
                    score = UPSTREAM.antisymmetric_score(
                        model,
                        torch.from_numpy(scaler.transform(forward)).to(device),
                        torch.from_numpy(scaler.transform(reverse)).to(device),
                    )
                pieces.append(score.detach().cpu().numpy().astype(np.float64))
            raw = np.concatenate(pieces)
            bounded = np.clip(alpha * raw, -BOUND, BOUND)
            if not np.isfinite(bounded).all():
                raise RuntimeError("nonfinite frozen scores")
            columns[f"{model_kind}_seed_{seed}"] = bounded
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            emit(
                "model_scored",
                symbol=args.symbol,
                source_date=args.source_date,
                model=model_kind,
                seed=seed,
                rows=len(starts),
            )
    assert path_events is not None
    columns["path_event_count"] = path_events
    table = pa.table(columns)
    atomic_parquet(output, table)
    atomic_json(
        output.with_suffix(".complete.json"),
        {
            "status": "complete",
            "contract": "phase10_frozen_scores_v1",
            "symbol": args.symbol,
            "source_date": args.source_date,
            "rows": table.num_rows,
            "device": args.device,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "input_files": len(block_paths(block_root, args.source_date)),
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": utc_now(),
        },
    )
    emit(
        "score_complete",
        output=str(output),
        rows=table.num_rows,
        seconds=time.monotonic() - started,
    )


def compact_stored_test(args: argparse.Namespace) -> None:
    if args.source_date not in TEST_DATES:
        raise ValueError("stored prediction compaction is test-only")
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_ARTIFACTS
        / "data/scores"
        / f"symbol={args.symbol}"
        / f"source_date={args.source_date}.parquet"
    )
    model_base = Path(args.model_base) if args.model_base else None
    values: dict[str, np.ndarray] = {}
    keys: dict[str, np.ndarray] | None = None
    for model_kind in MODELS:
        for seed in SEEDS:
            root = model_unit_root(args.symbol, model_kind, seed, model_base)
            path = root / "test_predictions.parquet"
            score_field = (
                "bounded_log99_score"
                if "bounded_log99_score" in pq.read_schema(path).names
                else "calibrated_antisymmetric_score"
            )
            table = pq.read_table(
                path,
                filters=[
                    ("source_date", "=", args.source_date),
                    ("orientation", "=", "forward"),
                ],
                columns=[
                    "source_date",
                    "start_block_index",
                    "path_event_count",
                    "duration_seconds",
                    score_field,
                ],
            )
            current = {
                "start": np.asarray(table["start_block_index"], dtype=np.int64),
                "events": np.asarray(table["path_event_count"], dtype=np.int64),
                "duration": np.asarray(table["duration_seconds"], dtype=np.float64),
            }
            if keys is None:
                keys = current
            elif not all(np.array_equal(keys[name], current[name]) for name in keys):
                raise RuntimeError(f"stored prediction key mismatch: {path}")
            values[f"{model_kind}_seed_{seed}"] = np.clip(
                np.asarray(table[score_field], dtype=np.float64), -BOUND, BOUND
            )
    assert keys is not None
    table = pa.table(
        {
            "symbol": np.repeat(args.symbol, len(keys["start"])),
            "source_date": np.repeat(args.source_date, len(keys["start"])),
            "segment_id": np.repeat("0000", len(keys["start"])),
            "path_start_block_index": keys["start"],
            "path_end_block_index": keys["start"] + K - 1,
            "path_duration_seconds": keys["duration"],
            **values,
            "path_event_count": keys["events"],
        }
    )
    atomic_parquet(output, table)
    atomic_json(
        output.with_suffix(".complete.json"),
        {
            "status": "complete",
            "contract": "phase10_stored_test_score_compaction_v1",
            "symbol": args.symbol,
            "source_date": args.source_date,
            "rows": table.num_rows,
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "completed_utc": utc_now(),
        },
    )
    emit("compact_test_complete", output=str(output), rows=table.num_rows)


def score_g0_nvda(args: argparse.Namespace) -> None:
    from src import ablation_runtime as ablation

    ablation.register_direct_feature_configs(
        {
            "g0_all": (
                "elapsed_ns",
                "realized_source_event_count",
                "completion_overshoot",
            )
        }
    )
    device = configure_device(args.device, args.threads)
    block_root = (
        Path(args.block_root) if args.block_root else DEFAULT_BLOCK_ROOTS["NVDA"]
    )
    phase09 = Path(args.phase09_root)
    phase06_unit_root = Path(args.phase06_unit_root)
    anchor = (
        PROJECT
        / "upstream/phases/05_antisymmetric_sequence_models/artifacts/data/primary_v1/B10_K6"
    )
    base_scaler, _ = ablation.load_immutable_full_scaler(anchor)
    scaler = ablation.MaskedScaler(base_scaler, "g0_all")
    day = read_model_day(block_root, args.source_date)
    starts = UPSTREAM.day_path_starts(day, K)
    scores: dict[str, np.ndarray] = {}
    events: np.ndarray | None = None
    for seed in SEEDS:
        root = (
            phase06_unit_root / "ladder_g0/ordered_tcn/seed=20260730"
            if seed == SEEDS[0]
            else phase09
            / "artifacts/units/measurement/g0_all/ordered_tcn"
            / f"seed={seed}"
        )
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        model = UPSTREAM.build_model("ordered_tcn")
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
        alpha = float(json.loads((root / "calibration.json").read_text())["alpha"])
        pieces = []
        for first in range(0, len(starts), args.batch_size):
            selected = starts[first : first + args.batch_size]
            forward, reverse, current_events = UPSTREAM.path_arrays(day, selected, K)
            if events is None:
                events = np.empty(len(starts), dtype=np.int64)
            events[first : first + len(selected)] = current_events.astype(np.int64)
            with torch.inference_mode():
                raw = UPSTREAM.antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)).to(device),
                    torch.from_numpy(scaler.transform(reverse)).to(device),
                )
            pieces.append(raw.detach().cpu().numpy().astype(np.float64))
        scores[f"g0_seed_{seed}"] = np.clip(
            alpha * np.concatenate(pieces), -BOUND, BOUND
        )
        del model
    assert events is not None
    duration = (day.block_end_ns[starts + K - 1] - day.block_start_ns[starts]) / 1e9
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_ARTIFACTS
        / "data/g0_scores/symbol=NVDA"
        / f"source_date={args.source_date}.parquet"
    )
    table = pa.table(
        {
            "symbol": np.repeat("NVDA", len(starts)),
            "source_date": np.repeat(args.source_date, len(starts)),
            "path_end_block_index": day.block_index[starts + K - 1],
            "path_event_count": events,
            "path_duration_seconds": duration,
            **scores,
        }
    )
    atomic_parquet(output, table)
    atomic_json(
        output.with_suffix(".complete.json"),
        {
            "status": "complete",
            "contract": "phase10_nvda_g0_scores_v1",
            "source_date": args.source_date,
            "rows": table.num_rows,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "completed_utc": utc_now(),
        },
    )
    emit("g0_score_complete", source_date=args.source_date, rows=table.num_rows)


def augment_g0_nvda(args: argparse.Namespace) -> None:
    source = Path(args.forecast_path)
    score_path = Path(args.g0_score_path)
    output = Path(args.output)
    table = pq.read_table(source)
    score = pq.read_table(score_path).slice(0, table.num_rows)
    if not np.array_equal(
        np.asarray(table["path_end_block_index"]),
        np.asarray(score["path_end_block_index"]),
    ):
        raise RuntimeError("G0/forecast key mismatch")
    median = np.median(
        np.column_stack(
            [np.asarray(score[f"g0_seed_{seed}"], dtype=np.float64) for seed in SEEDS]
        ),
        axis=1,
    )
    events = np.asarray(score["path_event_count"], dtype=np.float64)
    duration = np.asarray(score["path_duration_seconds"], dtype=np.float64)
    for name, values in (
        ("m2c_g0_sigma_block", median / K),
        ("m2c_g0_sigma_event", median / np.maximum(events, 1.0)),
        ("m2c_g0_sigma_second", median / np.maximum(duration, 1e-9)),
    ):
        table = table.append_column(name, pa.array(values, type=pa.float64()))
    atomic_parquet(output, table)
    atomic_json(
        output.with_suffix(".complete.json"),
        {
            "status": "complete",
            "contract": "phase10_nvda_g0_augmented_forecast_v1",
            "source_date": str(table["source_date"][0].as_py()),
            "rows": table.num_rows,
            "forecast_sha256": sha256(source),
            "g0_score_sha256": sha256(score_path),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "completed_utc": utc_now(),
        },
    )
    emit("g0_augment_complete", output=str(output), rows=table.num_rows)


def window_sum(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    numeric = values.astype(np.float64, copy=False)
    cumulative = np.concatenate(([0.0], np.cumsum(numeric, dtype=np.float64)))
    return cumulative[starts + width] - cumulative[starts]


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(np.abs(denominator), 1.0)


def feature_unit(args: argparse.Namespace) -> None:
    if args.symbol not in SYMBOLS or args.source_date not in ALL_DATES:
        raise ValueError("symbol/date outside Phase 10 contract")
    block_root = (
        Path(args.block_root) if args.block_root else DEFAULT_BLOCK_ROOTS[args.symbol]
    )
    score_path = (
        Path(args.score_path)
        if args.score_path
        else DEFAULT_ARTIFACTS
        / "data/scores"
        / f"symbol={args.symbol}"
        / f"source_date={args.source_date}.parquet"
    )
    output = (
        Path(args.output)
        if args.output
        else DEFAULT_ARTIFACTS
        / "data/forecast_rows"
        / f"symbol={args.symbol}"
        / f"source_date={args.source_date}.parquet"
    )
    started = time.monotonic()
    block = read_columns(block_root, args.source_date, BLOCK_COLUMNS)
    n = len(block["block_index"])
    offset = date.fromisoformat(args.source_date).toordinal() % STRIDE
    starts = np.arange(offset, n - K - HORIZON + 1, STRIDE, dtype=np.int64)
    ends = starts + K - 1
    future = ends + HORIZON
    end_mid = block["end_midprice_x2_nano"].astype(np.int64)
    outcome = np.sign(end_mid[future] - end_mid[ends]).astype(np.int8) + 1
    bid = block["end_best_bid_nano"][ends].astype(np.float64)
    ask = block["end_best_ask_nano"][ends].astype(np.float64)
    bid_l1 = block["end_bid_depth_l1"][ends].astype(np.float64)
    ask_l1 = block["end_ask_depth_l1"][ends].astype(np.float64)
    midpoint = (bid + ask) / 2.0
    microprice = safe_ratio(ask * bid_l1 + bid * ask_l1, bid_l1 + ask_l1)
    halfspread = np.maximum((ask - bid) / 2.0, 1.0)
    elapsed = window_sum(block["elapsed_ns"], starts, K) / 1e9
    events = window_sum(block["realized_source_event_count"], starts, K)
    overshoot = window_sum(block["completion_overshoot"], starts, K)
    adds = window_sum(block["cause_add_count"], starts, K)
    cancels = window_sum(block["cause_cancel_count"], starts, K)
    executions = window_sum(block["cause_displayed_execution_count"], starts, K)
    score = pq.read_table(score_path)
    score_end = np.asarray(score["path_end_block_index"], dtype=np.int64)
    expected_end = block["block_index"][ends]
    if not np.array_equal(
        score_end[: len(expected_end)], expected_end
    ) or score.num_rows < len(expected_end):
        raise RuntimeError("score/feature path key mismatch")
    if score.num_rows != len(
        UPSTREAM.day_path_starts(read_model_day(block_root, args.source_date), K)
    ):
        raise RuntimeError("score coverage mismatch")
    score = score.slice(0, len(expected_end))
    pooled = np.median(
        np.column_stack(
            [
                np.asarray(score[f"pooled_mlp_seed_{seed}"], dtype=np.float64)
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    ordered = np.median(
        np.column_stack(
            [
                np.asarray(score[f"ordered_tcn_seed_{seed}"], dtype=np.float64)
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    seconds_from_midnight = (
        (block["block_end_ns"][ends] // 1_000_000_000) % 86_400
    ).astype(np.float64)
    phase = 2.0 * np.pi * seconds_from_midnight / 86_400.0
    data: dict[str, Any] = {
        "symbol": np.repeat(args.symbol, len(starts)),
        "source_date": np.repeat(args.source_date, len(starts)),
        "segment_id": np.repeat("0000", len(starts)),
        "path_end_block_index": expected_end,
        "outcome_6block": outcome,
        "m0_log_spread": np.log1p(block["end_spread_ticks"][ends].astype(np.float64)),
        "m0_log_bid_depth_l1": np.log1p(bid_l1),
        "m0_log_ask_depth_l1": np.log1p(ask_l1),
        "m0_log_bid_depth_l5": np.log1p(
            block["end_bid_depth_l5"][ends].astype(np.float64)
        ),
        "m0_log_ask_depth_l5": np.log1p(
            block["end_ask_depth_l5"][ends].astype(np.float64)
        ),
        "m0_imbalance_l1": block["end_imbalance_l1"][ends].astype(np.float64),
        "m0_imbalance_l5": block["end_imbalance_l5"][ends].astype(np.float64),
        "m0_microprice_deviation": (microprice - midpoint) / halfspread,
        "m0_tod_sin": np.sin(phase),
        "m0_tod_cos": np.cos(phase),
        "m1_mid_log_return": window_sum(block["mid_log_return"], starts, K),
        "m1_mid_change_half_ticks": window_sum(
            block["mid_change_half_ticks"], starts, K
        ),
        "m1_mid_abs_variation_half_ticks": np.log1p(
            window_sum(block["mid_abs_variation_half_ticks"], starts, K)
        ),
        "m1_spread_abs_variation_ticks": np.log1p(
            window_sum(block["spread_abs_variation_ticks"], starts, K)
        ),
        "m1_ofi_l1": safe_ratio(
            window_sum(block["bid_depth_l1_net_change"], starts, K)
            - window_sum(block["ask_depth_l1_net_change"], starts, K),
            bid_l1 + ask_l1,
        ),
        "m1_ofi_l5": safe_ratio(
            window_sum(block["bid_depth_l5_net_change"], starts, K)
            - window_sum(block["ask_depth_l5_net_change"], starts, K),
            block["end_bid_depth_l5"][ends].astype(np.float64)
            + block["end_ask_depth_l5"][ends].astype(np.float64),
        ),
        "m1_gross_depth_variation": np.log1p(
            window_sum(block["bid_depth_l1_gross_variation"], starts, K)
            + window_sum(block["ask_depth_l1_gross_variation"], starts, K)
        ),
        "m1_bid_quote_change": window_sum(block["bid_quote_change_ticks"], starts, K),
        "m1_ask_quote_change": window_sum(block["ask_quote_change_ticks"], starts, K),
        "m1_depletion_imbalance": safe_ratio(
            window_sum(block["bid_best_depletion_count"], starts, K)
            - window_sum(block["ask_best_depletion_count"], starts, K),
            window_sum(block["depletion_plus_appearance_count"], starts, K),
        ),
        "m1_appearance_minus_depletion": window_sum(
            block["appearance_minus_depletion_count"], starts, K
        ),
        "m1_same_price_refill": np.log1p(
            window_sum(block["same_price_refill_count"], starts, K)
        ),
        "m1_add_share": safe_ratio(adds, events),
        "m1_cancel_share": safe_ratio(cancels, events),
        "m1_execution_share": safe_ratio(executions, events),
        "m1_log_event_intensity": np.log1p(events / np.maximum(elapsed, 1e-9)),
        "m1_log_elapsed_seconds": np.log1p(elapsed),
        "m1_log_event_count": np.log1p(events),
        "m1_log_overshoot": np.log1p(overshoot),
        "m2_ordered_sigma_block": ordered / K,
        "m2_ordered_sigma_event": ordered / np.maximum(events, 1.0),
        "m2_ordered_sigma_second": ordered / np.maximum(elapsed, 1e-9),
        "m2_pooled_sigma_block": pooled / K,
        "m2_pooled_sigma_event": pooled / np.maximum(events, 1.0),
        "m2_pooled_sigma_second": pooled / np.maximum(elapsed, 1e-9),
        "m2_temporal_gap": ordered - pooled,
    }
    table = pa.table(data)
    numeric = table.drop(["symbol", "source_date", "segment_id"])
    for field in numeric.schema:
        if (
            pa.types.is_floating(field.type)
            and pc.any(pc.invert(pc.is_finite(numeric[field.name]))).as_py()
        ):
            raise RuntimeError(f"nonfinite feature: {field.name}")
    atomic_parquet(output, table)
    counts = np.bincount(outcome, minlength=3)
    atomic_json(
        output.with_suffix(".complete.json"),
        {
            "status": "complete",
            "contract": "phase10_forecast_rows_v1",
            "symbol": args.symbol,
            "source_date": args.source_date,
            "rows": table.num_rows,
            "outcome_counts_down_flat_up": counts.tolist(),
            "horizon_blocks": HORIZON,
            "k": K,
            "stride": STRIDE,
            "causal_cutoff": "end of K6 path",
            "future_support_excluded": int(
                len(
                    UPSTREAM.day_path_starts(
                        read_model_day(block_root, args.source_date), K
                    )
                )
                - len(starts)
            ),
            "score_sha256": sha256(score_path),
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": utc_now(),
        },
    )
    emit(
        "feature_complete",
        output=str(output),
        rows=table.num_rows,
        seconds=time.monotonic() - started,
    )


def model_columns(table: pa.Table) -> dict[str, list[str]]:
    m0 = [name for name in table.column_names if name.startswith("m0_")]
    m1 = m0 + [name for name in table.column_names if name.startswith("m1_")]
    ordered = [name for name in table.column_names if name.startswith("m2_ordered_")]
    m2a = m1 + ordered
    m2b = m1 + [name for name in table.column_names if name.startswith("m2_")]
    result = {"M0": m0, "M1": m1, "M2a": m2a, "M2b": m2b}
    g0 = [name for name in table.column_names if name.startswith("m2c_")]
    if g0:
        result["M2c"] = m2b + g0
    return result


def day_equal_weights(days: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(days, return_counts=True)
    total = len(days)
    lookup = {
        key: total / (len(unique) * count)
        for key, count in zip(unique, counts, strict=True)
    }
    return np.asarray([lookup[value] for value in days], dtype=np.float64)


def prediction_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    one_hot = np.eye(3, dtype=np.float64)[y]
    confidence = probability.max(axis=1)
    correct = (probability.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= left) & (
            confidence < right if right < 1 else confidence <= right
        )
        if np.any(mask):
            ece += float(
                np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(confidence[mask]))
            )
    try:
        auc = float(
            roc_auc_score(
                y, probability, multi_class="ovr", average="macro", labels=[0, 1, 2]
            )
        )
    except ValueError:
        auc = float("nan")
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1, 2])),
        "brier_multiclass": float(
            np.mean(np.sum((probability - one_hot) ** 2, axis=1))
        ),
        "macro_auc_ovr": auc,
        "accuracy": float(np.mean(probability.argmax(axis=1) == y)),
        "ece10_top_label": ece,
    }


def permute_sigma(
    matrix: np.ndarray,
    table: pa.Table,
    columns: list[str],
    feature_names: list[str],
    seed: int,
    within_state: bool,
) -> np.ndarray:
    result = matrix.copy()
    sigma_positions = [
        columns.index(name) for name in columns if name.startswith("m2_")
    ]
    if not sigma_positions:
        return result
    days = np.asarray(table["source_date"])
    imbalance = np.asarray(table["m0_imbalance_l1"], dtype=np.float64)
    spread = np.asarray(table["m0_log_spread"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    for source_date in np.unique(days):
        day_index = np.flatnonzero(days == source_date)
        if not within_state:
            source = rng.permutation(day_index)
            result[np.ix_(day_index, sigma_positions)] = matrix[
                np.ix_(source, sigma_positions)
            ]
            continue
        # State-conditional negative control: 5x5 train/test-day local quantile cells.
        i_edges = np.unique(np.quantile(imbalance[day_index], np.linspace(0, 1, 6)))
        s_edges = np.unique(np.quantile(spread[day_index], np.linspace(0, 1, 6)))
        i_bin = np.searchsorted(i_edges[1:-1], imbalance[day_index], side="right")
        s_bin = np.searchsorted(s_edges[1:-1], spread[day_index], side="right")
        cell = i_bin * 5 + s_bin
        for value in np.unique(cell):
            target = day_index[cell == value]
            source = rng.permutation(target)
            result[np.ix_(target, sigma_positions)] = matrix[
                np.ix_(source, sigma_positions)
            ]
    return result


def fit_symbol(args: argparse.Namespace) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    root = Path(args.forecast_root)
    output = Path(args.output_root)
    tables: dict[str, pa.Table] = {}
    for source_date in ALL_DATES:
        path = root / f"source_date={source_date}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[source_date] = pq.read_table(path)
    train = pa.concat_tables([tables[value] for value in CALIBRATION_DATES])
    test = pa.concat_tables([tables[value] for value in TEST_DATES])
    groups = model_columns(train)
    all_features = groups.get("M2c", groups["M2b"])
    x_train_all = np.column_stack(
        [np.asarray(train[name], dtype=np.float64) for name in all_features]
    )
    x_test_all = np.column_stack(
        [np.asarray(test[name], dtype=np.float64) for name in all_features]
    )
    y_train = np.asarray(train["outcome_6block"], dtype=np.int64)
    y_test = np.asarray(test["outcome_6block"], dtype=np.int64)
    train_days = np.asarray(train["source_date"])
    test_days = np.asarray(test["source_date"])
    weights = day_equal_weights(train_days)
    metrics: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    started = time.monotonic()
    for kind in ("logistic", "hist_gradient_boosting"):
        for model_name, features in groups.items():
            positions = [all_features.index(name) for name in features]
            if kind == "logistic":
                estimator: Any = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=1.0,
                        solver="newton-cholesky",
                        max_iter=100,
                        tol=1e-6,
                        random_state=20260801,
                    ),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    estimator.fit(
                        x_train_all[:, positions],
                        y_train,
                        logisticregression__sample_weight=weights,
                    )
            else:
                estimator = HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=100,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=20260801,
                )
                estimator.fit(x_train_all[:, positions], y_train, sample_weight=weights)
            probability = estimator.predict_proba(x_test_all[:, positions])
            predictions[(kind, model_name)] = probability
            for source_date in TEST_DATES:
                mask = test_days == source_date
                values = prediction_metrics(y_test[mask], probability[mask])
                metrics.append(
                    {
                        "symbol": args.symbol,
                        "source_date": source_date,
                        "estimator": kind,
                        "model": model_name,
                        "permutation": "actual",
                        "replicate": -1,
                        "rows": int(np.sum(mask)),
                        "features": len(features),
                        **values,
                    }
                )
            emit(
                "forecast_model_fit",
                symbol=args.symbol,
                estimator=kind,
                model=model_name,
                features=len(features),
            )
    # Ten deterministic logistic negative-control replicates.  Train and test are
    # permuted independently within their own dates; outcomes and M0/M1 stay fixed.
    control_features = groups["M2b"]
    control_positions = [all_features.index(name) for name in control_features]
    control_train = x_train_all[:, control_positions]
    control_test = x_test_all[:, control_positions]
    for control, within_state in (
        ("date_permutation", False),
        ("state_bin_permutation", True),
    ):
        for replicate in range(10):
            seed = 20260801 + replicate + (1000 if within_state else 0)
            perm_train = permute_sigma(
                control_train,
                train,
                control_features,
                control_features,
                seed,
                within_state,
            )
            perm_test = permute_sigma(
                control_test,
                test,
                control_features,
                control_features,
                seed + 10_000,
                within_state,
            )
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="newton-cholesky",
                    max_iter=100,
                    tol=1e-6,
                    random_state=seed,
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(
                    perm_train, y_train, logisticregression__sample_weight=weights
                )
            probability = estimator.predict_proba(perm_test)
            for source_date in TEST_DATES:
                mask = test_days == source_date
                values = prediction_metrics(y_test[mask], probability[mask])
                metrics.append(
                    {
                        "symbol": args.symbol,
                        "source_date": source_date,
                        "estimator": "logistic",
                        "model": "M2b",
                        "permutation": control,
                        "replicate": replicate,
                        "rows": int(np.sum(mask)),
                        "features": len(control_features),
                        **values,
                    }
                )
    metric_table = pa.Table.from_pylist(metrics)
    increments: list[dict[str, Any]] = []
    rows = metric_table.to_pylist()
    baseline = {
        (row["source_date"], row["estimator"]): row
        for row in rows
        if row["model"] == "M1" and row["permutation"] == "actual"
    }
    for row in rows:
        if row["model"] in ("M2a", "M2b", "M2c"):
            base = baseline[(row["source_date"], row["estimator"])]
            increments.append(
                {
                    "symbol": args.symbol,
                    "source_date": row["source_date"],
                    "estimator": row["estimator"],
                    "model": row["model"],
                    "permutation": row["permutation"],
                    "replicate": row["replicate"],
                    "log_loss_increment_m1_minus_model": base["log_loss"]
                    - row["log_loss"],
                    "brier_increment_m1_minus_model": base["brier_multiclass"]
                    - row["brier_multiclass"],
                    "auc_increment_model_minus_m1": row["macro_auc_ovr"]
                    - base["macro_auc_ovr"],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / "model_metrics.parquet"
    increment_path = output / "paired_increments.parquet"
    atomic_parquet(metric_path, metric_table)
    atomic_parquet(increment_path, pa.Table.from_pylist(increments))
    support = []
    for source_date, table in tables.items():
        y = np.asarray(table["outcome_6block"], dtype=np.int64)
        count = np.bincount(y, minlength=3)
        support.append(
            {
                "symbol": args.symbol,
                "source_date": source_date,
                "split": "train" if source_date in CALIBRATION_DATES else "test",
                "rows": len(y),
                "down": int(count[0]),
                "flat": int(count[1]),
                "up": int(count[2]),
            }
        )
    support_path = output / "support_audits.parquet"
    atomic_parquet(support_path, pa.Table.from_pylist(support))
    atomic_json(
        output / "complete.json",
        {
            "status": "complete",
            "contract": "phase10_symbol_forecast_v1",
            "symbol": args.symbol,
            "train_dates": list(CALIBRATION_DATES),
            "test_dates": list(TEST_DATES),
            "horizon_blocks": HORIZON,
            "train_rows": train.num_rows,
            "test_rows": test.num_rows,
            "primary": "day-level log_loss(M1)-log_loss(M2b), L2 multinomial logistic",
            "negative_control_replicates": 10,
            "artifacts": [
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in (metric_path, increment_path, support_path)
            ],
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": utc_now(),
        },
    )
    emit("fit_symbol_complete", symbol=args.symbol, seconds=time.monotonic() - started)


def aggregate(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts)
    source = artifacts / "tables/by_symbol"
    table_names = (
        "model_metrics.parquet",
        "paired_increments.parquet",
        "support_audits.parquet",
    )
    combined: dict[str, pa.Table] = {}
    for name in table_names:
        pieces = []
        for symbol in SYMBOLS:
            complete = source / symbol / "complete.json"
            if (
                not complete.is_file()
                or json.loads(complete.read_text()).get("status") != "complete"
            ):
                raise RuntimeError(f"missing completed forecast symbol: {symbol}")
            pieces.append(pq.read_table(source / symbol / name))
        combined[name] = pa.concat_tables(pieces)
        atomic_parquet(artifacts / "tables" / name, combined[name])
    metric_rows = combined["model_metrics.parquet"].to_pylist()
    calibration_rows = [
        {
            "symbol": row["symbol"],
            "source_date": row["source_date"],
            "estimator": row["estimator"],
            "model": row["model"],
            "brier_multiclass": row["brier_multiclass"],
            "ece10_top_label": row["ece10_top_label"],
            "accuracy": row["accuracy"],
            "rows": row["rows"],
        }
        for row in metric_rows
        if row["permutation"] == "actual"
    ]
    atomic_parquet(
        artifacts / "tables/calibration_summary.parquet",
        pa.Table.from_pylist(calibration_rows),
    )
    lineage: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for source_date in ALL_DATES:
            forecast = (
                artifacts
                / "data/forecast_rows"
                / f"symbol={symbol}"
                / f"source_date={source_date}.parquet"
            )
            score = (
                artifacts
                / "data/scores"
                / f"symbol={symbol}"
                / f"source_date={source_date}.parquet"
            )
            manifest = json.loads(forecast.with_suffix(".complete.json").read_text())
            score_manifest = json.loads(score.with_suffix(".complete.json").read_text())
            forecast_rows = pq.ParquetFile(forecast).metadata.num_rows
            score_rows = pq.ParquetFile(score).metadata.num_rows
            keys = np.asarray(
                pq.read_table(forecast, columns=["path_end_block_index"])[
                    "path_end_block_index"
                ],
                dtype=np.int64,
            )
            if (
                manifest.get("status") != "complete"
                or score_manifest.get("status") != "complete"
                or int(manifest.get("rows", -1)) != forecast_rows
                or int(score_manifest.get("rows", -1)) != score_rows
                or manifest.get("sha256") != sha256(forecast)
                or score_manifest.get("sha256") != sha256(score)
                or score_rows != forecast_rows + 2
                or len(keys) != len(np.unique(keys))
                or (len(keys) > 1 and np.any(np.diff(keys) <= 0))
            ):
                raise RuntimeError(
                    f"forecast lineage/coverage audit failed: {symbol} {source_date}"
                )
            lineage.append(
                {
                    "symbol": symbol,
                    "source_date": source_date,
                    "split": "train" if source_date in CALIBRATION_DATES else "test",
                    "forecast_rows": forecast_rows,
                    "forecast_bytes": forecast.stat().st_size,
                    "forecast_sha256": sha256(forecast),
                    "score_rows": score_rows,
                    "score_bytes": score.stat().st_size,
                    "score_sha256": sha256(score),
                    "manifest_score_sha256_matches": manifest["score_sha256"]
                    == sha256(score),
                }
            )
    atomic_parquet(
        artifacts / "tables/input_lineage.parquet", pa.Table.from_pylist(lineage)
    )
    increments = combined["paired_increments.parquet"].to_pylist()
    summary: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for estimator in ("logistic", "hist_gradient_boosting"):
            actual = [
                row
                for row in increments
                if row["symbol"] == symbol
                and row["estimator"] == estimator
                and row["model"] == "M2b"
                and row["permutation"] == "actual"
            ]
            summary.append(
                {
                    "symbol": symbol,
                    "estimator": estimator,
                    "test_days": len(actual),
                    "mean_log_loss_increment_m1_minus_m2b": float(
                        np.mean(
                            [row["log_loss_increment_m1_minus_model"] for row in actual]
                        )
                    ),
                    "positive_days": sum(
                        row["log_loss_increment_m1_minus_model"] > 0 for row in actual
                    ),
                }
            )
    summary_path = artifacts / "tables/compact_summary.parquet"
    atomic_parquet(summary_path, pa.Table.from_pylist(summary))
    outputs = [artifacts / "tables" / name for name in table_names] + [
        artifacts / "tables/calibration_summary.parquet",
        artifacts / "tables/input_lineage.parquet",
        summary_path,
    ]
    atomic_json(
        artifacts / "tables/complete.json",
        {
            "status": "complete",
            "contract": "phase10_predictive_bridge_wave1_v1",
            "symbols": list(SYMBOLS),
            "train_dates": list(CALIBRATION_DATES),
            "test_dates": list(TEST_DATES),
            "horizon_blocks": HORIZON,
            "primary_model": "L2 multinomial logistic; M1 versus M2b",
            "artifacts": [
                {
                    "filename": path.name,
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in outputs
            ],
            "completed_utc": utc_now(),
        },
    )
    emit("aggregate_complete", symbols=len(SYMBOLS), files=len(outputs))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score-unit")
    score.add_argument("--symbol", required=True, choices=SYMBOLS)
    score.add_argument("--source-date", required=True, choices=ALL_DATES)
    score.add_argument("--block-root")
    score.add_argument("--model-base")
    score.add_argument("--output")
    score.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    score.add_argument("--threads", type=int, default=4)
    score.add_argument("--batch-size", type=int, default=4096)
    score.add_argument("--replace", action="store_true")
    score.set_defaults(function=score_unit)
    compact = commands.add_parser("compact-test")
    compact.add_argument("--symbol", required=True, choices=SYMBOLS)
    compact.add_argument("--source-date", required=True, choices=TEST_DATES)
    compact.add_argument("--model-base")
    compact.add_argument("--output")
    compact.set_defaults(function=compact_stored_test)
    g0score = commands.add_parser("score-g0-nvda")
    g0score.add_argument("--source-date", required=True, choices=ALL_DATES)
    g0score.add_argument("--phase09-root", required=True)
    g0score.add_argument("--phase06-unit-root", required=True)
    g0score.add_argument("--block-root")
    g0score.add_argument("--output")
    g0score.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    g0score.add_argument("--threads", type=int, default=8)
    g0score.add_argument("--batch-size", type=int, default=8192)
    g0score.set_defaults(function=score_g0_nvda)
    augment = commands.add_parser("augment-g0-nvda")
    augment.add_argument("--forecast-path", required=True)
    augment.add_argument("--g0-score-path", required=True)
    augment.add_argument("--output", required=True)
    augment.set_defaults(function=augment_g0_nvda)
    feature = commands.add_parser("feature-unit")
    feature.add_argument("--symbol", required=True, choices=SYMBOLS)
    feature.add_argument("--source-date", required=True, choices=ALL_DATES)
    feature.add_argument("--block-root")
    feature.add_argument("--score-path")
    feature.add_argument("--output")
    feature.set_defaults(function=feature_unit)
    fit = commands.add_parser("fit-symbol")
    fit.add_argument("--symbol", required=True, choices=SYMBOLS)
    fit.add_argument("--forecast-root", required=True)
    fit.add_argument("--output-root", required=True)
    fit.set_defaults(function=fit_symbol)
    collect = commands.add_parser("aggregate")
    collect.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    collect.set_defaults(function=aggregate)
    return value


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.function(parsed)
