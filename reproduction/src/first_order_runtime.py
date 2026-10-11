"""Phase 04 first-order factorized scoring runtime.

This module only reuses an already completed Phase 03 K=1 ordered-TCN.  It never
fits model weights and never writes into Phase 03 or the immutable upstream tree.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from src.cuda_runtime import UPSTREAM, configure_determinism, device_or_raise

DATES = (
    "2026-07-02",
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-13",
    "2026-07-14",
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
)
CALIBRATION_DATES = DATES[9:12]
TEST_DATES = DATES[12:]
K_VALUES = (1, 2, 6, 12)
SEEDS = (20260730, 20260731, 20260732)
EXPECTED_PATHS = {
    1: {"calibration": 671_864, "test": 582_772},
    2: {"calibration": 671_863, "test": 582_772},
    6: {"calibration": 671_859, "test": 582_767},
    12: {"calibration": 671_853, "test": 582_761},
}
BOUND = float(np.log(99.0))

SCORE_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("source_date", pa.string()),
        ("start_block_index", pa.uint32()),
        ("path_start_ns", pa.int64()),
        ("path_end_ns", pa.int64()),
        ("path_event_count", pa.int64()),
        ("duration_seconds", pa.float64()),
        ("target_b", pa.int16()),
        ("k", pa.int16()),
        ("seed", pa.int64()),
        ("raw_factorized_score", pa.float64()),
        ("local_alpha_score", pa.float64()),
        ("path_alpha_score", pa.float64()),
    ],
    metadata={
        b"schema_contract": b"first_order_factorized_scores_v1",
        b"orientation": b"forward_only_reverse_is_exact_negative",
    },
)


class ForcedResumeProbe(RuntimeError):
    """Expected one-time interruption used by the K12 throughput canary."""


RAW_PART_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("source_date", pa.string()),
        ("start_block_index", pa.uint32()),
        ("path_start_ns", pa.int64()),
        ("path_end_ns", pa.int64()),
        ("path_event_count", pa.int64()),
        ("duration_seconds", pa.float64()),
        ("target_b", pa.int16()),
        ("k", pa.int16()),
        ("seed", pa.int64()),
        ("raw_factorized_score", pa.float64()),
    ]
)


@dataclass(frozen=True)
class FirstOrderConfig:
    k: int
    seed: int
    device: str
    input_root: Path
    phase03_root: Path
    anchor_root: Path
    output_root: Path
    work_root: Path
    log_root: Path
    hourly_cost_usd: float = 0.69
    batch_size: int = 1_024
    resume: bool = True
    force_resume_probe: bool = False

    @property
    def unit_id(self) -> str:
        return f"K={self.k}/seed={self.seed}"

    @property
    def local_unit(self) -> Path:
        return (
            self.phase03_root
            / "units"
            / "B=10_K=1"
            / "ordered_tcn"
            / f"seed={self.seed}"
        )

    @property
    def comparison_unit(self) -> Path:
        if self.k == 6:
            return self.anchor_root / "ordered_tcn" / f"seed={self.seed}"
        return (
            self.phase03_root
            / "units"
            / f"B=10_K={self.k}"
            / "ordered_tcn"
            / f"seed={self.seed}"
        )

    @property
    def comparison_reference(self) -> str:
        if self.k == 6:
            return f"phase03_compact:B=10/K=6/model=ordered_tcn/seed={self.seed}"
        return str(self.comparison_unit)

    def scientific_config(self) -> dict[str, Any]:
        value = {
            "stage": "phase04_first_order_h1",
            "B": 10,
            "K": self.k,
            "seed": self.seed,
            "local_model": "phase03_B10_K1_ordered_tcn_same_seed",
            "calibration_modes": ["local_alpha_sum", "path_recalibrated"],
            "score_contracts": ["uncapped", "bounded_log99"],
            "calibration_dates": list(CALIBRATION_DATES),
            "test_dates": list(TEST_DATES),
            "path_stride": 3,
            "path_offset_rule": "date.toordinal()%3",
            "feature_projection": "continuous_path_37_v1",
            "local_scoring_contract": (
                "k1_selected_starts_v1" if self.k == 1 else "kgt1_all_local_blocks_v1"
            ),
            "batch_size": self.batch_size,
            "fp32": True,
            "tf32": False,
            "amp": False,
            "new_training": False,
            "new_reversible_null": False,
            "force_resume_probe": self.force_resume_probe,
        }
        return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table = table.cast(schema)
    table = table.replace_schema_metadata(schema.metadata)
    pq.write_table(table, temporary, compression="zstd", row_group_size=65_536)
    observed = pq.ParquetFile(temporary)
    if observed.metadata.num_rows != len(table) or not observed.schema_arrow.equals(
        schema, check_metadata=True
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"atomic parquet validation failed: {path}")
    os.replace(temporary, path)


def json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def factorize_local_scores(
    local_scores: np.ndarray, starts: np.ndarray, k: int
) -> np.ndarray:
    """Sum consecutive local scores without materializing K-window tensors."""
    scores = np.asarray(local_scores, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if k < 1 or scores.ndim != 1 or starts.ndim != 1:
        raise ValueError("invalid factorization shape/K")
    if len(starts) and (starts.min() < 0 or starts.max() + k > len(scores)):
        raise ValueError("factorized start outside local-score support")
    prefix = np.empty(len(scores) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(scores, out=prefix[1:])
    return prefix[starts + k] - prefix[starts]


def fit_path_alpha(
    scores_by_day: dict[str, np.ndarray],
) -> tuple[float, dict[str, Any]]:
    return UPSTREAM.fit_positive_day_equal_calibration(scores_by_day)


def metric_for_scores(
    score: np.ndarray,
    event_count: np.ndarray,
    duration_seconds: np.ndarray,
    k: int,
) -> dict[str, Any]:
    accumulator = UPSTREAM._metric_accumulator()
    UPSTREAM._update_metrics(
        accumulator,
        np.asarray(score, dtype=np.float64),
        np.asarray(event_count, dtype=np.float64),
        np.asarray(duration_seconds, dtype=np.float64),
    )
    return UPSTREAM._finalize_metrics(accumulator, k)


def load_local_reference(
    config: FirstOrderConfig, device: torch.device
) -> tuple[Any, Any, float, dict[str, Any]]:
    root = config.local_unit
    required = ("model.pt", "scaler.json", "calibration.json", "complete.json")
    if any(not (root / name).is_file() for name in required):
        raise FileNotFoundError(f"incomplete Phase 03 K1 reference: {root}")
    completion = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or completion.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError(f"Phase 03 K1 reference did not pass: {root}")
    scaler = UPSTREAM.ChannelScaler.from_dict(
        json.loads((root / "scaler.json").read_text(encoding="utf-8"))
    )
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    alpha = float(calibration["alpha"])
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    if (
        payload.get("model") != "ordered_tcn"
        or int(payload.get("seed", -1)) != config.seed
        or int(payload.get("B", -1)) != 10
        or int(payload.get("K", -1)) != 1
    ):
        raise RuntimeError(f"Phase 03 K1 model identity mismatch: {root}")
    model = UPSTREAM.build_model("ordered_tcn")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    reference = {
        "path": str(root),
        "model_bytes": (root / "model.pt").stat().st_size,
        "scaler_bytes": (root / "scaler.json").stat().st_size,
        "calibration_bytes": (root / "calibration.json").stat().st_size,
        "local_alpha": alpha,
        "model": "ordered_tcn",
        "B": 10,
        "K": 1,
        "seed": config.seed,
    }
    return model, scaler, alpha, reference


def _score_all_local_blocks(
    model: Any,
    scaler: Any,
    day: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    scores = np.empty(len(day.block_index), dtype=np.float64)
    reverse_scores = np.empty(len(day.block_index), dtype=np.float64)
    max_antisymmetry = 0.0
    involution_exact = True
    model.eval()
    with torch.no_grad():
        for first in range(0, len(scores), batch_size):
            if getattr(UPSTREAM, "STOP_REQUESTED", False):
                raise InterruptedError("checkpoint stop requested")
            selected = np.arange(first, min(first + batch_size, len(scores)))
            forward, reverse, _ = UPSTREAM.path_arrays(day, selected, 1)
            if first == 0:
                boundaries = np.empty(
                    (len(selected), 2, len(UPSTREAM.BOUNDARY_FIELDS)),
                    dtype=np.float64,
                )
                boundaries[:, 0, :] = day.starts[selected]
                boundaries[:, 1, :] = day.ends[selected]
                intervals = day.intervals[selected, None, :]
                once = UPSTREAM.reverse_continuous(boundaries, intervals)
                twice = UPSTREAM.reverse_continuous(*once)
                involution_exact = bool(
                    np.array_equal(twice[0], boundaries)
                    and np.array_equal(twice[1], intervals)
                )
            forward_tensor = torch.from_numpy(scaler.transform(forward)).to(device)
            reverse_tensor = torch.from_numpy(scaler.transform(reverse)).to(device)
            raw = UPSTREAM.antisymmetric_score(model, forward_tensor, reverse_tensor)
            reverse_raw = UPSTREAM.antisymmetric_score(
                model, reverse_tensor, forward_tensor
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            raw_np = raw.detach().cpu().numpy().astype(np.float64)
            reverse_np = reverse_raw.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(raw_np).all() or not np.isfinite(reverse_np).all():
                raise RuntimeError("nonfinite local antisymmetric score")
            max_antisymmetry = max(
                max_antisymmetry, float(np.max(np.abs(raw_np + reverse_np)))
            )
            scores[selected] = raw_np
            reverse_scores[selected] = reverse_np
    return scores, reverse_scores, max_antisymmetry, involution_exact


def _score_selected_local_blocks(
    model: Any,
    scaler: Any,
    day: Any,
    starts: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Reproduce the Phase 03 K1 selected-start CUDA batching contract."""
    selected_starts = np.asarray(starts, dtype=np.int64)
    scores = np.empty(len(selected_starts), dtype=np.float64)
    reverse_scores = np.empty(len(selected_starts), dtype=np.float64)
    max_antisymmetry = 0.0
    involution_exact = True
    model.eval()
    with torch.no_grad():
        for first in range(0, len(selected_starts), batch_size):
            if getattr(UPSTREAM, "STOP_REQUESTED", False):
                raise InterruptedError("checkpoint stop requested")
            selected = selected_starts[first : first + batch_size]
            forward, reverse, _ = UPSTREAM.path_arrays(day, selected, 1)
            if first == 0:
                boundaries = np.empty(
                    (len(selected), 2, len(UPSTREAM.BOUNDARY_FIELDS)),
                    dtype=np.float64,
                )
                boundaries[:, 0, :] = day.starts[selected]
                boundaries[:, 1, :] = day.ends[selected]
                intervals = day.intervals[selected, None, :]
                once = UPSTREAM.reverse_continuous(boundaries, intervals)
                twice = UPSTREAM.reverse_continuous(*once)
                involution_exact = bool(
                    np.array_equal(twice[0], boundaries)
                    and np.array_equal(twice[1], intervals)
                )
            forward_tensor = torch.from_numpy(scaler.transform(forward)).to(device)
            reverse_tensor = torch.from_numpy(scaler.transform(reverse)).to(device)
            raw = UPSTREAM.antisymmetric_score(model, forward_tensor, reverse_tensor)
            reverse_raw = UPSTREAM.antisymmetric_score(
                model, reverse_tensor, forward_tensor
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            raw_np = raw.detach().cpu().numpy().astype(np.float64)
            reverse_np = reverse_raw.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(raw_np).all() or not np.isfinite(reverse_np).all():
                raise RuntimeError("nonfinite selected local antisymmetric score")
            max_antisymmetry = max(
                max_antisymmetry, float(np.max(np.abs(raw_np + reverse_np)))
            )
            stop = first + len(selected)
            scores[first:stop] = raw_np
            reverse_scores[first:stop] = reverse_np
    return scores, reverse_scores, max_antisymmetry, involution_exact


def raw_day_table(
    *,
    config: FirstOrderConfig,
    model: Any,
    scaler: Any,
    source_date: str,
    device: torch.device,
) -> tuple[pa.Table, dict[str, Any]]:
    load_started = time.monotonic()
    day = UPSTREAM.read_day(config.input_root, source_date, 10)
    load_seconds = time.monotonic() - load_started
    starts = UPSTREAM.day_path_starts(day, config.k)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    score_started = time.monotonic()
    if config.k == 1:
        raw, reverse_raw, local_antisymmetry, involution = _score_selected_local_blocks(
            model, scaler, day, starts, device, config.batch_size
        )
    else:
        local_scores, reverse_local_scores, local_antisymmetry, involution = (
            _score_all_local_blocks(model, scaler, day, device, config.batch_size)
        )
        raw = factorize_local_scores(local_scores, starts, config.k)
        reverse_raw = factorize_local_scores(reverse_local_scores, starts, config.k)
    antisymmetry = float(np.max(np.abs(raw + reverse_raw)))
    score_seconds = time.monotonic() - score_started
    offsets = starts[:, None] + np.arange(config.k, dtype=np.int64)[None, :]
    events = (
        day.intervals[offsets, UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]]
        .sum(axis=1)
        .astype(np.int64)
    )
    start_ns = day.block_start_ns[starts].astype(np.int64)
    end_ns = day.block_end_ns[starts + config.k - 1].astype(np.int64)
    duration = (end_ns - start_ns).astype(np.float64) / 1e9
    if not np.isfinite(raw).all() or not np.isfinite(duration).all():
        raise RuntimeError(f"nonfinite factorized output: {source_date}")
    block_indices = day.block_index[starts].astype(np.uint32)
    pair_ids = [
        f"{source_date}:B10:K{config.k}:{int(index)}" for index in block_indices
    ]
    table = pa.Table.from_pydict(
        {
            "pair_id": pair_ids,
            "source_date": [source_date] * len(starts),
            "start_block_index": block_indices,
            "path_start_ns": start_ns,
            "path_end_ns": end_ns,
            "path_event_count": events,
            "duration_seconds": duration,
            "target_b": np.full(len(starts), 10, dtype=np.int16),
            "k": np.full(len(starts), config.k, dtype=np.int16),
            "seed": np.full(len(starts), config.seed, dtype=np.int64),
            "raw_factorized_score": raw,
        },
        schema=RAW_PART_SCHEMA,
    )
    return table, {
        "source_date": source_date,
        "rows": len(table),
        "max_antisymmetry_error": antisymmetry,
        "local_max_antisymmetry_error": local_antisymmetry,
        "reverse_involution_exact": involution,
        "first_block_index": int(block_indices[0]) if len(block_indices) else None,
        "last_block_index": int(block_indices[-1]) if len(block_indices) else None,
        "loader_seconds": load_seconds,
        "scoring_seconds": score_seconds,
        "paths_per_second": len(table) / score_seconds if score_seconds > 0 else None,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }


def validate_raw_part(
    path: Path,
    audit: dict[str, Any],
    config: FirstOrderConfig,
    source_date: str,
) -> None:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(RAW_PART_SCHEMA, check_metadata=True):
        raise RuntimeError(f"stale raw part schema: {path}")
    table = pq.read_table(path)
    day = UPSTREAM.read_day(config.input_root, source_date, 10)
    starts = UPSTREAM.day_path_starts(day, config.k)
    expected_indices = day.block_index[starts].astype(np.uint32)
    expected_pair_ids = [
        f"{source_date}:B10:K{config.k}:{int(index)}" for index in expected_indices
    ]
    score = np.asarray(table["raw_factorized_score"], dtype=np.float64)
    if (
        len(table) != len(starts)
        or int(audit.get("rows", -1)) != len(table)
        or audit.get("source_date") != source_date
        or audit.get("reverse_involution_exact") is not True
        or float(audit.get("max_antisymmetry_error", float("inf"))) >= 1e-6
        or table["source_date"].to_pylist() != [source_date] * len(table)
        or table["pair_id"].to_pylist() != expected_pair_ids
        or not np.array_equal(
            np.asarray(table["start_block_index"], dtype=np.uint32),
            expected_indices,
        )
        or not np.isfinite(score).all()
    ):
        raise RuntimeError(f"raw checkpoint part validation failed: {path}")


def _phase03_forward_batches(path: Path) -> Iterable[pa.Table]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=131_072):
        table = pa.Table.from_batches([batch])
        mask = pc.equal(table["orientation"], "forward")
        selected = table.filter(mask)
        if len(selected):
            yield selected


def validate_against_phase03(
    config: FirstOrderConfig,
    factorized_path: Path,
    local_alpha: float,
    path_alpha: float,
) -> dict[str, Any]:
    if config.k == 6:
        return validate_k6_compact_handoff(config, factorized_path)
    reference_path = config.comparison_unit / "test_predictions.parquet"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"missing Phase 03 comparison predictions: {reference_path}"
        )
    observed_file = pq.ParquetFile(factorized_path)
    expected_file = pq.ParquetFile(reference_path)
    expected_forward_rows = expected_file.metadata.num_rows // 2
    if observed_file.metadata.num_rows != expected_forward_rows:
        raise RuntimeError("Phase 03/factorized prediction row mismatch")
    identity = (
        "pair_id",
        "source_date",
        "start_block_index",
        "path_start_ns",
        "path_end_ns",
        "path_event_count",
        "duration_seconds",
        "target_b",
        "k",
    )
    max_raw = max_local = 0.0
    left = pq.read_table(factorized_path)
    right_all = pq.read_table(reference_path)
    right = right_all.filter(pc.equal(right_all["orientation"], "forward"))
    if len(left) != len(right):
        raise RuntimeError("Phase 03/factorized forward coverage mismatch")
    for name in identity:
        if not np.array_equal(np.asarray(left[name]), np.asarray(right[name])):
            raise RuntimeError(f"Phase 03 path identity mismatch: {name}")
    if len(right) and int(np.asarray(right["seed"])[0]) != config.seed:
        raise RuntimeError("Phase 03 comparison seed mismatch")
    if config.k == 1:
        raw = np.asarray(left["raw_factorized_score"], dtype=np.float64)
        expected_raw = np.asarray(right["raw_antisymmetric_score"], dtype=np.float64)
        local = np.asarray(left["local_alpha_score"], dtype=np.float64)
        expected_local = np.asarray(
            right["calibrated_antisymmetric_score"], dtype=np.float64
        )
        max_raw = float(np.max(np.abs(raw - expected_raw)))
        max_local = float(np.max(np.abs(local - expected_local)))
    rows = len(left)
    result = {
        "path_identity_exact": True,
        "rows": rows,
        "phase03_prediction": str(reference_path),
        "k1_raw_max_abs_error": max_raw if config.k == 1 else None,
        "k1_local_alpha_max_abs_error": max_local if config.k == 1 else None,
        "k1_alpha_abs_error": abs(local_alpha - path_alpha) if config.k == 1 else None,
    }
    if config.k == 1 and (
        max_raw >= 1e-6 or max_local >= 1e-6 or abs(local_alpha - path_alpha) >= 1e-8
    ):
        raise RuntimeError(f"K1 identity gate failed: {result}")
    return result


def validate_k6_compact_handoff(
    config: FirstOrderConfig, factorized_path: Path
) -> dict[str, Any]:
    """Validate K6 without copying the immutable Phase 03 prediction tree."""
    table = pq.read_table(factorized_path)
    expected_rows = EXPECTED_PATHS[6]["test"]
    if len(table) != expected_rows:
        raise RuntimeError("K6 compact-handoff row mismatch")
    offset = 0
    event_index = UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]
    for source_date in TEST_DATES:
        day = UPSTREAM.read_day(config.input_root, source_date, 10)
        starts = UPSTREAM.day_path_starts(day, 6)
        stop = offset + len(starts)
        observed = table.slice(offset, len(starts))
        offsets = starts[:, None] + np.arange(6, dtype=np.int64)[None, :]
        events = day.intervals[offsets, event_index].sum(axis=1).astype(np.int64)
        start_ns = day.block_start_ns[starts].astype(np.int64)
        end_ns = day.block_end_ns[starts + 5].astype(np.int64)
        duration = (end_ns - start_ns).astype(np.float64) / 1e9
        block_indices = day.block_index[starts].astype(np.uint32)
        pair_ids = [f"{source_date}:B10:K6:{int(index)}" for index in block_indices]
        exact = (
            observed["source_date"].to_pylist() == [source_date] * len(starts)
            and observed["pair_id"].to_pylist() == pair_ids
            and np.array_equal(
                np.asarray(observed["start_block_index"], dtype=np.uint32),
                block_indices,
            )
            and np.array_equal(
                np.asarray(observed["path_start_ns"], dtype=np.int64), start_ns
            )
            and np.array_equal(
                np.asarray(observed["path_end_ns"], dtype=np.int64), end_ns
            )
            and np.array_equal(
                np.asarray(observed["path_event_count"], dtype=np.int64), events
            )
            and np.array_equal(
                np.asarray(observed["duration_seconds"], dtype=np.float64), duration
            )
            and np.array_equal(
                np.asarray(observed["target_b"], dtype=np.int16),
                np.full(len(starts), 10, dtype=np.int16),
            )
            and np.array_equal(
                np.asarray(observed["k"], dtype=np.int16),
                np.full(len(starts), 6, dtype=np.int16),
            )
            and np.array_equal(
                np.asarray(observed["seed"], dtype=np.int64),
                np.full(len(starts), config.seed, dtype=np.int64),
            )
        )
        if not exact:
            raise RuntimeError(
                f"K6 deterministic input identity mismatch: {source_date}"
            )
        offset = stop
    if offset != len(table):
        raise RuntimeError("K6 deterministic input coverage mismatch")

    audit_path = config.phase03_root / "tables" / "support_audits.parquet"
    audits = pq.read_table(audit_path).to_pylist()
    matched = [
        row
        for row in audits
        if int(row["B"]) == 10
        and int(row["K"]) == 6
        and row["model"] == "ordered_tcn"
        and int(row["seed"]) == config.seed
    ]
    if len(matched) != 1:
        raise RuntimeError("K6 Phase 03 compact audit key mismatch")
    audit = matched[0]
    if (
        audit["completion_status"] != "complete"
        or audit["scientific_audit_passed"] is not True
        or audit["null_audit_passed"] is not True
        or audit["null_bias_passed"] is not True
        or int(audit["test_prediction_rows"]) != 2 * expected_rows
    ):
        raise RuntimeError("K6 Phase 03 compact audit failed")
    return {
        "path_identity_exact": True,
        "rows": len(table),
        "phase03_prediction": None,
        "phase03_compact_run_unit_id": audit["run_unit_id"],
        "path_identity_basis": (
            "verified_input_reconstruction_plus_phase03_compact_audit"
        ),
        "k1_raw_max_abs_error": None,
        "k1_local_alpha_max_abs_error": None,
        "k1_alpha_abs_error": None,
    }


def _read_raw_parts(paths: Iterable[Path]) -> pa.Table:
    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise RuntimeError("no raw score parts")
    return pa.concat_tables(tables, promote_options="default")


def _metrics_from_final(path: Path, k: int) -> dict[str, Any]:
    table = pq.read_table(path)
    by_mode: dict[str, Any] = {}
    for mode, field in (
        ("local_alpha_sum", "local_alpha_score"),
        ("path_recalibrated", "path_alpha_score"),
    ):
        by_contract: dict[str, Any] = {}
        score_all = np.asarray(table[field], dtype=np.float64)
        for contract, scores in (
            ("uncapped", score_all),
            ("bounded_log99", np.clip(score_all, -BOUND, BOUND)),
        ):
            by_day = {}
            for source_date in TEST_DATES:
                mask = np.asarray(pc.equal(table["source_date"], source_date))
                by_day[source_date] = metric_for_scores(
                    scores[mask],
                    np.asarray(table["path_event_count"])[mask],
                    np.asarray(table["duration_seconds"])[mask],
                    k,
                )
            by_contract[contract] = {"by_day": by_day}
        by_mode[mode] = by_contract
    return by_mode


def clean_resume_equivalence_probe(
    *,
    config: FirstOrderConfig,
    model: Any,
    scaler: Any,
    device: torch.device,
    parts: Path,
    local_alpha: float,
    path_alpha: float,
    final_path: Path,
) -> dict[str, Any]:
    """Recompute the entire canary without checkpoint reuse and compare exactly."""
    clean_tables: dict[str, pa.Table] = {}
    compared_rows = 0
    for source_date in CALIBRATION_DATES + TEST_DATES:
        clean, _audit = raw_day_table(
            config=config,
            model=model,
            scaler=scaler,
            source_date=source_date,
            device=device,
        )
        split = "calibration" if source_date in CALIBRATION_DATES else "test"
        resumed = pq.read_table(parts / f"{split}_{source_date}.parquet")
        if not clean.equals(resumed, check_metadata=True):
            raise RuntimeError(
                f"forced-resume raw part differs from clean recompute: {source_date}"
            )
        clean_tables[source_date] = clean
        compared_rows += len(clean)
    clean_scores = {
        source_date: np.asarray(
            clean_tables[source_date]["raw_factorized_score"], dtype=np.float64
        )
        for source_date in CALIBRATION_DATES
    }
    clean_alpha, _clean_calibration = fit_path_alpha(clean_scores)
    if clean_alpha != path_alpha:
        raise RuntimeError("forced-resume path alpha differs from clean recompute")
    clean_test_raw = pa.concat_tables(
        [clean_tables[source_date] for source_date in TEST_DATES],
        promote_options="default",
    )
    raw = np.asarray(clean_test_raw["raw_factorized_score"], dtype=np.float64)
    clean_final = (
        clean_test_raw.append_column(
            "local_alpha_score", pa.array(local_alpha * raw, type=pa.float64())
        )
        .append_column(
            "path_alpha_score", pa.array(clean_alpha * raw, type=pa.float64())
        )
        .cast(SCORE_SCHEMA)
        .replace_schema_metadata(SCORE_SCHEMA.metadata)
    )
    resumed_final = pq.read_table(final_path)
    if not clean_final.equals(resumed_final, check_metadata=True):
        raise RuntimeError("forced-resume final output differs from clean recompute")
    return {
        "passed": True,
        "compared_parts": 6,
        "compared_rows": compared_rows,
        "path_alpha_exact": True,
        "final_output_exact": True,
    }


def validate_complete(final: Path, config: FirstOrderConfig) -> dict[str, Any]:
    complete_path = final / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing completion record: {final}")
    value = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config") != config.scientific_config()
        or value.get("input", {}).get("root") != str(config.input_root)
        or value.get("references", {}).get("comparison") != config.comparison_reference
        or value.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError(f"completion config/audit mismatch: {final}")
    required = {
        "local_reference.json",
        "path_calibration.json",
        "test_factorized_scores.parquet",
        "metrics.json",
    }
    artifacts = value.get("artifacts", [])
    if {row.get("filename") for row in artifacts} != required:
        raise RuntimeError(f"completion artifact inventory mismatch: {final}")
    for row in artifacts:
        path = final / row["filename"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"completion artifact size mismatch: {path}")
        if row["filename"].endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != row[
                "rows"
            ] or not parquet.schema_arrow.equals(SCORE_SCHEMA, check_metadata=True):
                raise RuntimeError(f"completion parquet mismatch: {path}")

    def finite_tree(item: Any) -> bool:
        if isinstance(item, float):
            return bool(np.isfinite(item))
        if isinstance(item, dict):
            return all(finite_tree(value) for value in item.values())
        if isinstance(item, list):
            return all(finite_tree(value) for value in item)
        return True

    for filename in ("path_calibration.json", "metrics.json"):
        if not finite_tree(json.loads((final / filename).read_text(encoding="utf-8"))):
            raise RuntimeError(
                f"completion contains nonfinite numeric value: {filename}"
            )
    return value


def run_first_order_unit(config: FirstOrderConfig) -> dict[str, Any]:
    if config.k not in K_VALUES or config.seed not in SEEDS:
        raise ValueError("unit outside approved Phase 04 H1 scope")
    configure_determinism()
    device = device_or_raise(config.device)
    if device.type != "cuda":
        raise ValueError("Phase 04 production scoring requires qualified CUDA")
    final = config.output_root / config.unit_id
    work = config.work_root / config.unit_id
    if final.exists():
        if config.resume:
            return validate_complete(final, config)
        raise FileExistsError(final)
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / ".unit.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"unit already has a writer: {config.unit_id}") from error
    started = time.monotonic()
    runtime_path = work / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {"attempts": 0, "elapsed_seconds": 0.0, "measured_cost_usd": 0.0}
    )
    succeeded = False
    try:
        model, scaler, local_alpha, reference = load_local_reference(config, device)
        input_files = []
        for source_date in CALIBRATION_DATES + TEST_DATES:
            input_files.extend(
                UPSTREAM.partition_paths(config.input_root, source_date, 10)
            )
        if not input_files:
            raise FileNotFoundError(f"missing B10 input: {config.input_root}")
        exact = {
            "scientific_config": config.scientific_config(),
            "input": {
                "root": str(config.input_root),
                "files": len(input_files),
                "bytes": sum(path.stat().st_size for path in input_files),
            },
            "local_reference": reference,
            "comparison_path": config.comparison_reference,
        }
        config_path = work / "config.json"
        if config_path.is_file():
            if json.loads(config_path.read_text(encoding="utf-8")) != json_normalized(
                exact
            ):
                raise RuntimeError(f"stale Phase 04 checkpoint config: {work}")
        else:
            atomic_json(config_path, exact)
        staging = work / "staging"
        if (staging / "complete.json").is_file():
            validate_complete(staging, config)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            validated = validate_complete(final, config)
            shutil.rmtree(work)
            return validated
        parts = work / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        audits = []
        resumed_parts = 0
        forced_marker = work / "forced_resume_probe.json"
        for split, dates in (("calibration", CALIBRATION_DATES), ("test", TEST_DATES)):
            for source_date in dates:
                part = parts / f"{split}_{source_date}.parquet"
                audit_path = part.with_suffix(".audit.json")
                if part.is_file() and audit_path.is_file():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    resumed_parts += 1
                else:
                    table, audit = raw_day_table(
                        config=config,
                        model=model,
                        scaler=scaler,
                        source_date=source_date,
                        device=device,
                    )
                    atomic_parquet(part, table, RAW_PART_SCHEMA)
                    atomic_json(audit_path, audit)
                validate_raw_part(part, audit, config, source_date)
                audits.append({"split": split, **audit})
                checkpoint = {
                    "scientific_config": config.scientific_config(),
                    "completed_parts": sorted(
                        path.name for path in parts.glob("*.parquet")
                    ),
                    "last_source_date": source_date,
                    "last_block_index": audit.get("last_block_index"),
                }
                atomic_json(work / "checkpoint.json", checkpoint)
                if config.force_resume_probe and not forced_marker.is_file():
                    atomic_json(
                        forced_marker,
                        {
                            "status": "interruption_requested",
                            "completed_part": part.name,
                            "rows": audit["rows"],
                            "requested_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    raise ForcedResumeProbe(
                        "intentional K12 canary interruption after durable part"
                    )
        expected_cal = EXPECTED_PATHS[config.k]["calibration"]
        expected_test = EXPECTED_PATHS[config.k]["test"]
        cal_paths = [parts / f"calibration_{day}.parquet" for day in CALIBRATION_DATES]
        test_paths = [parts / f"test_{day}.parquet" for day in TEST_DATES]
        if (
            sum(pq.ParquetFile(path).metadata.num_rows for path in cal_paths)
            != expected_cal
        ):
            raise RuntimeError("calibration path count mismatch")
        if (
            sum(pq.ParquetFile(path).metadata.num_rows for path in test_paths)
            != expected_test
        ):
            raise RuntimeError("test path count mismatch")
        scores_by_day = {
            day: np.asarray(
                pq.read_table(parts / f"calibration_{day}.parquet")[
                    "raw_factorized_score"
                ],
                dtype=np.float64,
            )
            for day in CALIBRATION_DATES
        }
        path_alpha, path_calibration = fit_path_alpha(scores_by_day)
        staging.mkdir(parents=True, exist_ok=True)
        test_raw = _read_raw_parts(test_paths)
        raw = np.asarray(test_raw["raw_factorized_score"], dtype=np.float64)
        final_table = test_raw.append_column(
            "local_alpha_score", pa.array(local_alpha * raw, type=pa.float64())
        ).append_column(
            "path_alpha_score", pa.array(path_alpha * raw, type=pa.float64())
        )
        atomic_parquet(
            staging / "test_factorized_scores.parquet", final_table, SCORE_SCHEMA
        )
        identity = validate_against_phase03(
            config, staging / "test_factorized_scores.parquet", local_alpha, path_alpha
        )
        max_antisymmetry = max(float(row["max_antisymmetry_error"]) for row in audits)
        involution = all(bool(row["reverse_involution_exact"]) for row in audits)
        test_metrics = _metrics_from_final(
            staging / "test_factorized_scores.parquet", config.k
        )
        k1_mode_metric_max_abs = 0.0
        if config.k == 1:
            for contract in ("uncapped", "bounded_log99"):
                for source_date in TEST_DATES:
                    local_row = test_metrics["local_alpha_sum"][contract]["by_day"][
                        source_date
                    ]
                    path_row = test_metrics["path_recalibrated"][contract]["by_day"][
                        source_date
                    ]
                    for field in (
                        "nats_path",
                        "nats_block",
                        "nats_source_event",
                        "nats_second",
                        "paired_log_loss",
                        "posterior_saturation_share",
                        "reverse_like_share",
                        "calibrated_ovl",
                    ):
                        k1_mode_metric_max_abs = max(
                            k1_mode_metric_max_abs,
                            abs(float(local_row[field]) - float(path_row[field])),
                        )
        resume_equivalence = None
        if config.force_resume_probe:
            marker_valid = bool(
                forced_marker.is_file()
                and resumed_parts >= 1
                and json.loads(forced_marker.read_text(encoding="utf-8")).get("status")
                == "interruption_requested"
            )
            if marker_valid:
                resume_equivalence = clean_resume_equivalence_probe(
                    config=config,
                    model=model,
                    scaler=scaler,
                    device=device,
                    parts=parts,
                    local_alpha=local_alpha,
                    path_alpha=path_alpha,
                    final_path=staging / "test_factorized_scores.parquet",
                )
            forced_resume_exact = bool(
                marker_valid
                and resume_equivalence is not None
                and resume_equivalence.get("passed") is True
            )
        else:
            forced_resume_exact = True
        audit = {
            "passed": bool(
                identity["path_identity_exact"]
                and max_antisymmetry < 1e-6
                and involution
                and np.isfinite(local_alpha)
                and np.isfinite(path_alpha)
                and (config.k != 1 or k1_mode_metric_max_abs < 1e-8)
                and forced_resume_exact
            ),
            "path_identity": identity,
            "factorized_score_antisymmetry_max_abs_error": max_antisymmetry,
            "reverse_involution_exact": involution,
            "finite": bool(np.isfinite(raw).all()),
            "k1_mode_metric_max_abs_error": (
                k1_mode_metric_max_abs if config.k == 1 else None
            ),
            "forced_resume_exact": forced_resume_exact,
            "resume_equivalence": resume_equivalence,
            "resumed_parts": resumed_parts,
            "day_parts": audits,
        }
        if not audit["passed"]:
            raise RuntimeError(f"Phase 04 scientific audit failed: {audit}")
        atomic_json(staging / "local_reference.json", reference)
        atomic_json(staging / "path_calibration.json", path_calibration)
        metrics = {
            "B": 10,
            "K": config.k,
            "seed": config.seed,
            "local_alpha": local_alpha,
            "path_alpha": path_alpha,
            "test": test_metrics,
            "audit": audit,
            "new_training": False,
            "new_reversible_null": False,
        }
        atomic_json(staging / "metrics.json", metrics)
        elapsed = time.monotonic() - started
        cumulative_elapsed = float(runtime["elapsed_seconds"]) + elapsed
        cumulative_cost = (
            float(runtime["measured_cost_usd"])
            + elapsed * config.hourly_cost_usd / 3600.0
        )
        artifacts = []
        for path in sorted(staging.iterdir()):
            if not path.is_file() or path.name == "complete.json":
                continue
            row = {"filename": path.name, "bytes": path.stat().st_size}
            if path.suffix == ".parquet":
                row["rows"] = pq.ParquetFile(path).metadata.num_rows
            artifacts.append(row)
        completion = {
            "status": "complete",
            "unit_id": config.unit_id,
            "scientific_config": config.scientific_config(),
            "input": exact["input"],
            "references": {
                "local": reference,
                "comparison": config.comparison_reference,
            },
            "audit": audit,
            "runtime": {
                "attempts": int(runtime["attempts"]) + 1,
                "elapsed_seconds": cumulative_elapsed,
                "hourly_cost_usd": config.hourly_cost_usd,
                "measured_cost_usd": cumulative_cost,
            },
            "artifacts": artifacts,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(staging / "complete.json", completion)
        validate_complete(staging, config)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        validated = validate_complete(final, config)
        shutil.rmtree(work)
        return validated
    finally:
        if not succeeded and work.exists():
            elapsed = time.monotonic() - started
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": float(runtime["elapsed_seconds"]) + elapsed,
                    "measured_cost_usd": float(runtime["measured_cost_usd"])
                    + elapsed * config.hourly_cost_usd / 3600.0,
                },
            )
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()


def synthetic_smoke() -> dict[str, Any]:
    local = np.asarray([0.5, -0.25, 1.0, 0.75, -0.5], dtype=np.float64)
    starts = np.asarray([0, 2], dtype=np.int64)
    observed = factorize_local_scores(local, starts, 2)
    expected = np.asarray([0.25, 1.75])
    if not np.array_equal(observed, expected):
        raise RuntimeError("synthetic factorization failed")
    scores = {
        "d1": np.asarray([0.1, -0.2, 0.3]),
        "d2": np.asarray([0.4, 0.2, -0.1]),
        "d3": np.asarray([0.3, 0.1, 0.2]),
    }
    alpha, calibration = fit_path_alpha(scores)
    metric = metric_for_scores(
        np.clip(alpha * observed, -BOUND, BOUND),
        np.asarray([20, 20]),
        np.asarray([0.5, 0.75]),
        2,
    )
    if not np.isfinite(alpha) or not all(
        np.isfinite(float(metric[key]))
        for key in ("nats_path", "nats_block", "nats_source_event", "nats_second")
    ):
        raise RuntimeError("synthetic calibration/metric failed")
    with tempfile.TemporaryDirectory(prefix="phase04_smoke_") as folder:
        path = Path(folder) / "scores.parquet"
        table = pa.Table.from_pylist(
            [
                {
                    "pair_id": f"d1:B10:K2:{index}",
                    "source_date": "2026-07-21",
                    "start_block_index": index,
                    "path_start_ns": index * 100,
                    "path_end_ns": index * 100 + 50,
                    "path_event_count": 20,
                    "duration_seconds": 0.5,
                    "target_b": 10,
                    "k": 2,
                    "seed": 20260730,
                    "raw_factorized_score": float(raw),
                    "local_alpha_score": float(alpha * raw),
                    "path_alpha_score": float(alpha * raw),
                }
                for index, raw in enumerate(observed)
            ],
            schema=SCORE_SCHEMA,
        )
        atomic_parquet(path, table, SCORE_SCHEMA)
        parquet = pq.ParquetFile(path)
        artifact_atomic = bool(
            parquet.metadata.num_rows == 2
            and parquet.schema_arrow.equals(SCORE_SCHEMA, check_metadata=True)
        )
        if not artifact_atomic:
            raise RuntimeError("synthetic atomic artifact failed")
    return {
        "status": "passed",
        "factorized_scores": observed.tolist(),
        "alpha": alpha,
        "calibration_days": sorted(calibration["by_day"]),
        "bounded_nats_event": metric["nats_source_event"],
        "atomic_artifact_schema": artifact_atomic,
    }
