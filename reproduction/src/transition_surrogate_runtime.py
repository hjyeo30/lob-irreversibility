"""Phase 04 H2 transition-count surrogate construction and scoring runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import psutil

from src.cuda_runtime import configure_determinism, device_or_raise
from src.first_order_runtime import (
    BOUND,
    DATES,
    EXPECTED_PATHS,
    UPSTREAM,
    atomic_json,
    atomic_parquet,
)

DEVELOPMENT_DATES = DATES[:9]
TEST_DATES = DATES[12:]
H2_K_VALUES = (2, 6, 12)
MODEL_SEEDS = (20260730, 20260731, 20260732)
SURROGATE_SEEDS = tuple(range(20260800, 20260820))
STATE_AXES = (
    "spread",
    "depth",
    "imbalance",
    "recent_activity",
    "recent_volatility",
    "recent_duration",
)
CONSTRUCTION_CHECKPOINT_EDGES = 250_000
PERMUTATION_ROW_GROUP = 65_536
SCORING_BATCH_SIZE = 1_024
SCORING_CHECKPOINT_PATHS = 131_072
NY = ZoneInfo("America/New_York")
_BLOCK_ID_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _strict_schema(
    fields: Sequence[tuple[str, pa.DataType]], nullable: Sequence[str] = ()
) -> pa.Schema:
    allowed = set(nullable)
    return pa.schema(
        [pa.field(name, dtype, nullable=name in allowed) for name, dtype in fields]
    )


PERMUTATION_SCHEMA = _strict_schema(
    [
        ("source_date", pa.string()),
        ("segment_id", pa.uint32()),
        ("surrogate_position", pa.uint32()),
        ("source_block_index", pa.uint32()),
        ("coarse_from_state", pa.uint32()),
        ("coarse_to_state", pa.uint32()),
    ]
)

SURROGATE_METRIC_SCHEMA = _strict_schema(
    [
        ("stage", pa.string()),
        ("symbol", pa.string()),
        ("source_date", pa.string()),
        ("train_regime", pa.string()),
        ("test_regime", pa.string()),
        ("B", pa.int16()),
        ("K", pa.int16()),
        ("model", pa.string()),
        ("feature_config", pa.string()),
        ("seed", pa.int64()),
        ("surrogate_seed", pa.int64()),
        ("score_contract", pa.string()),
        ("calibration_mode", pa.string()),
        ("run_unit_id", pa.string()),
        ("n_paths", pa.int64()),
        ("realized_source_events", pa.int64()),
        ("elapsed_seconds", pa.float64()),
        ("nats_path", pa.float64()),
        ("nats_block", pa.float64()),
        ("nats_event", pa.float64()),
        ("nats_second", pa.float64()),
        ("paired_log_loss", pa.float64()),
        ("calibration_alpha", pa.float64()),
        ("posterior_saturation_share", pa.float64()),
        ("calibrated_ovl", pa.float64()),
        ("reverse_like_share", pa.float64()),
    ]
)

SURROGATE_CONTRAST_SCHEMA = _strict_schema(
    [
        ("B", pa.int16()),
        ("K", pa.int16()),
        ("seed", pa.int64()),
        ("surrogate_seed", pa.int64()),
        ("source_date", pa.string()),
        ("score_contract", pa.string()),
        ("contrast", pa.string()),
        ("real_n_paths", pa.int64()),
        ("surrogate_n_paths", pa.int64()),
        ("real_realized_source_events", pa.int64()),
        ("surrogate_realized_source_events", pa.int64()),
        ("nats_path", pa.float64()),
        ("nats_block", pa.float64()),
        ("nats_event", pa.float64()),
        ("nats_second", pa.float64()),
        ("positive_nats_event", pa.bool_()),
        ("real_run_unit_id", pa.string()),
        ("surrogate_run_unit_id", pa.string()),
    ]
)

LINKAGE_AUDIT_SCHEMA = _strict_schema(
    [
        ("source_date", pa.string()),
        ("status", pa.string()),
        ("surrogate_seed", pa.int64()),
        ("n_segments", pa.int64()),
        ("n_edges", pa.int64()),
        ("n_states", pa.int64()),
        ("n_directed_transitions", pa.int64()),
        ("occupancy_max_abs_difference", pa.int64()),
        ("transition_max_abs_difference", pa.int64()),
        ("current_max_abs_difference", pa.int64()),
        ("eligible_links", pa.int64()),
        ("fixed_links", pa.int64()),
        ("endpoint_excluded_links", pa.int64()),
        ("changed_links", pa.int64()),
        ("K2_paths", pa.int64()),
        ("K6_paths", pa.int64()),
        ("K12_paths", pa.int64()),
        ("changed_linkage_share", pa.float64()),
        ("pooled_changed_linkage_share", pa.float64()),
        ("adjacent_exact_boundary_share", pa.float64()),
        ("edge_multiset_exact", pa.bool_()),
        ("payload_exact", pa.bool_()),
        ("endpoints_exact", pa.bool_()),
        ("consumed_once", pa.bool_()),
        ("passed", pa.bool_()),
    ]
)

ONE_STEP_SCHEMA = _strict_schema(
    [
        ("source_date", pa.string()),
        ("status", pa.string()),
        ("classifier_status", pa.string()),
        ("surrogate_seed", pa.int64()),
        ("edge_rows", pa.int64()),
        ("realized_source_events", pa.int64()),
        ("k1_score_max_abs_error", pa.float64()),
        ("d1_uncapped_abs_error", pa.float64()),
        ("d1_bounded_abs_error", pa.float64()),
        ("classifier_balanced_accuracy", pa.float64()),
        ("classifier_roc_auc", pa.float64()),
        ("classifier_pooled_balanced_accuracy", pa.float64()),
        ("classifier_pooled_roc_auc", pa.float64()),
        ("token_identity_exact", pa.bool_()),
        ("passed", pa.bool_()),
    ],
    nullable=(
        "classifier_balanced_accuracy",
        "classifier_roc_auc",
        "classifier_pooled_balanced_accuracy",
        "classifier_pooled_roc_auc",
    ),
)

H2_LINEAGE_SCHEMA = _strict_schema(
    [
        ("source", pa.string()),
        ("B", pa.int16()),
        ("K", pa.int16()),
        ("path", pa.string()),
        ("files", pa.int64()),
        ("rows", pa.int64()),
        ("bytes", pa.int64()),
        ("status", pa.string()),
    ]
)

H2_AXIS_SCHEMA = _strict_schema(
    [
        ("K", pa.int16()),
        ("gate", pa.string()),
        ("status", pa.string()),
        ("threshold_json", pa.string()),
        ("observed_json", pa.string()),
        ("decision", pa.string()),
    ]
)

K6_PREDICTION_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("source_date", pa.string()),
        ("start_block_index", pa.int64()),
        ("path_start_ns", pa.int64()),
        ("path_end_ns", pa.int64()),
        ("path_event_count", pa.int64()),
        ("duration_seconds", pa.float64()),
        ("target_b", pa.int16()),
        ("k", pa.int16()),
        ("model", pa.string()),
        ("seed", pa.int64()),
        ("orientation", pa.string()),
        ("label", pa.uint8()),
        ("raw_antisymmetric_score", pa.float64()),
        ("calibrated_antisymmetric_score", pa.float64()),
        ("probability_forward", pa.float64()),
    ]
)


@dataclass
class SegmentableDay:
    source_date: str
    block_index: np.ndarray
    block_start_ns: np.ndarray
    block_end_ns: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    intervals: np.ndarray
    segments: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CoarseStateDefinition:
    cutpoints: dict[str, tuple[float, float]]
    development_dates: tuple[str, ...] = DEVELOPMENT_DATES
    quantile_method: str = "linear"
    bin_side: str = "right"
    clock_boundaries_seconds: tuple[int, int] = (
        11 * 3600 + 50 * 60,
        13 * 3600 + 40 * 60,
    )
    recent_sentinel: int = 3
    state_contract: str = "phase04_h2_coarse_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cutpoints": {key: list(value) for key, value in self.cutpoints.items()},
            "development_dates": list(self.development_dates),
            "clock_boundaries_seconds": list(self.clock_boundaries_seconds),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CoarseStateDefinition":
        return cls(
            cutpoints={
                key: tuple(map(float, pair)) for key, pair in value["cutpoints"].items()
            },
            development_dates=tuple(value["development_dates"]),
            quantile_method=value["quantile_method"],
            bin_side=value["bin_side"],
            clock_boundaries_seconds=tuple(map(int, value["clock_boundaries_seconds"])),
            recent_sentinel=int(value["recent_sentinel"]),
            state_contract=value["state_contract"],
        )


@dataclass(frozen=True)
class ConstructionConfig:
    surrogate_seed: int
    source_date: str
    input_root: Path
    state_path: Path
    output_root: Path
    work_root: Path
    resume: bool = True
    checkpoint_edges: int = CONSTRUCTION_CHECKPOINT_EDGES
    row_group_size: int = PERMUTATION_ROW_GROUP
    interrupt_after_edges: int | None = None
    failure_sentinel: Path | None = None

    @property
    def unit_id(self) -> str:
        return f"replicate={self.surrogate_seed}/source_date={self.source_date}"


@dataclass(frozen=True)
class ScoringConfig:
    k: int
    model_seed: int
    surrogate_seed: int
    device: str
    input_root: Path
    permutation_root: Path
    model_root: Path
    output_root: Path
    work_root: Path
    hourly_cost_usd: float = 0.0
    resume: bool = True
    batch_size: int = SCORING_BATCH_SIZE
    checkpoint_paths: int = SCORING_CHECKPOINT_PATHS
    interrupt_after_paths: int | None = None
    failure_sentinel: Path | None = None

    @property
    def unit_id(self) -> str:
        return (
            f"K={self.k}/model_seed={self.model_seed}/replicate={self.surrogate_seed}"
        )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def validate_arrow_contract(table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=True):
        raise RuntimeError(
            f"Arrow schema mismatch observed={table.schema} expected={schema}"
        )
    nullable = {field.name for field in schema if field.nullable}
    for name in schema.names:
        if name not in nullable and table[name].null_count:
            raise RuntimeError(f"unexpected nulls in non-nullable field: {name}")


def _validate_permutation_keys(
    table: pa.Table, source_date: str, expected_block_ids: np.ndarray | None = None
) -> None:
    validate_arrow_contract(table, PERMUTATION_SCHEMA)
    if len(table) <= 0 or set(table["source_date"].to_pylist()) != {source_date}:
        raise RuntimeError("invalid H2 permutation date/rows")
    position = np.asarray(table["surrogate_position"], dtype=np.int64)
    block = np.asarray(table["source_block_index"], dtype=np.int64)
    segment = np.asarray(table["segment_id"], dtype=np.int64)
    if not np.array_equal(position, np.arange(len(table), dtype=np.int64)):
        raise RuntimeError(
            "H2 permutation position key has duplicates, gaps, or reordering"
        )
    if len(np.unique(block)) != len(table):
        raise RuntimeError(
            "H2 permutation source block key has duplicates/missing rows"
        )
    if expected_block_ids is not None:
        expected = np.asarray(expected_block_ids, dtype=np.int64)
        if len(block) != len(expected) or not np.array_equal(
            np.sort(block), np.sort(expected)
        ):
            raise RuntimeError(
                "H2 permutation source block keys do not equal the current date block set"
            )
    if len(segment) > 1 and np.any(np.diff(segment) < 0):
        raise RuntimeError("H2 permutation segment key is decreasing")


def _expected_block_ids(
    input_root: Path, source_date: str, path_resolver: Any | None = None
) -> np.ndarray:
    key = (str(input_root.resolve()), source_date)
    cached = _BLOCK_ID_CACHE.get(key)
    if cached is not None:
        return cached
    resolver = path_resolver or UPSTREAM.partition_paths
    paths = resolver(input_root, source_date, 10)
    if not paths:
        raise FileNotFoundError(f"missing B10 input: {source_date}")
    parts = []
    for path in paths:
        column = pq.read_table(path, columns=["block_index"])["block_index"]
        if column.null_count:
            raise RuntimeError(f"null block_index in B10 input: {path}")
        parts.append(np.asarray(column, dtype=np.int64))
    result = np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)
    if len(result) <= 0 or (len(result) > 1 and np.any(np.diff(result) <= 0)):
        raise RuntimeError(
            f"current B10 block_index is empty, duplicate, or decreasing: {source_date}"
        )
    result.setflags(write=False)
    _BLOCK_ID_CACHE[key] = result
    return result


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"unit already has a writer: {path}") from error
    return handle


def publish_failure_sentinel(
    path: Path | None, *, unit_id: str, stage: str, error: BaseException
) -> None:
    if path is None or isinstance(error, InterruptedError):
        return
    lock = path.with_suffix(path.suffix + ".lock")
    handle = _lock(lock)
    try:
        if not path.is_file():
            atomic_json(
                path,
                {
                    "status": "failed",
                    "unit_id": unit_id,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "created_utc": datetime.now(UTC).isoformat(),
                },
            )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def quarantine_stale_work(
    work: Path, exact: dict[str, Any], *, resume: bool, work_root: Path
) -> None:
    """Move an incompatible or explicitly non-resumed unit aside before locking it."""
    config_path = work / "config.json"
    if not work.exists() or not any(work.iterdir()):
        return
    compatible = config_path.is_file() and json.loads(
        config_path.read_text(encoding="utf-8")
    ) == _jsonable(exact)
    if resume and compatible:
        return
    quarantine = (
        work_root
        / "quarantine"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        / work.relative_to(work_root)
    )
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, quarantine)
    prior_runtime = quarantine / "runtime.json"
    if prior_runtime.is_file():
        runtime = json.loads(prior_runtime.read_text(encoding="utf-8"))
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / "runtime.json", runtime)


def _logical_input_inventory(
    root: Path, dates: Sequence[str], paths: Sequence[Path]
) -> dict[str, Any]:
    partitions = []
    for path in paths:
        try:
            logical_path = str(path.relative_to(root))
        except ValueError:
            logical_path = path.name
        partitions.append({"path": logical_path, "bytes": path.stat().st_size})
    return {
        "contract": "phase02_B10_test_partitions_v1",
        "B": 10,
        "dates": list(dates),
        "files": len(paths),
        "bytes": sum(p.stat().st_size for p in paths),
        "partitions": partitions,
    }


def _input_inventory(root: Path, dates: Sequence[str]) -> dict[str, Any]:
    paths = [path for day in dates for path in UPSTREAM.partition_paths(root, day, 10)]
    if not paths:
        raise FileNotFoundError(root)
    return _logical_input_inventory(root, dates, paths)


def read_segmentable_day(root: Path, source_date: str) -> SegmentableDay:
    paths = UPSTREAM.partition_paths(root, source_date, 10)
    if not paths:
        raise FileNotFoundError(f"missing B10 input: {source_date}")
    tables = [
        pq.ParquetFile(path).read(columns=list(UPSTREAM.REQUIRED_COLUMNS))
        for path in paths
    ]
    table = pa.concat_tables(tables, promote_options="default")
    original = np.asarray(table["block_index"], dtype=np.int64)
    if len(original) > 1 and np.any(np.diff(original) <= 0):
        raise RuntimeError(
            f"original source cursor duplicate/decreasing: {source_date}"
        )

    def vec(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.int64)

    def mat(names: Iterable[str]) -> np.ndarray:
        result = np.column_stack(
            [np.asarray(table[name], dtype=np.float64) for name in names]
        )
        if not np.isfinite(result).all():
            raise RuntimeError(f"nonfinite H2 input: {source_date}")
        return result

    starts = mat(f"start_{field}" for field in UPSTREAM.BOUNDARY_FIELDS)
    ends = mat(f"end_{field}" for field in UPSTREAM.BOUNDARY_FIELDS)
    intervals = mat(UPSTREAM.INTERVAL_FIELDS)
    block_start = vec("block_start_ns")
    block_end = vec("block_end_ns")
    # Multiple completion-closed source events can share an exchange
    # timestamp.  Such blocks have zero (not negative) elapsed time and are
    # valid support; Phase 05 established the project-wide rule that path
    # duration is nonnegative while only the aggregate denominator must be
    # strictly positive.
    if np.any(block_end < block_start):
        raise RuntimeError(f"invalid block duration: {source_date}")
    event = intervals[:, UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]]
    if np.any(event <= 0):
        raise RuntimeError(f"invalid source-event count: {source_date}")
    breaks = (
        np.flatnonzero(
            (np.diff(original) != 1)
            | (block_end[:-1] != block_start[1:])
            | np.any(ends[:-1] != starts[1:], axis=1)
        )
        + 1
    )
    bounds = np.concatenate(([0], breaks, [len(original)]))
    segments = tuple(
        (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a
    )
    return SegmentableDay(
        source_date, original, block_start, block_end, starts, ends, intervals, segments
    )


def _numeric_axes(day: SegmentableDay) -> dict[str, np.ndarray]:
    b = {
        name: UPSTREAM.BOUNDARY_FIELDS.index(name)
        for name in ("spread_ticks", "bid_depth_l5", "ask_depth_l5", "imbalance_l1")
    }
    i = UPSTREAM.INTERVAL_INDEX
    return {
        "spread": day.ends[:, b["spread_ticks"]],
        "depth": np.log1p(
            day.ends[:, b["bid_depth_l5"]] + day.ends[:, b["ask_depth_l5"]]
        ),
        "imbalance": day.ends[:, b["imbalance_l1"]],
        "recent_activity": np.log1p(day.intervals[:, i["realized_source_event_count"]]),
        "recent_volatility": np.log1p(
            day.intervals[:, i["mid_abs_variation_half_ticks"]]
        ),
        "recent_duration": np.log1p(day.intervals[:, i["elapsed_ns"]] / 1e9),
    }


def fit_coarse_state(input_root: Path) -> dict[str, Any]:
    parts: dict[str, list[np.ndarray]] = {axis: [] for axis in STATE_AXES}
    rows = 0
    for source_date in DEVELOPMENT_DATES:
        day = read_segmentable_day(input_root, source_date)
        rows += len(day.block_index)
        for axis, values in _numeric_axes(day).items():
            parts[axis].append(values)
    cutpoints = {
        axis: tuple(
            np.quantile(np.concatenate(values), (1 / 3, 2 / 3), method="linear").astype(
                np.float64
            )
        )
        for axis, values in parts.items()
    }
    if any(not np.isfinite(pair).all() for pair in cutpoints.values()):
        raise RuntimeError("nonfinite H2 coarse-state cutpoint")
    definition = CoarseStateDefinition(cutpoints=cutpoints)
    return {"status": "fitted", "development_rows": rows, **definition.to_dict()}


def _clock_bins(
    ns: np.ndarray, source_date: str, cutpoints: tuple[int, int]
) -> np.ndarray:
    noon = datetime.fromisoformat(source_date + "T12:00:00").replace(tzinfo=NY)
    offset = int(noon.utcoffset().total_seconds())
    seconds = (ns // 1_000_000_000 + offset) % 86_400
    return np.searchsorted(np.asarray(cutpoints), seconds, side="right").astype(
        np.int16
    )


def state_ids(day: SegmentableDay, definition: CoarseStateDefinition) -> np.ndarray:
    n = len(day.block_index)
    bidx = {
        name: UPSTREAM.BOUNDARY_FIELDS.index(name)
        for name in ("spread_ticks", "bid_depth_l5", "ask_depth_l5", "imbalance_l1")
    }
    values_from = {
        "spread": day.starts[:, bidx["spread_ticks"]],
        "depth": np.log1p(
            day.starts[:, bidx["bid_depth_l5"]] + day.starts[:, bidx["ask_depth_l5"]]
        ),
        "imbalance": day.starts[:, bidx["imbalance_l1"]],
    }
    values_to = {
        "spread": day.ends[:, bidx["spread_ticks"]],
        "depth": np.log1p(
            day.ends[:, bidx["bid_depth_l5"]] + day.ends[:, bidx["ask_depth_l5"]]
        ),
        "imbalance": day.ends[:, bidx["imbalance_l1"]],
    }
    recent = _numeric_axes(day)
    for axis in ("recent_activity", "recent_volatility", "recent_duration"):
        prior = np.empty(n, dtype=np.float64)
        prior.fill(np.nan)
        for first, last in day.segments:
            if last - first > 1:
                prior[first + 1 : last] = recent[axis][first : last - 1]
        values_from[axis] = prior
        values_to[axis] = recent[axis]
    bins_from: dict[str, np.ndarray] = {}
    bins_to: dict[str, np.ndarray] = {}
    segment_starts = np.asarray(
        [first for first, _last in day.segments], dtype=np.int64
    )
    for axis in STATE_AXES:
        from_result = np.searchsorted(
            np.asarray(definition.cutpoints[axis]), values_from[axis], side="right"
        ).astype(np.int16)
        to_result = np.searchsorted(
            np.asarray(definition.cutpoints[axis]), values_to[axis], side="right"
        ).astype(np.int16)
        if axis.startswith("recent_"):
            from_result[segment_starts] = definition.recent_sentinel
        bins_from[axis] = from_result
        bins_to[axis] = to_result

    def encode(clock_ns: np.ndarray, bins: dict[str, np.ndarray]) -> np.ndarray:
        encoded = _clock_bins(
            clock_ns, day.source_date, definition.clock_boundaries_seconds
        ).astype(np.int64)
        for axis in ("spread", "depth", "imbalance"):
            encoded = encoded * 3 + bins[axis]
        for axis in ("recent_activity", "recent_volatility", "recent_duration"):
            encoded = encoded * 4 + bins[axis]
        return encoded

    encoded = np.column_stack(
        (encode(day.block_start_ns, bins_from), encode(day.block_end_ns, bins_to))
    )
    if (
        encoded.shape != (n, 2)
        or np.any(encoded < 0)
        or np.any(encoded > np.iinfo(np.uint32).max)
    ):
        raise RuntimeError("invalid H2 state encoding")
    return encoded.astype(np.uint32)


def _transition_counts(
    edge_order: np.ndarray, states: np.ndarray
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for edge in edge_order:
        key = (int(states[edge, 0]), int(states[edge, 1]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def randomized_euler(
    edge_positions: np.ndarray,
    states: np.ndarray,
    *,
    surrogate_seed: int,
    source_date: str,
    segment_ordinal: int,
    checkpoint_root: Path | None = None,
    checkpoint_edges: int = CONSTRUCTION_CHECKPOINT_EDGES,
    interrupt_after_edges: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(edge_positions, dtype=np.int64)
    if len(original) <= 2:
        return original.copy(), {
            "eligible": 0,
            "fixed": max(0, len(original) - 1),
            "endpoint": max(0, len(original) - 1),
            "changed": 0,
        }
    head, tail = int(original[0]), int(original[-1])
    movable = original[1:-1]
    adjacency: dict[int, list[int]] = {}
    for edge in movable:
        adjacency.setdefault(int(states[edge, 0]), []).append(int(edge))
    day_seed = int(source_date.replace("-", ""))
    for state in sorted(adjacency):
        adjacency[state].sort()
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [surrogate_seed, day_seed, segment_ordinal, state]
                )
            )
        )
        rng.shuffle(adjacency[state])
    cursor = {state: 0 for state in adjacency}
    vertex_stack = [int(states[head, 1])]
    edge_stack: list[int] = []
    postorder: list[int] = []
    state_path = (
        checkpoint_root / "traversal.json" if checkpoint_root is not None else None
    )
    if state_path is not None and state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            saved.get("contract") != "phase04_h2_hierholzer_checkpoint_v1"
            or int(saved.get("surrogate_seed", -1)) != surrogate_seed
            or saved.get("source_date") != source_date
            or int(saved.get("segment_ordinal", -1)) != segment_ordinal
            or int(saved.get("head", -1)) != head
            or int(saved.get("tail", -1)) != tail
        ):
            raise RuntimeError("stale H2 traversal checkpoint")
        cursor = {int(key): int(value) for key, value in saved["cursor"].items()}

        def load_vector(name: str) -> list[int]:
            table = pq.read_table(checkpoint_root / f"{name}.parquet")
            return np.asarray(table["value"], dtype=np.int64).tolist()

        vertex_stack = load_vector("vertex_stack")
        edge_stack = load_vector("edge_stack")
        postorder = load_vector("postorder_edges")
    selected_count = sum(cursor.values())

    def save_state() -> None:
        if checkpoint_root is None:
            return
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([("position", pa.int64()), ("value", pa.int64())])
        for name, values in (
            ("vertex_stack", vertex_stack),
            ("edge_stack", edge_stack),
            ("postorder_edges", postorder),
        ):
            table = pa.Table.from_arrays(
                [
                    pa.array(np.arange(len(values)), type=pa.int64()),
                    pa.array(values, type=pa.int64()),
                ],
                schema=schema,
            )
            atomic_parquet(checkpoint_root / f"{name}.parquet", table, schema)
        atomic_json(
            state_path,
            {
                "contract": "phase04_h2_hierholzer_checkpoint_v1",
                "surrogate_seed": surrogate_seed,
                "source_date": source_date,
                "segment_ordinal": segment_ordinal,
                "head": head,
                "tail": tail,
                "cursor": {str(key): value for key, value in sorted(cursor.items())},
                "selected_edges": sum(cursor.values()),
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )

    while vertex_stack:
        vertex = vertex_stack[-1]
        edges = adjacency.get(vertex, [])
        selected = cursor.get(vertex, 0)
        if selected < len(edges):
            edge = edges[selected]
            cursor[vertex] = selected + 1
            selected_count += 1
            edge_stack.append(edge)
            vertex_stack.append(int(states[edge, 1]))
            if checkpoint_root is not None and (
                selected_count % checkpoint_edges == 0
                or (
                    interrupt_after_edges is not None
                    and selected_count >= interrupt_after_edges
                )
            ):
                save_state()
                if (
                    interrupt_after_edges is not None
                    and selected_count >= interrupt_after_edges
                ):
                    raise InterruptedError("synthetic construction interruption")
        else:
            vertex_stack.pop()
            if edge_stack:
                postorder.append(edge_stack.pop())
    middle = np.asarray(postorder[::-1], dtype=np.int64)
    result = np.concatenate(([head], middle, [tail])).astype(np.int64)
    if (
        len(result) != len(original)
        or set(result.tolist()) != set(original.tolist())
        or int(states[result[0], 0]) != int(states[original[0], 0])
        or int(states[result[-1], 1]) != int(states[original[-1], 1])
        or np.any(states[result[:-1], 1] != states[result[1:], 0])
    ):
        raise RuntimeError("randomized Hierholzer traversal failed")
    original_successor = {
        int(a): int(b) for a, b in zip(original[:-1], original[1:], strict=True)
    }
    surrogate_successor = {
        int(a): int(b) for a, b in zip(result[:-1], result[1:], strict=True)
    }
    endpoint_incoming = {head, int(original[-2]), int(result[-2])}
    eligible = fixed = changed = 0
    outdegree = {state: len(edges) for state, edges in adjacency.items()}
    for incoming in sorted(set(original_successor) & set(surrogate_successor)):
        if incoming in endpoint_incoming:
            continue
        middle_state = int(states[incoming, 1])
        if outdegree.get(middle_state, 0) < 2:
            fixed += 1
            continue
        eligible += 1
        changed += original_successor[incoming] != surrogate_successor[incoming]
    return result, {
        "eligible": eligible,
        "fixed": fixed,
        "endpoint": len(endpoint_incoming),
        "changed": int(changed),
    }


def _path_count(permutation: pa.Table, source_date: str, k: int) -> int:
    segments = np.asarray(permutation["segment_id"], dtype=np.uint32)
    positions = np.asarray(permutation["surrogate_position"], dtype=np.int64)
    offset = date.fromisoformat(source_date).toordinal() % 3
    count = 0
    for segment in np.unique(segments):
        selected = positions[segments == segment]
        if len(selected) < k:
            continue
        count += int(
            np.count_nonzero((selected[: len(selected) - k + 1] - offset) % 3 == 0)
        )
    return count


def _original_path_count(day: SegmentableDay, k: int) -> int:
    offset = date.fromisoformat(day.source_date).toordinal() % 3
    return sum(
        int(
            np.count_nonzero(
                (np.arange(first, last - k + 1, dtype=np.int64) - offset) % 3 == 0
            )
        )
        for first, last in day.segments
        if last - first >= k
    )


def _construction_exact(
    config: ConstructionConfig, definition: CoarseStateDefinition
) -> dict[str, Any]:
    return {
        "contract": "phase04_h2_construction_v1",
        "surrogate_seed": config.surrogate_seed,
        "source_date": config.source_date,
        "input": _input_inventory(config.input_root, (config.source_date,)),
        "state": definition.to_dict(),
        "checkpoint_edges": config.checkpoint_edges,
        "row_group_size": config.row_group_size,
        "rng": "numpy.PCG64.SeedSequence",
        "traversal": "fixed_head_tail_hierholzer_head_consume_postorder_reverse_v1",
    }


def _construction_unit_implementation(config: ConstructionConfig) -> dict[str, Any]:
    if (
        config.surrogate_seed not in SURROGATE_SEEDS
        or config.source_date not in TEST_DATES
    ):
        raise ValueError("construction unit outside approved H2 scope")
    definition = CoarseStateDefinition.from_dict(
        json.loads(config.state_path.read_text(encoding="utf-8"))
    )
    final = (
        config.output_root
        / f"replicate={config.surrogate_seed}"
        / f"source_date={config.source_date}"
    )
    work = config.work_root / "construction" / config.unit_id
    if (final / "complete.json").is_file():
        return validate_construction_complete(final, config, definition)
    exact = _construction_exact(config, definition)
    quarantine_stale_work(work, exact, resume=config.resume, work_root=config.work_root)
    work.mkdir(parents=True, exist_ok=True)
    handle = _lock(work / ".unit.lock")
    started = time.monotonic()
    runtime_path = work / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {"attempts": 0, "elapsed_seconds": 0.0}
    )
    succeeded = False
    try:
        config_path = work / "config.json"
        if not config_path.is_file():
            atomic_json(config_path, exact)
        day = read_segmentable_day(config.input_root, config.source_date)
        states = state_ids(day, definition)
        rows: list[dict[str, Any]] = []
        eligibility = changed = fixed = endpoint = 0
        original_order = np.arange(len(day.block_index), dtype=np.int64)
        surrogate_parts = []
        processed = 0
        parts_root = work / "parts"
        parts_root.mkdir(parents=True, exist_ok=True)
        for segment_id, (first, last) in enumerate(day.segments):
            part_path = parts_root / f"segment={segment_id}.parquet"
            audit_path = parts_root / f"segment={segment_id}.audit.json"
            if part_path.is_file() and audit_path.is_file():
                part = pq.read_table(part_path)
                if (
                    not part.schema.equals(PERMUTATION_SCHEMA, check_metadata=True)
                    or len(part) != last - first
                ):
                    raise RuntimeError("stale H2 completed segment part")
                order_ids = np.asarray(part["source_block_index"], dtype=np.int64)
                order = np.searchsorted(day.block_index, order_ids)
                if not np.array_equal(day.block_index[order], order_ids):
                    raise RuntimeError("completed segment block identity mismatch")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                part_rows = part.to_pylist()
            else:
                order, audit = randomized_euler(
                    np.arange(first, last),
                    states,
                    surrogate_seed=config.surrogate_seed,
                    source_date=config.source_date,
                    segment_ordinal=segment_id,
                    checkpoint_root=work / "traversal" / f"segment={segment_id}",
                    checkpoint_edges=config.checkpoint_edges,
                    interrupt_after_edges=config.interrupt_after_edges,
                )
                part_rows = [
                    {
                        "source_date": config.source_date,
                        "segment_id": segment_id,
                        "surrogate_position": first + local_position,
                        "source_block_index": int(day.block_index[edge]),
                        "coarse_from_state": int(states[edge, 0]),
                        "coarse_to_state": int(states[edge, 1]),
                    }
                    for local_position, edge in enumerate(order)
                ]
                atomic_parquet(
                    part_path,
                    pa.Table.from_pylist(part_rows, schema=PERMUTATION_SCHEMA),
                    PERMUTATION_SCHEMA,
                )
                atomic_json(audit_path, audit)
            surrogate_parts.append(order)
            eligibility += audit["eligible"]
            changed += audit["changed"]
            fixed += audit["fixed"]
            endpoint += audit["endpoint"]
            rows.extend(part_rows)
            processed += len(order)
            running_elapsed = (
                float(runtime["elapsed_seconds"]) + time.monotonic() - started
            )
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": running_elapsed,
                },
            )
            atomic_json(
                work / "checkpoint.json",
                {
                    "state": "checkpointed",
                    "next_segment_ordinal": segment_id + 1,
                    "consumed_edges": processed,
                    "eligibility": eligibility,
                    "changed": changed,
                    "fixed": fixed,
                    "endpoint": endpoint,
                    "elapsed_seconds": running_elapsed,
                },
            )
        surrogate_order = (
            np.concatenate(surrogate_parts)
            if surrogate_parts
            else np.asarray([], dtype=np.int64)
        )
        table = pa.Table.from_pylist(rows, schema=PERMUTATION_SCHEMA)
        if len(table) != len(day.block_index):
            raise RuntimeError("permutation row mismatch")
        transition_original = _transition_counts(original_order, states)
        transition_surrogate = _transition_counts(surrogate_order, states)
        all_keys = set(transition_original) | set(transition_surrogate)
        transition_diff = max(
            (
                abs(transition_original.get(k, 0) - transition_surrogate.get(k, 0))
                for k in all_keys
            ),
            default=0,
        )
        current_diff = 0
        for i, j in all_keys:
            real = transition_original.get((i, j), 0) - transition_original.get(
                (j, i), 0
            )
            synthetic = transition_surrogate.get((i, j), 0) - transition_surrogate.get(
                (j, i), 0
            )
            current_diff = max(current_diff, abs(real - synthetic))
        exact_edges = np.array_equal(np.sort(surrogate_order), original_order)
        adjacent = 0
        adjacency_total = 0
        for order in surrogate_parts:
            if len(order) > 1:
                adjacent += int(
                    np.count_nonzero(
                        np.all(day.ends[order[:-1]] == day.starts[order[1:]], axis=1)
                    )
                )
                adjacency_total += len(order) - 1
        date_share = changed / eligibility if eligibility else float("nan")
        observed_paths = {
            k: _path_count(table, config.source_date, k) for k in H2_K_VALUES
        }
        expected_paths = {k: _original_path_count(day, k) for k in H2_K_VALUES}
        path_counts_exact = observed_paths == expected_paths
        audit = {
            "source_date": config.source_date,
            "status": "constructed",
            "surrogate_seed": config.surrogate_seed,
            "n_segments": len(day.segments),
            "n_edges": len(day.block_index),
            "n_states": len(np.unique(states)),
            "n_directed_transitions": len(transition_original),
            "occupancy_max_abs_difference": 0,
            "transition_max_abs_difference": transition_diff,
            "current_max_abs_difference": current_diff,
            "eligible_links": eligibility,
            "fixed_links": fixed,
            "endpoint_excluded_links": endpoint,
            "changed_links": changed,
            "K2_paths": observed_paths[2],
            "K6_paths": observed_paths[6],
            "K12_paths": observed_paths[12],
            "changed_linkage_share": date_share,
            "pooled_changed_linkage_share": date_share,
            "adjacent_exact_boundary_share": adjacent / adjacency_total
            if adjacency_total
            else 1.0,
            "edge_multiset_exact": exact_edges,
            "payload_exact": exact_edges,
            "endpoints_exact": True,
            "consumed_once": exact_edges,
            "path_counts_exact": path_counts_exact,
            "expected_paths": {f"K{k}": expected_paths[k] for k in H2_K_VALUES},
            "passed": bool(
                exact_edges
                and transition_diff == 0
                and current_diff == 0
                and path_counts_exact
                and eligibility > 0
                and date_share >= 0.75
            ),
        }
        if not audit["passed"]:
            raise RuntimeError(f"H2 construction audit failed: {audit}")
        staging = work / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        atomic_parquet(staging / "permutation.parquet", table, PERMUTATION_SCHEMA)
        atomic_json(staging / "linkage_audit.json", audit)
        complete = {
            "status": "complete",
            "unit_id": config.unit_id,
            "scientific_config": exact,
            "runtime": {
                "attempts": int(runtime["attempts"]) + 1,
                "elapsed_seconds": float(runtime["elapsed_seconds"])
                + time.monotonic()
                - started,
            },
            "artifacts": [
                {
                    "filename": "permutation.parquet",
                    "rows": len(table),
                    "bytes": (staging / "permutation.parquet").stat().st_size,
                },
                {
                    "filename": "linkage_audit.json",
                    "bytes": (staging / "linkage_audit.json").stat().st_size,
                },
            ],
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(staging / "complete.json", complete)
        validate_construction_complete(
            staging, config, definition, expected_block_ids=day.block_index
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        shutil.rmtree(work)
        return validate_construction_complete(
            final, config, definition, expected_block_ids=day.block_index
        )
    finally:
        if not succeeded and work.is_dir():
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": float(runtime["elapsed_seconds"])
                    + time.monotonic()
                    - started,
                },
            )
        handle.close()


def validate_construction_complete(
    final: Path,
    config: ConstructionConfig,
    definition: CoarseStateDefinition,
    *,
    expected_block_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    value = json.loads((final / "complete.json").read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config")
        != _jsonable(_construction_exact(config, definition))
    ):
        raise RuntimeError(f"invalid H2 construction completion: {final}")
    table = pq.read_table(final / "permutation.parquet")
    validate_arrow_contract(table, PERMUTATION_SCHEMA)
    expected = (
        expected_block_ids
        if expected_block_ids is not None
        else _expected_block_ids(config.input_root, config.source_date)
    )
    _validate_permutation_keys(table, config.source_date, expected)
    permutation_entry = next(
        (
            item
            for item in value.get("artifacts", [])
            if item.get("filename") == "permutation.parquet"
        ),
        None,
    )
    if (
        not permutation_entry
        or int(permutation_entry.get("rows", -1)) != len(table)
        or permutation_entry.get("bytes")
        != (final / "permutation.parquet").stat().st_size
    ):
        raise RuntimeError("H2 permutation row/byte inventory mismatch")
    audit = json.loads((final / "linkage_audit.json").read_text(encoding="utf-8"))
    if (
        audit.get("passed") is not True
        or int(audit.get("n_edges", -1)) != len(table)
        or audit.get("path_counts_exact") is not True
        or any(
            int(audit[f"K{k}_paths"])
            != int(audit.get("expected_paths", {}).get(f"K{k}", -1))
            for k in H2_K_VALUES
        )
    ):
        raise RuntimeError("H2 linkage/path-count audit failed")
    return value


@dataclass
class MetricAccumulator:
    paths: int = 0
    score_sum: float = 0.0
    event_sum: float = 0.0
    seconds_sum: float = 0.0
    loss_sum: float = 0.0
    saturation_sum: int = 0
    reverse_like_sum: int = 0
    overlap_sum: float = 0.0

    def update(
        self, score: np.ndarray, events: np.ndarray, seconds: np.ndarray
    ) -> None:
        if (
            not np.isfinite(score).all()
            or not np.isfinite(events).all()
            or not np.isfinite(seconds).all()
        ):
            raise RuntimeError("nonfinite H2 score or denominator")
        if np.any(events <= 0) or np.any(seconds < 0):
            raise RuntimeError("nonpositive H2 score denominator")
        clipped = np.clip(score, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-clipped))
        self.paths += len(score)
        self.score_sum += float(score.sum())
        self.event_sum += float(events.sum())
        self.seconds_sum += float(seconds.sum())
        self.loss_sum += float(np.logaddexp(0.0, -clipped).sum())
        self.saturation_sum += int(
            np.count_nonzero((probability < 0.01) | (probability > 0.99))
        )
        self.reverse_like_sum += int(np.count_nonzero(score < 0))
        self.overlap_sum += float((1.0 - np.abs(2.0 * probability - 1.0)).sum())

    def final(self, k: int) -> dict[str, Any]:
        if self.paths <= 0 or self.event_sum <= 0 or self.seconds_sum <= 0:
            raise RuntimeError("empty H2 score accumulator")
        result = {
            "paths": self.paths,
            "source_event_sum": int(self.event_sum),
            "duration_seconds_sum": self.seconds_sum,
            "nats_path": self.score_sum / self.paths,
            "nats_block": self.score_sum / (self.paths * k),
            "nats_source_event": self.score_sum / self.event_sum,
            "nats_second": self.score_sum / self.seconds_sum,
            "paired_log_loss": self.loss_sum / self.paths,
            "posterior_saturation_share": self.saturation_sum / self.paths,
            "reverse_like_share": self.reverse_like_sum / self.paths,
            "calibrated_ovl": self.overlap_sum / self.paths,
        }
        if not np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all():
            raise RuntimeError("nonfinite H2 metric accumulator output")
        return result


def _load_model(config: ScoringConfig) -> tuple[Any, Any, float, dict[str, Any]]:
    required = ("model.pt", "scaler.json", "calibration.json")
    if any(not (config.model_root / name).is_file() for name in required):
        raise FileNotFoundError(config.model_root)
    completion_identity = validate_model_completion(
        config.model_root, config.k, config.model_seed
    )
    scaler = UPSTREAM.ChannelScaler.from_dict(
        json.loads((config.model_root / "scaler.json").read_text(encoding="utf-8"))
    )
    calibration = json.loads(
        (config.model_root / "calibration.json").read_text(encoding="utf-8")
    )
    alpha = float(calibration["alpha"])
    if not np.isfinite(alpha):
        raise RuntimeError("nonfinite H2 calibration alpha")
    payload = torch.load(
        config.model_root / "model.pt", map_location="cpu", weights_only=False
    )
    core_valid = (
        payload.get("model") == "ordered_tcn"
        and int(payload.get("seed", -1)) == config.model_seed
    )
    dimensional_valid = (
        payload.get("K") is not None
        and payload.get("B") is not None
        and int(payload["K"]) == config.k
        and int(payload["B"]) == 10
    )
    # The immutable Phase-05 K6 anchor predates B/K fields in model.pt.  Its
    # complete_manifest is the authoritative dimensional identity and binds
    # the payload through the same spec_hash.  Newer Phase-03 payloads must
    # continue to carry B and K directly.
    manifest_path = config.model_root / "complete_manifest.json"
    legacy_valid = False
    if not dimensional_valid and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_valid = bool(
            int(manifest.get("target_b", -1)) == 10
            and int(manifest.get("k", -1)) == config.k
            and manifest.get("spec_hash") == payload.get("spec_hash")
        )
    if not core_valid or not (dimensional_valid or legacy_valid):
        raise RuntimeError("H2 model identity mismatch")
    model = UPSTREAM.build_model("ordered_tcn")
    model.load_state_dict(payload["model_state"], strict=True)
    device = device_or_raise(config.device)
    model.to(device)
    model.eval()
    identity = model_validation_identity(
        config, completion_identity=completion_identity, device=str(device)
    )
    return model, scaler, alpha, identity


def model_validation_identity(
    config: ScoringConfig,
    *,
    completion_identity: dict[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    required = ("model.pt", "scaler.json", "calibration.json")
    completion = completion_identity or validate_model_completion(
        config.model_root, config.k, config.model_seed
    )
    artifacts = {
        name: {"filename": name, "bytes": (config.model_root / name).stat().st_size}
        for name in required
    }
    return {
        "device": device or str(torch.device(config.device)),
        "artifacts": artifacts,
        "completion": completion,
    }


def validate_model_completion(
    model_root: Path, k: int, model_seed: int
) -> dict[str, Any]:
    manifest_path = model_root / "complete_manifest.json"
    complete_path = model_root / "complete.json"
    path = manifest_path if manifest_path.is_file() else complete_path
    if not path.is_file():
        raise RuntimeError(f"model completion descriptor missing: {model_root}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError("model completion status is not complete")
    if manifest_path.is_file():
        if (
            int(value.get("target_b", -1)) != 10
            or int(value.get("k", -1)) != k
            or value.get("model") != "ordered_tcn"
            or int(value.get("seed", -1)) != model_seed
        ):
            raise RuntimeError("immutable model completion identity mismatch")
    else:
        unit_id = str(value.get("unit_id", value.get("logical_unit_id", "")))
        if unit_id and unit_id != f"B=10_K={k}/ordered_tcn/seed={model_seed}":
            raise RuntimeError("Phase03 model completion unit identity mismatch")
    artifacts = value.get("artifacts", {})
    index = (
        artifacts
        if isinstance(artifacts, dict)
        else {
            str(item.get("filename")): item
            for item in artifacts
            if isinstance(item, dict)
        }
    )
    for name in ("model.pt", "scaler.json", "calibration.json"):
        entry = index.get(name)
        if entry is None:
            candidates = [item for key, item in index.items() if Path(key).name == name]
            entry = candidates[0] if len(candidates) == 1 else None
        file_path = model_root / name
        if entry is None or int(entry.get("bytes", -1)) != file_path.stat().st_size:
            raise RuntimeError(f"model completion artifact mismatch: {file_path}")
    return {
        "descriptor": path.name,
        "bytes": path.stat().st_size,
        "status": "complete",
        "unit_id": f"B=10_K={k}/ordered_tcn/seed={model_seed}",
    }


def _edge_tokens(
    day: SegmentableDay, edge_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = edge_matrix.reshape(-1)
    n, k = edge_matrix.shape
    boundaries = np.stack((day.starts[flat], day.ends[flat]), axis=1)
    intervals = day.intervals[flat, None, :]
    forward = UPSTREAM.make_tokens(boundaries, intervals).reshape(n, k, -1)
    rb, ri = UPSTREAM.reverse_continuous(boundaries, intervals)
    reverse_edge = UPSTREAM.make_tokens(rb, ri).reshape(n, k, -1)
    reverse = reverse_edge[:, ::-1, :]
    event_index = UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]
    elapsed_index = UPSTREAM.INTERVAL_INDEX["elapsed_ns"]
    events = day.intervals[edge_matrix, event_index].sum(axis=1)
    seconds = day.intervals[edge_matrix, elapsed_index].sum(axis=1) / 1e9
    return forward, reverse, events, seconds


def original_k1_reference_tokens(
    day: SegmentableDay, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent original K1 assembly; does not call the surrogate assembler."""
    selected = np.asarray(positions, dtype=np.int64)
    boundaries = np.stack((day.starts[selected], day.ends[selected]), axis=1)
    intervals = day.intervals[selected, None, :]
    forward = UPSTREAM.make_tokens(boundaries, intervals)
    reverse_boundaries, reverse_intervals = UPSTREAM.reverse_continuous(
        boundaries.copy(), intervals.copy()
    )
    reverse = UPSTREAM.make_tokens(reverse_boundaries, reverse_intervals)
    return forward, reverse


def surrogate_k1_tokens(
    day: SegmentableDay, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Surrogate K1 assembly through the general permuted-edge path."""
    forward, reverse, _events, _seconds = _edge_tokens(
        day, np.asarray(positions, dtype=np.int64)[:, None]
    )
    return forward, reverse


def score_k1_edges(
    day: SegmentableDay,
    *,
    model_root: Path,
    model_seed: int,
    device_name: str = "cpu",
    batch_size: int = SCORING_BATCH_SIZE,
) -> tuple[np.ndarray, float]:
    """Score each labeled real edge once for the mandatory H2 one-step join audit."""
    configure_determinism()
    config = ScoringConfig(
        k=1,
        model_seed=model_seed,
        surrogate_seed=SURROGATE_SEEDS[0],
        device=device_name,
        input_root=Path("."),
        permutation_root=Path("."),
        model_root=model_root,
        output_root=Path("."),
        work_root=Path("."),
        batch_size=batch_size,
    )
    model, scaler, alpha, _identity = _load_model(config)
    device = next(model.parameters()).device
    scores = np.empty(len(day.block_index), dtype=np.float64)
    with torch.no_grad():
        for first in range(0, len(scores), batch_size):
            selected = np.arange(
                first, min(first + batch_size, len(scores)), dtype=np.int64
            )
            forward, reverse = original_k1_reference_tokens(day, selected)
            raw = UPSTREAM.antisymmetric_score(
                model,
                torch.from_numpy(scaler.transform(forward)).to(device),
                torch.from_numpy(scaler.transform(reverse)).to(device),
            )
            scores[first : first + len(selected)] = (
                alpha * raw.detach().cpu().numpy().astype(np.float64)
            )
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite original K1 reference score")
    return scores, alpha


def score_k1_positions(
    day: SegmentableDay,
    positions: np.ndarray,
    *,
    model_root: Path,
    model_seed: int,
    device_name: str = "cpu",
    batch_size: int = SCORING_BATCH_SIZE,
) -> tuple[np.ndarray, float]:
    """Independently construct and score K1 tokens in the supplied edge order."""
    configure_determinism()
    config = ScoringConfig(
        k=1,
        model_seed=model_seed,
        surrogate_seed=SURROGATE_SEEDS[0],
        device=device_name,
        input_root=Path("."),
        permutation_root=Path("."),
        model_root=model_root,
        output_root=Path("."),
        work_root=Path("."),
        batch_size=batch_size,
    )
    model, scaler, alpha, _identity = _load_model(config)
    device = next(model.parameters()).device
    ordered = np.asarray(positions, dtype=np.int64)
    if np.any(ordered < 0) or np.any(ordered >= len(day.block_index)):
        raise ValueError("K1 edge position outside day")
    scores = np.empty(len(ordered), dtype=np.float64)
    with torch.no_grad():
        for first in range(0, len(scores), batch_size):
            selected = ordered[first : first + batch_size, None]
            forward, reverse = surrogate_k1_tokens(day, selected[:, 0])
            ft = torch.from_numpy(scaler.transform(forward)).to(device)
            rt = torch.from_numpy(scaler.transform(reverse)).to(device)
            raw = UPSTREAM.antisymmetric_score(model, ft, rt)
            scores[first : first + len(selected)] = (
                alpha * raw.detach().cpu().numpy().astype(np.float64)
            )
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite H2 K1 validation score")
    return scores, alpha


def _starts_for_permutation(
    table: pa.Table, source_date: str, k: int
) -> tuple[np.ndarray, np.ndarray]:
    segment = np.asarray(table["segment_id"], dtype=np.uint32)
    positions = np.asarray(table["surrogate_position"], dtype=np.int64)
    source_index = np.asarray(table["source_block_index"], dtype=np.int64)
    starts: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    offset = date.fromisoformat(source_date).toordinal() % 3
    for sid in np.unique(segment):
        selected = np.flatnonzero(segment == sid)
        if len(selected) < k:
            continue
        candidates = selected[: len(selected) - k + 1]
        candidates = candidates[(positions[candidates] - offset) % 3 == 0]
        if len(candidates):
            starts.append(positions[candidates])
            matrices.append(
                np.stack(
                    [source_index[candidates + shift] for shift in range(k)], axis=1
                )
            )
    return (
        np.concatenate(starts) if starts else np.asarray([], dtype=np.int64),
        np.concatenate(matrices) if matrices else np.empty((0, k), dtype=np.int64),
    )


def _permutation_inventory(
    config: ScoringConfig,
    expected_block_ids_by_date: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    result = []
    for source_date in TEST_DATES:
        root = (
            config.permutation_root
            / f"replicate={config.surrogate_seed}"
            / f"source_date={source_date}"
        )
        complete_path = root / "complete.json"
        permutation_path = root / "permutation.parquet"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        table = pq.read_table(permutation_path)
        if (
            complete.get("status") != "complete"
            or complete.get("unit_id")
            != f"replicate={config.surrogate_seed}/source_date={source_date}"
        ):
            raise RuntimeError("H2 scoring construction completion identity mismatch")
        permutation_entry = next(
            (
                item
                for item in complete.get("artifacts", [])
                if item.get("filename") == "permutation.parquet"
            ),
            None,
        )
        if (
            not permutation_entry
            or int(permutation_entry.get("rows", -1)) != len(table)
            or permutation_entry.get("bytes") != permutation_path.stat().st_size
        ):
            raise RuntimeError("H2 scoring permutation inventory mismatch")
        expected_block_ids = (
            expected_block_ids_by_date[source_date]
            if expected_block_ids_by_date is not None
            else _expected_block_ids(config.input_root, source_date)
        )
        _validate_permutation_keys(table, source_date, expected_block_ids)
        result.append(
            {
                "source_date": source_date,
                "unit_id": complete["unit_id"],
                "artifact": "permutation.parquet",
                "rows": len(table),
                "bytes": permutation_path.stat().st_size,
                "completion_bytes": complete_path.stat().st_size,
                "construction_contract": complete.get("scientific_config", {}).get(
                    "contract"
                ),
            }
        )
    return result


def _scoring_exact(
    config: ScoringConfig, model_identity: dict[str, Any], alpha: float
) -> dict[str, Any]:
    return {
        "contract": "phase04_h2_scoring_v1",
        "K": config.k,
        "model_seed": config.model_seed,
        "surrogate_seed": config.surrogate_seed,
        "input": _input_inventory(config.input_root, TEST_DATES),
        "permutations": _permutation_inventory(config),
        "model_identity": model_identity,
        "alpha": alpha,
        "device": config.device,
        "fp32": True,
        "tf32": False,
        "amp": False,
        "batch_size": config.batch_size,
        "checkpoint_paths": config.checkpoint_paths,
        "score_contracts": ["uncapped", "bounded_log99"],
    }


def _scoring_unit_implementation(config: ScoringConfig) -> dict[str, Any]:
    if (
        config.k not in H2_K_VALUES
        or config.model_seed not in MODEL_SEEDS
        or config.surrogate_seed not in SURROGATE_SEEDS
    ):
        raise ValueError("scoring unit outside approved H2 scope")
    configure_determinism()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, scaler, alpha, model_identity = _load_model(config)
    device = next(model.parameters()).device
    final = (
        config.output_root
        / f"K={config.k}"
        / f"model_seed={config.model_seed}"
        / f"replicate={config.surrogate_seed}"
    )
    work = config.work_root / "scoring" / config.unit_id
    if (final / "complete.json").is_file():
        return validate_scoring_complete(final, config, model_identity, alpha)
    exact = _scoring_exact(config, model_identity, alpha)
    quarantine_stale_work(work, exact, resume=config.resume, work_root=config.work_root)
    work.mkdir(parents=True, exist_ok=True)
    handle = _lock(work / ".unit.lock")
    started = time.monotonic()
    runtime_path = work / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {"attempts": 0, "elapsed_seconds": 0.0, "measured_cost_usd": 0.0}
    )
    loader_wait_seconds = float(runtime.get("loader_wait_seconds", 0.0))
    process = psutil.Process()
    max_rss_bytes = int(runtime.get("max_rss_bytes", process.memory_info().rss))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    succeeded = False
    checkpoint_path = work / "checkpoint.json"
    checkpoint_value: dict[str, Any] | None = None
    try:
        config_path = work / "config.json"
        if not config_path.is_file():
            atomic_json(config_path, exact)
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else {
                "date_index": 0,
                "path_cursor": 0,
                "processed_paths": 0,
                "metrics": {},
            }
        )
        metrics_by_date: dict[str, dict[str, MetricAccumulator]] = {}
        for day_name, contracts in checkpoint.get("metrics", {}).items():
            metrics_by_date[day_name] = {
                name: MetricAccumulator(**value) for name, value in contracts.items()
            }
        max_antisymmetry = float(checkpoint.get("max_antisymmetry", 0.0))
        involution = bool(checkpoint.get("involution", True))
        processed = int(checkpoint.get("processed_paths", 0))
        for date_index, source_date in enumerate(TEST_DATES):
            if date_index < int(checkpoint.get("date_index", 0)):
                continue
            load_started = time.monotonic()
            day = read_segmentable_day(config.input_root, source_date)
            permutation = pq.read_table(
                config.permutation_root
                / f"replicate={config.surrogate_seed}"
                / f"source_date={source_date}"
                / "permutation.parquet"
            )
            loader_wait_seconds += time.monotonic() - load_started
            _, block_ids = _starts_for_permutation(permutation, source_date, config.k)
            construction_audit = json.loads(
                (
                    config.permutation_root
                    / f"replicate={config.surrogate_seed}"
                    / f"source_date={source_date}"
                    / "linkage_audit.json"
                ).read_text(encoding="utf-8")
            )
            if construction_audit.get("passed") is not True or len(block_ids) != int(
                construction_audit[f"K{config.k}_paths"]
            ):
                raise RuntimeError(
                    f"H2 scoring date path-count mismatch: {source_date} K={config.k}"
                )
            positions = np.searchsorted(day.block_index, block_ids)
            if np.any(positions >= len(day.block_index)) or not np.array_equal(
                day.block_index[positions], block_ids
            ):
                raise RuntimeError("surrogate block ID lookup mismatch")
            cursor = (
                int(checkpoint.get("path_cursor", 0))
                if date_index == int(checkpoint.get("date_index", 0))
                else 0
            )
            metrics_by_date.setdefault(
                source_date,
                {"uncapped": MetricAccumulator(), "bounded_log99": MetricAccumulator()},
            )
            with torch.no_grad():
                for first in range(cursor, len(positions), config.batch_size):
                    selected = positions[first : first + config.batch_size]
                    load_started = time.monotonic()
                    forward, reverse, events, seconds = _edge_tokens(day, selected)
                    loader_wait_seconds += time.monotonic() - load_started
                    if first == 0:
                        flat = selected[:1].reshape(-1)
                        boundaries = np.stack(
                            (day.starts[flat], day.ends[flat]), axis=1
                        )
                        intervals = day.intervals[flat, None, :]
                        once = UPSTREAM.reverse_continuous(boundaries, intervals)
                        twice = UPSTREAM.reverse_continuous(*once)
                        involution = (
                            involution
                            and np.array_equal(boundaries, twice[0])
                            and np.array_equal(intervals, twice[1])
                        )
                    ft = torch.from_numpy(scaler.transform(forward)).to(device)
                    rt = torch.from_numpy(scaler.transform(reverse)).to(device)
                    raw = UPSTREAM.antisymmetric_score(model, ft, rt)
                    reverse_raw = UPSTREAM.antisymmetric_score(model, rt, ft)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    raw_np = raw.detach().cpu().numpy().astype(np.float64)
                    reverse_np = reverse_raw.detach().cpu().numpy().astype(np.float64)
                    if (
                        not np.isfinite(raw_np).all()
                        or not np.isfinite(reverse_np).all()
                    ):
                        raise RuntimeError("nonfinite H2 raw/calibrated score")
                    max_antisymmetry = max(
                        max_antisymmetry, float(np.max(np.abs(raw_np + reverse_np)))
                    )
                    score = alpha * raw_np
                    metrics_by_date[source_date]["uncapped"].update(
                        score, events, seconds
                    )
                    metrics_by_date[source_date]["bounded_log99"].update(
                        np.clip(score, -BOUND, BOUND), events, seconds
                    )
                    processed += len(selected)
                    next_cursor = first + len(selected)
                    max_rss_bytes = max(max_rss_bytes, int(process.memory_info().rss))
                    running_elapsed = (
                        float(runtime["elapsed_seconds"]) + time.monotonic() - started
                    )
                    running_cost = (
                        float(runtime["measured_cost_usd"])
                        + (time.monotonic() - started) * config.hourly_cost_usd / 3600
                    )
                    checkpoint_value = {
                        "state": "checkpointed",
                        "date_index": date_index + (next_cursor == len(positions)),
                        "path_cursor": 0
                        if next_cursor == len(positions)
                        else next_cursor,
                        "processed_paths": processed,
                        "metrics": {
                            d: {c: asdict(a) for c, a in values.items()}
                            for d, values in metrics_by_date.items()
                        },
                        "max_antisymmetry": max_antisymmetry,
                        "involution": involution,
                        "elapsed_seconds": running_elapsed,
                        "accrued_cost_usd": running_cost,
                    }
                    if (
                        processed % config.checkpoint_paths < config.batch_size
                        or next_cursor == len(positions)
                    ):
                        atomic_json(checkpoint_path, checkpoint_value)
                        atomic_json(
                            runtime_path,
                            {
                                "attempts": int(runtime["attempts"]) + 1,
                                "elapsed_seconds": running_elapsed,
                                "measured_cost_usd": running_cost,
                                "loader_wait_seconds": loader_wait_seconds,
                                "max_rss_bytes": max_rss_bytes,
                            },
                        )
                    if (
                        config.interrupt_after_paths is not None
                        and processed >= config.interrupt_after_paths
                    ):
                        atomic_json(checkpoint_path, checkpoint_value)
                        raise InterruptedError("synthetic scoring interruption")
        if max_antisymmetry >= 1e-6 or not involution:
            raise RuntimeError("H2 scoring structural audit failed")
        metrics = {
            day: {
                contract: accumulator.final(config.k)
                for contract, accumulator in values.items()
            }
            for day, values in metrics_by_date.items()
        }
        if set(metrics) != set(TEST_DATES):
            raise RuntimeError("H2 scoring missing test date")
        if (
            sum(int(values["uncapped"]["paths"]) for values in metrics.values())
            != EXPECTED_PATHS[config.k]["test"]
        ):
            raise RuntimeError("H2 scoring total test path-count mismatch")
        staging = work / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        atomic_json(
            staging / "metrics.json",
            {
                "K": config.k,
                "model_seed": config.model_seed,
                "surrogate_seed": config.surrogate_seed,
                "alpha": alpha,
                "by_day": metrics,
                "audit": {
                    "score_antisymmetry_max_abs_error": max_antisymmetry,
                    "reverse_involution_exact": involution,
                    "checkpoint_resume_supported": True,
                },
            },
        )
        final_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        atomic_json(staging / "final_checkpoint.json", final_checkpoint)
        elapsed = float(runtime["elapsed_seconds"]) + time.monotonic() - started
        cost = (
            float(runtime["measured_cost_usd"])
            + (time.monotonic() - started) * config.hourly_cost_usd / 3600
        )
        input_bytes = int(
            exact["input"]["bytes"]
            + sum(item["bytes"] for item in exact["permutations"])
        )
        vram_peak = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        telemetry = {
            "attempts": int(runtime["attempts"]) + 1,
            "elapsed_seconds": elapsed,
            "hourly_cost_usd": config.hourly_cost_usd,
            "measured_cost_usd": cost,
            "processed_paths": processed,
            "paths_per_second": processed / elapsed,
            "loader_wait_seconds": loader_wait_seconds,
            "loader_wait_share": loader_wait_seconds / elapsed,
            "max_rss_bytes": max_rss_bytes,
            "peak_vram_allocated_bytes": vram_peak,
            "input_bytes": input_bytes,
            "input_bytes_per_second": input_bytes / elapsed,
            "usd_per_second": config.hourly_cost_usd / 3600.0,
            "usd_per_path": cost / processed,
        }
        if not np.isfinite(
            np.asarray(
                [
                    value
                    for value in telemetry.values()
                    if isinstance(value, (int, float))
                ],
                dtype=np.float64,
            )
        ).all():
            raise RuntimeError("nonfinite H2 scoring telemetry")
        complete = {
            "status": "complete",
            "unit_id": config.unit_id,
            "scientific_config": exact,
            "runtime": telemetry,
            "artifacts": [
                {
                    "filename": "metrics.json",
                    "bytes": (staging / "metrics.json").stat().st_size,
                },
                {
                    "filename": "final_checkpoint.json",
                    "bytes": (staging / "final_checkpoint.json").stat().st_size,
                },
            ],
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(staging / "complete.json", complete)
        validate_scoring_complete(staging, config, model_identity, alpha)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        shutil.rmtree(work)
        return validate_scoring_complete(final, config, model_identity, alpha)
    except (KeyboardInterrupt, InterruptedError):
        if checkpoint_value is not None:
            atomic_json(checkpoint_path, checkpoint_value)
        raise
    finally:
        if not succeeded and work.is_dir():
            attempt_elapsed = time.monotonic() - started
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": float(runtime["elapsed_seconds"])
                    + attempt_elapsed,
                    "measured_cost_usd": float(runtime["measured_cost_usd"])
                    + attempt_elapsed * config.hourly_cost_usd / 3600,
                    "loader_wait_seconds": loader_wait_seconds,
                    "max_rss_bytes": max_rss_bytes,
                },
            )
        handle.close()


run_construction_unit = _construction_unit_implementation
run_scoring_unit = _scoring_unit_implementation


def validate_scoring_complete(
    final: Path, config: ScoringConfig, model_identity: dict[str, Any], alpha: float
) -> dict[str, Any]:
    value = json.loads((final / "complete.json").read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config")
        != _jsonable(_scoring_exact(config, model_identity, alpha))
    ):
        raise RuntimeError(f"invalid H2 scoring completion: {final}")
    metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    if (
        set(metrics.get("by_day", {})) != set(TEST_DATES)
        or metrics.get("audit", {}).get("reverse_involution_exact") is not True
    ):
        raise RuntimeError("invalid H2 score metrics")
    numeric = []
    for contracts in metrics["by_day"].values():
        for row in contracts.values():
            numeric.extend(
                value for value in row.values() if isinstance(value, (int, float))
            )
    runtime = value.get("runtime", {})
    numeric.extend(item for item in runtime.values() if isinstance(item, (int, float)))
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise RuntimeError("nonfinite H2 scoring completion")
    checkpoint = json.loads(
        (final / "final_checkpoint.json").read_text(encoding="utf-8")
    )
    if (
        int(checkpoint.get("date_index", -1)) != len(TEST_DATES)
        or int(checkpoint.get("processed_paths", -1))
        != EXPECTED_PATHS[config.k]["test"]
    ):
        raise RuntimeError("invalid H2 final checkpoint/path count")
    return value


def verify_manifest_sha(root: Path, filenames: Sequence[str]) -> dict[str, Any]:
    manifest = json.loads((root / "complete_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("immutable manifest incomplete")
    for filename in filenames:
        expected = manifest.get("artifacts", {}).get(filename, {})
        path = root / filename
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or not expected.get("sha256")
        ):
            raise RuntimeError(f"immutable artifact inventory mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"immutable artifact SHA mismatch: {path}")
    return manifest


def validate_k6_prediction_envelope(
    table: pa.Table, model_seed: int, expected_paths: int = EXPECTED_PATHS[6]["test"]
) -> None:
    if not table.schema.equals(K6_PREDICTION_SCHEMA, check_metadata=True):
        raise RuntimeError("immutable K6 prediction schema mismatch")
    if len(table) != 2 * expected_paths:
        raise RuntimeError("immutable K6 paired prediction row count mismatch")
    for name in (
        "raw_antisymmetric_score",
        "calibrated_antisymmetric_score",
        "probability_forward",
        "duration_seconds",
    ):
        if (
            table[name].null_count
            or not np.isfinite(np.asarray(table[name], dtype=np.float64)).all()
        ):
            raise RuntimeError(f"immutable K6 prediction nonfinite/null field: {name}")
    orientation = np.asarray(table["orientation"], dtype=object)
    label = np.asarray(table["label"], dtype=np.uint8)
    if (
        np.count_nonzero(orientation == "forward") != expected_paths
        or np.count_nonzero(orientation == "reverse") != expected_paths
        or np.any(label[orientation == "forward"] != 1)
        or np.any(label[orientation == "reverse"] != 0)
    ):
        raise RuntimeError("immutable K6 orientation/label mismatch")
    if (
        set(np.asarray(table["target_b"], dtype=np.int16).tolist()) != {10}
        or set(np.asarray(table["k"], dtype=np.int16).tolist()) != {6}
        or set(np.asarray(table["model"], dtype=object).tolist()) != {"ordered_tcn"}
        or set(np.asarray(table["seed"], dtype=np.int64).tolist()) != {model_seed}
    ):
        raise RuntimeError("immutable K6 row identity mismatch")


def run_k6_cpu_parity(
    *,
    input_root: Path,
    model_root: Path,
    phase03_metrics_path: Path,
    output_path: Path,
    model_seed: int = 20260730,
) -> dict[str, Any]:
    """Fail-closed real-path parity before any immutable K6 surrogate scoring."""
    manifest = verify_manifest_sha(
        model_root,
        ("model.pt", "scaler.json", "calibration.json", "test_predictions.parquet"),
    )
    if (
        int(manifest.get("target_b", -1)) != 10
        or int(manifest.get("k", -1)) != 6
        or manifest.get("model") != "ordered_tcn"
        or int(manifest.get("seed", -1)) != model_seed
    ):
        raise RuntimeError("immutable K6 manifest identity mismatch")
    dummy = ScoringConfig(
        k=6,
        model_seed=model_seed,
        surrogate_seed=SURROGATE_SEEDS[0],
        device="cpu",
        input_root=input_root,
        permutation_root=Path("unused"),
        model_root=model_root,
        output_root=Path("unused"),
        work_root=Path("unused"),
    )
    configure_determinism()
    model, scaler, alpha, _identity = _load_model(dummy)
    stored_all = pq.read_table(model_root / "test_predictions.parquet")
    validate_k6_prediction_envelope(stored_all, model_seed)
    stored = stored_all.filter(pc.equal(stored_all["orientation"], "forward"))
    reverse_stored = stored_all.filter(pc.equal(stored_all["orientation"], "reverse"))
    if len(stored) != EXPECTED_PATHS[6]["test"]:
        raise RuntimeError("immutable K6 forward prediction count mismatch")
    if len(reverse_stored) != len(stored):
        raise RuntimeError("immutable K6 reverse prediction count mismatch")
    phase03 = pq.read_table(phase03_metrics_path).to_pylist()
    metric_index = {
        (row["source_date"], row["score_contract"]): row
        for row in phase03
        if int(row["B"]) == 10
        and int(row["K"]) == 6
        and row["model"] == "ordered_tcn"
        and int(row["seed"]) == model_seed
    }
    if len(metric_index) != 6:
        raise RuntimeError("K6 compact parity metric key mismatch")
    max_raw = max_calibrated = max_reverse_raw = max_reverse_calibrated = (
        max_reverse_probability
    ) = max_current_antisymmetry = max_metric = 0.0
    rows = 0
    by_day: dict[str, Any] = {}
    with torch.no_grad():
        for source_date in TEST_DATES:
            day = read_segmentable_day(input_root, source_date)
            offset = date.fromisoformat(source_date).toordinal() % 3
            starts = np.concatenate(
                [
                    np.arange(first, last - 6 + 1, dtype=np.int64)[
                        (np.arange(first, last - 6 + 1, dtype=np.int64) - offset) % 3
                        == 0
                    ]
                    for first, last in day.segments
                    if last - first >= 6
                ]
            )
            accumulators = {
                "uncapped": MetricAccumulator(),
                "bounded_log99": MetricAccumulator(),
            }
            raw_parts: list[np.ndarray] = []
            calibrated_parts: list[np.ndarray] = []
            reverse_raw_parts: list[np.ndarray] = []
            for first in range(0, len(starts), SCORING_BATCH_SIZE):
                selected = starts[first : first + SCORING_BATCH_SIZE]
                matrix = selected[:, None] + np.arange(6, dtype=np.int64)[None, :]
                forward, reverse, events, seconds = _edge_tokens(day, matrix)
                forward_tensor = torch.from_numpy(scaler.transform(forward))
                reverse_tensor = torch.from_numpy(scaler.transform(reverse))
                raw = (
                    UPSTREAM.antisymmetric_score(model, forward_tensor, reverse_tensor)
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
                reverse_raw = (
                    UPSTREAM.antisymmetric_score(model, reverse_tensor, forward_tensor)
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
                calibrated = alpha * raw
                raw_parts.append(raw)
                calibrated_parts.append(calibrated)
                reverse_raw_parts.append(reverse_raw)
                accumulators["uncapped"].update(calibrated, events, seconds)
                accumulators["bounded_log99"].update(
                    np.clip(calibrated, -BOUND, BOUND), events, seconds
                )
            raw_value = np.concatenate(raw_parts)
            calibrated_value = np.concatenate(calibrated_parts)
            reverse_raw_value = np.concatenate(reverse_raw_parts)
            reverse_calibrated_value = alpha * reverse_raw_value
            expected = stored.filter(pc.equal(stored["source_date"], source_date))
            expected_reverse = reverse_stored.filter(
                pc.equal(reverse_stored["source_date"], source_date)
            )
            expected_index = np.asarray(expected["start_block_index"], dtype=np.int64)
            expected_pair_id = np.asarray(
                [
                    f"{source_date}:B10:K6:{int(value)}"
                    for value in day.block_index[starts]
                ],
                dtype=object,
            )
            expected_events = (
                day.intervals[
                    starts[:, None] + np.arange(6)[None, :],
                    UPSTREAM.INTERVAL_INDEX["realized_source_event_count"],
                ]
                .sum(axis=1)
                .astype(np.int64)
            )
            expected_seconds = (
                day.intervals[
                    starts[:, None] + np.arange(6)[None, :],
                    UPSTREAM.INTERVAL_INDEX["elapsed_ns"],
                ].sum(axis=1)
                / 1e9
            )
            identity = (
                len(expected) == len(starts)
                and np.array_equal(expected_index, day.block_index[starts])
                and np.array_equal(
                    np.asarray(expected["pair_id"], dtype=object), expected_pair_id
                )
                and np.array_equal(
                    np.asarray(expected_reverse["pair_id"], dtype=object),
                    expected_pair_id,
                )
                and np.array_equal(
                    np.asarray(expected["path_start_ns"]), day.block_start_ns[starts]
                )
                and np.array_equal(
                    np.asarray(expected["path_end_ns"]), day.block_end_ns[starts + 5]
                )
                and np.array_equal(
                    np.asarray(expected_reverse["start_block_index"], dtype=np.int64),
                    day.block_index[starts],
                )
                and np.array_equal(
                    np.asarray(expected_reverse["path_start_ns"], dtype=np.int64),
                    day.block_start_ns[starts],
                )
                and np.array_equal(
                    np.asarray(expected_reverse["path_end_ns"], dtype=np.int64),
                    day.block_end_ns[starts + 5],
                )
                and np.array_equal(
                    np.asarray(expected["path_event_count"], dtype=np.int64),
                    expected_events,
                )
                and np.array_equal(
                    np.asarray(expected["duration_seconds"], dtype=np.float64),
                    expected_seconds,
                )
                and np.array_equal(
                    np.asarray(expected_reverse["path_event_count"], dtype=np.int64),
                    expected_events,
                )
                and np.array_equal(
                    np.asarray(expected_reverse["duration_seconds"], dtype=np.float64),
                    expected_seconds,
                )
                and set(np.asarray(expected["orientation"], dtype=object))
                == {"forward"}
                and set(np.asarray(expected_reverse["orientation"], dtype=object))
                == {"reverse"}
                and set(np.asarray(expected["label"], dtype=np.uint8).tolist()) == {1}
                and set(np.asarray(expected_reverse["label"], dtype=np.uint8).tolist())
                == {0}
                and set(np.asarray(expected["target_b"], dtype=np.int16).tolist())
                == {10}
                and set(np.asarray(expected["k"], dtype=np.int16).tolist()) == {6}
                and set(np.asarray(expected["model"], dtype=object).tolist())
                == {"ordered_tcn"}
                and set(np.asarray(expected["seed"], dtype=np.int64).tolist())
                == {model_seed}
                and np.array_equal(
                    np.asarray(
                        expected_reverse["raw_antisymmetric_score"], dtype=np.float64
                    ),
                    -np.asarray(expected["raw_antisymmetric_score"], dtype=np.float64),
                )
                and np.array_equal(
                    np.asarray(
                        expected_reverse["calibrated_antisymmetric_score"],
                        dtype=np.float64,
                    ),
                    -np.asarray(
                        expected["calibrated_antisymmetric_score"], dtype=np.float64
                    ),
                )
                and np.allclose(
                    np.asarray(
                        expected_reverse["probability_forward"], dtype=np.float64
                    ),
                    1.0 - np.asarray(expected["probability_forward"], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-15,
                )
            )
            if not identity:
                raise RuntimeError(f"K6 parity path identity mismatch: {source_date}")
            raw_error = np.abs(
                raw_value
                - np.asarray(expected["raw_antisymmetric_score"], dtype=np.float64)
            )
            calibrated_error = np.abs(
                calibrated_value
                - np.asarray(
                    expected["calibrated_antisymmetric_score"], dtype=np.float64
                )
            )
            reverse_raw_error = np.abs(
                reverse_raw_value
                - np.asarray(
                    expected_reverse["raw_antisymmetric_score"], dtype=np.float64
                )
            )
            reverse_calibrated_error = np.abs(
                reverse_calibrated_value
                - np.asarray(
                    expected_reverse["calibrated_antisymmetric_score"], dtype=np.float64
                )
            )
            reverse_probability = 1.0 / (
                1.0 + np.exp(-np.clip(reverse_calibrated_value, -700.0, 700.0))
            )
            reverse_probability_error = np.abs(
                reverse_probability
                - np.asarray(expected_reverse["probability_forward"], dtype=np.float64)
            )
            current_antisymmetry = np.abs(raw_value + reverse_raw_value)
            if not all(
                np.isfinite(value).all()
                for value in (
                    raw_error,
                    calibrated_error,
                    reverse_raw_error,
                    reverse_calibrated_error,
                    reverse_probability_error,
                    current_antisymmetry,
                )
            ):
                raise RuntimeError("nonfinite K6 parity score error")
            max_raw = max(max_raw, float(np.max(raw_error)))
            max_calibrated = max(max_calibrated, float(np.max(calibrated_error)))
            max_reverse_raw = max(max_reverse_raw, float(np.max(reverse_raw_error)))
            max_reverse_calibrated = max(
                max_reverse_calibrated, float(np.max(reverse_calibrated_error))
            )
            max_reverse_probability = max(
                max_reverse_probability, float(np.max(reverse_probability_error))
            )
            max_current_antisymmetry = max(
                max_current_antisymmetry, float(np.max(current_antisymmetry))
            )
            day_result: dict[str, Any] = {}
            for contract, accumulator in accumulators.items():
                observed = accumulator.final(6)
                reference = metric_index[(source_date, contract)]
                fields = {
                    "nats_path": "nats_path",
                    "nats_block": "nats_block",
                    "nats_source_event": "nats_event",
                    "nats_second": "nats_second",
                    "paired_log_loss": "paired_log_loss",
                    "posterior_saturation_share": "posterior_saturation_share",
                    "calibrated_ovl": "calibrated_ovl",
                    "reverse_like_share": "reverse_like_share",
                }
                errors = {
                    left: abs(float(observed[left]) - float(reference[right]))
                    for left, right in fields.items()
                }
                if not np.isfinite(
                    np.asarray(list(errors.values()), dtype=np.float64)
                ).all():
                    raise RuntimeError("nonfinite K6 parity metric error")
                max_metric = max(max_metric, max(errors.values()))
                day_result[contract] = errors
                if int(observed["paths"]) != int(reference["n_paths"]) or int(
                    observed["source_event_sum"]
                ) != int(reference["realized_source_events"]):
                    raise RuntimeError("K6 parity denominator mismatch")
            by_day[source_date] = day_result
            rows += len(starts)
    calibration = json.loads(
        (model_root / "calibration.json").read_text(encoding="utf-8")
    )
    alpha_error = abs(alpha - float(calibration["alpha"]))
    finite = bool(
        np.isfinite(
            [
                max_raw,
                max_calibrated,
                max_reverse_raw,
                max_reverse_calibrated,
                max_reverse_probability,
                max_current_antisymmetry,
                alpha_error,
                max_metric,
            ]
        ).all()
    )
    passed = bool(
        finite
        and rows == EXPECTED_PATHS[6]["test"]
        and max_raw < 1e-6
        and max_calibrated < 1e-6
        and max_reverse_raw < 1e-6
        and max_reverse_calibrated < 1e-6
        and max_reverse_probability < 1e-6
        and max_current_antisymmetry < 1e-6
        and alpha_error < 1e-12
        and max_metric < 1e-8
    )
    result = {
        "status": "passed" if passed else "failed",
        "model_seed": model_seed,
        "rows": rows,
        "model_unit_id": f"B=10_K=6/ordered_tcn/seed={model_seed}",
        "prediction_bytes": (model_root / "test_predictions.parquet").stat().st_size,
        "raw_max_abs_error": max_raw,
        "calibrated_max_abs_error": max_calibrated,
        "reverse_raw_max_abs_error": max_reverse_raw,
        "reverse_calibrated_max_abs_error": max_reverse_calibrated,
        "reverse_probability_max_abs_error": max_reverse_probability,
        "current_antisymmetry_max_abs_error": max_current_antisymmetry,
        "alpha_abs_error": alpha_error,
        "metric_max_abs_error": max_metric,
        "by_day": by_day,
        "finite": finite,
        "manifest_identity": {
            "status": manifest["status"],
            "B": manifest["target_b"],
            "K": manifest["k"],
            "model": manifest["model"],
            "seed": manifest["seed"],
        },
        "thresholds": {
            "raw": 1e-6,
            "calibrated": 1e-6,
            "reverse_raw": 1e-6,
            "reverse_calibrated": 1e-6,
            "reverse_probability": 1e-6,
            "current_antisymmetry": 1e-6,
            "alpha": 1e-12,
            "metric": 1e-8,
        },
    }
    atomic_json(output_path, result)
    if not passed:
        raise RuntimeError(f"K6 CPU parity failed: {result}")
    return result


def _balanced_accuracy(labels: np.ndarray, probability: np.ndarray) -> float:
    predicted = probability >= 0.5
    return float(
        0.5 * (np.mean(predicted[labels == 1]) + np.mean(~predicted[labels == 0]))
    )


def _midrank_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    first = 0
    while first < len(order):
        last = first + 1
        while (
            last < len(order) and probability[order[last]] == probability[order[first]]
        ):
            last += 1
        ranks[order[first:last]] = 0.5 * (first + 1 + last)
        first = last
    n1 = int(np.count_nonzero(labels == 1))
    n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def diagnostic_logistic_oof(tokens: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {
            "status": "unavailable",
            "balanced_accuracy": None,
            "roc_auc": None,
            "labels": None,
            "probability": None,
        }
    x = np.repeat(tokens.astype(np.float64), 2, axis=0)
    y = np.tile(np.asarray([0, 1], dtype=np.int8), len(tokens))
    pair_groups = np.repeat(groups, 2)
    probability = np.empty(len(y), dtype=np.float64)
    status = "complete"
    for fold in range(5):
        test = pair_groups == fold
        train = ~test
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            tol=1e-10,
            max_iter=1000,
            fit_intercept=True,
            class_weight=None,
        )
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        if int(model.n_iter_[0]) >= 1000:
            status = "nonconverged"
    if not np.isfinite(probability).all():
        return {
            "status": "unavailable",
            "balanced_accuracy": None,
            "roc_auc": None,
            "labels": None,
            "probability": None,
        }
    return {
        "status": status,
        "balanced_accuracy": _balanced_accuracy(y, probability),
        "roc_auc": _midrank_auc(y, probability),
        "labels": y,
        "probability": probability,
    }


def diagnostic_logistic(tokens: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    value = diagnostic_logistic_oof(tokens, groups)
    return {key: value[key] for key in ("status", "balanced_accuracy", "roc_auc")}


def pooled_logistic_oof(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(value["status"]) for value in values]
    if any(status == "unavailable" for status in statuses):
        return {"status": "unavailable", "balanced_accuracy": None, "roc_auc": None}
    labels = np.concatenate(
        [np.asarray(value["labels"], dtype=np.int8) for value in values]
    )
    probability = np.concatenate(
        [np.asarray(value["probability"], dtype=np.float64) for value in values]
    )
    status = "nonconverged" if "nonconverged" in statuses else "complete"
    return {
        "status": status,
        "balanced_accuracy": _balanced_accuracy(labels, probability),
        "roc_auc": _midrank_auc(labels, probability),
    }


def synthetic_smoke(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    state_vertices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint32)
    states = np.column_stack((state_vertices[:-1], state_vertices[1:]))
    edges = np.arange(10, dtype=np.int64)
    first, audit_a = randomized_euler(
        edges,
        states,
        surrogate_seed=20260800,
        source_date="2026-07-21",
        segment_ordinal=0,
    )
    second, audit_b = randomized_euler(
        edges,
        states,
        surrogate_seed=20260800,
        source_date="2026-07-21",
        segment_ordinal=0,
    )
    if not np.array_equal(first, second) or set(first) != set(edges):
        raise RuntimeError("H2 Euler determinism smoke failed")
    resume_root = root / "traversal-checkpoint"
    try:
        randomized_euler(
            edges,
            states,
            surrogate_seed=20260800,
            source_date="2026-07-21",
            segment_ordinal=0,
            checkpoint_root=resume_root,
            checkpoint_edges=2,
            interrupt_after_edges=3,
        )
    except InterruptedError:
        pass
    else:
        raise RuntimeError("H2 Euler forced-interruption smoke did not interrupt")
    resumed, resumed_audit = randomized_euler(
        edges,
        states,
        surrogate_seed=20260800,
        source_date="2026-07-21",
        segment_ordinal=0,
        checkpoint_root=resume_root,
        checkpoint_edges=2,
    )
    if not np.array_equal(first, resumed) or audit_a != resumed_audit:
        raise RuntimeError("H2 Euler checkpoint/resume smoke failed")
    table = pa.Table.from_pylist(
        [
            {
                "source_date": "2026-07-21",
                "segment_id": 0,
                "surrogate_position": i,
                "source_block_index": int(edge),
                "coarse_from_state": int(states[edge, 0]),
                "coarse_to_state": int(states[edge, 1]),
            }
            for i, edge in enumerate(first)
        ],
        schema=PERMUTATION_SCHEMA,
    )
    atomic_parquet(root / "permutation.parquet", table, PERMUTATION_SCHEMA)
    accumulator = MetricAccumulator()
    score = np.asarray([1.0, -0.5, 0.25])
    accumulator.update(score, np.asarray([10, 11, 12]), np.asarray([0.1, 0.2, 0.3]))
    metric = accumulator.final(2)
    diagnostic = diagnostic_logistic(np.zeros((10, 37)), np.repeat(np.arange(5), 2))
    boundary = np.ones((4, len(UPSTREAM.BOUNDARY_FIELDS)), dtype=np.float64)
    boundary[1] = 9.0
    boundary[2] = boundary[0]
    end_boundary = boundary.copy()
    end_boundary[1] = 8.0
    interval = np.ones((4, len(UPSTREAM.INTERVAL_FIELDS)), dtype=np.float64)
    interval[0] *= 100.0
    interval[:, UPSTREAM.INTERVAL_INDEX["elapsed_ns"]] = 1_000_000.0
    segment_day = SegmentableDay(
        "2026-07-21",
        np.arange(4),
        np.asarray([1, 2, 10, 11]) * 1_000_000_000,
        np.asarray([2, 3, 11, 12]) * 1_000_000_000,
        boundary,
        end_boundary,
        interval,
        ((0, 2), (2, 4)),
    )
    definition = CoarseStateDefinition(
        cutpoints={axis: (1.0, 2.0) for axis in STATE_AXES}
    )
    segment_states = state_ids(segment_day, definition)
    segment_reset = bool(
        segment_states[2, 0] == segment_states[0, 0]
        and segment_states[2, 0] != segment_states[1, 1]
    )
    expected_k2 = _original_path_count(segment_day, 2)
    path_rows = [
        {
            "source_date": "2026-07-21",
            "segment_id": sid,
            "surrogate_position": pos,
            "source_block_index": pos,
            "coarse_from_state": int(segment_states[pos, 0]),
            "coarse_to_state": int(segment_states[pos, 1]),
        }
        for sid, (a, b) in enumerate(segment_day.segments)
        for pos in range(a, b)
    ]
    path_table = pa.Table.from_pylist(path_rows, schema=PERMUTATION_SCHEMA)
    wrong_path_count_rejected = (
        _path_count(path_table.slice(0, 1), "2026-07-21", 2) != expected_k2
    )
    nan_rejected = False
    try:
        MetricAccumulator().update(
            np.asarray([np.nan]), np.asarray([1.0]), np.asarray([1.0])
        )
    except RuntimeError:
        nan_rejected = True
    nullability_rejected = False
    bad = table.set_column(
        table.schema.get_field_index("source_date"),
        "source_date",
        pa.array([None] + ["2026-07-21"] * 9, type=pa.string()),
    )
    try:
        validate_arrow_contract(bad, PERMUTATION_SCHEMA)
    except RuntimeError:
        nullability_rejected = True
    duplicate_key_rejected = False
    duplicate_blocks = np.asarray(table["source_block_index"], dtype=np.int64).copy()
    duplicate_blocks[-1] = duplicate_blocks[0]
    duplicate_table = table.set_column(
        table.schema.get_field_index("source_block_index"),
        "source_block_index",
        pa.array(duplicate_blocks, type=table.schema.field("source_block_index").type),
    )
    try:
        _validate_permutation_keys(duplicate_table, "2026-07-21")
    except RuntimeError:
        duplicate_key_rejected = True
    alien_rows = [
        {
            "source_date": "2026-07-21",
            "segment_id": 0,
            "surrogate_position": i,
            "source_block_index": block,
            "coarse_from_state": 0,
            "coarse_to_state": 0,
        }
        for i, block in enumerate([0, 1, 2, 3, 999])
    ]
    alien_table = pa.Table.from_pylist(alien_rows, schema=PERMUTATION_SCHEMA)
    alien_key_rejected = False
    try:
        _validate_permutation_keys(
            alien_table, "2026-07-21", np.arange(5, dtype=np.int64)
        )
    except RuntimeError:
        alien_key_rejected = True
    projection_root = root / "projection-input"
    projection_paths = []
    for index, values in enumerate(([0, 1, 2], [3, 4])):
        path = projection_root / f"part-{index}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "block_index": pa.array(values, type=pa.int64()),
                    "unused": pa.array([1] * len(values), type=pa.int64()),
                }
            ),
            path,
        )
        projection_paths.append(path)
    resolver = lambda _root, _date, _b: projection_paths
    projected_first = _expected_block_ids(projection_root, "2026-07-21", resolver)
    projected_second = _expected_block_ids(projection_root, "2026-07-21", resolver)
    projected_block_id_cache = (
        projected_first is projected_second
        and np.array_equal(projected_first, np.arange(5, dtype=np.int64))
        and projected_first.flags.writeable is False
    )
    portable_roots = (root / "host-a", root / "host-b")
    input_identities = []
    model_identities = []
    permutation_identities = []
    for host_root in portable_roots:
        input_paths = []
        for source_date in TEST_DATES:
            path = host_root / "inputs" / f"date={source_date}" / "part-0.parquet"
            _atomic_text(path, f"fixture-{source_date}")
            input_paths.append(path)
        input_identities.append(
            _logical_input_inventory(host_root / "inputs", TEST_DATES, input_paths)
        )
        model_root = (
            host_root / "models" / "B=10_K=6" / "ordered_tcn" / f"seed={MODEL_SEEDS[0]}"
        )
        artifacts = []
        for name in ("model.pt", "scaler.json", "calibration.json"):
            artifact = model_root / name
            _atomic_text(artifact, f"fixture-{name}")
            artifacts.append({"filename": name, "bytes": artifact.stat().st_size})
        atomic_json(
            model_root / "complete.json",
            {
                "status": "complete",
                "logical_unit_id": f"B=10_K=6/ordered_tcn/seed={MODEL_SEEDS[0]}",
                "artifacts": artifacts,
            },
        )
        config = ScoringConfig(
            k=6,
            model_seed=MODEL_SEEDS[0],
            surrogate_seed=SURROGATE_SEEDS[0],
            device="cpu",
            input_root=host_root / "inputs",
            permutation_root=host_root / "permutations",
            model_root=model_root,
            output_root=host_root / "units",
            work_root=host_root / "checkpoints",
        )
        model_identities.append(model_validation_identity(config, device="cpu"))
        for source_date in TEST_DATES:
            unit = (
                config.permutation_root
                / f"replicate={SURROGATE_SEEDS[0]}"
                / f"source_date={source_date}"
            )
            unit.mkdir(parents=True, exist_ok=True)
            row = {
                "source_date": source_date,
                "segment_id": 0,
                "surrogate_position": 0,
                "source_block_index": 0,
                "coarse_from_state": 0,
                "coarse_to_state": 0,
            }
            permutation_path = unit / "permutation.parquet"
            atomic_parquet(
                permutation_path,
                pa.Table.from_pylist([row], schema=PERMUTATION_SCHEMA),
                PERMUTATION_SCHEMA,
            )
            atomic_json(
                unit / "complete.json",
                {
                    "status": "complete",
                    "unit_id": f"replicate={SURROGATE_SEEDS[0]}/source_date={source_date}",
                    "scientific_config": {"contract": "phase04_h2_construction_v1"},
                    "artifacts": [
                        {
                            "filename": "permutation.parquet",
                            "rows": 1,
                            "bytes": permutation_path.stat().st_size,
                        }
                    ],
                },
            )
        permutation_identities.append(
            _permutation_inventory(
                config,
                {
                    source_date: np.asarray([0], dtype=np.int64)
                    for source_date in TEST_DATES
                },
            )
        )
    physical_root_independent = (
        input_identities[0] == input_identities[1]
        and model_identities[0] == model_identities[1]
        and permutation_identities[0] == permutation_identities[1]
    )
    changed_share = audit_a["changed"] / audit_a["eligible"]
    if (
        audit_a != audit_b
        or changed_share < 0.75
        or metric["paths"] != 3
        or pq.ParquetFile(root / "permutation.parquet").metadata.num_rows != 10
        or not all(
            (
                segment_reset,
                wrong_path_count_rejected,
                nan_rejected,
                nullability_rejected,
                duplicate_key_rejected,
                alien_key_rejected,
                projected_block_id_cache,
                physical_root_independent,
            )
        )
    ):
        raise RuntimeError("H2 bounded smoke failed")
    return {
        "passed": True,
        "euler_deterministic": True,
        "checkpoint_resume_exact": True,
        "edge_multiset_exact": True,
        "multi_segment_boundary_reset": segment_reset,
        "wrong_path_count_rejected": wrong_path_count_rejected,
        "nan_rejected": nan_rejected,
        "nullability_rejected": nullability_rejected,
        "duplicate_or_missing_key_rejected": duplicate_key_rejected,
        "alien_block_key_rejected": alien_key_rejected,
        "projected_block_id_cache": projected_block_id_cache,
        "physical_root_independent_identity": physical_root_independent,
        "changed_linkage_share": changed_share,
        "audit": audit_a,
        "metric_paths": 3,
        "diagnostic": diagnostic,
    }
