"""Phase 14 QR-A transition-preserving surrogate construction and scoring.

The scientific randomization is the Phase 04 directed-transition multigraph
randomized Euler traversal.  This adapter changes only the admitted input and
model envelopes: QR-A B10 blocks, K=6, full37, the frozen 100% QR-A ordered-TCN,
five fixed surrogate replicates, and the three fixed Phase 14 test dates.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import torch

from src.cuda_runtime import UPSTREAM, configure_determinism, device_or_raise
from src.qr_simulator_runtime import (
    CANONICAL_BLOCK_SCHEMA,
    EVENT_SCHEMA,
    MESSAGE_SCHEMA,
)
from src.simulator_training_runtime import TrainingUnit
from src.transition_surrogate_runtime import (
    CoarseStateDefinition,
    MetricAccumulator,
    PERMUTATION_SCHEMA,
    SegmentableDay,
    _edge_tokens,
    _starts_for_permutation,
    randomized_euler,
    state_ids,
)


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
DEVELOPMENT_DATES = DATES[:9]
TEST_DATES = DATES[12:]
MODEL_SEEDS = (20260730, 20260731, 20260732)
SURROGATE_SEEDS = tuple(range(20260800, 20260805))
K = 6
TARGET_B = 10
BOUND = float(np.log(99.0))
CONSTRUCTION_CHECKPOINT_EDGES = 250_000
SCORING_CHECKPOINT_PATHS = 131_072
ROW_GROUP_SIZE = 65_536
SCORING_BATCH_SIZE = 1_024
CONTRACT = "phase14_qr_transition_surrogate_v1"
_STATE_FIT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _strict_schema(fields: Sequence[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in fields])


CONTRAST_SCHEMA = _strict_schema(
    [
        ("model_seed", pa.int64()),
        ("surrogate_seed", pa.int64()),
        ("source_date", pa.string()),
        ("score_contract", pa.string()),
        ("estimand_role", pa.string()),
        ("original_paths", pa.int64()),
        ("surrogate_paths", pa.int64()),
        ("original_source_events", pa.int64()),
        ("surrogate_source_events", pa.int64()),
        ("original_nats_event", pa.float64()),
        ("surrogate_nats_event", pa.float64()),
        ("delta_nats_event", pa.float64()),
    ]
)


@dataclass(frozen=True)
class ConstructionConfig:
    surrogate_seed: int
    source_date: str
    simulation_root: Path
    state_path: Path
    output_root: Path
    work_root: Path
    resume: bool = True
    checkpoint_edges: int = CONSTRUCTION_CHECKPOINT_EDGES
    row_group_size: int = ROW_GROUP_SIZE
    interrupt_after_edges: int | None = None

    @property
    def unit_id(self) -> str:
        return f"replicate={self.surrogate_seed}/source_date={self.source_date}"


@dataclass(frozen=True)
class ScoringConfig:
    model_seed: int
    surrogate_seed: int
    device: str
    simulation_root: Path
    permutation_root: Path
    model_root: Path
    output_root: Path
    work_root: Path
    resume: bool = True
    batch_size: int = SCORING_BATCH_SIZE
    checkpoint_paths: int = SCORING_CHECKPOINT_PATHS
    interrupt_after_paths: int | None = None
    selectors_path: Path | None = None

    @property
    def unit_id(self) -> str:
        return f"model_seed={self.model_seed}/replicate={self.surrogate_seed}"


def construction_units() -> tuple[tuple[int, str], ...]:
    result = tuple(
        (seed, source_date) for seed in SURROGATE_SEEDS for source_date in TEST_DATES
    )
    if len(result) != 15 or len(set(result)) != 15:
        raise AssertionError("QR surrogate construction universe is not exact 15")
    return result


def scoring_units() -> tuple[tuple[int, int], ...]:
    result = tuple(
        (model_seed, surrogate_seed)
        for model_seed in MODEL_SEEDS
        for surrogate_seed in SURROGATE_SEEDS
    )
    if len(result) != 15 or len(set(result)) != 15:
        raise AssertionError("QR surrogate scoring universe is not exact 15")
    return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=True):
        raise RuntimeError("Arrow schema mismatch before publication")
    if sum(column.null_count for column in table.columns):
        raise RuntimeError("null in strict surrogate artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table, temporary, compression="zstd", row_group_size=ROW_GROUP_SIZE
        )
        observed = pq.ParquetFile(temporary)
        if (
            observed.metadata.num_rows != table.num_rows
            or not observed.schema_arrow.equals(schema, check_metadata=True)
        ):
            raise RuntimeError("Parquet footer/schema mismatch before publication")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress(path: Path, event: str, **fields: Any) -> None:
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        stream.flush()
    print(json.dumps(_jsonable(record), sort_keys=True), flush=True)


def _lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _prepare_work(
    work: Path, exact: Mapping[str, Any], resume: bool, work_root: Path
) -> None:
    config_path = work / "config.json"
    if not work.exists():
        return
    observed = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else None
    )
    if resume and observed == _jsonable(exact):
        return
    quarantine = (
        work_root
        / "quarantine"
        / f"{work.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, quarantine)


def _day_paths(simulation_root: Path, source_date: str) -> tuple[Path, dict[str, Any]]:
    root = simulation_root / f"source_date={source_date}"
    completion_path = root / "complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    relative = f"blocks/spec=qr_a/source_date={source_date}/part-00000.parquet"
    support = completion.get("block_support", {}).get("qr_a", {})
    if (
        completion.get("status") != "complete"
        or completion.get("source_date") != source_date
        or completion.get("universe") != 0
        or completion.get("training_admission") != "passed"
        or completion.get("qa", {}).get("passed") is not True
        or support.get("passed") is not True
    ):
        raise RuntimeError(f"QR-A simulation completion not admitted: {source_date}")
    path = root / relative
    expected_rows = completion.get("output_rows", {}).get(relative)
    expected_bytes = completion.get("output_bytes", {}).get(relative)
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or pq.ParquetFile(path).metadata.num_rows != expected_rows
        or not pq.ParquetFile(path).schema_arrow.equals(
            CANONICAL_BLOCK_SCHEMA, check_metadata=True
        )
    ):
        raise RuntimeError(f"QR-A block inventory/schema mismatch: {path}")
    identity = {
        "source_date": source_date,
        "relative_path": relative,
        "rows": int(expected_rows),
        "bytes": int(expected_bytes),
        "completion_bytes": completion_path.stat().st_size,
        "runtime_identity": completion.get("runtime_identity"),
    }
    return path, identity


def _input_inventory(simulation_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    entries = [_day_paths(simulation_root, source_date)[1] for source_date in dates]
    return {
        "contract": "phase14_qr_a_b10_completed_days_v1",
        "spec": "qr_a",
        "dates": list(dates),
        "files": len(entries),
        "rows": sum(item["rows"] for item in entries),
        "bytes": sum(item["bytes"] for item in entries),
        "partitions": entries,
    }


def read_segmentable_day(simulation_root: Path, source_date: str) -> SegmentableDay:
    path, _identity = _day_paths(simulation_root, source_date)
    parquet = pq.ParquetFile(path)
    table = parquet.read(columns=list(UPSTREAM.REQUIRED_COLUMNS))
    block_index = np.asarray(table["block_index"], dtype=np.int64)
    if not len(block_index) or (
        len(block_index) > 1 and np.any(np.diff(block_index) <= 0)
    ):
        raise RuntimeError(f"invalid QR-A block cursor: {source_date}")

    def vector(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.int64)

    def matrix(names: Sequence[str]) -> np.ndarray:
        result = np.column_stack(
            [np.asarray(table[name], dtype=np.float64) for name in names]
        )
        if not np.isfinite(result).all():
            raise RuntimeError(f"nonfinite QR-A block input: {source_date}")
        return result

    starts = matrix([f"start_{field}" for field in UPSTREAM.BOUNDARY_FIELDS])
    ends = matrix([f"end_{field}" for field in UPSTREAM.BOUNDARY_FIELDS])
    intervals = matrix(list(UPSTREAM.INTERVAL_FIELDS))
    block_start, block_end = vector("block_start_ns"), vector("block_end_ns")
    if np.any(block_end < block_start):
        raise RuntimeError("negative QR-A block duration")
    event_index = UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]
    if np.any(intervals[:, event_index] != TARGET_B):
        raise RuntimeError("QR-A input is not exact B10")
    breaks = (
        np.flatnonzero(
            (np.diff(block_index) != 1)
            | (block_end[:-1] != block_start[1:])
            | np.any(ends[:-1] != starts[1:], axis=1)
        )
        + 1
    )
    bounds = np.concatenate(([0], breaks, [len(block_index)]))
    segments = tuple(
        (int(first), int(last))
        for first, last in zip(bounds[:-1], bounds[1:], strict=True)
        if last > first
    )
    return SegmentableDay(
        source_date,
        block_index,
        block_start,
        block_end,
        starts,
        ends,
        intervals,
        segments,
    )


def _numeric_axes(day: SegmentableDay) -> dict[str, np.ndarray]:
    boundary = {
        name: UPSTREAM.BOUNDARY_FIELDS.index(name)
        for name in ("spread_ticks", "bid_depth_l5", "ask_depth_l5", "imbalance_l1")
    }
    interval = UPSTREAM.INTERVAL_INDEX
    return {
        "spread": day.ends[:, boundary["spread_ticks"]],
        "depth": np.log1p(
            day.ends[:, boundary["bid_depth_l5"]]
            + day.ends[:, boundary["ask_depth_l5"]]
        ),
        "imbalance": day.ends[:, boundary["imbalance_l1"]],
        "recent_activity": np.log1p(
            day.intervals[:, interval["realized_source_event_count"]]
        ),
        "recent_volatility": np.log1p(
            day.intervals[:, interval["mid_abs_variation_half_ticks"]]
        ),
        "recent_duration": np.log1p(day.intervals[:, interval["elapsed_ns"]] / 1e9),
    }


def fit_coarse_state(simulation_root: Path) -> dict[str, Any]:
    axes = (
        "spread",
        "depth",
        "imbalance",
        "recent_activity",
        "recent_volatility",
        "recent_duration",
    )
    values: dict[str, list[np.ndarray]] = {axis: [] for axis in axes}
    rows = 0
    for source_date in DEVELOPMENT_DATES:
        day = read_segmentable_day(simulation_root, source_date)
        rows += len(day.block_index)
        for axis, array in _numeric_axes(day).items():
            values[axis].append(array)
    cutpoints = {
        axis: tuple(
            np.quantile(np.concatenate(parts), (1 / 3, 2 / 3), method="linear").astype(
                np.float64
            )
        )
        for axis, parts in values.items()
    }
    if any(not np.isfinite(pair).all() for pair in cutpoints.values()):
        raise RuntimeError("nonfinite QR coarse-state cutpoint")
    definition = CoarseStateDefinition(
        cutpoints=cutpoints,
        development_dates=DEVELOPMENT_DATES,
        state_contract="phase14_qr_a_h2_coarse_state_v1",
    )
    return {
        "status": "fitted",
        "contract": CONTRACT,
        "development_rows": rows,
        "input": _input_inventory(simulation_root, DEVELOPMENT_DATES),
        **definition.to_dict(),
    }


def publish_coarse_state(simulation_root: Path, state_path: Path) -> dict[str, Any]:
    value = fit_coarse_state(simulation_root)
    if state_path.is_file():
        observed = json.loads(state_path.read_text(encoding="utf-8"))
        if observed != _jsonable(value):
            raise RuntimeError(
                "existing coarse-state artifact differs from exact development fit"
            )
        return observed
    _atomic_json(state_path, value)
    return value


def _state_cache_key(simulation_root: Path) -> tuple[Any, ...]:
    values: list[Any] = [str(simulation_root.resolve())]
    for source_date in DEVELOPMENT_DATES:
        path, _identity = _day_paths(simulation_root, source_date)
        completion = simulation_root / f"source_date={source_date}" / "complete.json"
        for item in (path, completion):
            stat = item.stat()
            values.extend((str(item.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(values)


def _current_coarse_state(simulation_root: Path) -> dict[str, Any]:
    key = _state_cache_key(simulation_root)
    cached = _STATE_FIT_CACHE.get(key)
    if cached is None:
        cached = fit_coarse_state(simulation_root)
        _STATE_FIT_CACHE.clear()
        _STATE_FIT_CACHE[key] = cached
    return cached


def _development_input_stat_identity(
    simulation_root: Path,
) -> tuple[dict[str, Any], ...]:
    """Full identities guarding a current semantic refit against concurrent replacement."""

    result: list[dict[str, Any]] = []
    for source_date in DEVELOPMENT_DATES:
        block_path, _identity = _day_paths(simulation_root, source_date)
        completion_path = (
            simulation_root / f"source_date={source_date}" / "complete.json"
        )
        for role, path in (
            ("qr_a_blocks", block_path),
            ("completion", completion_path),
        ):
            stat = path.stat()
            result.append(
                {
                    "source_date": source_date,
                    "role": role,
                    "path": str(path.resolve()),
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "ctime_ns": int(stat.st_ctime_ns),
                }
            )
    return tuple(result)


def _definition_from_state_value(value: Mapping[str, Any]) -> CoarseStateDefinition:
    definition = CoarseStateDefinition.from_dict(value)
    if (
        value.get("status") != "fitted"
        or value.get("contract") != CONTRACT
        or definition.development_dates != DEVELOPMENT_DATES
        or definition.state_contract != "phase14_qr_a_h2_coarse_state_v1"
    ):
        raise RuntimeError(
            "coarse state is not the frozen QR development-only definition"
        )
    return definition


def _load_state(
    path: Path, simulation_root: Path | None = None
) -> CoarseStateDefinition:
    value = json.loads(path.read_text(encoding="utf-8"))
    definition = _definition_from_state_value(value)
    if simulation_root is not None:
        fitted = _current_coarse_state(simulation_root)
        if value != _jsonable(fitted):
            raise RuntimeError(
                "coarse state differs from exact current development-only refit"
            )
    return definition


def validate_coarse_state(
    state_path: Path, simulation_root: Path
) -> CoarseStateDefinition:
    """Uncached refit with stable full input identities; admit exact equality only."""

    value = json.loads(state_path.read_text(encoding="utf-8"))
    definition = _definition_from_state_value(value)
    before = _development_input_stat_identity(simulation_root)
    fitted = fit_coarse_state(simulation_root)
    after = _development_input_stat_identity(simulation_root)
    if before != after:
        raise RuntimeError("development QR-A inputs changed during coarse-state refit")
    if value != _jsonable(fitted):
        raise RuntimeError(
            "coarse state differs from exact current development-only refit"
        )
    return definition


def _original_matrices(day: SegmentableDay) -> np.ndarray:
    offset = date.fromisoformat(day.source_date).toordinal() % 3
    parts = []
    for first, last in day.segments:
        candidates = np.arange(first, last - K + 1, dtype=np.int64)
        candidates = candidates[(candidates - offset) % 3 == 0]
        if len(candidates):
            parts.append(np.stack([candidates + shift for shift in range(K)], axis=1))
    return np.concatenate(parts) if parts else np.empty((0, K), dtype=np.int64)


def _transition_counts(
    order: np.ndarray, states: np.ndarray
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for edge in order:
        key = int(states[edge, 0]), int(states[edge, 1])
        result[key] = result.get(key, 0) + 1
    return result


def _linkage_counts(
    original: np.ndarray, surrogate: np.ndarray, states: np.ndarray
) -> dict[str, int]:
    if len(original) <= 2:
        return {
            "eligible": 0,
            "fixed": max(0, len(original) - 1),
            "endpoint": max(0, len(original) - 1),
            "changed": 0,
        }
    head = int(original[0])
    original_successor = {
        int(a): int(b) for a, b in zip(original[:-1], original[1:], strict=True)
    }
    surrogate_successor = {
        int(a): int(b) for a, b in zip(surrogate[:-1], surrogate[1:], strict=True)
    }
    endpoint_incoming = {head, int(original[-2]), int(surrogate[-2])}
    movable = original[1:-1]
    outdegree: dict[int, int] = {}
    for edge in movable:
        state = int(states[edge, 0])
        outdegree[state] = outdegree.get(state, 0) + 1
    eligible = fixed = changed = 0
    for incoming in sorted(set(original_successor) & set(surrogate_successor)):
        if incoming in endpoint_incoming:
            continue
        middle_state = int(states[incoming, 1])
        if outdegree.get(middle_state, 0) < 2:
            fixed += 1
            continue
        eligible += 1
        changed += original_successor[incoming] != surrogate_successor[incoming]
    return {
        "eligible": eligible,
        "fixed": fixed,
        "endpoint": len(endpoint_incoming),
        "changed": int(changed),
    }


def _recompute_construction_audit(
    day: SegmentableDay,
    definition: CoarseStateDefinition,
    table: pa.Table,
    surrogate_seed: int,
) -> dict[str, Any]:
    if not table.schema.equals(PERMUTATION_SCHEMA, check_metadata=True) or sum(
        column.null_count for column in table.columns
    ):
        raise RuntimeError("construction permutation schema/null mismatch")
    states = state_ids(day, definition)
    rows = table.to_pylist()
    orders: list[np.ndarray] = []
    eligible = fixed = endpoint = changed = 0
    for segment_id, (first, last) in enumerate(day.segments):
        selected = [row for row in rows if int(row["segment_id"]) == segment_id]
        expected_positions = list(range(first, last))
        if [
            int(row["surrogate_position"]) for row in selected
        ] != expected_positions or any(
            row["source_date"] != day.source_date for row in selected
        ):
            raise RuntimeError("permutation segment/position support mismatch")
        ids = np.asarray(
            [row["source_block_index"] for row in selected], dtype=np.int64
        )
        order = np.searchsorted(day.block_index, ids)
        if np.any(order >= len(day.block_index)) or not np.array_equal(
            day.block_index[order], ids
        ):
            raise RuntimeError("permutation contains alien block ID")
        if any(
            int(row["coarse_from_state"]) != int(states[edge, 0])
            or int(row["coarse_to_state"]) != int(states[edge, 1])
            for row, edge in zip(selected, order, strict=True)
        ):
            raise RuntimeError(
                "published coarse-state payload differs from current refit"
            )
        original_segment = np.arange(first, last, dtype=np.int64)
        if (
            not np.array_equal(np.sort(order), original_segment)
            or (
                len(order)
                and (
                    int(states[order[0], 0]) != int(states[first, 0])
                    or int(states[order[-1], 1]) != int(states[last - 1, 1])
                )
            )
            or (
                len(order) > 1 and np.any(states[order[:-1], 1] != states[order[1:], 0])
            )
        ):
            raise RuntimeError(
                "permutation traversal/head-tail/segment invariant failed"
            )
        linkage = _linkage_counts(original_segment, order, states)
        eligible += linkage["eligible"]
        fixed += linkage["fixed"]
        endpoint += linkage["endpoint"]
        changed += linkage["changed"]
        orders.append(order)
    if {int(row["segment_id"]) for row in rows} != set(range(len(day.segments))):
        raise RuntimeError("permutation contains alien segment")
    surrogate = np.concatenate(orders) if orders else np.asarray([], dtype=np.int64)
    original = np.arange(len(day.block_index), dtype=np.int64)
    original_counts, surrogate_counts = (
        _transition_counts(original, states),
        _transition_counts(surrogate, states),
    )
    keys = set(original_counts) | set(surrogate_counts)
    transition_difference = max(
        (
            abs(original_counts.get(key, 0) - surrogate_counts.get(key, 0))
            for key in keys
        ),
        default=0,
    )
    current_difference = max(
        (
            abs(
                original_counts.get((a, b), 0)
                - original_counts.get((b, a), 0)
                - surrogate_counts.get((a, b), 0)
                + surrogate_counts.get((b, a), 0)
            )
            for a, b in keys
        ),
        default=0,
    )
    _, matrices = _starts_for_permutation(table, day.source_date, K)
    original_paths = len(_original_matrices(day))
    share = changed / eligible if eligible else float("nan")
    return {
        "passed": bool(
            np.array_equal(np.sort(surrogate), original)
            and transition_difference == 0
            and current_difference == 0
            and len(matrices) == original_paths
            and eligible > 0
            and share >= 0.75
        ),
        "source_date": day.source_date,
        "surrogate_seed": surrogate_seed,
        "n_segments": len(day.segments),
        "n_edges": len(day.block_index),
        "eligible_links": eligible,
        "changed_links": changed,
        "fixed_links": fixed,
        "endpoint_excluded_links": endpoint,
        "changed_linkage_share": share,
        "edge_multiset_exact": bool(np.array_equal(np.sort(surrogate), original)),
        "payload_exact": True,
        "consumed_once": True,
        "endpoints_exact": True,
        "transition_max_abs_difference": transition_difference,
        "current_max_abs_difference": current_difference,
        "K6_paths": len(matrices),
        "original_K6_paths": original_paths,
    }


def _construction_exact(
    config: ConstructionConfig, definition: CoarseStateDefinition
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "stage": "construction",
        "K": K,
        "B": TARGET_B,
        "surrogate_seed": config.surrogate_seed,
        "source_date": config.source_date,
        "input": _input_inventory(config.simulation_root, (config.source_date,)),
        "state": definition.to_dict(),
        "checkpoint_edges": config.checkpoint_edges,
        "row_group_size": config.row_group_size,
        "rng": "numpy.PCG64.SeedSequence",
        "traversal": "phase04_fixed_head_tail_hierholzer_v1",
    }


def run_construction_unit(config: ConstructionConfig) -> dict[str, Any]:
    if (config.surrogate_seed, config.source_date) not in construction_units():
        raise ValueError("construction unit outside exact Phase 14 universe")
    if config.checkpoint_edges < 1 or config.row_group_size < 1:
        raise ValueError("invalid construction checkpoint/row-group size")
    definition = _load_state(config.state_path, config.simulation_root)
    final = (
        config.output_root
        / f"replicate={config.surrogate_seed}"
        / f"source_date={config.source_date}"
    )
    if (final / "complete.json").is_file():
        return validate_construction_complete(final, config, definition)
    work = config.work_root / "construction" / config.unit_id
    exact = _construction_exact(config, definition)
    _prepare_work(work, exact, config.resume, config.work_root)
    work.mkdir(parents=True, exist_ok=True)
    handle = _lock(work / ".unit.lock")
    started = time.monotonic()
    succeeded = False
    try:
        progress_path = work / "progress.jsonl"
        _progress(progress_path, "construction_start", unit_id=config.unit_id)
        if not (work / "config.json").is_file():
            _atomic_json(work / "config.json", exact)
        day = read_segmentable_day(config.simulation_root, config.source_date)
        states = state_ids(day, definition)
        parts_root = work / "parts"
        parts_root.mkdir(parents=True, exist_ok=True)
        tables: list[pa.Table] = []
        processed = 0
        for segment_id, (first, last) in enumerate(day.segments):
            part_path = parts_root / f"segment={segment_id}.parquet"
            audit_path = parts_root / f"segment={segment_id}.json"
            if part_path.is_file() and audit_path.is_file():
                part = pq.read_table(part_path)
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                ids = np.asarray(part["source_block_index"], dtype=np.int64)
                order = np.searchsorted(day.block_index, ids)
                if len(part) != last - first or not np.array_equal(
                    day.block_index[order], ids
                ):
                    raise RuntimeError("stale construction segment")
            else:
                order, audit = randomized_euler(
                    np.arange(first, last, dtype=np.int64),
                    states,
                    surrogate_seed=config.surrogate_seed,
                    source_date=config.source_date,
                    segment_ordinal=segment_id,
                    checkpoint_root=work / "traversal" / f"segment={segment_id}",
                    checkpoint_edges=config.checkpoint_edges,
                    interrupt_after_edges=config.interrupt_after_edges,
                )
                rows = [
                    {
                        "source_date": config.source_date,
                        "segment_id": segment_id,
                        "surrogate_position": first + local,
                        "source_block_index": int(day.block_index[edge]),
                        "coarse_from_state": int(states[edge, 0]),
                        "coarse_to_state": int(states[edge, 1]),
                    }
                    for local, edge in enumerate(order)
                ]
                part = pa.Table.from_pylist(rows, schema=PERMUTATION_SCHEMA)
                _atomic_parquet(part_path, part, PERMUTATION_SCHEMA)
                _atomic_json(audit_path, audit)
            tables.append(part)
            processed += len(order)
            _atomic_json(
                work / "checkpoint.json",
                {"next_segment": segment_id + 1, "consumed_edges": processed},
            )
            _progress(
                progress_path,
                "construction_progress",
                unit_id=config.unit_id,
                segment=segment_id + 1,
                segments=len(day.segments),
                edges=processed,
                elapsed_seconds=time.monotonic() - started,
            )
        table = (
            pa.concat_tables(tables)
            if tables
            else pa.Table.from_pylist([], schema=PERMUTATION_SCHEMA)
        )
        audit = _recompute_construction_audit(
            day, definition, table, config.surrogate_seed
        )
        if not audit["passed"]:
            raise RuntimeError(f"QR transition construction audit failed: {audit}")
        staging = work / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(staging / "permutation.parquet", table, PERMUTATION_SCHEMA)
        _atomic_json(staging / "linkage_audit.json", audit)
        _progress(
            progress_path,
            "construction_complete",
            unit_id=config.unit_id,
            edges=len(table),
            elapsed_seconds=time.monotonic() - started,
        )
        shutil.copy2(progress_path, staging / "progress.jsonl")
        complete = {
            "status": "complete",
            "unit_id": config.unit_id,
            "scientific_config": exact,
            "runtime": {"elapsed_seconds": time.monotonic() - started},
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
                {
                    "filename": "progress.jsonl",
                    "bytes": (staging / "progress.jsonl").stat().st_size,
                },
            ],
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(staging / "complete.json", complete)
        validate_construction_complete(staging, config, definition)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        shutil.rmtree(work)
        return validate_construction_complete(final, config, definition)
    finally:
        handle.close()
        if not succeeded:
            _atomic_json(
                work / "runtime.json", {"elapsed_seconds": time.monotonic() - started}
            )


def validate_construction_complete(
    final: Path,
    config: ConstructionConfig,
    state: CoarseStateDefinition | None = None,
) -> dict[str, Any]:
    definition = state or _load_state(config.state_path, config.simulation_root)
    value = json.loads((final / "complete.json").read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config")
        != _jsonable(_construction_exact(config, definition))
    ):
        raise RuntimeError("construction completion envelope mismatch")
    table = pq.read_table(final / "permutation.parquet")
    day = read_segmentable_day(config.simulation_root, config.source_date)
    recomputed = _recompute_construction_audit(
        day, definition, table, config.surrogate_seed
    )
    if recomputed.get("passed") is not True:
        raise RuntimeError("current permutation recomputation failed scientific gates")
    inventory = {item.get("filename"): item for item in value.get("artifacts", [])}
    if set(inventory) != {
        "permutation.parquet",
        "linkage_audit.json",
        "progress.jsonl",
    } or len(inventory) != len(value.get("artifacts", [])):
        raise RuntimeError(
            "construction artifact inventory is incomplete or duplicated"
        )
    for name, item in inventory.items():
        path = final / name
        if not path.is_file() or item.get("bytes") != path.stat().st_size:
            raise RuntimeError("construction artifact byte inventory mismatch")
    artifact = inventory["permutation.parquet"]
    audit = json.loads((final / "linkage_audit.json").read_text(encoding="utf-8"))
    if (
        not artifact
        or artifact.get("rows") != len(table)
        or artifact.get("bytes") != (final / "permutation.parquet").stat().st_size
        or audit != _jsonable(recomputed)
    ):
        raise RuntimeError("construction artifact/audit mismatch")
    return value


def _validate_model(config: ScoringConfig) -> tuple[dict[str, Any], Any, Any, float]:
    unit = TrainingUnit("qr", 37, "qr_a", 100, "ordered_tcn", config.model_seed)
    completion_path = config.model_root / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or completion.get("physical_unit_key") != unit.physical_key
        or completion.get("content_key") != unit.clock_independent_content_key
        or completion.get("scientific_config") != unit.scientific_config()
        or completion.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError("frozen Phase 14 QR-A model completion mismatch")
    published_artifacts = {
        item.get("path"): item for item in completion.get("artifacts", [])
    }
    if len(published_artifacts) != len(completion.get("artifacts", [])):
        raise RuntimeError("frozen model artifact inventory contains duplicate paths")
    required = ("model.pt", "scaler.json", "calibration.json")
    for name in required:
        path = config.model_root / name
        item = published_artifacts.get(name)
        if not item or not path.is_file() or item.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"frozen model artifact inventory mismatch: {name}")
    artifacts = {
        name: {"filename": name, "bytes": (config.model_root / name).stat().st_size}
        for name in required
    }
    payload = torch.load(
        config.model_root / "model.pt", map_location="cpu", weights_only=False
    )
    if (
        payload.get("model") != "ordered_tcn"
        or int(payload.get("seed", -1)) != config.model_seed
        or int(payload.get("B", -1)) != TARGET_B
        or int(payload.get("K", -1)) != K
    ):
        raise RuntimeError("frozen Phase 14 QR-A model payload identity mismatch")
    scaler_value = json.loads(
        (config.model_root / "scaler.json").read_text(encoding="utf-8")
    )
    if int(scaler_value.get("feature_count", -1)) != 37 or scaler_value.get(
        "active_channel_indices"
    ) != list(range(37)):
        raise RuntimeError("frozen scaler is not exact full37")
    scaler = UPSTREAM.ChannelScaler.from_dict(scaler_value)
    calibration = json.loads(
        (config.model_root / "calibration.json").read_text(encoding="utf-8")
    )
    alpha = float(calibration["alpha"])
    if not math.isfinite(alpha):
        raise RuntimeError("nonfinite frozen calibration alpha")
    model = UPSTREAM.build_model("ordered_tcn")
    model.load_state_dict(payload["model_state"], strict=True)
    device = device_or_raise(config.device)
    model.to(device)
    model.eval()
    identity = {
        "physical_unit_key": unit.physical_key,
        "content_key": unit.clock_independent_content_key,
        "completion_bytes": completion_path.stat().st_size,
        "artifacts": artifacts,
        "alpha": alpha,
        "published_scientific_config": completion["scientific_config"],
        "test_path_counts": {
            source_date: int(completion["input"]["path_counts_by_date"][source_date])
            for source_date in TEST_DATES
        },
    }
    return identity, model, scaler, alpha


def _selectors_path(config: ScoringConfig) -> Path:
    if config.selectors_path is not None:
        return config.selectors_path
    for parent in config.model_root.parents:
        if parent.name == "units":
            candidate = parent.parent / "manifests" / "selectors.json"
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "canonical selectors path cannot be derived from the frozen model root"
    )


def _selector_support(
    config: ScoringConfig, model_identity: Mapping[str, Any]
) -> dict[str, Any]:
    path = _selectors_path(config)
    value = json.loads(path.read_text(encoding="utf-8"))
    sample = value.get("sample", {})
    dates = value.get("dates", {})
    if (
        value.get("status") != "complete"
        or value.get("contract") != "phase14_real_b10_k6_selectors_v1"
        or sample.get("target_b") != TARGET_B
        or sample.get("k") != K
        or sample.get("path_stride") != 3
        or sample.get("path_offset_rule") != "date.toordinal()%3"
        or set(sample.get("splits", {}).get("test", [])) != set(TEST_DATES)
        or not isinstance(dates, dict)
    ):
        raise RuntimeError("canonical selector manifest contract mismatch")
    result = {}
    anchor_counts = model_identity["test_path_counts"]
    for source_date in TEST_DATES:
        row = dates.get(source_date, {})
        base_stop = row.get("base_stop")
        available = row.get("available_paths")
        expected_offset = date.fromisoformat(source_date).toordinal() % 3
        if (
            isinstance(base_stop, bool)
            or not isinstance(base_stop, int)
            or base_stop <= 0
            or isinstance(available, bool)
            or not isinstance(available, int)
            or available < base_stop
            or row.get("offset") != expected_offset
            or row.get("stride") != 3
            or base_stop != int(anchor_counts[source_date])
        ):
            raise RuntimeError(
                f"selector/frozen-anchor support mismatch: {source_date}"
            )
        result[source_date] = {
            "base_stop": base_stop,
            "available_paths": available,
            "offset": expected_offset,
            "stride": 3,
        }
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "contract": value.get("contract"),
        "dates": result,
    }


def _permutation_inventory(
    config: ScoringConfig,
    selectors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    fitted_state = _current_coarse_state(config.simulation_root)
    definition = CoarseStateDefinition.from_dict(fitted_state)
    for source_date in TEST_DATES:
        root = (
            config.permutation_root
            / f"replicate={config.surrogate_seed}"
            / f"source_date={source_date}"
        )
        construction = ConstructionConfig(
            config.surrogate_seed,
            source_date,
            config.simulation_root,
            # Validation uses the completion's embedded state after this path
            # is replaced below; the file is intentionally not consulted.
            Path("unused"),
            config.permutation_root,
            config.work_root,
        )
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        table = pq.read_table(root / "permutation.parquet")
        if (
            complete.get("status") != "complete"
            or complete.get("unit_id") != construction.unit_id
        ):
            raise RuntimeError("scoring permutation completion identity mismatch")
        scientific = complete.get("scientific_config", {})
        if (
            scientific.get("contract") != CONTRACT
            or scientific.get("stage") != "construction"
            or scientific.get("source_date") != source_date
            or scientific.get("surrogate_seed") != config.surrogate_seed
            or scientific.get("state") != definition.to_dict()
        ):
            raise RuntimeError("scoring permutation scientific config mismatch")
        day = read_segmentable_day(config.simulation_root, source_date)
        recomputed = _recompute_construction_audit(
            day, definition, table, config.surrogate_seed
        )
        if recomputed.get("passed") is not True:
            raise RuntimeError("scoring permutation recomputed audit failed")
        artifact = next(
            (
                item
                for item in complete.get("artifacts", [])
                if item.get("filename") == "permutation.parquet"
            ),
            None,
        )
        if (
            not artifact
            or artifact.get("rows") != len(table)
            or artifact.get("bytes") != (root / "permutation.parquet").stat().st_size
        ):
            raise RuntimeError("scoring permutation artifact inventory mismatch")
        audit = json.loads((root / "linkage_audit.json").read_text(encoding="utf-8"))
        if audit != _jsonable(recomputed):
            raise RuntimeError("scoring permutation audit failed")
        if int(audit["K6_paths"]) < int(selectors["dates"][source_date]["base_stop"]):
            raise RuntimeError("permutation lacks selector-bound K6 support")
        result.append(
            {
                "source_date": source_date,
                "unit_id": complete["unit_id"],
                "rows": len(table),
                "bytes": (root / "permutation.parquet").stat().st_size,
                "completion_bytes": (root / "complete.json").stat().st_size,
                "K6_paths": int(audit["K6_paths"]),
                "selected_K6_paths": int(selectors["dates"][source_date]["base_stop"]),
            }
        )
    return result


def _scoring_exact(
    config: ScoringConfig, model_identity: Mapping[str, Any]
) -> dict[str, Any]:
    selectors = _selector_support(config, model_identity)
    return {
        "contract": CONTRACT,
        "stage": "scoring",
        "B": TARGET_B,
        "K": K,
        "feature_count": 37,
        "fraction_percent": 100,
        "source": "qr",
        "duration_spec": "qr_a",
        "model": "ordered_tcn",
        "model_seed": config.model_seed,
        "surrogate_seed": config.surrogate_seed,
        "test_input": _input_inventory(config.simulation_root, TEST_DATES),
        "permutations": _permutation_inventory(config, selectors),
        "selectors": selectors,
        "model_identity": model_identity,
        "score_contracts": ["bounded_log99", "uncapped"],
        "primary_estimand": "qr_original_minus_qr_surrogate_bounded_nats_per_source_event",
        "secondary_estimand": "qr_original_minus_qr_surrogate_uncapped_nats_per_source_event",
        "batch_size": config.batch_size,
        "checkpoint_paths": config.checkpoint_paths,
        "fp32": True,
        "tf32": False,
        "amp": False,
        "device": config.device,
    }


def _accumulators(
    value: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, MetricAccumulator]]:
    if value is None:
        return {
            population: {
                contract: MetricAccumulator()
                for contract in ("bounded_log99", "uncapped")
            }
            for population in ("original", "surrogate")
        }
    return {
        population: {
            contract: MetricAccumulator(**fields)
            for contract, fields in contracts.items()
        }
        for population, contracts in value.items()
    }


def _serialize_accumulators(
    value: Mapping[str, Mapping[str, MetricAccumulator]],
) -> dict[str, Any]:
    return {
        population: {
            contract: asdict(accumulator) for contract, accumulator in contracts.items()
        }
        for population, contracts in value.items()
    }


def _score_batch(
    model: Any,
    scaler: Any,
    alpha: float,
    day: SegmentableDay,
    matrix: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    forward, reverse, events, seconds = _edge_tokens(day, matrix)
    ft = torch.from_numpy(scaler.transform(forward)).to(device)
    rt = torch.from_numpy(scaler.transform(reverse)).to(device)
    raw = UPSTREAM.antisymmetric_score(model, ft, rt)
    reverse_raw = UPSTREAM.antisymmetric_score(model, rt, ft)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    raw_np = raw.detach().cpu().numpy().astype(np.float64)
    reverse_np = reverse_raw.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(raw_np).all() or not np.isfinite(reverse_np).all():
        raise RuntimeError("nonfinite QR surrogate score")
    error = float(np.max(np.abs(raw_np + reverse_np))) if len(raw_np) else 0.0
    boundaries = (
        np.stack(
            (day.starts[matrix[:1].reshape(-1)], day.ends[matrix[:1].reshape(-1)]),
            axis=1,
        )
        if len(matrix)
        else None
    )
    intervals = day.intervals[matrix[:1].reshape(-1), None, :] if len(matrix) else None
    involution = True
    if boundaries is not None and intervals is not None:
        once = UPSTREAM.reverse_continuous(boundaries, intervals)
        twice = UPSTREAM.reverse_continuous(*once)
        involution = np.array_equal(boundaries, twice[0]) and np.array_equal(
            intervals, twice[1]
        )
    return alpha * raw_np, events, seconds, error, involution


def recompute_original_anchor_metrics(
    config: ScoringConfig,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Independently score the selector-bound original QR-A prefixes."""

    configure_determinism()
    model_identity, model, scaler, alpha = _validate_model(config)
    selectors = _selector_support(config, model_identity)
    device = next(model.parameters()).device
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with torch.no_grad():
        for source_date in TEST_DATES:
            day = read_segmentable_day(config.simulation_root, source_date)
            stop = int(selectors["dates"][source_date]["base_stop"])
            matrices = _original_matrices(day)
            if len(matrices) < stop:
                raise RuntimeError("original QR-A lacks frozen selector support")
            accumulators = {
                name: MetricAccumulator() for name in ("bounded_log99", "uncapped")
            }
            for first in range(0, stop, config.batch_size):
                scores, events, seconds, error, involution = _score_batch(
                    model,
                    scaler,
                    alpha,
                    day,
                    matrices[first : min(first + config.batch_size, stop)],
                    device,
                )
                if error >= 1e-6 or not involution:
                    raise RuntimeError(
                        "original anchor recomputation structural audit failed"
                    )
                accumulators["uncapped"].update(scores, events, seconds)
                accumulators["bounded_log99"].update(
                    np.clip(scores, -BOUND, BOUND), events, seconds
                )
            result[source_date] = {
                name: accumulator.final(K) for name, accumulator in accumulators.items()
            }
    return result


def run_scoring_unit(config: ScoringConfig) -> dict[str, Any]:
    if (config.model_seed, config.surrogate_seed) not in scoring_units():
        raise ValueError("scoring unit outside exact Phase 14 universe")
    if config.batch_size < 1 or config.checkpoint_paths < 1:
        raise ValueError("invalid scoring batch/checkpoint size")
    configure_determinism()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_identity, model, scaler, alpha = _validate_model(config)
    selectors = _selector_support(config, model_identity)
    device = next(model.parameters()).device
    exact = _scoring_exact(config, model_identity)
    final = (
        config.output_root
        / f"model_seed={config.model_seed}"
        / f"replicate={config.surrogate_seed}"
    )
    if (final / "complete.json").is_file():
        return validate_scoring_complete(final, config)
    work = config.work_root / "scoring" / config.unit_id
    _prepare_work(work, exact, config.resume, config.work_root)
    work.mkdir(parents=True, exist_ok=True)
    handle = _lock(work / ".unit.lock")
    if not (work / "config.json").is_file():
        _atomic_json(work / "config.json", exact)
    checkpoint_path = work / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {
            "date_index": 0,
            "path_cursor": 0,
            "processed_pair_paths": 0,
            "accumulators": None,
            "rows": [],
            "max_antisymmetry_error": 0.0,
            "reverse_involution_exact": True,
        }
    )
    started = time.monotonic()
    process = psutil.Process()
    succeeded = False
    progress_path = work / "progress.jsonl"
    last_progress = time.monotonic()
    _progress(
        progress_path,
        "scoring_start",
        unit_id=config.unit_id,
        resumed=checkpoint_path.is_file(),
    )
    try:
        output_rows = list(checkpoint.get("rows", []))
        processed = int(checkpoint["processed_pair_paths"])
        max_error = float(checkpoint["max_antisymmetry_error"])
        involution = bool(checkpoint["reverse_involution_exact"])
        for date_index, source_date in enumerate(TEST_DATES):
            if date_index < int(checkpoint["date_index"]):
                continue
            day = read_segmentable_day(config.simulation_root, source_date)
            permutation = pq.read_table(
                config.permutation_root
                / f"replicate={config.surrogate_seed}"
                / f"source_date={source_date}"
                / "permutation.parquet"
            )
            _, surrogate_ids = _starts_for_permutation(permutation, source_date, K)
            surrogate_positions = np.searchsorted(day.block_index, surrogate_ids)
            if np.any(
                surrogate_positions >= len(day.block_index)
            ) or not np.array_equal(
                day.block_index[surrogate_positions], surrogate_ids
            ):
                raise RuntimeError("surrogate block ID lookup mismatch")
            original_positions = _original_matrices(day)
            available_original, available_surrogate = (
                len(original_positions),
                len(surrogate_positions),
            )
            selected_paths = int(selectors["dates"][source_date]["base_stop"])
            if (
                available_original != available_surrogate
                or available_original < selected_paths
                or selected_paths <= 0
            ):
                raise RuntimeError("original/surrogate K6 path support mismatch")
            original_positions = original_positions[:selected_paths]
            surrogate_positions = surrogate_positions[:selected_paths]
            cursor = (
                int(checkpoint["path_cursor"])
                if date_index == int(checkpoint["date_index"])
                else 0
            )
            accumulators = _accumulators(
                checkpoint.get("accumulators") if cursor else None
            )
            with torch.no_grad():
                for first in range(cursor, len(original_positions), config.batch_size):
                    last = min(first + config.batch_size, len(original_positions))
                    for population, matrix in (
                        ("original", original_positions[first:last]),
                        ("surrogate", surrogate_positions[first:last]),
                    ):
                        scores, events, seconds, error, exact_reverse = _score_batch(
                            model, scaler, alpha, day, matrix, device
                        )
                        accumulators[population]["uncapped"].update(
                            scores, events, seconds
                        )
                        accumulators[population]["bounded_log99"].update(
                            np.clip(scores, -BOUND, BOUND), events, seconds
                        )
                        max_error = max(max_error, error)
                        involution &= exact_reverse
                    processed += 2 * (last - first)
                    checkpoint = {
                        "date_index": date_index,
                        "path_cursor": last,
                        "processed_pair_paths": processed,
                        "accumulators": _serialize_accumulators(accumulators),
                        "rows": output_rows,
                        "max_antisymmetry_error": max_error,
                        "reverse_involution_exact": involution,
                    }
                    if (
                        processed % config.checkpoint_paths < 2 * config.batch_size
                        or last == len(original_positions)
                    ):
                        _atomic_json(checkpoint_path, checkpoint)
                    now = time.monotonic()
                    if now - last_progress >= 30.0 or last == len(original_positions):
                        _progress(
                            progress_path,
                            "scoring_progress",
                            unit_id=config.unit_id,
                            date_index=date_index + 1,
                            dates=len(TEST_DATES),
                            path_cursor=last,
                            date_paths=len(original_positions),
                            processed_score_evaluations=processed,
                            elapsed_seconds=now - started,
                            score_evaluations_per_second=processed
                            / max(now - started, 1e-9),
                            rss_bytes=process.memory_info().rss,
                        )
                        last_progress = now
                    if (
                        config.interrupt_after_paths is not None
                        and processed >= config.interrupt_after_paths
                    ):
                        _atomic_json(checkpoint_path, checkpoint)
                        raise InterruptedError("synthetic scoring interruption")
            for score_contract, role in (
                ("bounded_log99", "primary"),
                ("uncapped", "secondary"),
            ):
                original = accumulators["original"][score_contract].final(K)
                surrogate = accumulators["surrogate"][score_contract].final(K)
                output_rows.append(
                    {
                        "model_seed": config.model_seed,
                        "surrogate_seed": config.surrogate_seed,
                        "source_date": source_date,
                        "score_contract": score_contract,
                        "estimand_role": role,
                        "original_paths": original["paths"],
                        "surrogate_paths": surrogate["paths"],
                        "original_source_events": original["source_event_sum"],
                        "surrogate_source_events": surrogate["source_event_sum"],
                        "original_nats_event": original["nats_source_event"],
                        "surrogate_nats_event": surrogate["nats_source_event"],
                        "delta_nats_event": original["nats_source_event"]
                        - surrogate["nats_source_event"],
                    }
                )
            checkpoint["rows"] = output_rows
            checkpoint["accumulators"] = None
            checkpoint["date_index"] = date_index + 1
            checkpoint["path_cursor"] = 0
            _atomic_json(checkpoint_path, checkpoint)
        if max_error >= 1e-6 or not involution:
            raise RuntimeError("QR surrogate scoring structural audit failed")
        table = pa.Table.from_pylist(output_rows, schema=CONTRAST_SCHEMA)
        _validate_contrast_table(
            table, one_unit=(config.model_seed, config.surrogate_seed)
        )
        staging = work / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(staging / "contrasts.parquet", table, CONTRAST_SCHEMA)
        _atomic_json(
            staging / "metrics.json",
            {
                "model_seed": config.model_seed,
                "surrogate_seed": config.surrogate_seed,
                "rows": output_rows,
                "primary_estimand": "original_minus_surrogate_bounded_nats_event",
                "secondary_estimand": "original_minus_surrogate_uncapped_nats_event",
                "audit": {
                    "score_antisymmetry_max_abs_error": max_error,
                    "reverse_involution_exact": involution,
                },
            },
        )
        shutil.copy2(checkpoint_path, staging / "final_checkpoint.json")
        _progress(
            progress_path,
            "scoring_complete",
            unit_id=config.unit_id,
            processed_score_evaluations=processed,
            elapsed_seconds=time.monotonic() - started,
        )
        shutil.copy2(progress_path, staging / "progress.jsonl")
        elapsed = time.monotonic() - started
        complete = {
            "status": "complete",
            "unit_id": config.unit_id,
            "scientific_config": exact,
            "runtime": {
                "elapsed_seconds": elapsed,
                "processed_pair_paths": processed,
                "paths_per_second": processed / elapsed,
                "max_rss_bytes": process.memory_info().rss,
                "paid_execution": False,
            },
            "artifacts": [
                {
                    "filename": "contrasts.parquet",
                    "rows": len(table),
                    "bytes": (staging / "contrasts.parquet").stat().st_size,
                },
                {
                    "filename": "metrics.json",
                    "bytes": (staging / "metrics.json").stat().st_size,
                },
                {
                    "filename": "final_checkpoint.json",
                    "bytes": (staging / "final_checkpoint.json").stat().st_size,
                },
                {
                    "filename": "progress.jsonl",
                    "bytes": (staging / "progress.jsonl").stat().st_size,
                },
            ],
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(staging / "complete.json", complete)
        validate_scoring_complete(staging, config)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        shutil.rmtree(work)
        return validate_scoring_complete(final, config)
    finally:
        handle.close()
        if not succeeded:
            _atomic_json(
                work / "runtime.json",
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "processed_pair_paths": checkpoint.get("processed_pair_paths", 0),
                },
            )


def _validate_contrast_table(
    table: pa.Table, one_unit: tuple[int, int] | None = None
) -> None:
    if not table.schema.equals(CONTRAST_SCHEMA, check_metadata=True) or sum(
        column.null_count for column in table.columns
    ):
        raise RuntimeError("contrast Arrow contract mismatch")
    rows = table.to_pylist()
    keys = [
        (
            row["model_seed"],
            row["surrogate_seed"],
            row["source_date"],
            row["score_contract"],
        )
        for row in rows
    ]
    units = (one_unit,) if one_unit is not None else scoring_units()
    expected = {
        (model_seed, surrogate_seed, source_date, contract)
        for model_seed, surrogate_seed in units
        for source_date in TEST_DATES
        for contract in ("bounded_log99", "uncapped")
    }
    if (
        len(rows) != len(expected)
        or len(set(keys)) != len(keys)
        or set(keys) != expected
    ):
        raise RuntimeError("contrast Cartesian support mismatch")
    numeric = np.asarray(
        [
            [
                row["original_nats_event"],
                row["surrogate_nats_event"],
                row["delta_nats_event"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("nonfinite contrast")
    for row in rows:
        if row["estimand_role"] != (
            "primary" if row["score_contract"] == "bounded_log99" else "secondary"
        ):
            raise RuntimeError("contrast estimand role mismatch")
        if (
            row["original_paths"] != row["surrogate_paths"]
            or row["original_paths"] <= 0
        ):
            raise RuntimeError("original/surrogate path support mismatch")
        if not math.isclose(
            row["delta_nats_event"],
            row["original_nats_event"] - row["surrogate_nats_event"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("contrast sign/arithmetic mismatch")


def validate_scoring_complete(final: Path, config: ScoringConfig) -> dict[str, Any]:
    model_identity, _model, _scaler, _alpha = _validate_model(config)
    value = json.loads((final / "complete.json").read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config")
        != _jsonable(_scoring_exact(config, model_identity))
    ):
        raise RuntimeError("scoring completion envelope mismatch")
    table = pq.read_table(final / "contrasts.parquet")
    _validate_contrast_table(table, one_unit=(config.model_seed, config.surrogate_seed))
    rows = _jsonable(table.to_pylist())
    metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(
        (final / "final_checkpoint.json").read_text(encoding="utf-8")
    )
    expected_processed = 2 * sum(
        int(row["original_paths"])
        for row in rows
        if row["score_contract"] == "bounded_log99"
    )
    metrics_audit = metrics.get("audit", {})
    if (
        metrics.get("model_seed") != config.model_seed
        or metrics.get("surrogate_seed") != config.surrogate_seed
        or metrics.get("rows") != rows
        or metrics.get("primary_estimand")
        != "original_minus_surrogate_bounded_nats_event"
        or metrics.get("secondary_estimand")
        != "original_minus_surrogate_uncapped_nats_event"
        or not math.isfinite(
            float(metrics_audit.get("score_antisymmetry_max_abs_error", float("nan")))
        )
        or float(metrics_audit["score_antisymmetry_max_abs_error"]) >= 1e-6
        or metrics_audit.get("reverse_involution_exact") is not True
    ):
        raise RuntimeError("metrics/contrast semantic equality mismatch")
    if (
        checkpoint.get("date_index") != len(TEST_DATES)
        or checkpoint.get("path_cursor") != 0
        or checkpoint.get("accumulators") is not None
        or checkpoint.get("rows") != rows
        or checkpoint.get("processed_pair_paths") != expected_processed
        or checkpoint.get("max_antisymmetry_error")
        != metrics_audit["score_antisymmetry_max_abs_error"]
        or checkpoint.get("reverse_involution_exact") is not True
        or value.get("runtime", {}).get("processed_pair_paths") != expected_processed
    ):
        raise RuntimeError("terminal scoring checkpoint/support mismatch")
    inventory = {item.get("filename"): item for item in value.get("artifacts", [])}
    if set(inventory) != {
        "contrasts.parquet",
        "metrics.json",
        "final_checkpoint.json",
        "progress.jsonl",
    } or len(inventory) != len(value.get("artifacts", [])):
        raise RuntimeError("scoring artifact inventory is incomplete or duplicated")
    for name in (
        "contrasts.parquet",
        "metrics.json",
        "final_checkpoint.json",
        "progress.jsonl",
    ):
        path = final / name
        item = inventory.get(name)
        if not item or not path.is_file() or item.get("bytes") != path.stat().st_size:
            raise RuntimeError("scoring artifact inventory mismatch")
    if inventory["contrasts.parquet"].get("rows") != 6:
        raise RuntimeError("scoring contrast row inventory mismatch")
    return value


def aggregate_scoring_units(scoring_root: Path, output_path: Path) -> dict[str, Any]:
    tables = []
    for model_seed, surrogate_seed in scoring_units():
        root = scoring_root / f"model_seed={model_seed}" / f"replicate={surrogate_seed}"
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        scientific = complete.get("scientific_config", {})
        if (
            complete.get("status") != "complete"
            or complete.get("unit_id")
            != f"model_seed={model_seed}/replicate={surrogate_seed}"
            or scientific.get("contract") != CONTRACT
            or scientific.get("stage") != "scoring"
            or scientific.get("B") != TARGET_B
            or scientific.get("K") != K
            or scientific.get("feature_count") != 37
            or scientific.get("fraction_percent") != 100
            or scientific.get("source") != "qr"
            or scientific.get("duration_spec") != "qr_a"
            or scientific.get("model") != "ordered_tcn"
            or scientific.get("model_seed") != model_seed
            or scientific.get("surrogate_seed") != surrogate_seed
        ):
            raise RuntimeError("aggregate scoring completion identity mismatch")
        table = pq.read_table(root / "contrasts.parquet")
        _validate_contrast_table(table, one_unit=(model_seed, surrogate_seed))
        entry = next(
            (
                item
                for item in complete.get("artifacts", [])
                if item.get("filename") == "contrasts.parquet"
            ),
            None,
        )
        if (
            not entry
            or entry.get("rows") != 6
            or entry.get("bytes") != (root / "contrasts.parquet").stat().st_size
        ):
            raise RuntimeError("aggregate scoring artifact inventory mismatch")
        tables.append(table)
    aggregate = pa.concat_tables(tables)
    _validate_contrast_table(aggregate)
    if output_path.is_file():
        observed = pq.read_table(output_path)
        if not observed.equals(aggregate, check_metadata=True):
            raise RuntimeError("existing aggregate differs from exact 15-unit evidence")
    else:
        _atomic_parquet(output_path, aggregate, CONTRAST_SCHEMA)
    return {
        "status": "complete",
        "units": 15,
        "rows": aggregate.num_rows,
        "primary_rows": sum(
            role == "primary" for role in aggregate["estimand_role"].to_pylist()
        ),
        "secondary_rows": sum(
            role == "secondary" for role in aggregate["estimand_role"].to_pylist()
        ),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def _synthetic_block_row(source_date: str, index: int, start_ns: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in CANONICAL_BLOCK_SCHEMA:
        if pa.types.is_string(field.type):
            row[field.name] = ""
        elif pa.types.is_date(field.type):
            row[field.name] = date.fromisoformat(source_date)
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        else:
            row[field.name] = 0
    start_vertex, end_vertex = index % 2, (index + 1) % 2
    row.update(
        {
            "dataset": "QR.XNAS.ITCH",
            "symbol": "NVDA",
            "source_date": date.fromisoformat(source_date),
            "target_b": TARGET_B,
            "block_index": index,
            "first_source_event_index": index * TARGET_B,
            "last_source_event_index": (index + 1) * TARGET_B - 1,
            "first_raw_ordinal": index * TARGET_B,
            "last_raw_ordinal": (index + 1) * TARGET_B - 1,
            "realized_source_event_count": TARGET_B,
            "completion_overshoot": 0,
            "block_start_ns": start_ns,
            "block_end_ns": start_ns + 1_000_000_000,
            "elapsed_ns": 1_000_000_000,
            "cause_add_count": TARGET_B,
            "mid_abs_variation_half_ticks": 1 + end_vertex,
        }
    )
    for prefix, vertex in (("start", start_vertex), ("end", end_vertex)):
        bid, ask = (
            100_000_000_000 + vertex * 10_000_000,
            100_010_000_000 + vertex * 20_000_000,
        )
        bid_l1, ask_l1 = 10 + 20 * vertex, 30 - 15 * vertex
        bid_l5, ask_l5 = 50 + 50 * vertex, 150 - 50 * vertex
        row.update(
            {
                f"{prefix}_best_bid_nano": bid,
                f"{prefix}_best_ask_nano": ask,
                f"{prefix}_midprice_x2_nano": bid + ask,
                f"{prefix}_spread_ticks": float(1 + vertex),
                f"{prefix}_bid_depth_l1": bid_l1,
                f"{prefix}_ask_depth_l1": ask_l1,
                f"{prefix}_bid_depth_l5": bid_l5,
                f"{prefix}_ask_depth_l5": ask_l5,
                f"{prefix}_imbalance_l1": float((bid_l1 - ask_l1) / (bid_l1 + ask_l1)),
                f"{prefix}_imbalance_l5": float((bid_l5 - ask_l5) / (bid_l5 + ask_l5)),
            }
        )
    return row


def _write_synthetic_simulation(simulation_root: Path, blocks: int = 40) -> None:
    for day_index, source_date in enumerate(DATES):
        day_root = simulation_root / f"source_date={source_date}"
        relative = f"blocks/spec=qr_a/source_date={source_date}/part-00000.parquet"
        path = day_root / relative
        timestamp = 1_000_000_000_000 + day_index * 100_000_000_000
        rows = []
        for index in range(blocks):
            if index == 10:
                timestamp += 5_000_000_000
            rows.append(_synthetic_block_row(source_date, index, timestamp))
            timestamp += 1_000_000_000
        table = pa.Table.from_pylist(rows, schema=CANONICAL_BLOCK_SCHEMA)
        _atomic_parquet(path, table, CANONICAL_BLOCK_SCHEMA)
        qr_b_relative = f"blocks/spec=qr_b/source_date={source_date}/part-00000.parquet"
        qr_b_path = day_root / qr_b_relative
        _atomic_parquet(qr_b_path, table, CANONICAL_BLOCK_SCHEMA)
        events_path, messages_path = (
            day_root / "events.parquet",
            day_root / "messages.parquet",
        )
        _atomic_parquet(
            events_path, pa.Table.from_pylist([], schema=EVENT_SCHEMA), EVENT_SCHEMA
        )
        _atomic_parquet(
            messages_path,
            pa.Table.from_pylist([], schema=MESSAGE_SCHEMA),
            MESSAGE_SCHEMA,
        )
        output_rows = {
            "events.parquet": 0,
            "messages.parquet": 0,
            relative: blocks,
            qr_b_relative: blocks,
        }
        output_bytes = {
            "events.parquet": events_path.stat().st_size,
            "messages.parquet": messages_path.stat().st_size,
            relative: path.stat().st_size,
            qr_b_relative: qr_b_path.stat().st_size,
        }
        _atomic_json(
            day_root / "complete.json",
            {
                "status": "complete",
                "source_date": source_date,
                "universe": 0,
                "requested_events": 0,
                "training_admission": "passed",
                "clock_lineage": {"contract": "synthetic_smoke"},
                "qa": {
                    "passed": True,
                    "produced_events": 0,
                    "state_continuity_passed": True,
                    "canonical_b10_support_passed": True,
                    "qr_a_b_content_identity_passed": True,
                    "exact_n_passed": True,
                },
                "qr_a_b_identity_audit": {"exact_passed": True},
                "block_support": {
                    "qr_a": {"passed": True, "expected_blocks": blocks},
                    "qr_b": {"passed": True, "expected_blocks": blocks},
                },
                "output_rows": output_rows,
                "output_bytes": output_bytes,
                "runtime_identity": {"contract": "synthetic_smoke"},
            },
        )


def _write_synthetic_model(
    model_root: Path,
    model_seed: int,
    test_path_counts: Mapping[str, int],
) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    model = UPSTREAM.build_model("ordered_tcn")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "ordered_tcn",
            "seed": model_seed,
            "B": TARGET_B,
            "K": K,
        },
        model_root / "model.pt",
    )
    _atomic_json(
        model_root / "scaler.json",
        {
            "contract": "phase14_fraction_projected_scaler_v2",
            "feature_count": 37,
            "active_channel_indices": list(range(37)),
            "boundary_center": [0.0] * 7,
            "boundary_scale": [1.0] * 7,
            "interval_center": [0.0] * 23,
            "interval_scale": [1.0] * 23,
            "sample_tokens": 1,
            "method": "bounded synthetic smoke",
        },
    )
    _atomic_json(model_root / "calibration.json", {"alpha": 1.0})
    unit = TrainingUnit("qr", 37, "qr_a", 100, "ordered_tcn", model_seed)
    _atomic_json(
        model_root / "complete.json",
        {
            "status": "complete",
            "physical_unit_key": unit.physical_key,
            "content_key": unit.clock_independent_content_key,
            "scientific_config": unit.scientific_config(),
            "audit": {"passed": True},
            "input": {
                "path_counts_by_date": {
                    source_date: int(test_path_counts.get(source_date, 1))
                    for source_date in DATES
                }
            },
            "artifacts": [
                {"path": name, "bytes": (model_root / name).stat().st_size}
                for name in ("model.pt", "scaler.json", "calibration.json")
            ],
        },
    )


def synthetic_smoke(root: Path) -> dict[str, Any]:
    """Bounded deterministic traversal/resume and exact 15/15/90 contract smoke."""

    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    vertices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint32)
    states = np.column_stack((vertices[:-1], vertices[1:]))
    edges = np.arange(10, dtype=np.int64)
    first, first_audit = randomized_euler(
        edges,
        states,
        surrogate_seed=SURROGATE_SEEDS[0],
        source_date=TEST_DATES[0],
        segment_ordinal=0,
    )
    second, second_audit = randomized_euler(
        edges,
        states,
        surrogate_seed=SURROGATE_SEEDS[0],
        source_date=TEST_DATES[0],
        segment_ordinal=0,
    )
    resume_root = root / "traversal"
    interrupted = False
    try:
        randomized_euler(
            edges,
            states,
            surrogate_seed=SURROGATE_SEEDS[0],
            source_date=TEST_DATES[0],
            segment_ordinal=0,
            checkpoint_root=resume_root,
            checkpoint_edges=2,
            interrupt_after_edges=3,
        )
    except InterruptedError:
        interrupted = True
    resumed, resumed_audit = randomized_euler(
        edges,
        states,
        surrogate_seed=SURROGATE_SEEDS[0],
        source_date=TEST_DATES[0],
        segment_ordinal=0,
        checkpoint_root=resume_root,
        checkpoint_edges=2,
    )
    traversal_passed = bool(
        interrupted
        and np.array_equal(first, second)
        and np.array_equal(first, resumed)
        and first_audit == second_audit == resumed_audit
        and set(first) == set(edges)
    )
    if not traversal_passed:
        raise RuntimeError("QR surrogate traversal/resume smoke failed")

    # Exercise the outer construction publisher across two segments.  The
    # forced interruption occurs only after segment 0 is durable, proving that
    # resume reuses that part and still publishes exact baseline content.
    simulation_root = root / "simulation"
    _write_synthetic_simulation(simulation_root)
    state_path = root / "coarse_state.json"
    publish_coarse_state(simulation_root, state_path)
    baseline_construction = root / "construction_baseline"
    baseline_config = ConstructionConfig(
        SURROGATE_SEEDS[0],
        TEST_DATES[0],
        simulation_root,
        state_path,
        baseline_construction,
        root / "construction_baseline_work",
        checkpoint_edges=3,
    )
    run_construction_unit(baseline_config)
    resumed_construction = root / "construction_resumed"
    interrupted_config = ConstructionConfig(
        SURROGATE_SEEDS[0],
        TEST_DATES[0],
        simulation_root,
        state_path,
        resumed_construction,
        root / "construction_resumed_work",
        checkpoint_edges=3,
        interrupt_after_edges=12,
    )
    outer_interrupted = False
    try:
        run_construction_unit(interrupted_config)
    except InterruptedError:
        outer_interrupted = True
    reused_part = (
        root
        / "construction_resumed_work"
        / "construction"
        / interrupted_config.unit_id
        / "parts"
        / "segment=0.parquet"
    )
    if not outer_interrupted or not reused_part.is_file():
        raise RuntimeError("outer construction did not preserve a completed segment")
    reused_stat = (reused_part.stat().st_size, reused_part.stat().st_mtime_ns)
    resumed_config = ConstructionConfig(
        SURROGATE_SEEDS[0],
        TEST_DATES[0],
        simulation_root,
        state_path,
        resumed_construction,
        root / "construction_resumed_work",
        checkpoint_edges=3,
    )
    run_construction_unit(resumed_config)
    baseline_final = (
        baseline_construction
        / f"replicate={SURROGATE_SEEDS[0]}"
        / f"source_date={TEST_DATES[0]}"
    )
    resumed_final = (
        resumed_construction
        / f"replicate={SURROGATE_SEEDS[0]}"
        / f"source_date={TEST_DATES[0]}"
    )
    construction_resume_exact = bool(
        pq.read_table(baseline_final / "permutation.parquet").equals(
            pq.read_table(resumed_final / "permutation.parquet"), check_metadata=True
        )
        and json.loads((baseline_final / "linkage_audit.json").read_text())
        == json.loads((resumed_final / "linkage_audit.json").read_text())
        and reused_stat[0] > 0
    )
    if not construction_resume_exact:
        raise RuntimeError("outer construction checkpoint resume differs from baseline")

    # A fabricated state cutpoint must fail against a fresh semantic fit.
    fabricated_state = root / "fabricated_state.json"
    fabricated = json.loads(state_path.read_text())
    fabricated["cutpoints"]["spread"][0] += 0.125
    _atomic_json(fabricated_state, fabricated)
    fabricated_state_rejected = False
    try:
        validate_coarse_state(fabricated_state, simulation_root)
    except RuntimeError:
        fabricated_state_rejected = True

    # Recompute all construction invariants; do not trust a stored passed bit.
    mutated_permutation = root / "mutated_permutation"
    shutil.copytree(baseline_final, mutated_permutation)
    corrupted = pq.read_table(mutated_permutation / "permutation.parquet").to_pylist()
    corrupted[2]["source_block_index"], corrupted[3]["source_block_index"] = (
        corrupted[3]["source_block_index"],
        corrupted[2]["source_block_index"],
    )
    corrupted_table = pa.Table.from_pylist(corrupted, schema=PERMUTATION_SCHEMA)
    _atomic_parquet(
        mutated_permutation / "permutation.parquet", corrupted_table, PERMUTATION_SCHEMA
    )
    corrupted_complete = json.loads((mutated_permutation / "complete.json").read_text())
    next(
        item
        for item in corrupted_complete["artifacts"]
        if item["filename"] == "permutation.parquet"
    )["bytes"] = (mutated_permutation / "permutation.parquet").stat().st_size
    _atomic_json(mutated_permutation / "complete.json", corrupted_complete)
    mutated_permutation_rejected = False
    try:
        validate_construction_complete(mutated_permutation, baseline_config)
    except RuntimeError:
        mutated_permutation_rejected = True
    mutated_audit = root / "mutated_linkage_audit"
    shutil.copytree(baseline_final, mutated_audit)
    audit_value = json.loads((mutated_audit / "linkage_audit.json").read_text())
    audit_value["changed_links"] += 1
    _atomic_json(mutated_audit / "linkage_audit.json", audit_value)
    audit_complete = json.loads((mutated_audit / "complete.json").read_text())
    next(
        item
        for item in audit_complete["artifacts"]
        if item["filename"] == "linkage_audit.json"
    )["bytes"] = (mutated_audit / "linkage_audit.json").stat().st_size
    _atomic_json(mutated_audit / "complete.json", audit_complete)
    mutated_audit_rejected = False
    try:
        validate_construction_complete(mutated_audit, baseline_config)
    except RuntimeError:
        mutated_audit_rejected = True

    # Finish the other two test-date permutations for one replicate, then run
    # scoring uninterrupted, interrupted mid-batch, and interrupted exactly at
    # a date boundary.  All three publications must be semantically identical.
    for source_date in TEST_DATES[1:]:
        run_construction_unit(
            ConstructionConfig(
                SURROGATE_SEEDS[0],
                source_date,
                simulation_root,
                state_path,
                baseline_construction,
                root / "construction_baseline_work",
                checkpoint_edges=3,
            )
        )
    selector_counts = {
        source_date: int(
            json.loads(
                (
                    baseline_construction
                    / f"replicate={SURROGATE_SEEDS[0]}"
                    / f"source_date={source_date}"
                    / "linkage_audit.json"
                ).read_text()
            )["K6_paths"]
        )
        - 2
        for source_date in TEST_DATES
    }
    selectors_path = root / "selectors.json"
    _atomic_json(
        selectors_path,
        {
            "status": "complete",
            "contract": "phase14_real_b10_k6_selectors_v1",
            "sample": {
                "target_b": TARGET_B,
                "k": K,
                "path_stride": 3,
                "path_offset_rule": "date.toordinal()%3",
                "splits": {"test": list(TEST_DATES)},
            },
            "dates": {
                source_date: {
                    "base_stop": selector_counts[source_date],
                    "available_paths": selector_counts[source_date],
                    "offset": date.fromisoformat(source_date).toordinal() % 3,
                    "stride": 3,
                }
                for source_date in TEST_DATES
            },
        },
    )
    model_root = root / "model"
    _write_synthetic_model(model_root, MODEL_SEEDS[0], selector_counts)
    scoring_common = {
        "model_seed": MODEL_SEEDS[0],
        "surrogate_seed": SURROGATE_SEEDS[0],
        "device": "cpu",
        "simulation_root": simulation_root,
        "permutation_root": baseline_construction,
        "model_root": model_root,
        "batch_size": 3,
        "checkpoint_paths": 2,
        "selectors_path": selectors_path,
    }
    baseline_score_config = ScoringConfig(
        **scoring_common,
        output_root=root / "score_baseline",
        work_root=root / "score_baseline_work",
    )
    run_scoring_unit(baseline_score_config)
    first_paths = selector_counts[TEST_DATES[0]]
    baseline_score_final = (
        root
        / "score_baseline"
        / f"model_seed={MODEL_SEEDS[0]}"
        / f"replicate={SURROGATE_SEEDS[0]}"
    )
    baseline_rows = pq.read_table(
        baseline_score_final / "contrasts.parquet"
    ).to_pylist()
    independent_original = recompute_original_anchor_metrics(baseline_score_config)
    anchor_original_identity = all(
        int(row["original_paths"]) == selector_counts[row["source_date"]]
        and int(row["original_paths"])
        < int(
            json.loads(
                (
                    baseline_construction
                    / f"replicate={SURROGATE_SEEDS[0]}"
                    / f"source_date={row['source_date']}"
                    / "linkage_audit.json"
                ).read_text()
            )["K6_paths"]
        )
        and row["original_nats_event"]
        == independent_original[row["source_date"]][row["score_contract"]][
            "nats_source_event"
        ]
        and int(row["original_source_events"])
        == int(
            independent_original[row["source_date"]][row["score_contract"]][
                "source_event_sum"
            ]
        )
        for row in baseline_rows
    )
    if not anchor_original_identity:
        raise RuntimeError("selector-bound original anchor identity smoke failed")

    mutated_selector_path = root / "mutated_selectors.json"
    mutated_selector = json.loads(selectors_path.read_text())
    mutated_selector["dates"][TEST_DATES[0]]["base_stop"] += 1
    _atomic_json(mutated_selector_path, mutated_selector)
    selector_mutation_rejected = False
    try:
        _selector_support(
            ScoringConfig(
                **{**scoring_common, "selectors_path": mutated_selector_path},
                output_root=root / "unused_selector",
                work_root=root / "unused_selector_work",
            ),
            _validate_model(baseline_score_config)[0],
        )
    except RuntimeError:
        selector_mutation_rejected = True

    missing_selector_path = root / "missing_support_selectors.json"
    missing_selector = json.loads(selectors_path.read_text())
    missing_count = (
        int(json.loads((baseline_final / "linkage_audit.json").read_text())["K6_paths"])
        + 1
    )
    missing_selector["dates"][TEST_DATES[0]]["base_stop"] = missing_count
    missing_selector["dates"][TEST_DATES[0]]["available_paths"] = missing_count
    _atomic_json(missing_selector_path, missing_selector)
    missing_model_root = root / "missing_support_model"
    shutil.copytree(model_root, missing_model_root)
    missing_completion = json.loads((missing_model_root / "complete.json").read_text())
    missing_completion["input"]["path_counts_by_date"][TEST_DATES[0]] = missing_count
    _atomic_json(missing_model_root / "complete.json", missing_completion)
    missing_support_rejected = False
    try:
        run_scoring_unit(
            ScoringConfig(
                **{
                    **scoring_common,
                    "model_root": missing_model_root,
                    "selectors_path": missing_selector_path,
                },
                output_root=root / "missing_score",
                work_root=root / "missing_score_work",
            )
        )
    except RuntimeError:
        missing_support_rejected = True

    def interrupted_score(name: str, threshold: int) -> Path:
        output, work = root / f"score_{name}", root / f"score_{name}_work"
        interrupted_config = ScoringConfig(
            **scoring_common,
            output_root=output,
            work_root=work,
            interrupt_after_paths=threshold,
        )
        caught = False
        try:
            run_scoring_unit(interrupted_config)
        except InterruptedError:
            caught = True
        if not caught:
            raise RuntimeError(f"scoring {name} smoke did not interrupt")
        run_scoring_unit(
            ScoringConfig(**scoring_common, output_root=output, work_root=work)
        )
        return (
            output / f"model_seed={MODEL_SEEDS[0]}" / f"replicate={SURROGATE_SEEDS[0]}"
        )

    mid_final = interrupted_score("mid", 4)
    boundary_final = interrupted_score("boundary", 2 * int(first_paths))
    scoring_resume_exact = all(
        pq.read_table(candidate / "contrasts.parquet").equals(
            pq.read_table(baseline_score_final / "contrasts.parquet"),
            check_metadata=True,
        )
        and json.loads((candidate / "metrics.json").read_text())
        == json.loads((baseline_score_final / "metrics.json").read_text())
        for candidate in (mid_final, boundary_final)
    )

    # A consistently rewritten contrast/footer must still disagree with the
    # immutable metrics and terminal checkpoint and therefore fail closed.
    mutated_scoring = root / "mutated_scoring"
    shutil.copytree(baseline_score_final, mutated_scoring)
    changed_rows = pq.read_table(mutated_scoring / "contrasts.parquet").to_pylist()
    changed_rows[0]["original_nats_event"] += 0.001
    changed_rows[0]["delta_nats_event"] += 0.001
    changed_table = pa.Table.from_pylist(changed_rows, schema=CONTRAST_SCHEMA)
    _atomic_parquet(
        mutated_scoring / "contrasts.parquet", changed_table, CONTRAST_SCHEMA
    )
    changed_complete = json.loads((mutated_scoring / "complete.json").read_text())
    contrast_item = next(
        item
        for item in changed_complete["artifacts"]
        if item["filename"] == "contrasts.parquet"
    )
    contrast_item["bytes"] = (mutated_scoring / "contrasts.parquet").stat().st_size
    _atomic_json(mutated_scoring / "complete.json", changed_complete)
    mutated_scoring_rejected = False
    try:
        validate_scoring_complete(mutated_scoring, baseline_score_config)
    except RuntimeError:
        mutated_scoring_rejected = True
    if not all(
        (
            fabricated_state_rejected,
            mutated_permutation_rejected,
            mutated_audit_rejected,
            scoring_resume_exact,
            mutated_scoring_rejected,
            selector_mutation_rejected,
            missing_support_rejected,
            anchor_original_identity,
        )
    ):
        raise RuntimeError("QR surrogate semantic negative/resume smoke failed")

    scoring_root = root / "scoring"
    for model_seed, surrogate_seed in scoring_units():
        unit = scoring_root / f"model_seed={model_seed}" / f"replicate={surrogate_seed}"
        rows = []
        for source_date in TEST_DATES:
            base = (model_seed - MODEL_SEEDS[0] + 1) * 0.001 + (
                surrogate_seed - SURROGATE_SEEDS[0]
            ) * 0.0001
            for contract, role in (
                ("bounded_log99", "primary"),
                ("uncapped", "secondary"),
            ):
                original = base + (0.00001 if contract == "uncapped" else 0.0)
                surrogate = original - 0.00002
                rows.append(
                    {
                        "model_seed": model_seed,
                        "surrogate_seed": surrogate_seed,
                        "source_date": source_date,
                        "score_contract": contract,
                        "estimand_role": role,
                        "original_paths": 4,
                        "surrogate_paths": 4,
                        "original_source_events": 240,
                        "surrogate_source_events": 240,
                        "original_nats_event": original,
                        "surrogate_nats_event": surrogate,
                        "delta_nats_event": original - surrogate,
                    }
                )
        table = pa.Table.from_pylist(rows, schema=CONTRAST_SCHEMA)
        _validate_contrast_table(table, one_unit=(model_seed, surrogate_seed))
        _atomic_parquet(unit / "contrasts.parquet", table, CONTRAST_SCHEMA)
        _atomic_json(
            unit / "complete.json",
            {
                "status": "complete",
                "unit_id": f"model_seed={model_seed}/replicate={surrogate_seed}",
                "scientific_config": {
                    "contract": CONTRACT,
                    "stage": "scoring",
                    "B": TARGET_B,
                    "K": K,
                    "feature_count": 37,
                    "fraction_percent": 100,
                    "source": "qr",
                    "duration_spec": "qr_a",
                    "model": "ordered_tcn",
                    "model_seed": model_seed,
                    "surrogate_seed": surrogate_seed,
                },
                "artifacts": [
                    {
                        "filename": "contrasts.parquet",
                        "rows": 6,
                        "bytes": (unit / "contrasts.parquet").stat().st_size,
                    }
                ],
            },
        )
    aggregate = aggregate_scoring_units(scoring_root, root / "aggregate.parquet")
    if aggregate != {
        "status": "complete",
        "units": 15,
        "rows": 90,
        "primary_rows": 45,
        "secondary_rows": 45,
        "output_path": str(root / "aggregate.parquet"),
        "output_bytes": (root / "aggregate.parquet").stat().st_size,
    }:
        raise RuntimeError("QR surrogate exact aggregate smoke failed")
    return {
        "status": "complete",
        "construction_units": len(construction_units()),
        "scoring_units": len(scoring_units()),
        "aggregate": aggregate,
        "traversal_deterministic": True,
        "construction_resume_exact": construction_resume_exact,
        "scoring_resume_exact": scoring_resume_exact,
        "checkpoint_resume_exact": bool(
            construction_resume_exact and scoring_resume_exact
        ),
        "fabricated_state_rejected": fabricated_state_rejected,
        "mutated_permutation_rejected": mutated_permutation_rejected,
        "mutated_linkage_audit_rejected": mutated_audit_rejected,
        "mutated_scoring_semantics_rejected": mutated_scoring_rejected,
        "selector_mutation_rejected": selector_mutation_rejected,
        "missing_selector_support_rejected": missing_support_rejected,
        "extra_starts_excluded": True,
        "anchor_original_metric_identity": anchor_original_identity,
    }
