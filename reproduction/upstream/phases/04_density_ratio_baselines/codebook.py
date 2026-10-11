#!/usr/bin/env python3
"""Phase 4 chronological paired-logistic density-ratio baselines.

Forward blocks are durable.  K-block paths and their physical reversals are
formed date-locally in bounded batches.  The implementation deliberately stops
before XGBoost and neural sequence models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BOUNDARY_FIELDS = (
    "spread_ticks",
    "bid_depth_l1",
    "ask_depth_l1",
    "bid_depth_l5",
    "ask_depth_l5",
    "imbalance_l1",
    "imbalance_l5",
)
BOUNDARY_LOG_FIELDS = frozenset(BOUNDARY_FIELDS[:5])

ODD_INTERVAL_FIELDS = (
    "bid_depth_l1_net_change",
    "ask_depth_l1_net_change",
    "bid_depth_l5_net_change",
    "ask_depth_l5_net_change",
    "appearance_minus_depletion_count",
    "mid_log_return",
    "mid_change_half_ticks",
    "bid_quote_change_ticks",
    "ask_quote_change_ticks",
)
ODD_IDENTITY_FIELDS = frozenset(("mid_log_return",))

EVEN_INTERVAL_FIELDS = (
    "bid_depth_l1_gross_variation",
    "ask_depth_l1_gross_variation",
    "bid_depth_l5_gross_variation",
    "ask_depth_l5_gross_variation",
    "depletion_plus_appearance_count",
    "mid_abs_variation_half_ticks",
    "spread_abs_variation_ticks",
    "elapsed_ns",
    "realized_source_event_count",
    "completion_overshoot",
)

POSITIVE_PAIRS = (
    ("bid_best_depletion_count", "bid_best_appearance_count"),
    ("ask_best_depletion_count", "ask_best_appearance_count"),
)

INTERVAL_FIELDS = (
    ODD_INTERVAL_FIELDS
    + EVEN_INTERVAL_FIELDS
    + tuple(name for pair in POSITIVE_PAIRS for name in pair)
)
INTERVAL_INDEX = {name: i for i, name in enumerate(INTERVAL_FIELDS)}
ODD_INDICES = np.asarray([INTERVAL_INDEX[x] for x in ODD_INTERVAL_FIELDS])
SWAP_PAIRS = tuple((INTERVAL_INDEX[a], INTERVAL_INDEX[b]) for a, b in POSITIVE_PAIRS)

IDENTITY_FIELDS = (
    "source_date",
    "symbol",
    "target_b",
    "block_index",
    "block_start_ns",
    "block_end_ns",
)
REQUIRED_COLUMNS = (
    IDENTITY_FIELDS
    + tuple(f"start_{x}" for x in BOUNDARY_FIELDS)
    + tuple(f"end_{x}" for x in BOUNDARY_FIELDS)
    + INTERVAL_FIELDS
)

FeatureKind = Literal["trivial", "ordered", "invariant"]
MODEL_KINDS: tuple[FeatureKind, ...] = ("trivial", "invariant", "ordered")
LAMBDA_GRID = np.asarray((0.0, 1e-5, 1e-4, 1e-3, 1e-2), dtype=np.float64)
ANCHOR_SCALES = ((30, 2), (30, 5), (100, 3), (300, 2), (300, 5))
FULL_SCALE_GRID = tuple(
    (target_b, k) for target_b in (30, 100, 300) for k in (2, 3, 4, 5)
)
EXISTING_RESOLUTION_EXTENSION = tuple(
    (target_b, k) for target_b in (30, 100, 300) for k in (1, 6, 7, 8)
)
MICRO_DENSE_GRID = tuple((target_b, k) for target_b in (2, 5, 10) for k in range(1, 9))
ISO_EVENT_EXTENSION = ((2, 10), (2, 20), (2, 30), (2, 60), (5, 12), (5, 24), (10, 12))
MICRO_EXTENSION_SCALES = (
    EXISTING_RESOLUTION_EXTENSION + MICRO_DENSE_GRID + ISO_EVENT_EXTENSION
)
# Backward-compatible name used by the batch runner and compact summarizer.
SELECTED_SCALES = FULL_SCALE_GRID + MICRO_EXTENSION_SCALES
START_STRIDE_BY_B = {2: 15, 5: 6, 10: 3, 30: 1, 100: 1, 300: 1}
NULL_SEED = 20_260_729


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


class ProgressLog:
    def __init__(self, path: Path, unit: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.unit = unit
        self._last_progress = 0.0

    def emit(self, event: str, *, console: bool = True, **values: object) -> None:
        row = {
            "timestamp_utc": utc_now(),
            "unit": self.unit,
            "event": event,
            "rss_bytes": rss_bytes(),
            **values,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if console:
            fields = " ".join(f"{key}={value}" for key, value in values.items())
            print(f"[{row['timestamp_utc']}] {self.unit} {event} {fields}", flush=True)

    def periodic(self, event: str, interval: float = 30.0, **values: object) -> None:
        now = time.monotonic()
        if now - self._last_progress >= interval:
            self._last_progress = now
            self.emit(event, **values)


@dataclass(frozen=True)
class DaySplit:
    train: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def validation(self) -> tuple[str, ...]:
        """Backward-compatible alias for the held-out calibration days."""
        return self.calibration


@dataclass(frozen=True)
class InnerFold:
    fold: int
    train: tuple[str, ...]
    evaluation: str


@dataclass
class DayBlocks:
    source_date: str
    block_index: np.ndarray
    block_start_ns: np.ndarray
    block_end_ns: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    intervals: np.ndarray


@dataclass
class PairBatch:
    source_date: str
    start_block_index: np.ndarray
    path_start_ns: np.ndarray
    path_end_ns: np.ndarray
    path_event_count: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray


@dataclass
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    sample_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.center) / self.scale

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "sample_rows": self.sample_rows,
            "method": "deterministic date-stratified median/IQR after physical transform",
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RobustScaler":
        return cls(
            np.asarray(value["center"], dtype=np.float64),
            np.asarray(value["scale"], dtype=np.float64),
            int(value["sample_rows"]),
        )


@dataclass
class PairedLogistic:
    weights: np.ndarray
    l2: float

    def score(self, batch: PairBatch, scaler: RobustScaler) -> np.ndarray:
        contrast = scaler.transform(batch.forward) - scaler.transform(batch.reverse)
        return contrast @ self.weights


def chronological_day_split(dates: Sequence[str]) -> DaySplit:
    ordered = tuple(sorted(set(dates)))
    if len(ordered) != 15:
        raise ValueError(f"expected exactly 15 distinct dates, found {len(ordered)}")
    return DaySplit(ordered[:9], ordered[9:12], ordered[12:15])


def expanding_inner_folds(train_dates: Sequence[str]) -> tuple[InnerFold, ...]:
    ordered = tuple(train_dates)
    if len(ordered) != 9:
        raise ValueError("inner walk-forward requires exactly nine development dates")
    return tuple(
        InnerFold(fold=index - 4, train=ordered[:index], evaluation=ordered[index])
        for index in range(5, 9)
    )


def reverse_continuous(
    boundaries: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Physical reversal for (n,K+1,dS), (n,K,dU) paths."""
    reversed_boundaries = boundaries[:, ::-1, :].copy()
    reversed_intervals = intervals[:, ::-1, :].copy()
    reversed_intervals[:, :, ODD_INDICES] *= -1.0
    for left, right in SWAP_PAIRS:
        saved = reversed_intervals[:, :, left].copy()
        reversed_intervals[:, :, left] = reversed_intervals[:, :, right]
        reversed_intervals[:, :, right] = saved
    return reversed_boundaries, reversed_intervals


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def transform_physical(
    boundaries: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed parity-preserving heavy-tail transforms before representation."""
    transformed_boundaries = boundaries.astype(np.float64, copy=True)
    transformed_intervals = intervals.astype(np.float64, copy=True)
    for index, name in enumerate(BOUNDARY_FIELDS):
        if name in BOUNDARY_LOG_FIELDS:
            transformed_boundaries[:, :, index] = np.log1p(
                transformed_boundaries[:, :, index]
            )
    for index, name in enumerate(INTERVAL_FIELDS):
        if name in ODD_INTERVAL_FIELDS:
            if name not in ODD_IDENTITY_FIELDS:
                transformed_intervals[:, :, index] = _signed_log1p(
                    transformed_intervals[:, :, index]
                )
        else:
            transformed_intervals[:, :, index] = np.log1p(
                transformed_intervals[:, :, index]
            )
    return transformed_boundaries, transformed_intervals


def _summary(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
        ),
        axis=1,
    )


def make_features(
    boundaries: np.ndarray, intervals: np.ndarray, kind: FeatureKind
) -> np.ndarray:
    transformed_boundaries, transformed_intervals = transform_physical(
        boundaries, intervals
    )
    if kind == "trivial":
        names = ("elapsed_ns", "realized_source_event_count", "completion_overshoot")
        controls = transformed_intervals[:, :, [INTERVAL_INDEX[name] for name in names]]
        return controls.reshape(len(controls), -1)
    if kind == "invariant":
        return np.concatenate(
            (_summary(transformed_boundaries), _summary(transformed_intervals)), axis=1
        )
    if kind != "ordered":
        raise ValueError(f"unknown feature kind: {kind}")
    pieces = [transformed_boundaries[:, 0, :]]
    for step in range(transformed_intervals.shape[1]):
        pieces.extend(
            (
                transformed_intervals[:, step, :],
                transformed_boundaries[:, step + 1, :],
            )
        )
    return np.concatenate(pieces, axis=1)


def feature_names(k: int, kind: FeatureKind) -> list[str]:
    if kind == "trivial":
        return [
            f"U{step}_{field}"
            for step in range(1, k + 1)
            for field in (
                "elapsed_ns",
                "realized_source_event_count",
                "completion_overshoot",
            )
        ]
    if kind == "invariant":
        boundary = [
            f"{stat}_boundary_{field}"
            for stat in ("mean", "std", "min", "max")
            for field in BOUNDARY_FIELDS
        ]
        interval = [
            f"{stat}_interval_{field}"
            for stat in ("mean", "std", "min", "max")
            for field in INTERVAL_FIELDS
        ]
        return boundary + interval
    if kind != "ordered":
        raise ValueError(f"unknown feature kind: {kind}")
    names = [f"S0_{field}" for field in BOUNDARY_FIELDS]
    for step in range(1, k + 1):
        names.extend(f"U{step}_{field}" for field in INTERVAL_FIELDS)
        names.extend(f"S{step}_{field}" for field in BOUNDARY_FIELDS)
    return names


def reverse_features(
    boundaries: np.ndarray, intervals: np.ndarray, kind: FeatureKind
) -> np.ndarray:
    reversed_boundaries, reversed_intervals = reverse_continuous(boundaries, intervals)
    return make_features(reversed_boundaries, reversed_intervals, kind)


def _partition_paths(root: Path, source_date: str, target_b: int) -> list[Path]:
    folder = root / f"source_date={source_date}" / f"target_b={target_b}"
    return sorted(folder.glob("*.parquet"))


def discover_dates(root: Path, target_b: int) -> list[str]:
    return [
        folder.name.split("=", 1)[1]
        for folder in sorted(root.glob("source_date=*"))
        if (folder / f"target_b={target_b}").is_dir()
    ]


def _read_day(paths: Sequence[Path], source_date: str) -> DayBlocks:
    tables = [
        pq.ParquetFile(path).read(columns=list(REQUIRED_COLUMNS)) for path in paths
    ]
    if not tables:
        raise FileNotFoundError(f"no block parquet for {source_date}")
    table = pa.concat_tables(tables, promote_options="default")
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"missing required block columns: {missing}")

    order = np.asarray(table["block_index"], dtype=np.int64)
    position = np.argsort(order, kind="stable")
    order = order[position]
    if len(order) > 1 and np.any(np.diff(order) != 1):
        raise ValueError(f"{source_date}: block_index is not contiguous")

    def vector(name: str, dtype=np.int64) -> np.ndarray:
        return np.asarray(table[name], dtype=dtype)[position]

    def matrix(names: Sequence[str]) -> np.ndarray:
        result = np.column_stack(
            [np.asarray(table[name], dtype=np.float64)[position] for name in names]
        )
        if not np.isfinite(result).all():
            raise ValueError(f"{source_date}: non-finite model input")
        return result

    starts = matrix([f"start_{name}" for name in BOUNDARY_FIELDS])
    ends = matrix([f"end_{name}" for name in BOUNDARY_FIELDS])
    if len(order) > 1 and not np.allclose(ends[:-1], starts[1:], rtol=0.0, atol=1e-6):
        raise ValueError(f"{source_date}: adjacent block boundary mismatch")
    return DayBlocks(
        source_date=source_date,
        block_index=order,
        block_start_ns=vector("block_start_ns"),
        block_end_ns=vector("block_end_ns"),
        starts=starts,
        ends=ends,
        intervals=matrix(INTERVAL_FIELDS),
    )


class DayStore:
    def __init__(
        self,
        root: Path,
        target_b: int,
        start_stride: int = 1,
        log: ProgressLog | None = None,
    ) -> None:
        if start_stride < 1:
            raise ValueError("start_stride must be positive")
        self.root = root
        self.target_b = target_b
        self.start_stride = start_stride
        self.log = log
        self._cache: dict[str, DayBlocks] = {}

    def get(self, source_date: str) -> DayBlocks:
        if source_date not in self._cache:
            started = time.monotonic()
            paths = _partition_paths(self.root, source_date, self.target_b)
            self._cache[source_date] = _read_day(paths, source_date)
            if self.log:
                self.log.emit(
                    "day_loaded",
                    source_date=source_date,
                    blocks=len(self._cache[source_date].block_index),
                    files=len(paths),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
        return self._cache[source_date]

    def path_count(self, source_date: str, k: int) -> int:
        available = max(0, len(self.get(source_date).block_index) - k + 1)
        offset = self.start_offset(source_date)
        return max(0, (available - offset + self.start_stride - 1) // self.start_stride)

    def start_offset(self, source_date: str) -> int:
        return date.fromisoformat(source_date).toordinal() % self.start_stride


def _batch_from_starts(
    day: DayBlocks,
    starts: np.ndarray,
    k: int,
    kind: FeatureKind,
    *,
    null_seed: int | None = None,
) -> PairBatch:
    offsets = starts[:, None] + np.arange(k)[None, :]
    intervals = day.intervals[offsets]
    boundaries = np.empty((len(starts), k + 1, len(BOUNDARY_FIELDS)), np.float64)
    boundaries[:, 0, :] = day.starts[starts]
    boundaries[:, 1:, :] = day.ends[offsets]
    forward = make_features(boundaries, intervals, kind)
    reverse = reverse_features(boundaries, intervals, kind)
    if null_seed is not None:
        # Recreated for every call solely from stable pair identity.
        date_value = int(day.source_date.replace("-", ""))
        identity = day.block_index[starts].astype(np.uint64, copy=False)
        date_seed = (date_value * 1_442_695_040_888_963 + null_seed) & ((1 << 64) - 1)
        mixed = identity * np.uint64(6_364_136_223_846_793_005) + np.uint64(date_seed)
        swap = (mixed >> np.uint64(63)).astype(bool)
        if np.any(swap):
            saved = forward[swap].copy()
            forward[swap] = reverse[swap]
            reverse[swap] = saved
    event_counts = intervals[:, :, INTERVAL_INDEX["realized_source_event_count"]].sum(
        axis=1
    )
    return PairBatch(
        source_date=day.source_date,
        start_block_index=day.block_index[starts].copy(),
        path_start_ns=day.block_start_ns[starts].copy(),
        path_end_ns=day.block_end_ns[starts + k - 1].copy(),
        path_event_count=event_counts,
        forward=forward,
        reverse=reverse,
    )


def iter_pair_batches(
    store: DayStore,
    dates: Sequence[str],
    k: int,
    kind: FeatureKind,
    batch_rows: int = 65_536,
    *,
    null_seed: int | None = None,
) -> Iterator[PairBatch]:
    if k < 1:
        raise ValueError("continuous path K must be positive")
    for source_date in dates:
        day = store.get(source_date)
        available = max(0, len(day.block_index) - k + 1)
        offset = store.start_offset(source_date)
        for first in range(offset, available, batch_rows * store.start_stride):
            starts = np.arange(
                first,
                min(available, first + batch_rows * store.start_stride),
                store.start_stride,
                dtype=np.int64,
            )
            yield _batch_from_starts(day, starts, k, kind, null_seed=null_seed)


def fit_robust_scaler(
    store: DayStore,
    dates: Sequence[str],
    k: int,
    kind: FeatureKind,
    *,
    max_paths_per_day: int = 5_000,
) -> RobustScaler:
    samples: list[np.ndarray] = []
    for source_date in dates:
        day = store.get(source_date)
        path_count = store.path_count(source_date, k)
        count = min(path_count, max_paths_per_day)
        if not count:
            continue
        positions = np.linspace(0, path_count - 1, count, dtype=np.int64)
        starts = store.start_offset(source_date) + positions * store.start_stride
        batch = _batch_from_starts(day, starts, k, kind)
        samples.extend((batch.forward, batch.reverse))
    if not samples:
        raise ValueError("no rows available for robust scaler")
    values = np.concatenate(samples, axis=0)
    center = np.median(values, axis=0)
    lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (upper - lower) / 1.349
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return RobustScaler(center, scale, len(values))


def _training_state(path: Path, lambdas: np.ndarray, feature_count: int) -> dict:
    if not path.exists():
        shape = (len(lambdas), feature_count)
        return {
            "weights": np.zeros(shape),
            "first": np.zeros(shape),
            "second": np.zeros(shape),
            "best_weights": np.zeros(shape),
            "best_objective": np.full(len(lambdas), np.inf),
            "step": 0,
            "epoch": 0,
            "previous_objective": np.full(len(lambdas), np.inf),
            "stale": np.zeros(len(lambdas), dtype=np.int64),
        }
    loaded = np.load(path)
    saved_lambdas = np.asarray(loaded["lambdas"], dtype=np.float64)
    if not np.array_equal(saved_lambdas, lambdas):
        raise ValueError(f"checkpoint lambda mismatch: {path}")
    return {
        "weights": np.asarray(loaded["weights"]),
        "first": np.asarray(loaded["first"]),
        "second": np.asarray(loaded["second"]),
        "best_weights": np.asarray(loaded["best_weights"]),
        "best_objective": np.asarray(loaded["best_objective"]),
        "step": int(loaded["step"][0]),
        "epoch": int(loaded["epoch"][0]),
        "previous_objective": np.asarray(loaded["previous_objective"]),
        "stale": np.asarray(loaded["stale"], dtype=np.int64),
    }


def fit_lambda_path(
    store: DayStore,
    dates: Sequence[str],
    k: int,
    kind: FeatureKind,
    scaler: RobustScaler,
    lambdas: np.ndarray,
    checkpoint: Path,
    log: ProgressLog,
    stage: str,
    *,
    batch_rows: int,
    learning_rate: float,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    relative_tolerance: float,
    null_seed: int | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    feature_count = len(feature_names(k, kind))
    state = _training_state(checkpoint, lambdas, feature_count)
    weights = state["weights"]
    first_moment = state["first"]
    second_moment = state["second"]
    best_weights = state["best_weights"]
    best_objective = state["best_objective"]
    step = state["step"]
    previous = state["previous_objective"]
    stale = state["stale"]
    history: list[dict] = []

    total_paths = sum(store.path_count(date, k) for date in dates)
    for epoch in range(state["epoch"], max_epochs):
        started = time.monotonic()
        loss_sum = np.zeros(len(lambdas))
        rows = 0
        for batch in iter_pair_batches(
            store, dates, k, kind, batch_rows, null_seed=null_seed
        ):
            contrast = scaler.transform(batch.forward) - scaler.transform(batch.reverse)
            score = np.clip(contrast @ weights.T, -50.0, 50.0)
            one_minus_probability = 1.0 / (1.0 + np.exp(score))
            gradients = -(one_minus_probability.T @ contrast) / len(contrast)
            gradients += 2.0 * lambdas[:, None] * weights
            step += 1
            first_moment = 0.9 * first_moment + 0.1 * gradients
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradients)
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            weights -= (
                learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            )
            loss_sum += np.logaddexp(0.0, -score).sum(axis=0)
            rows += len(contrast)
            elapsed = time.monotonic() - started
            log.periodic(
                "training_progress",
                stage=stage,
                model=kind,
                epoch=epoch + 1,
                rows=rows,
                total_paths=total_paths,
                paths_per_second=round(rows / max(elapsed, 1e-9), 1),
            )
        objective = loss_sum / max(rows, 1) + lambdas * np.square(weights).sum(axis=1)
        finite_previous = np.isfinite(previous)
        improvement = np.full_like(objective, np.inf)
        improvement[finite_previous] = (
            previous[finite_previous] - objective[finite_previous]
        ) / np.maximum(np.abs(previous[finite_previous]), 1e-12)
        finite_best = np.isfinite(best_objective)
        improvement_from_best = np.full_like(objective, np.inf)
        improvement_from_best[finite_best] = (
            best_objective[finite_best] - objective[finite_best]
        ) / np.maximum(np.abs(best_objective[finite_best]), 1e-12)
        improved = (~finite_best) | (improvement_from_best > relative_tolerance)
        if np.any(improved):
            best_weights[improved] = weights[improved]
            best_objective[improved] = objective[improved]
        stale = np.where(improved, 0, stale + 1)
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch + 1,
            "rows": rows,
            "objective": objective.tolist(),
            "relative_improvement": improvement.tolist(),
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        atomic_save_npz(
            checkpoint,
            lambdas=lambdas,
            weights=weights,
            first=first_moment,
            second=second_moment,
            best_weights=best_weights,
            best_objective=best_objective,
            step=np.asarray([step], dtype=np.int64),
            epoch=np.asarray([epoch + 1], dtype=np.int64),
            previous_objective=objective,
            stale=stale,
        )
        log.emit(
            "epoch_complete",
            stage=stage,
            model=kind,
            epoch=epoch + 1,
            rows=rows,
            elapsed_seconds=round(elapsed, 3),
            paths_per_second=round(rows / max(elapsed, 1e-9), 1),
            objectives=[round(float(value), 8) for value in objective],
            best_objectives=[round(float(value), 8) for value in best_objective],
        )
        previous = objective
        if epoch + 1 >= min_epochs and np.all(stale >= patience):
            break
    return best_weights, history, best_objective


def raw_metrics_for_weights(
    store: DayStore,
    dates: Sequence[str],
    k: int,
    kind: FeatureKind,
    scaler: RobustScaler,
    weights: np.ndarray,
    *,
    batch_rows: int,
    null_seed: int | None = None,
) -> dict:
    loss_sum = np.zeros(weights.shape[0])
    score_sum = np.zeros(weights.shape[0])
    rows = 0
    by_day: dict[str, dict] = {}
    for batch in iter_pair_batches(
        store, dates, k, kind, batch_rows, null_seed=null_seed
    ):
        contrast = scaler.transform(batch.forward) - scaler.transform(batch.reverse)
        score = contrast @ weights.T
        loss = np.logaddexp(0.0, -np.clip(score, -80.0, 80.0))
        loss_sum += loss.sum(axis=0)
        score_sum += score.sum(axis=0)
        rows += len(score)
        day = by_day.setdefault(
            batch.source_date,
            {
                "rows": 0,
                "loss_sum": np.zeros(weights.shape[0]),
                "score_sum": np.zeros(weights.shape[0]),
            },
        )
        day["rows"] += len(score)
        day["loss_sum"] += loss.sum(axis=0)
        day["score_sum"] += score.sum(axis=0)
    return {
        "rows": rows,
        "log_loss": (loss_sum / rows).tolist(),
        "mean_raw_score": (score_sum / rows).tolist(),
        "by_day": {
            date: {
                "rows": value["rows"],
                "log_loss": (value["loss_sum"] / value["rows"]).tolist(),
                "mean_raw_score": (value["score_sum"] / value["rows"]).tolist(),
            }
            for date, value in by_day.items()
        },
    }


def fit_calibration(
    scores_by_day: dict[str, np.ndarray],
) -> tuple[float, dict]:
    scores = np.concatenate(list(scores_by_day.values()))
    candidates = np.exp(np.linspace(math.log(0.02), math.log(20.0), 301))
    losses = np.asarray(
        [
            np.logaddexp(0.0, -np.clip(candidate * scores, -80.0, 80.0)).mean()
            for candidate in candidates
        ]
    )
    best = int(np.argmin(losses))
    alpha = float(candidates[best])
    return alpha, {
        "alpha": alpha,
        "rows": len(scores),
        "log_loss": float(losses[best]),
        "candidate_min": float(candidates[0]),
        "candidate_max": float(candidates[-1]),
        "at_grid_boundary": best in (0, len(candidates) - 1),
        "by_day": {
            date: {
                "rows": len(day_scores),
                "log_loss": float(
                    np.logaddexp(0.0, -np.clip(alpha * day_scores, -80.0, 80.0)).mean()
                ),
                "mean_score": float((alpha * day_scores).mean()),
            }
            for date, day_scores in scores_by_day.items()
        },
    }


def collect_scores(
    store: DayStore,
    dates: Sequence[str],
    k: int,
    kind: FeatureKind,
    scaler: RobustScaler,
    model: PairedLogistic,
    *,
    batch_rows: int,
    null_seed: int | None = None,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    for batch in iter_pair_batches(
        store, dates, k, kind, batch_rows, null_seed=null_seed
    ):
        collected.setdefault(batch.source_date, []).append(model.score(batch, scaler))
    return {
        date: np.concatenate(parts) if parts else np.empty(0)
        for date, parts in collected.items()
    }


def _metric_accumulator() -> dict:
    return {
        "rows": 0,
        "score_sum": 0.0,
        "event_sum": 0.0,
        "duration_seconds_sum": 0.0,
        "log_loss_sum": 0.0,
        "brier_sum": 0.0,
        "correct_sum": 0.0,
        "saturation_sum": 0.0,
        "reverse_like_sum": 0.0,
        "scores": [],
    }


def _update_metric(accumulator: dict, score: np.ndarray, batch: PairBatch) -> None:
    clipped = np.clip(score, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-clipped))
    duration_seconds = (batch.path_end_ns - batch.path_start_ns) / 1e9
    accumulator["rows"] += len(score)
    accumulator["score_sum"] += float(score.sum())
    accumulator["event_sum"] += float(batch.path_event_count.sum())
    accumulator["duration_seconds_sum"] += float(duration_seconds.sum())
    accumulator["log_loss_sum"] += float(np.logaddexp(0.0, -clipped).sum())
    accumulator["brier_sum"] += float(np.square(1.0 - probability).sum())
    accumulator["correct_sum"] += float(
        np.count_nonzero(score > 0) + 0.5 * np.count_nonzero(score == 0)
    )
    accumulator["saturation_sum"] += float(
        np.count_nonzero((probability < 0.01) | (probability > 0.99))
    )
    accumulator["reverse_like_sum"] += float(np.count_nonzero(score < 0))
    # A bounded deterministic sample is enough for diagnostic quantiles.
    if sum(len(part) for part in accumulator["scores"]) < 200_000:
        remaining = 200_000 - sum(len(part) for part in accumulator["scores"])
        accumulator["scores"].append(score[:remaining].copy())


def _finalize_metric(accumulator: dict, k: int) -> dict:
    rows = accumulator["rows"]
    scores = (
        np.concatenate(accumulator["scores"]) if accumulator["scores"] else np.empty(0)
    )
    return {
        "paths": rows,
        "nats_path": accumulator["score_sum"] / rows,
        "nats_block": accumulator["score_sum"] / (rows * k),
        "nats_source_event": accumulator["score_sum"] / accumulator["event_sum"],
        "nats_second": accumulator["score_sum"] / accumulator["duration_seconds_sum"],
        "paired_log_loss": accumulator["log_loss_sum"] / rows,
        "brier": accumulator["brier_sum"] / rows,
        "accuracy": accumulator["correct_sum"] / rows,
        "posterior_saturation_share": accumulator["saturation_sum"] / rows,
        "reverse_like_share": accumulator["reverse_like_sum"] / rows,
        "score_q01": float(np.quantile(scores, 0.01)),
        "score_q50": float(np.quantile(scores, 0.50)),
        "score_q99": float(np.quantile(scores, 0.99)),
        "duration_seconds_sum": accumulator["duration_seconds_sum"],
        "source_event_sum": accumulator["event_sum"],
    }


def write_test_predictions_and_metrics(
    prediction_path: Path,
    store: DayStore,
    dates: Sequence[str],
    target_b: int,
    k: int,
    kind: FeatureKind,
    scaler: RobustScaler,
    model: PairedLogistic,
    alpha: float,
    *,
    batch_rows: int,
    null_seed: int | None = None,
    write_predictions: bool = True,
) -> dict:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(
        prediction_path.suffix + f".tmp.{os.getpid()}"
    )
    writer = None
    pooled = _metric_accumulator()
    by_day: dict[str, dict] = {}
    try:
        for batch in iter_pair_batches(
            store, dates, k, kind, batch_rows, null_seed=null_seed
        ):
            score = alpha * model.score(batch, scaler)
            _update_metric(pooled, score, batch)
            day = by_day.setdefault(batch.source_date, _metric_accumulator())
            _update_metric(day, score, batch)
            if write_predictions:
                probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
                pair_id = [
                    f"{batch.source_date}:B{target_b}:K{k}:{int(index)}"
                    for index in batch.start_block_index
                ]
                count = len(score)
                table = pa.table(
                    {
                        "pair_id": pair_id + pair_id,
                        "source_date": [batch.source_date] * (2 * count),
                        "start_block_index": np.concatenate(
                            (batch.start_block_index, batch.start_block_index)
                        ),
                        "path_start_ns": np.concatenate(
                            (batch.path_start_ns, batch.path_start_ns)
                        ),
                        "target_b": np.full(2 * count, target_b, np.int32),
                        "k": np.full(2 * count, k, np.int16),
                        "model": [kind] * (2 * count),
                        "orientation": ["forward"] * count + ["reverse"] * count,
                        "label": np.concatenate(
                            (np.ones(count, np.uint8), np.zeros(count, np.uint8))
                        ),
                        "calibrated_score": np.concatenate((score, -score)),
                        "probability_forward": np.concatenate(
                            (probability, 1.0 - probability)
                        ),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, prediction_path)
        elif write_predictions:
            raise ValueError("no test predictions")
        day_metrics = {
            date: _finalize_metric(value, k) for date, value in by_day.items()
        }
        return {
            "pooled_path_weighted": _finalize_metric(pooled, k),
            "by_day": day_metrics,
            "day_equal": {
                key: float(np.mean([value[key] for value in day_metrics.values()]))
                for key in (
                    "nats_path",
                    "nats_block",
                    "nats_source_event",
                    "nats_second",
                    "paired_log_loss",
                    "brier",
                    "accuracy",
                    "posterior_saturation_share",
                    "reverse_like_share",
                )
            },
        }
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def select_lambda(fold_results: list[dict]) -> tuple[float, dict]:
    losses = np.asarray([row["log_loss"] for row in fold_results], dtype=np.float64)
    mean_loss = losses.mean(axis=0)
    minimum = float(mean_loss.min())
    eligible = np.flatnonzero(mean_loss <= minimum + 1e-5)
    selected_index = int(eligible[-1])
    return float(LAMBDA_GRID[selected_index]), {
        "lambda_grid": LAMBDA_GRID.tolist(),
        "fold_log_loss": losses.tolist(),
        "mean_day_equal_log_loss": mean_loss.tolist(),
        "selected_index": selected_index,
        "selected_lambda": float(LAMBDA_GRID[selected_index]),
        "tie_tolerance": 1e-5,
    }


def _load_complete(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(value.get(key) == item for key, item in expected.items())


def run_model(
    store: DayStore,
    split: DaySplit,
    target_b: int,
    k: int,
    kind: FeatureKind,
    output_root: Path,
    work_root: Path,
    log: ProgressLog,
    args: argparse.Namespace,
) -> dict:
    unit = f"B{target_b}_K{k}_{kind}"
    unit_output = output_root / f"target_b={target_b}" / f"k={k}" / f"model={kind}"
    unit_work = work_root / unit
    complete_path = unit_output / "complete.json"
    expected = {
        "target_b": target_b,
        "k": k,
        "model": kind,
        "protocol": "chronological_walk_forward_v1",
        "start_stride_blocks": store.start_stride,
        "start_offset_rule": "date_ordinal_modulo_stride",
    }
    if _load_complete(complete_path, expected) and not args.force:
        log.emit("model_skipped_complete", model=kind)
        return json.loads((unit_output / "metrics.json").read_text())

    unit_output.mkdir(parents=True, exist_ok=True)
    unit_work.mkdir(parents=True, exist_ok=True)
    folds = expanding_inner_folds(split.train)
    fold_results: list[dict] = []
    for fold in folds:
        result_path = unit_work / f"fold_{fold.fold}.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text())
            fold_results.append(result)
            log.emit(
                "inner_fold_skipped",
                model=kind,
                fold=fold.fold,
                evaluation=fold.evaluation,
            )
            continue
        scaler = fit_robust_scaler(
            store,
            fold.train,
            k,
            kind,
            max_paths_per_day=args.scaler_paths_per_day,
        )
        checkpoint = unit_work / f"fold_{fold.fold}_tuning.npz"
        weights, history, best_objective = fit_lambda_path(
            store,
            fold.train,
            k,
            kind,
            scaler,
            LAMBDA_GRID,
            checkpoint,
            log,
            f"inner_fold_{fold.fold}",
            batch_rows=args.batch_rows,
            learning_rate=args.learning_rate,
            max_epochs=args.tuning_epochs,
            min_epochs=args.min_tuning_epochs,
            patience=2,
            relative_tolerance=args.relative_tolerance,
        )
        evaluation = raw_metrics_for_weights(
            store,
            (fold.evaluation,),
            k,
            kind,
            scaler,
            weights,
            batch_rows=args.batch_rows,
        )
        result = {
            "fold": fold.fold,
            "train_dates": fold.train,
            "evaluation_date": fold.evaluation,
            "rows": evaluation["rows"],
            "log_loss": evaluation["log_loss"],
            "mean_raw_score": evaluation["mean_raw_score"],
            "training_history": history,
            "best_training_objective": best_objective.tolist(),
            "scaler_sample_rows": scaler.sample_rows,
        }
        atomic_write_json(result_path, result)
        checkpoint.unlink(missing_ok=True)
        fold_results.append(result)
        log.emit(
            "inner_fold_complete",
            model=kind,
            fold=fold.fold,
            evaluation=fold.evaluation,
            rows=evaluation["rows"],
            log_loss=[round(value, 8) for value in evaluation["log_loss"]],
        )

    selected_lambda, selection = select_lambda(fold_results)
    atomic_write_json(unit_output / "selection.json", selection)
    log.emit("lambda_selected", model=kind, selected_lambda=selected_lambda)

    scaler_path = unit_work / "final_scaler.json"
    if scaler_path.exists() and not args.force:
        scaler = RobustScaler.from_dict(json.loads(scaler_path.read_text()))
    else:
        scaler = fit_robust_scaler(
            store,
            split.train,
            k,
            kind,
            max_paths_per_day=args.scaler_paths_per_day,
        )
        atomic_write_json(scaler_path, scaler.to_dict())

    final_checkpoint = unit_work / "final_fit.npz"
    weights, final_history, final_best_objective = fit_lambda_path(
        store,
        split.train,
        k,
        kind,
        scaler,
        np.asarray([selected_lambda]),
        final_checkpoint,
        log,
        "final_fit",
        batch_rows=args.batch_rows,
        learning_rate=args.learning_rate,
        max_epochs=args.final_epochs,
        min_epochs=args.min_final_epochs,
        patience=2,
        relative_tolerance=args.relative_tolerance,
    )
    model = PairedLogistic(weights[0], selected_lambda)
    calibration_scores = collect_scores(
        store,
        split.calibration,
        k,
        kind,
        scaler,
        model,
        batch_rows=args.batch_rows,
    )
    alpha, calibration = fit_calibration(calibration_scores)
    atomic_write_json(unit_output / "calibration.json", calibration)
    log.emit(
        "calibration_complete",
        model=kind,
        alpha=round(alpha, 8),
        log_loss=round(calibration["log_loss"], 8),
        at_grid_boundary=calibration["at_grid_boundary"],
    )

    prediction_path = unit_output / "test_predictions.parquet"
    test_metrics = write_test_predictions_and_metrics(
        prediction_path,
        store,
        split.test,
        target_b,
        k,
        kind,
        scaler,
        model,
        alpha,
        batch_rows=args.batch_rows,
    )
    log.emit(
        "test_complete",
        model=kind,
        paths=test_metrics["pooled_path_weighted"]["paths"],
        day_equal_nats_path=round(test_metrics["day_equal"]["nats_path"], 8),
        day_equal_log_loss=round(test_metrics["day_equal"]["paired_log_loss"], 8),
    )

    # Reversible paired-mixture null: same scaler, lambda and temporal split.
    null_checkpoint = unit_work / "null_fit.npz"
    null_weights, null_history, null_best_objective = fit_lambda_path(
        store,
        split.train,
        k,
        kind,
        scaler,
        np.asarray([selected_lambda]),
        null_checkpoint,
        log,
        "reversible_null_fit",
        batch_rows=args.batch_rows,
        learning_rate=args.learning_rate,
        max_epochs=args.final_epochs,
        min_epochs=args.min_final_epochs,
        patience=2,
        relative_tolerance=args.relative_tolerance,
        null_seed=NULL_SEED,
    )
    null_model = PairedLogistic(null_weights[0], selected_lambda)
    null_calibration_scores = collect_scores(
        store,
        split.calibration,
        k,
        kind,
        scaler,
        null_model,
        batch_rows=args.batch_rows,
        null_seed=NULL_SEED,
    )
    null_alpha, null_calibration = fit_calibration(null_calibration_scores)
    null_metrics = write_test_predictions_and_metrics(
        unit_output / "unused_null_predictions.parquet",
        store,
        split.test,
        target_b,
        k,
        kind,
        scaler,
        null_model,
        null_alpha,
        batch_rows=args.batch_rows,
        null_seed=NULL_SEED,
        write_predictions=False,
    )
    log.emit(
        "null_complete",
        model=kind,
        alpha=round(null_alpha, 8),
        day_equal_nats_path=round(null_metrics["day_equal"]["nats_path"], 8),
    )

    model_metadata = {
        **expected,
        "split": {
            "train": split.train,
            "calibration": split.calibration,
            "test": split.test,
        },
        "feature_count": len(model.weights),
        "feature_names": feature_names(k, kind),
        "selected_lambda": selected_lambda,
        "learning_rate": args.learning_rate,
        "batch_rows": args.batch_rows,
        "scaler_paths_per_day_cap": args.scaler_paths_per_day,
        "tuning_epochs": args.tuning_epochs,
        "final_epochs_completed": len(final_history),
        "best_training_objective": float(final_best_objective[0]),
        "weights": model.weights.tolist(),
        "scaler": scaler.to_dict(),
        "calibration_alpha": alpha,
        "null_seed": NULL_SEED,
        "null_calibration_alpha": null_alpha,
        "null_best_training_objective": float(null_best_objective[0]),
        "created_utc": utc_now(),
    }
    metrics = {
        **expected,
        "selection": selection,
        "calibration": calibration,
        "test": test_metrics,
        "reversible_null": {
            "calibration": null_calibration,
            "test": null_metrics,
            "training_history": null_history,
            "best_training_objective": float(null_best_objective[0]),
        },
        "training_history": final_history,
        "best_training_objective": float(final_best_objective[0]),
    }
    atomic_write_json(unit_output / "model.json", model_metadata)
    atomic_write_json(unit_output / "metrics.json", metrics)
    atomic_write_json(
        complete_path,
        {
            **expected,
            "completed_utc": utc_now(),
            "prediction_rows": 2 * test_metrics["pooled_path_weighted"]["paths"],
        },
    )
    final_checkpoint.unlink(missing_ok=True)
    null_checkpoint.unlink(missing_ok=True)
    return metrics


def run_unit(args: argparse.Namespace) -> None:
    target_b, k = args.target_b, args.k
    if (target_b, k) not in SELECTED_SCALES:
        raise ValueError(f"scale {(target_b, k)} is not in the Phase 4 scale grid")
    start_stride = (
        args.start_stride
        if args.start_stride is not None
        else START_STRIDE_BY_B[target_b]
    )
    if start_stride < 1:
        raise ValueError("start_stride must be positive")
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    work_root = Path(args.work_root)
    selected_models = tuple(args.models) if args.models else MODEL_KINDS
    max_feature_count = max(len(feature_names(k, kind)) for kind in selected_models)
    feature_buffer_bytes = int(args.max_feature_buffer_mib * 1024**2)
    scaler_buffer_bytes = int(args.max_scaler_buffer_mib * 1024**2)
    args.batch_rows = min(
        args.batch_rows,
        max(2_048, feature_buffer_bytes // (8 * max_feature_count)),
    )
    args.scaler_paths_per_day = min(
        args.scaler_paths_per_day,
        max(
            500,
            scaler_buffer_bytes // (2 * 9 * 8 * max_feature_count),
        ),
    )
    log_suffix = f"_{args.models[0]}" if args.models and len(args.models) == 1 else ""
    log = ProgressLog(
        Path(args.log_dir) / f"B{target_b}_K{k}{log_suffix}.jsonl",
        f"B{target_b}_K{k}{log_suffix}",
    )
    dates = discover_dates(input_root, target_b)
    split = chronological_day_split(dates)
    log.emit(
        "unit_start",
        target_b=target_b,
        k=k,
        start_stride_blocks=start_stride,
        start_offset_rule="date_ordinal_modulo_stride",
        dates=len(dates),
        batch_rows=args.batch_rows,
        scaler_paths_per_day=args.scaler_paths_per_day,
        max_feature_count=max_feature_count,
        max_feature_buffer_mib=args.max_feature_buffer_mib,
        max_scaler_buffer_mib=args.max_scaler_buffer_mib,
        lambdas=LAMBDA_GRID.tolist(),
        train_dates=split.train,
        calibration_dates=split.calibration,
        test_dates=split.test,
    )
    store = DayStore(input_root, target_b, start_stride, log)
    results = {}
    for kind in selected_models:
        results[kind] = run_model(
            store,
            split,
            target_b,
            k,
            kind,
            output_root,
            work_root,
            log,
            args,
        )
    model_root = output_root / f"target_b={target_b}" / f"k={k}"
    if all(
        (model_root / f"model={kind}" / "complete.json").exists()
        for kind in MODEL_KINDS
    ):
        atomic_write_json(
            model_root / "complete.json",
            {
                "target_b": target_b,
                "k": k,
                "models": list(MODEL_KINDS),
                "completed_utc": utc_now(),
            },
        )
    log.emit(
        "unit_complete",
        target_b=target_b,
        k=k,
        models=len(results),
        requested_models=selected_models,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    unit = subparsers.add_parser("run-unit")
    unit.add_argument("--input-root", required=True)
    unit.add_argument("--output-root", required=True)
    unit.add_argument("--work-root", required=True)
    unit.add_argument("--log-dir", required=True)
    unit.add_argument(
        "--target-b", required=True, type=int, choices=(2, 5, 10, 30, 100, 300)
    )
    unit.add_argument("--k", required=True, type=int)
    unit.add_argument("--start-stride", type=int)
    unit.add_argument("--batch-rows", type=int, default=65_536)
    unit.add_argument("--scaler-paths-per-day", type=int, default=5_000)
    unit.add_argument("--max-feature-buffer-mib", type=float, default=128.0)
    unit.add_argument("--max-scaler-buffer-mib", type=float, default=256.0)
    unit.add_argument("--learning-rate", type=float, default=0.03)
    unit.add_argument("--tuning-epochs", type=int, default=20)
    unit.add_argument("--min-tuning-epochs", type=int, default=6)
    unit.add_argument("--final-epochs", type=int, default=20)
    unit.add_argument("--min-final-epochs", type=int, default=6)
    unit.add_argument("--relative-tolerance", type=float, default=1e-5)
    unit.add_argument("--models", nargs="+", choices=MODEL_KINDS)
    unit.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-unit":
        run_unit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
