"""Deterministic multi-level Queue-Reactive simulation and Arrow publication.

The module deliberately owns only the Phase 14 *source* contract.  It does not
construct B10 blocks or train a model.  One QR transition is one source group;
an optional synthetic ``M`` message caused by a reference-price move remains in
that source group and therefore never changes the requested event count.

Calibration is development-only by construction: callers must supply the exact
allowed development dates and every input row is checked against that set.
Simulation content is independent of the clock specification.  Each durable
event row stores QR-A's exponential clock and QR-B's typed, same-date real clock;
content and full L5 state are never duplicated between clock trees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil


SCHEMA_VERSION = "qr_simulator_v1"
RUNTIME_VERSION = "qr_simulator_runtime_v2"
SIMULATOR_ALGORITHM_VERSION = "multi_l5_qr_gillespie_v1"
SIDES = (-1, 1)  # bid, ask
KINDS = ("limit", "cancel", "market")
CALIBRATION_KINDS = KINDS + ("source_only",)
MIN_LEVELS = 5
REDRAW_CONTRACT = "full_l5_independent_limit_jump_side_major_level_major_v1"


def _schema(fields: Sequence[tuple[str, pa.DataType]], contract: str) -> pa.Schema:
    return pa.schema(
        [pa.field(name, dtype, nullable=False) for name, dtype in fields],
        metadata={
            b"schema_contract": contract.encode(),
            b"schema_version": SCHEMA_VERSION.encode(),
        },
    )


# Parquet canonicalizes the list child name to ``element``.  Naming it here is
# necessary for metadata-aware Arrow schema equality after a round trip.
L5_PRICE_TYPE = pa.list_(pa.field("element", pa.int64()), list_size=MIN_LEVELS)
L5_DEPTH_TYPE = pa.list_(pa.field("element", pa.uint64()), list_size=MIN_LEVELS)

# A replay-derived, development-only sufficient input.  ``duration_ns`` is the
# exposure since the preceding source event.  ``source_only`` rows use
# side/level/size = 0: they preserve wall-clock exposure from real source groups
# outside A/C/T-F-C support and estimate the explicit global no-op hazard.
# Full pre-event L5 state is needed because all 30 competing clocks accrue
# exposure during every interval.
CALIBRATION_EVENT_SCHEMA = _schema(
    [
        ("source_date", pa.date32()),
        ("event_ordinal", pa.uint64()),
        ("timestamp_ns", pa.int64()),
        ("duration_ns", pa.uint64()),
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("kind", pa.string()),
        ("size", pa.uint64()),
        ("pre_bid_depth_l5", L5_DEPTH_TYPE),
        ("pre_ask_depth_l5", L5_DEPTH_TYPE),
    ],
    "qr_calibration_events_v1",
)

# One row is one conditional intensity.  Jump support/probability is repeated
# per row to keep the calibration artifact self-contained and scan-friendly.
PARAMETER_SCHEMA = _schema(
    [
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("queue_bin", pa.uint16()),
        ("queue_lower_inclusive", pa.uint64()),
        ("queue_upper_inclusive", pa.uint64()),
        ("kind", pa.string()),
        ("exposure_ns", pa.uint64()),
        ("event_count", pa.uint64()),
        ("intensity_per_second", pa.float64()),
        ("jump_sizes", pa.list_(pa.field("element", pa.uint64()))),
        ("jump_probabilities", pa.list_(pa.field("element", pa.float64()))),
        ("source_only_exposure_ns", pa.uint64()),
        ("source_only_event_count", pa.uint64()),
        ("source_only_intensity_per_second", pa.float64()),
        ("calibration_queue_bins", pa.uint16()),
        ("rate_prior_events", pa.float64()),
        ("rate_prior_seconds", pa.float64()),
        ("development_dates", pa.list_(pa.field("element", pa.date32()))),
    ],
    "qr_parameters_v1",
)

CALIBRATION_HISTOGRAM_SCHEMA = _schema(
    [
        ("source_date", pa.date32()),
        ("checkpoint_unit", pa.string()),
        ("statistic", pa.string()),
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("kind", pa.string()),
        ("value", pa.uint64()),
        ("count", pa.uint64()),
    ],
    "qr_calibration_date_histogram_v2",
)

CALIBRATION_BIN_COUNT_SCHEMA = _schema(
    [
        ("source_date", pa.date32()),
        ("checkpoint_unit", pa.string()),
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("queue_bin", pa.uint16()),
        ("exposure_ns", pa.uint64()),
        ("limit_count", pa.uint64()),
        ("cancel_count", pa.uint64()),
        ("market_count", pa.uint64()),
    ],
    "qr_calibration_date_bin_counts_v2",
)

# One row is one QR transition/source group.  Content and L5 state are durable
# exactly once.  Both clock specifications are columns on that same row, which
# makes QR-A/QR-B content identity structural rather than a duplicated-tree QA.
EVENT_SCHEMA = _schema(
    [
        ("simulation_day", pa.string()),
        ("source_date", pa.date32()),
        ("universe", pa.uint16()),
        ("event_ordinal", pa.uint64()),
        ("source_group_id", pa.uint64()),
        ("first_raw_ordinal", pa.uint64()),
        ("last_raw_ordinal", pa.uint64()),
        ("qr_a_timestamp_ns", pa.int64()),
        ("qr_a_duration_ns", pa.uint64()),
        ("qr_b_timestamp_ns", pa.int64()),
        ("qr_b_duration_ns", pa.uint64()),
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("kind", pa.string()),
        ("size", pa.uint64()),
        ("applied_size", pa.uint64()),
        ("pre_reference_price_nano", pa.int64()),
        ("post_reference_price_nano", pa.int64()),
        ("reference_shift_ticks", pa.int8()),
        ("synthetic_m", pa.bool_()),
        ("pre_bid_price_l5", L5_PRICE_TYPE),
        ("pre_ask_price_l5", L5_PRICE_TYPE),
        ("post_bid_price_l5", L5_PRICE_TYPE),
        ("post_ask_price_l5", L5_PRICE_TYPE),
        ("pre_bid_depth_l5", L5_DEPTH_TYPE),
        ("pre_ask_depth_l5", L5_DEPTH_TYPE),
        ("post_bid_depth_l5", L5_DEPTH_TYPE),
        ("post_ask_depth_l5", L5_DEPTH_TYPE),
    ],
    "qr_atomic_events_v1",
)

# The message adapter is intentionally normalized: a market event expands to
# T/F/C; a reference shift appends M.  All rows retain the parent source group.
MESSAGE_SCHEMA = _schema(
    [
        ("simulation_day", pa.string()),
        ("source_date", pa.date32()),
        ("event_ordinal", pa.uint64()),
        ("source_group_id", pa.uint64()),
        ("raw_ordinal", pa.uint64()),
        ("message_ordinal", pa.uint8()),
        ("qr_a_timestamp_ns", pa.int64()),
        ("qr_b_timestamp_ns", pa.int64()),
        ("action", pa.string()),
        ("side", pa.int8()),
        ("level", pa.uint8()),
        ("size", pa.uint64()),
        ("synthetic", pa.bool_()),
    ],
    "qr_source_adapter_messages_v1",
)

REAL_CLOCK_SCHEMA = _schema(
    [
        ("dataset", pa.string()),
        ("symbol", pa.string()),
        ("regime", pa.string()),
        ("source_date", pa.date32()),
        ("split", pa.string()),
        ("event_ordinal", pa.uint64()),
        ("duration_ns", pa.uint64()),
    ],
    "qr_real_clock_input_v1",
)

# Descriptive aliases form the supported import surface; the shorter names are
# retained for compact phase-local code.
QR_CALIBRATION_EVENT_SCHEMA = CALIBRATION_EVENT_SCHEMA
QR_PARAMETER_SCHEMA = PARAMETER_SCHEMA
QR_EVENT_SCHEMA = EVENT_SCHEMA
QR_MESSAGE_SCHEMA = MESSAGE_SCHEMA
QR_REAL_CLOCK_SCHEMA = REAL_CLOCK_SCHEMA

# Exact copy of the immutable upstream ``continuous_blocks_v1`` Arrow contract.
# Unlike simulator source schemas, upstream fields are nullable at the Arrow
# level; publication QA below additionally requires every value to be non-null.
CANONICAL_BLOCK_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("symbol", pa.string()),
        ("source_date", pa.date32()),
        ("target_b", pa.uint16()),
        ("block_index", pa.uint32()),
        ("first_source_event_index", pa.uint64()),
        ("last_source_event_index", pa.uint64()),
        ("first_raw_ordinal", pa.uint64()),
        ("last_raw_ordinal", pa.uint64()),
        ("realized_source_event_count", pa.uint16()),
        ("completion_overshoot", pa.uint16()),
        ("block_start_ns", pa.int64()),
        ("block_end_ns", pa.int64()),
        ("elapsed_ns", pa.int64()),
        ("flags_or", pa.uint8()),
        ("start_best_bid_nano", pa.int64()),
        ("start_best_ask_nano", pa.int64()),
        ("start_midprice_x2_nano", pa.int64()),
        ("start_spread_ticks", pa.float32()),
        ("start_bid_depth_l1", pa.uint64()),
        ("start_ask_depth_l1", pa.uint64()),
        ("start_bid_depth_l5", pa.uint64()),
        ("start_ask_depth_l5", pa.uint64()),
        ("start_imbalance_l1", pa.float32()),
        ("start_imbalance_l5", pa.float32()),
        ("end_best_bid_nano", pa.int64()),
        ("end_best_ask_nano", pa.int64()),
        ("end_midprice_x2_nano", pa.int64()),
        ("end_spread_ticks", pa.float32()),
        ("end_bid_depth_l1", pa.uint64()),
        ("end_ask_depth_l1", pa.uint64()),
        ("end_bid_depth_l5", pa.uint64()),
        ("end_ask_depth_l5", pa.uint64()),
        ("end_imbalance_l1", pa.float32()),
        ("end_imbalance_l5", pa.float32()),
        ("bid_depth_l1_net_change", pa.int64()),
        ("ask_depth_l1_net_change", pa.int64()),
        ("bid_depth_l5_net_change", pa.int64()),
        ("ask_depth_l5_net_change", pa.int64()),
        ("bid_depth_l1_gross_variation", pa.uint64()),
        ("ask_depth_l1_gross_variation", pa.uint64()),
        ("bid_depth_l5_gross_variation", pa.uint64()),
        ("ask_depth_l5_gross_variation", pa.uint64()),
        ("bid_best_depletion_count", pa.uint32()),
        ("ask_best_depletion_count", pa.uint32()),
        ("bid_best_appearance_count", pa.uint32()),
        ("ask_best_appearance_count", pa.uint32()),
        ("depletion_plus_appearance_count", pa.uint32()),
        ("appearance_minus_depletion_count", pa.int32()),
        ("mid_log_return", pa.float64()),
        ("mid_change_half_ticks", pa.int32()),
        ("bid_quote_change_ticks", pa.int32()),
        ("ask_quote_change_ticks", pa.int32()),
        ("mid_abs_variation_half_ticks", pa.uint64()),
        ("spread_abs_variation_ticks", pa.uint64()),
        ("same_price_refill_count", pa.uint32()),
        ("cause_add_count", pa.uint32()),
        ("cause_cancel_count", pa.uint32()),
        ("cause_replace_count", pa.uint32()),
        ("cause_displayed_execution_count", pa.uint32()),
        ("cause_other_count", pa.uint32()),
    ],
    metadata={
        b"schema_version": b"continuous_blocks_v1",
        b"scientific_unit": b"completion-closed source-event-count block",
        b"price_unit": b"1e-9 currency units",
    },
)
QR_BLOCK_SCHEMA = CANONICAL_BLOCK_SCHEMA


@dataclass(frozen=True)
class CalibrationConfig:
    development_dates: tuple[date, ...]
    queue_bins: int = 8
    rate_prior_events: float = 0.5
    rate_prior_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.development_dates:
            raise ValueError("development_dates must be non-empty")
        if len(set(self.development_dates)) != len(self.development_dates):
            raise ValueError("development_dates must be unique")
        if self.queue_bins < 1 or self.queue_bins > 65_535:
            raise ValueError("queue_bins outside uint16 range")
        if self.rate_prior_events < 0 or self.rate_prior_seconds <= 0:
            raise ValueError("invalid nonnegative rate smoothing prior")


@dataclass(frozen=True)
class QRParameters:
    """Immutable dense lookup representation of calibrated QR parameters."""

    bin_upper: np.ndarray  # uint64 [2, L, Q], last bound is uint64 max
    rates: np.ndarray  # float64 [2, L, Q, 3]
    jump_sizes: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    jump_probabilities: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    source_only_exposure_ns: int
    source_only_event_count: int
    source_only_rate: float
    calibration_config: CalibrationConfig
    table: pa.Table

    @property
    def levels(self) -> int:
        return int(self.rates.shape[1])

    @property
    def queue_bins(self) -> int:
        return int(self.rates.shape[2])

    def validate(self) -> None:
        if self.levels < MIN_LEVELS or self.rates.shape[0] != 2:
            raise ValueError("QR requires bid/ask L5 or deeper")
        if self.rates.shape[-1] != len(KINDS):
            raise ValueError("rate kind dimension mismatch")
        if self.bin_upper.shape != self.rates.shape[:3]:
            raise ValueError("bin/rate shape mismatch")
        if not np.all(np.isfinite(self.rates)) or np.any(self.rates <= 0):
            raise ValueError("all smoothed intensities must be finite and positive")
        if (
            self.source_only_exposure_ns <= 0
            or self.source_only_event_count < 0
            or not math.isfinite(self.source_only_rate)
            or self.source_only_rate <= 0
        ):
            raise ValueError("invalid global source_only hazard")
        expected_source_rate = (
            self.source_only_event_count + self.calibration_config.rate_prior_events
        ) / (
            self.source_only_exposure_ns / 1e9
            + self.calibration_config.rate_prior_seconds
        )
        if self.source_only_rate != expected_source_rate:
            raise ValueError(
                "source_only rate does not match calibrated count/exposure/prior"
            )
        if len(self.jump_sizes) != 2 or len(self.jump_probabilities) != 2:
            raise ValueError("jump distributions require both sides")
        for si in range(2):
            if (
                len(self.jump_sizes[si]) != self.levels
                or len(self.jump_probabilities[si]) != self.levels
            ):
                raise ValueError("jump distribution level dimension mismatch")
            for li in range(self.levels):
                if len(self.jump_sizes[si][li]) != len(KINDS) or len(
                    self.jump_probabilities[si][li]
                ) != len(KINDS):
                    raise ValueError("jump distribution kind dimension mismatch")
                for sizes, probabilities in zip(
                    self.jump_sizes[si][li],
                    self.jump_probabilities[si][li],
                    strict=True,
                ):
                    if (
                        len(sizes) == 0
                        or len(sizes) != len(probabilities)
                        or np.any(np.asarray(sizes) <= 0)
                        or not np.all(np.isfinite(probabilities))
                        or np.any(np.asarray(probabilities) < 0)
                        or not np.isclose(
                            np.asarray(probabilities).sum(), 1.0, atol=1e-12
                        )
                    ):
                        raise ValueError("invalid positive jump-size distribution")
        _require_schema(self.table, PARAMETER_SCHEMA, "parameter table")

    def matches_calibration_config(self, config: CalibrationConfig) -> bool:
        return self.calibration_config == config


@dataclass(frozen=True)
class SimulationConfig:
    simulation_day: str
    source_date: date
    requested_events: int
    seed: int
    start_timestamp_ns: int
    initial_reference_price_nano: int
    tick_size_nano: int
    universe: int = 0
    dataset: str = "XNAS.ITCH"
    symbol: str = "NVDA"
    regime: str = "daytime"
    split: str = "development"
    target_b: int = 10
    chunk_rows: int = 65_536
    progress_seconds: float = 30.0
    initial_midprice_x2_nano: int | None = None
    initial_bid_price_l5: tuple[int, ...] | None = None
    initial_ask_price_l5: tuple[int, ...] | None = None
    initial_bid_depth_l5: tuple[int, ...] | None = None
    initial_ask_depth_l5: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.requested_events < 1:
            raise ValueError("requested_events must be positive")
        if self.tick_size_nano <= 0:
            raise ValueError("tick_size_nano must be positive")
        if not 0 <= self.universe <= 65_535:
            raise ValueError("universe outside uint16 range")
        if self.target_b != 10:
            raise ValueError("Phase 14 v1 target_b is fixed at 10")
        if (
            self.regime != "daytime"
            or not self.dataset
            or not self.symbol
            or not self.split
        ):
            raise ValueError("invalid source identity for Phase 14 clock lineage")
        if self.split not in {"development", "calibration", "test"}:
            raise ValueError("split must be development, calibration or test")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        if not 30.0 <= self.progress_seconds <= 60.0:
            raise ValueError("progress_seconds must be in [30, 60]")
        arrays = (
            self.initial_bid_price_l5,
            self.initial_ask_price_l5,
            self.initial_bid_depth_l5,
            self.initial_ask_depth_l5,
        )
        if self.initial_midprice_x2_nano is None or any(
            value is None or len(value) != 5 for value in arrays
        ):
            raise ValueError("typed initial bid/ask price/depth L5 is required")
        bid_price = tuple(int(x) for x in self.initial_bid_price_l5 or ())
        ask_price = tuple(int(x) for x in self.initial_ask_price_l5 or ())
        bid_depth = tuple(int(x) for x in self.initial_bid_depth_l5 or ())
        ask_depth = tuple(int(x) for x in self.initial_ask_depth_l5 or ())
        if not all(
            bid_price[i] > bid_price[i + 1] and ask_price[i] < ask_price[i + 1]
            for i in range(4)
        ):
            raise ValueError("initial L5 prices must be strictly ordered")
        spread = ask_price[0] - bid_price[0]
        if spread not in {self.tick_size_nano * value for value in range(1, 5)}:
            raise ValueError("initial spread must be an integer 1-4 ticks")
        if any(price % self.tick_size_nano for price in (*bid_price, *ask_price)):
            raise ValueError("initial L5 prices must lie on the integer nano tick grid")
        if any(value <= 0 for value in (*bid_depth, *ask_depth)):
            raise ValueError("initial L5 depths must be positive")
        if int(self.initial_midprice_x2_nano) != bid_price[0] + ask_price[0]:
            raise ValueError("initial midpoint_x2 does not match best quotes")
        if self.initial_reference_price_nano != int(self.initial_midprice_x2_nano) // 2:
            raise ValueError("initial reference price must be floor(midpoint_x2/2)")


@dataclass
class RealClockArtifact:
    """Typed QR-B intervention with exact date/split/ordinal lineage."""

    table: pa.Table | None
    artifact_path: str
    manifest_path: str
    manifest: Mapping[str, Any]
    _validated: set[tuple[Any, ...]] = field(
        default_factory=set, init=False, repr=False
    )

    def _key(self, config: SimulationConfig) -> tuple[Any, ...]:
        return (
            config.dataset,
            config.symbol,
            config.regime,
            config.source_date,
            config.split,
            config.requested_events,
            config.start_timestamp_ns,
        )

    def _batches(
        self, columns: Sequence[str] | None = None
    ) -> Iterator[pa.RecordBatch]:
        if self.table is not None:
            table = self.table if columns is None else self.table.select(columns)
            yield from table.to_batches(max_chunksize=65_536)
        else:
            yield from pq.ParquetFile(self.artifact_path).iter_batches(
                batch_size=65_536, columns=columns
            )

    def validate_for(self, config: SimulationConfig) -> dict[str, Any]:
        if self.table is not None:
            _require_schema(self.table, REAL_CLOCK_SCHEMA, "QR-B clock artifact")
        else:
            parquet = pq.ParquetFile(self.artifact_path)
            if not parquet.schema_arrow.equals(REAL_CLOCK_SCHEMA, check_metadata=True):
                raise ValueError("QR-B clock physical schema mismatch")
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("schema_contract") != "qr_real_clock_input_v1"
        ):
            raise ValueError("QR-B clock manifest is not a completed typed artifact")
        expected_identity = {
            "dataset": config.dataset,
            "symbol": config.symbol,
            "regime": config.regime,
            "source_date": config.source_date.isoformat(),
            "split": config.split,
            "rows": config.requested_events,
            "start_timestamp_ns": config.start_timestamp_ns,
        }
        for key, expected in expected_identity.items():
            observed = self.manifest.get(key)
            if observed != expected:
                raise ValueError(
                    f"QR-B clock manifest {key} mismatch: {observed!r} != {expected!r}"
                )
        manifest_rows = int(self.manifest.get("rows", -1))
        physical_rows = (
            self.table.num_rows
            if self.table is not None
            else pq.ParquetFile(self.artifact_path).metadata.num_rows
        )
        if physical_rows != config.requested_events or manifest_rows != physical_rows:
            raise ValueError("QR-B clock row coverage mismatch")
        cursor = total_duration = 0
        identity = (
            ("dataset", config.dataset),
            ("symbol", config.symbol),
            ("regime", config.regime),
            ("source_date", config.source_date),
            ("split", config.split),
        )
        for batch in self._batches():
            table = pa.Table.from_batches([batch], schema=REAL_CLOCK_SCHEMA)
            for name, expected in identity:
                if set(table[name].to_pylist()) != {expected}:
                    raise ValueError(f"QR-B clock table {name} lineage mismatch")
            ordinal = table["event_ordinal"].to_numpy(zero_copy_only=False)
            if not np.array_equal(
                ordinal, np.arange(cursor, cursor + len(ordinal), dtype=np.uint64)
            ):
                raise ValueError(
                    "QR-B clock ordinals must exactly cover 0..N-1 in order"
                )
            duration = table["duration_ns"].to_numpy(zero_copy_only=False)
            total_duration += sum(int(x) for x in duration)
            cursor += len(ordinal)
        if cursor != config.requested_events:
            raise ValueError("QR-B streamed ordinal coverage mismatch")
        if config.start_timestamp_ns + total_duration > np.iinfo(np.int64).max:
            raise ValueError("QR-B timestamp would overflow int64")
        self._validated.add(self._key(config))
        return {"rows": cursor, "total_duration_ns": total_duration, "passed": True}

    def iter_durations(
        self, config: SimulationConfig, start_ordinal: int = 0
    ) -> Iterator[int]:
        if self._key(config) not in self._validated:
            self.validate_for(config)
        cursor = 0
        for batch in self._batches(columns=["duration_ns"]):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            end = cursor + len(values)
            offset = max(0, start_ordinal - cursor)
            if end > start_ordinal:
                for value in values[offset:]:
                    yield int(value)
            cursor = end

    def durations_for(self, config: SimulationConfig) -> np.ndarray:
        """Compact-test convenience; production generation uses ``iter_durations``."""

        return np.fromiter(
            self.iter_durations(config), dtype=np.uint64, count=config.requested_events
        )


def real_clock_artifact(
    table: pa.Table,
    *,
    artifact_path: str,
    manifest_path: str,
    manifest: Mapping[str, Any],
) -> RealClockArtifact:
    """Construct a typed clock artifact; scientific identity is checked on use."""

    _require_schema(table, REAL_CLOCK_SCHEMA, "QR-B clock artifact")
    if not artifact_path or not manifest_path:
        raise ValueError("QR-B clock artifact and manifest paths are required")
    return RealClockArtifact(table, artifact_path, manifest_path, dict(manifest))


def load_real_clock_artifact(
    artifact_path: Path | str, manifest_path: Path | str
) -> RealClockArtifact:
    artifact, manifest_file = Path(artifact_path), Path(manifest_path)
    if not artifact.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("QR-B clock artifact and manifest must both exist")
    parquet = pq.ParquetFile(artifact)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("artifact_bytes") != artifact.stat().st_size:
        raise ValueError("QR-B clock artifact byte lineage mismatch")
    return RealClockArtifact(None, str(artifact), str(manifest_file), manifest)


def _require_schema(table: pa.Table, schema: pa.Schema, label: str) -> None:
    if not table.schema.equals(schema, check_metadata=True):
        raise ValueError(
            f"{label} schema mismatch\nexpected: {schema}\nobserved: {table.schema}"
        )


def _as_table(value: pa.Table | pa.RecordBatch | Path | str) -> pa.Table:
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    return pq.read_table(Path(value))


def _side_index(side: int) -> int:
    if side == -1:
        return 0
    if side == 1:
        return 1
    raise ValueError(f"invalid side {side}; expected -1 (bid) or +1 (ask)")


def _strictly_positive_distribution(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(values, dtype=np.uint64)
    positive = positive[positive > 0]
    if not len(positive):
        return np.array([1], dtype=np.uint64), np.array([1.0], dtype=np.float64)
    sizes, counts = np.unique(positive, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    return sizes, probs


def _parameters_from_counts(
    upper: np.ndarray,
    exposure: np.ndarray,
    counts: np.ndarray,
    jump_s: list[list[list[np.ndarray]]],
    jump_p: list[list[list[np.ndarray]]],
    config: CalibrationConfig,
    source_only_exposure_ns: int,
    source_only_event_count: int,
) -> QRParameters:
    source_only_rate = (source_only_event_count + config.rate_prior_events) / (
        source_only_exposure_ns / 1e9 + config.rate_prior_seconds
    )
    rates = np.empty((*upper.shape, len(KINDS)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for si, side in enumerate(SIDES):
        for li in range(MIN_LEVELS):
            lower = 0
            for bi in range(config.queue_bins):
                seconds = float(exposure[si, li, bi]) / 1e9
                rates[si, li, bi] = (
                    counts[si, li, bi].astype(np.float64) + config.rate_prior_events
                ) / (seconds + config.rate_prior_seconds)
                bound = int(upper[si, li, bi])
                for ki, kind in enumerate(KINDS):
                    rows.append(
                        {
                            "side": side,
                            "level": li + 1,
                            "queue_bin": bi,
                            "queue_lower_inclusive": lower,
                            "queue_upper_inclusive": bound,
                            "kind": kind,
                            "exposure_ns": int(exposure[si, li, bi]),
                            "event_count": int(counts[si, li, bi, ki]),
                            "intensity_per_second": float(rates[si, li, bi, ki]),
                            "jump_sizes": jump_s[si][li][ki].tolist(),
                            "jump_probabilities": jump_p[si][li][ki].tolist(),
                            "source_only_exposure_ns": source_only_exposure_ns,
                            "source_only_event_count": source_only_event_count,
                            "source_only_intensity_per_second": source_only_rate,
                            "calibration_queue_bins": config.queue_bins,
                            "rate_prior_events": config.rate_prior_events,
                            "rate_prior_seconds": config.rate_prior_seconds,
                            "development_dates": list(config.development_dates),
                        }
                    )
                lower = bound + 1 if bound < np.iinfo(np.uint64).max else bound
    result = QRParameters(
        upper,
        rates,
        tuple(tuple(tuple(x for x in level) for level in side) for side in jump_s),
        tuple(tuple(tuple(x for x in level) for level in side) for side in jump_p),
        source_only_exposure_ns,
        source_only_event_count,
        source_only_rate,
        config,
        pa.Table.from_pylist(rows, schema=PARAMETER_SCHEMA),
    )
    result.validate()
    return result


def _histogram_quantiles(histogram: Counter[int], bins: int) -> np.ndarray:
    total = sum(histogram.values())
    if total <= 0:
        raise ValueError("empty queue-depth histogram")
    ordered = sorted(histogram.items())
    targets = [math.ceil((i + 1) * total / bins) for i in range(bins)]
    result = np.empty(bins, dtype=np.uint64)
    cumulative = index = 0
    for value, count in ordered:
        cumulative += count
        while index < bins and cumulative >= targets[index]:
            result[index] = value
            index += 1
    for bi in range(1, bins - 1):
        previous = int(result[bi - 1])
        if previous == np.iinfo(np.uint64).max:
            raise ValueError("queue depth leaves no room for requested bins")
        result[bi] = max(int(result[bi]), previous + 1)
    result[-1] = np.iinfo(np.uint64).max
    return result


def _calibration_checkpoint_identity(
    path: Path, source_date: date, config: CalibrationConfig
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    source_date_index = parquet.schema_arrow.get_field_index("source_date")
    if source_date_index < 0:
        raise ValueError("calibration input has no source_date column")
    for row_group_i in range(parquet.metadata.num_row_groups):
        statistics = (
            parquet.metadata.row_group(row_group_i).column(source_date_index).statistics
        )
        if (
            statistics is None
            or not statistics.has_min_max
            or statistics.min != source_date
            or statistics.max != source_date
            or statistics.null_count != 0
        ):
            raise ValueError("calibration input row-group date metadata mismatch")
    return {
        "contract": "qr_calibration_restart_v2",
        "source_date": source_date.isoformat(),
        "input_path": str(path.resolve()),
        "input_rows": parquet.metadata.num_rows,
        "input_bytes": path.stat().st_size,
        "input_mtime_ns": path.stat().st_mtime_ns,
        "input_schema": str(parquet.schema_arrow),
        "development_dates": [value.isoformat() for value in config.development_dates],
        "queue_bins": config.queue_bins,
        "rate_prior_events": config.rate_prior_events,
        "rate_prior_seconds": config.rate_prior_seconds,
        "runtime_version": RUNTIME_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def _publish_calibration_checkpoint(
    final: Path,
    table: pa.Table,
    schema: pa.Schema,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    if final.exists():
        raise FileExistsError(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    atomic_parquet(staging / "statistics.parquet", table, schema)
    completion = {
        "status": "complete",
        "identity": dict(identity),
        "summary": dict(summary),
        "statistics_rows": table.num_rows,
        "statistics_bytes": (staging / "statistics.parquet").stat().st_size,
        "statistics_schema": schema.metadata[b"schema_contract"].decode(),
    }
    _validate_calibration_statistics(
        table,
        completion,
        identity,
        "histogram"
        if schema.equals(CALIBRATION_HISTOGRAM_SCHEMA, check_metadata=True)
        else "bin_counts",
    )
    atomic_json(staging / "complete.json", completion)
    os.replace(staging, final)


def _load_calibration_checkpoint(
    final: Path,
    schema: pa.Schema,
    identity: Mapping[str, Any],
    unit: str,
    expected_summary: Mapping[str, Any] | None = None,
) -> tuple[pa.Table, Mapping[str, Any]]:
    completion_path, statistics = final / "complete.json", final / "statistics.parquet"
    if not completion_path.is_file() or not statistics.is_file():
        raise RuntimeError(f"incomplete calibration checkpoint: {final}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(statistics)
    if (
        completion.get("status") != "complete"
        or completion.get("identity") != dict(identity)
        or completion.get("statistics_rows") != parquet.metadata.num_rows
        or completion.get("statistics_bytes") != statistics.stat().st_size
        or not parquet.schema_arrow.equals(schema, check_metadata=True)
    ):
        raise RuntimeError(f"stale or invalid calibration checkpoint: {final}")
    table = parquet.read()
    _validate_calibration_statistics(
        table, completion, identity, unit, expected_summary=expected_summary
    )
    return table, completion


def _validate_calibration_statistics(
    table: pa.Table,
    completion: Mapping[str, Any],
    identity: Mapping[str, Any],
    unit: str,
    expected_summary: Mapping[str, Any] | None = None,
) -> None:
    """Validate date-local sufficient statistics without content checksums."""

    source_date = date.fromisoformat(str(identity["source_date"]))
    rows = table.to_pylist()
    if not rows or any(
        row["source_date"] != source_date or row["checkpoint_unit"] != unit
        for row in rows
    ):
        raise RuntimeError("calibration checkpoint physical unit identity mismatch")
    summary = completion.get("summary")
    if not isinstance(summary, dict) or int(summary.get("input_rows", -1)) != int(
        identity["input_rows"]
    ):
        raise RuntimeError("calibration checkpoint input-row accounting mismatch")
    if expected_summary is not None and any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise RuntimeError("calibration checkpoint cross-stage accounting mismatch")
    if unit == "histogram":
        depth_totals = {(side, level): 0 for side in SIDES for level in range(1, 6)}
        event_counts = {kind: 0 for kind in CALIBRATION_KINDS}
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            key = (
                row["statistic"],
                row["side"],
                row["level"],
                row["kind"],
                row["value"],
            )
            if key in seen or int(row["count"]) <= 0 or int(row["value"]) <= 0:
                raise RuntimeError("invalid or duplicate histogram statistic")
            seen.add(key)
            side, level = int(row["side"]), int(row["level"])
            if side not in SIDES or not 1 <= level <= 5:
                raise RuntimeError("histogram dimension outside contract")
            if row["statistic"] == "depth" and row["kind"] == "":
                depth_totals[(side, level)] += int(row["count"])
            elif row["statistic"] == "jump" and row["kind"] in KINDS:
                event_counts[str(row["kind"])] += int(row["count"])
            else:
                raise RuntimeError("histogram statistic/kind contract mismatch")
        input_rows = int(identity["input_rows"])
        if any(value != input_rows for value in depth_totals.values()):
            raise RuntimeError("histogram depth totals do not equal input rows")
        source_only = int(summary.get("source_only_count", -1))
        event_counts["source_only"] = source_only
        if source_only < 0 or sum(event_counts.values()) != input_rows:
            raise RuntimeError("histogram event accounting does not equal input rows")
        if (
            summary.get("event_counts") != event_counts
            or int(summary.get("duration_sum_ns", -1)) < 0
        ):
            raise RuntimeError("histogram completion semantic totals mismatch")
    elif unit == "bin_counts":
        bins = int(identity["queue_bins"])
        expected_keys = {
            (side, level, bi)
            for side in SIDES
            for level in range(1, 6)
            for bi in range(bins)
        }
        observed_keys: set[tuple[int, int, int]] = set()
        exposures = {(side, level): 0 for side in SIDES for level in range(1, 6)}
        event_counts = {kind: 0 for kind in KINDS}
        for row in rows:
            key = (int(row["side"]), int(row["level"]), int(row["queue_bin"]))
            if key in observed_keys:
                raise RuntimeError("duplicate bin-count statistic")
            observed_keys.add(key)
            exposures[key[:2]] += int(row["exposure_ns"])
            for kind, column in zip(
                KINDS, ("limit_count", "cancel_count", "market_count"), strict=True
            ):
                event_counts[kind] += int(row[column])
        if observed_keys != expected_keys:
            raise RuntimeError("bin-count dimensions are incomplete")
        duration = int(summary.get("duration_sum_ns", -1))
        if duration < 0 or any(value != duration for value in exposures.values()):
            raise RuntimeError("bin-count exposure totals mismatch")
        if summary.get("event_counts") != event_counts:
            raise RuntimeError("bin-count event totals mismatch")
    else:
        raise RuntimeError(f"unknown calibration checkpoint unit: {unit}")


def _calibration_batch_arrays(
    batch: pa.RecordBatch,
    config: CalibrationConfig,
    source_date: date,
    last_cursor: tuple[int, int] | None,
) -> tuple[pa.Table, tuple[int, int]]:
    table = pa.Table.from_batches([batch], schema=CALIBRATION_EVENT_SCHEMA)
    dates = table["source_date"].to_pylist()
    if set(dates) != {source_date} or source_date not in config.development_dates:
        raise ValueError("partitioned calibration date lineage mismatch")
    ordinals = table["event_ordinal"].to_numpy(zero_copy_only=False)
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False)
    sides = table["side"].to_numpy(zero_copy_only=False)
    levels = table["level"].to_numpy(zero_copy_only=False)
    kinds = table["kind"].to_pylist()
    sizes = table["size"].to_numpy(zero_copy_only=False)
    bid = np.asarray(table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64)
    ask = np.asarray(table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64)
    if (
        bid.shape != (table.num_rows, 5)
        or ask.shape != bid.shape
        or np.any(bid == 0)
        or np.any(ask == 0)
    ):
        raise ValueError("partitioned calibration requires positive complete L5")
    cursor = last_cursor
    for ordinal, timestamp, side, level, kind, size in zip(
        ordinals, timestamps, sides, levels, kinds, sizes, strict=True
    ):
        if cursor is not None and (
            int(ordinal) <= cursor[0] or int(timestamp) < cursor[1]
        ):
            raise ValueError("partitioned calibration rows are not strictly ordered")
        cursor = (int(ordinal), int(timestamp))
        if kind == "source_only":
            if (int(side), int(level), int(size)) != (0, 0, 0):
                raise ValueError("malformed source_only calibration row")
        elif (
            kind not in KINDS
            or int(side) not in SIDES
            or not 1 <= int(level) <= 5
            or int(size) <= 0
        ):
            raise ValueError("malformed mapped calibration row")
    assert cursor is not None
    return table, cursor


def _recompute_calibration_date_statistics(
    path: Path,
    source_date: date,
    config: CalibrationConfig,
    unit: str,
    *,
    upper: np.ndarray | None = None,
    date_i: int = 1,
    date_n: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[pa.Table, dict[str, Any]]:
    """Boundedly recompute one date for value-exact checkpoint admission."""

    parquet = pq.ParquetFile(path)
    rows_done = duration_sum = source_only_count = 0
    last_cursor = None
    event_counts = {kind: 0 for kind in CALIBRATION_KINDS}
    depth_hist = [[Counter() for _ in range(5)] for _ in SIDES]
    jump_hist = [[[Counter() for _ in KINDS] for _ in range(5)] for _ in SIDES]
    exposure = (
        None if upper is None else np.zeros((2, 5, config.queue_bins), dtype=np.uint64)
    )
    counts = (
        None
        if upper is None
        else np.zeros((2, 5, config.queue_bins, 3), dtype=np.uint64)
    )
    chunk_n = math.ceil(parquet.metadata.num_rows / 65_536)
    started = time.monotonic()
    for chunk_i, batch in enumerate(parquet.iter_batches(batch_size=65_536), start=1):
        batch_table, last_cursor = _calibration_batch_arrays(
            batch, config, source_date, last_cursor
        )
        bid = np.asarray(batch_table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64)
        ask = np.asarray(batch_table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64)
        duration = batch_table["duration_ns"].to_numpy(zero_copy_only=False)
        if unit == "histogram":
            for si, values in enumerate((bid, ask)):
                for li in range(5):
                    unique, frequencies = np.unique(values[:, li], return_counts=True)
                    depth_hist[si][li].update(
                        {
                            int(value): int(count)
                            for value, count in zip(unique, frequencies, strict=True)
                        }
                    )
        elif (
            unit == "bin_counts"
            and upper is not None
            and exposure is not None
            and counts is not None
        ):
            depth = np.stack([bid, ask], axis=1)
            for si in range(2):
                for li in range(5):
                    bins = np.searchsorted(upper[si, li], depth[:, si, li], side="left")
                    for bi in range(config.queue_bins):
                        exposure[si, li, bi] += np.uint64(
                            sum(int(value) for value in duration[bins == bi])
                        )
        else:
            raise ValueError(
                "checkpoint recomputation requires a valid unit/global bin grid"
            )
        for row, (side, level, kind, size) in enumerate(
            zip(
                batch_table["side"].to_pylist(),
                batch_table["level"].to_pylist(),
                batch_table["kind"].to_pylist(),
                batch_table["size"].to_pylist(),
                strict=True,
            )
        ):
            event_counts[kind] += 1
            if kind == "source_only":
                source_only_count += 1
            elif unit == "histogram":
                jump_hist[_side_index(int(side))][int(level) - 1][KINDS.index(kind)][
                    int(size)
                ] += 1
            else:
                assert upper is not None and counts is not None
                si, li, ki = _side_index(int(side)), int(level) - 1, KINDS.index(kind)
                depths = bid if si == 0 else ask
                bi = int(np.searchsorted(upper[si, li], depths[row, li], side="left"))
                counts[si, li, bi, ki] += 1
        rows_done += batch.num_rows
        duration_sum += sum(int(value) for value in duration)
        if progress_callback is not None:
            elapsed = max(time.monotonic() - started, 1e-9)
            throughput = rows_done / elapsed
            progress_callback(
                {
                    "stage": f"calibration_verify_{unit}_dates",
                    "date_i": date_i,
                    "date_n": date_n,
                    "chunk_i": chunk_i,
                    "chunk_n": chunk_n,
                    "rows_processed": rows_done,
                    "rows_total": parquet.metadata.num_rows,
                    "bytes_processed": int(
                        path.stat().st_size * rows_done / parquet.metadata.num_rows
                    ),
                    "bytes_total": path.stat().st_size,
                    "throughput_rows_second": throughput,
                    "eta_seconds": (parquet.metadata.num_rows - rows_done)
                    / max(throughput, 1e-9),
                }
            )
    if unit == "histogram":
        rows = []
        for si, side in enumerate(SIDES):
            for li in range(5):
                rows.extend(
                    {
                        "source_date": source_date,
                        "checkpoint_unit": unit,
                        "statistic": "depth",
                        "side": side,
                        "level": li + 1,
                        "kind": "",
                        "value": value,
                        "count": count,
                    }
                    for value, count in sorted(depth_hist[si][li].items())
                )
                for ki, kind in enumerate(KINDS):
                    rows.extend(
                        {
                            "source_date": source_date,
                            "checkpoint_unit": unit,
                            "statistic": "jump",
                            "side": side,
                            "level": li + 1,
                            "kind": kind,
                            "value": value,
                            "count": count,
                        }
                        for value, count in sorted(jump_hist[si][li][ki].items())
                    )
        return pa.Table.from_pylist(rows, schema=CALIBRATION_HISTOGRAM_SCHEMA), {
            "input_rows": rows_done,
            "duration_sum_ns": duration_sum,
            "source_only_count": source_only_count,
            "event_counts": event_counts,
        }
    assert exposure is not None and counts is not None
    rows = [
        {
            "source_date": source_date,
            "checkpoint_unit": unit,
            "side": side,
            "level": li + 1,
            "queue_bin": bi,
            "exposure_ns": int(exposure[si, li, bi]),
            "limit_count": int(counts[si, li, bi, 0]),
            "cancel_count": int(counts[si, li, bi, 1]),
            "market_count": int(counts[si, li, bi, 2]),
        }
        for si, side in enumerate(SIDES)
        for li in range(5)
        for bi in range(config.queue_bins)
    ]
    return pa.Table.from_pylist(rows, schema=CALIBRATION_BIN_COUNT_SCHEMA), {
        "input_rows": rows_done,
        "duration_sum_ns": duration_sum,
        "event_counts": {kind: event_counts[kind] for kind in KINDS},
    }


def calibrate_qr_partitioned(
    paths_by_date: Mapping[str, Path | str],
    config: CalibrationConfig,
    checkpoint_root: Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> QRParameters:
    """Restartable two-stage, bounded-memory calibration over nine date files."""

    expected_dates = tuple(value.isoformat() for value in config.development_dates)
    if tuple(paths_by_date) != expected_dates:
        raise ValueError(
            "partitioned calibration paths must follow exact development-date order"
        )
    paths = {key: Path(value).resolve() for key, value in paths_by_date.items()}
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths.values())
    total_bytes = sum(path.stat().st_size for path in paths.values())
    stage1_tables: list[pa.Table] = []
    for date_i, source_date_text in enumerate(expected_dates, start=1):
        source_date, path = (
            date.fromisoformat(source_date_text),
            paths[source_date_text],
        )
        identity = _calibration_checkpoint_identity(path, source_date, config)
        final = checkpoint_root / "histograms" / f"source_date={source_date_text}"
        try:
            table, completion = _load_calibration_checkpoint(
                final, CALIBRATION_HISTOGRAM_SCHEMA, identity, "histogram"
            )
            recomputed, recomputed_summary = _recompute_calibration_date_statistics(
                path,
                source_date,
                config,
                "histogram",
                date_i=date_i,
                date_n=len(expected_dates),
                progress_callback=progress_callback,
            )
            if (
                not table.equals(recomputed, check_metadata=True)
                or completion.get("summary") != recomputed_summary
            ):
                raise RuntimeError(
                    "histogram checkpoint differs from bounded input recomputation"
                )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            if final.exists():
                stale = final.parent / f".{final.name}.stale.{uuid.uuid4().hex}"
                os.replace(final, stale)
            depth_hist = [[Counter() for _ in range(5)] for _ in SIDES]
            jump_hist = [[[Counter() for _ in KINDS] for _ in range(5)] for _ in SIDES]
            last_cursor = None
            rows_done = duration_sum = source_only_count = 0
            event_counts = {kind: 0 for kind in CALIBRATION_KINDS}
            parquet = pq.ParquetFile(path)
            chunk_n = math.ceil(parquet.metadata.num_rows / 65_536)
            date_started = time.monotonic()
            for chunk_i, batch in enumerate(
                parquet.iter_batches(batch_size=65_536), start=1
            ):
                batch_table, last_cursor = _calibration_batch_arrays(
                    batch, config, source_date, last_cursor
                )
                bid = np.asarray(
                    batch_table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64
                )
                ask = np.asarray(
                    batch_table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64
                )
                for si, values in enumerate((bid, ask)):
                    for li in range(5):
                        unique, counts = np.unique(values[:, li], return_counts=True)
                        depth_hist[si][li].update(
                            {
                                int(x): int(n)
                                for x, n in zip(unique, counts, strict=True)
                            }
                        )
                for side, level, kind, size in zip(
                    batch_table["side"].to_pylist(),
                    batch_table["level"].to_pylist(),
                    batch_table["kind"].to_pylist(),
                    batch_table["size"].to_pylist(),
                    strict=True,
                ):
                    if kind == "source_only":
                        source_only_count += 1
                    else:
                        jump_hist[_side_index(int(side))][int(level) - 1][
                            KINDS.index(kind)
                        ][int(size)] += 1
                    event_counts[kind] += 1
                rows_done += batch.num_rows
                duration_sum += sum(
                    int(x) for x in batch_table["duration_ns"].to_pylist()
                )
                if progress_callback is not None:
                    elapsed = max(time.monotonic() - date_started, 1e-9)
                    throughput = rows_done / elapsed
                    progress_callback(
                        {
                            "stage": "calibration_histogram_dates",
                            "date_i": date_i,
                            "date_n": len(expected_dates),
                            "chunk_i": chunk_i,
                            "chunk_n": chunk_n,
                            "rows_processed": rows_done,
                            "rows_total": parquet.metadata.num_rows,
                            "bytes_processed": int(
                                path.stat().st_size
                                * rows_done
                                / parquet.metadata.num_rows
                            ),
                            "bytes_total": path.stat().st_size,
                            "throughput_rows_second": throughput,
                            "eta_seconds": (parquet.metadata.num_rows - rows_done)
                            / max(throughput, 1e-9),
                        }
                    )
            rows = []
            for si, side in enumerate(SIDES):
                for li in range(5):
                    rows.extend(
                        {
                            "source_date": source_date,
                            "checkpoint_unit": "histogram",
                            "statistic": "depth",
                            "side": side,
                            "level": li + 1,
                            "kind": "",
                            "value": value,
                            "count": count,
                        }
                        for value, count in sorted(depth_hist[si][li].items())
                    )
                    for ki, kind in enumerate(KINDS):
                        rows.extend(
                            {
                                "source_date": source_date,
                                "checkpoint_unit": "histogram",
                                "statistic": "jump",
                                "side": side,
                                "level": li + 1,
                                "kind": kind,
                                "value": value,
                                "count": count,
                            }
                            for value, count in sorted(jump_hist[si][li][ki].items())
                        )
            table = pa.Table.from_pylist(rows, schema=CALIBRATION_HISTOGRAM_SCHEMA)
            _publish_calibration_checkpoint(
                final,
                table,
                CALIBRATION_HISTOGRAM_SCHEMA,
                identity,
                {
                    "input_rows": rows_done,
                    "duration_sum_ns": duration_sum,
                    "source_only_count": source_only_count,
                    "event_counts": event_counts,
                },
            )
        stage1_tables.append(table)
    depth_hist = [[Counter() for _ in range(5)] for _ in SIDES]
    jump_hist = [[[Counter() for _ in KINDS] for _ in range(5)] for _ in SIDES]
    source_only_exposure = source_only_count = 0
    for source_date_text, table in zip(expected_dates, stage1_tables, strict=True):
        completion = json.loads(
            (
                checkpoint_root
                / "histograms"
                / f"source_date={source_date_text}"
                / "complete.json"
            ).read_text()
        )
        source_only_exposure += int(completion["summary"]["duration_sum_ns"])
        source_only_count += int(completion["summary"]["source_only_count"])
        for row in table.to_pylist():
            si, li = _side_index(int(row["side"])), int(row["level"]) - 1
            target = (
                depth_hist[si][li]
                if row["statistic"] == "depth"
                else jump_hist[si][li][KINDS.index(row["kind"])]
            )
            target[int(row["value"])] += int(row["count"])
    upper = np.stack(
        [
            np.stack(
                [
                    _histogram_quantiles(depth_hist[si][li], config.queue_bins)
                    for li in range(5)
                ]
            )
            for si in range(2)
        ]
    )
    exposure = np.zeros((2, 5, config.queue_bins), dtype=np.uint64)
    counts = np.zeros((2, 5, config.queue_bins, 3), dtype=np.uint64)
    upper_identity = upper.tolist()
    for date_i, source_date_text in enumerate(expected_dates, start=1):
        source_date, path = (
            date.fromisoformat(source_date_text),
            paths[source_date_text],
        )
        identity = {
            **_calibration_checkpoint_identity(path, source_date, config),
            "queue_bin_upper": upper_identity,
        }
        final = checkpoint_root / "bin_counts" / f"source_date={source_date_text}"
        try:
            histogram_summary = json.loads(
                (
                    checkpoint_root
                    / "histograms"
                    / f"source_date={source_date_text}"
                    / "complete.json"
                ).read_text()
            )["summary"]
            expected_bin_summary = {
                "input_rows": int(histogram_summary["input_rows"]),
                "duration_sum_ns": int(histogram_summary["duration_sum_ns"]),
                "event_counts": {
                    kind: int(histogram_summary["event_counts"][kind]) for kind in KINDS
                },
            }
            table, completion = _load_calibration_checkpoint(
                final,
                CALIBRATION_BIN_COUNT_SCHEMA,
                identity,
                "bin_counts",
                expected_summary=expected_bin_summary,
            )
            recomputed, recomputed_summary = _recompute_calibration_date_statistics(
                path,
                source_date,
                config,
                "bin_counts",
                upper=upper,
                date_i=date_i,
                date_n=len(expected_dates),
                progress_callback=progress_callback,
            )
            if (
                not table.equals(recomputed, check_metadata=True)
                or completion.get("summary") != recomputed_summary
            ):
                raise RuntimeError(
                    "bin-count checkpoint differs from bounded input recomputation"
                )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            if final.exists():
                os.replace(
                    final, final.parent / f".{final.name}.stale.{uuid.uuid4().hex}"
                )
            local_exposure = np.zeros_like(exposure)
            local_counts = np.zeros_like(counts)
            last_cursor = None
            parquet = pq.ParquetFile(path)
            rows_done = 0
            chunk_n = math.ceil(parquet.metadata.num_rows / 65_536)
            date_started = time.monotonic()
            for chunk_i, batch in enumerate(
                parquet.iter_batches(batch_size=65_536), start=1
            ):
                batch_table, last_cursor = _calibration_batch_arrays(
                    batch, config, source_date, last_cursor
                )
                duration = batch_table["duration_ns"].to_numpy(zero_copy_only=False)
                depth = np.stack(
                    [
                        np.asarray(
                            batch_table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64
                        ),
                        np.asarray(
                            batch_table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64
                        ),
                    ],
                    axis=1,
                )
                for si in range(2):
                    for li in range(5):
                        bins = np.searchsorted(
                            upper[si, li], depth[:, si, li], side="left"
                        )
                        for bi in range(config.queue_bins):
                            local_exposure[si, li, bi] += np.uint64(
                                sum(int(x) for x in duration[bins == bi])
                            )
                for row, kind in enumerate(batch_table["kind"].to_pylist()):
                    if kind == "source_only":
                        continue
                    si = _side_index(int(batch_table["side"][row].as_py()))
                    li = int(batch_table["level"][row].as_py()) - 1
                    ki = KINDS.index(kind)
                    bi = int(
                        np.searchsorted(upper[si, li], depth[row, si, li], side="left")
                    )
                    local_counts[si, li, bi, ki] += 1
                rows_done += batch.num_rows
                if progress_callback is not None:
                    elapsed = max(time.monotonic() - date_started, 1e-9)
                    throughput = rows_done / elapsed
                    progress_callback(
                        {
                            "stage": "calibration_bin_count_dates",
                            "date_i": date_i,
                            "date_n": len(expected_dates),
                            "chunk_i": chunk_i,
                            "chunk_n": chunk_n,
                            "rows_processed": rows_done,
                            "rows_total": parquet.metadata.num_rows,
                            "bytes_processed": int(
                                path.stat().st_size
                                * rows_done
                                / parquet.metadata.num_rows
                            ),
                            "bytes_total": path.stat().st_size,
                            "throughput_rows_second": throughput,
                            "eta_seconds": (parquet.metadata.num_rows - rows_done)
                            / max(throughput, 1e-9),
                        }
                    )
            rows = [
                {
                    "source_date": source_date,
                    "checkpoint_unit": "bin_counts",
                    "side": side,
                    "level": li + 1,
                    "queue_bin": bi,
                    "exposure_ns": int(local_exposure[si, li, bi]),
                    "limit_count": int(local_counts[si, li, bi, 0]),
                    "cancel_count": int(local_counts[si, li, bi, 1]),
                    "market_count": int(local_counts[si, li, bi, 2]),
                }
                for si, side in enumerate(SIDES)
                for li in range(5)
                for bi in range(config.queue_bins)
            ]
            table = pa.Table.from_pylist(rows, schema=CALIBRATION_BIN_COUNT_SCHEMA)
            histogram_summary = json.loads(
                (
                    checkpoint_root
                    / "histograms"
                    / f"source_date={source_date_text}"
                    / "complete.json"
                ).read_text()
            )["summary"]
            _publish_calibration_checkpoint(
                final,
                table,
                CALIBRATION_BIN_COUNT_SCHEMA,
                identity,
                {
                    "input_rows": rows_done,
                    "duration_sum_ns": int(histogram_summary["duration_sum_ns"]),
                    "event_counts": {
                        kind: int(local_counts[:, :, :, ki].sum())
                        for ki, kind in enumerate(KINDS)
                    },
                },
            )
        for row in table.to_pylist():
            si, li, bi = (
                _side_index(int(row["side"])),
                int(row["level"]) - 1,
                int(row["queue_bin"]),
            )
            exposure[si, li, bi] += np.uint64(row["exposure_ns"])
            counts[si, li, bi] += np.asarray(
                [row["limit_count"], row["cancel_count"], row["market_count"]],
                dtype=np.uint64,
            )
    jump_s, jump_p = [], []
    for si in range(2):
        side_s, side_p = [], []
        for li in range(5):
            level_s, level_p = [], []
            for ki in range(3):
                histogram = jump_hist[si][li][ki]
                if histogram:
                    js = np.asarray(sorted(histogram), dtype=np.uint64)
                    weights = np.asarray(
                        [histogram[int(x)] for x in js], dtype=np.float64
                    )
                    jp = weights / weights.sum()
                else:
                    js, jp = np.array([1], dtype=np.uint64), np.array([1.0])
                level_s.append(js)
                level_p.append(jp)
            side_s.append(level_s)
            side_p.append(level_p)
        jump_s.append(side_s)
        jump_p.append(side_p)
    return _parameters_from_counts(
        upper,
        exposure,
        counts,
        jump_s,
        jump_p,
        config,
        source_only_exposure,
        source_only_count,
    )


def _calibrate_qr_parquet(
    path: Path,
    config: CalibrationConfig,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> QRParameters:
    """Two-pass bounded-memory calibration using exact discrete histograms."""

    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(CALIBRATION_EVENT_SCHEMA, check_metadata=True):
        raise ValueError("calibration input schema mismatch")
    identity = (path.stat().st_size, path.stat().st_mtime_ns, parquet.metadata.num_rows)
    if identity[2] == 0:
        raise ValueError("cannot calibrate from zero rows")
    depth_hist = [[Counter() for _ in range(MIN_LEVELS)] for _ in SIDES]
    jump_hist = [[[Counter() for _ in KINDS] for _ in range(MIN_LEVELS)] for _ in SIDES]
    seen_dates: set[date] = set()
    source_only_event_count = 0
    last_cursor: dict[date, tuple[int, int]] = {}
    total_rows = int(parquet.metadata.num_rows)
    chunks_per_pass = math.ceil(total_rows / 65_536)
    progress_started = time.monotonic()
    pass_rows = 0
    for chunk_i, batch in enumerate(parquet.iter_batches(batch_size=65_536), start=1):
        table = pa.Table.from_batches([batch], schema=CALIBRATION_EVENT_SCHEMA)
        dates = table["source_date"].to_pylist()
        ordinals = table["event_ordinal"].to_numpy(zero_copy_only=False)
        timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False)
        sides = table["side"].to_numpy(zero_copy_only=False)
        levels = table["level"].to_numpy(zero_copy_only=False)
        kinds = table["kind"].to_pylist()
        sizes = table["size"].to_numpy(zero_copy_only=False)
        bid = np.asarray(table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64)
        ask = np.asarray(table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64)
        if (
            bid.shape != (table.num_rows, 5)
            or ask.shape != bid.shape
            or np.any(bid == 0)
            or np.any(ask == 0)
        ):
            raise ValueError("calibration requires positive complete L5 state")
        for source_date, ordinal, timestamp, side, level, kind, size in zip(
            dates, ordinals, timestamps, sides, levels, kinds, sizes, strict=True
        ):
            if source_date not in config.development_dates:
                raise ValueError(f"non-development date in calibration: {source_date}")
            previous = last_cursor.get(source_date)
            if previous is not None and (
                int(ordinal) <= previous[0] or int(timestamp) < previous[1]
            ):
                raise ValueError("calibration rows are not strictly event-ordered")
            last_cursor[source_date] = int(ordinal), int(timestamp)
            seen_dates.add(source_date)
            if kind == "source_only":
                if int(side) != 0 or int(level) != 0 or int(size) != 0:
                    raise ValueError("source_only requires side=level=size=0")
                source_only_event_count += 1
            elif (
                kind in KINDS
                and int(side) in SIDES
                and 1 <= int(level) <= 5
                and int(size) > 0
            ):
                jump_hist[_side_index(int(side))][int(level) - 1][KINDS.index(kind)][
                    int(size)
                ] += 1
            else:
                raise ValueError("invalid counted calibration event")
        for si, values in enumerate((bid, ask)):
            for li in range(5):
                unique, frequencies = np.unique(values[:, li], return_counts=True)
                depth_hist[si][li].update(
                    {int(x): int(n) for x, n in zip(unique, frequencies, strict=True)}
                )
        pass_rows += batch.num_rows
        if progress_callback is not None:
            elapsed = max(time.monotonic() - progress_started, 1e-9)
            progress_callback(
                {
                    "stage": "calibration_pass1_histograms",
                    "chunk_i": chunk_i,
                    "chunk_n": chunks_per_pass * 2,
                    "rows_processed": pass_rows,
                    "rows_total": total_rows * 2,
                    "bytes_processed": int(identity[0] * pass_rows / total_rows),
                    "bytes_total": identity[0] * 2,
                    "throughput_rows_second": pass_rows / elapsed,
                    "eta_seconds": (total_rows * 2 - pass_rows)
                    / max(pass_rows / elapsed, 1e-9),
                }
            )
    if seen_dates != set(config.development_dates):
        raise ValueError(
            f"development date coverage incomplete: {sorted(set(config.development_dates) - seen_dates)}"
        )
    upper = np.stack(
        [
            np.stack(
                [
                    _histogram_quantiles(depth_hist[si][li], config.queue_bins)
                    for li in range(5)
                ]
            )
            for si in range(2)
        ]
    )
    exposure = np.zeros((2, 5, config.queue_bins), dtype=np.uint64)
    counts = np.zeros((2, 5, config.queue_bins, 3), dtype=np.uint64)
    source_only_exposure_ns = 0
    parquet = pq.ParquetFile(path)
    pass_rows = 0
    for chunk_i, batch in enumerate(parquet.iter_batches(batch_size=65_536), start=1):
        table = pa.Table.from_batches([batch], schema=CALIBRATION_EVENT_SCHEMA)
        duration = table["duration_ns"].to_numpy(zero_copy_only=False)
        source_only_exposure_ns += sum(int(x) for x in duration)
        sides = table["side"].to_numpy(zero_copy_only=False)
        levels = table["level"].to_numpy(zero_copy_only=False)
        kinds = table["kind"].to_pylist()
        depth = np.stack(
            [
                np.asarray(table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64),
                np.asarray(table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64),
            ],
            axis=1,
        )
        for si in range(2):
            for li in range(5):
                bins = np.searchsorted(upper[si, li], depth[:, si, li], side="left")
                for bi in range(config.queue_bins):
                    exposure[si, li, bi] += np.uint64(
                        sum(int(x) for x in duration[bins == bi])
                    )
        for row, kind in enumerate(kinds):
            if kind == "source_only":
                continue
            si, li, ki = (
                _side_index(int(sides[row])),
                int(levels[row]) - 1,
                KINDS.index(kind),
            )
            bi = int(np.searchsorted(upper[si, li], depth[row, si, li], side="left"))
            counts[si, li, bi, ki] += 1
        pass_rows += batch.num_rows
        if progress_callback is not None:
            completed = total_rows + pass_rows
            elapsed = max(time.monotonic() - progress_started, 1e-9)
            progress_callback(
                {
                    "stage": "calibration_pass2_counts",
                    "chunk_i": chunks_per_pass + chunk_i,
                    "chunk_n": chunks_per_pass * 2,
                    "rows_processed": completed,
                    "rows_total": total_rows * 2,
                    "bytes_processed": identity[0]
                    + int(identity[0] * pass_rows / total_rows),
                    "bytes_total": identity[0] * 2,
                    "throughput_rows_second": completed / elapsed,
                    "eta_seconds": (total_rows * 2 - completed)
                    / max(completed / elapsed, 1e-9),
                }
            )
    if (
        path.stat().st_size,
        path.stat().st_mtime_ns,
        pq.ParquetFile(path).metadata.num_rows,
    ) != identity:
        raise RuntimeError("calibration artifact changed between deterministic passes")
    jump_s, jump_p = [], []
    for si in range(2):
        side_s, side_p = [], []
        for li in range(5):
            level_s, level_p = [], []
            for ki in range(3):
                histogram = jump_hist[si][li][ki]
                if histogram:
                    js = np.asarray(sorted(histogram), dtype=np.uint64)
                    weights = np.asarray(
                        [histogram[int(x)] for x in js], dtype=np.float64
                    )
                    jp = weights / weights.sum()
                else:
                    js, jp = np.array([1], dtype=np.uint64), np.array([1.0])
                level_s.append(js)
                level_p.append(jp)
            side_s.append(level_s)
            side_p.append(level_p)
        jump_s.append(side_s)
        jump_p.append(side_p)
    return _parameters_from_counts(
        upper,
        exposure,
        counts,
        jump_s,
        jump_p,
        config,
        source_only_exposure_ns,
        source_only_event_count,
    )


def calibrate_qr(
    events: pa.Table | pa.RecordBatch | Path | str,
    config: CalibrationConfig,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> QRParameters:
    """Fit deterministic queue-conditioned intensities using development rows only.

    Bins use the nearest-rank empirical quantiles independently for side/level.
    Every event interval contributes exposure to all queues at their pre-event
    depth.  A Gamma-style fixed pseudo-count/exposure makes unseen cells usable
    without a data-dependent fallback.
    """

    if isinstance(events, (Path, str)):
        return _calibrate_qr_parquet(Path(events), config, progress_callback)

    table = _as_table(events)
    _require_schema(table, CALIBRATION_EVENT_SCHEMA, "calibration input")
    if table.num_rows == 0:
        raise ValueError("cannot calibrate from zero rows")

    dates = table["source_date"].to_pylist()
    allowed = set(config.development_dates)
    unexpected = sorted({d for d in dates if d not in allowed})
    if unexpected:
        raise ValueError(f"non-development dates in calibration input: {unexpected}")
    if set(dates) != allowed:
        missing = sorted(allowed - set(dates))
        raise ValueError(f"development date coverage incomplete: {missing}")

    ordinals = table["event_ordinal"].to_numpy(zero_copy_only=False)
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False)
    duration = table["duration_ns"].to_numpy(zero_copy_only=False).astype(np.uint64)
    sides = table["side"].to_numpy(zero_copy_only=False).astype(np.int8)
    levels = table["level"].to_numpy(zero_copy_only=False).astype(np.uint8)
    kinds = table["kind"].to_pylist()
    sizes = table["size"].to_numpy(zero_copy_only=False).astype(np.uint64)
    bid = np.asarray(table["pre_bid_depth_l5"].to_pylist(), dtype=np.uint64)
    ask = np.asarray(table["pre_ask_depth_l5"].to_pylist(), dtype=np.uint64)
    if bid.shape != (table.num_rows, MIN_LEVELS) or ask.shape != bid.shape:
        raise ValueError("calibration input must contain exact complete L5 depths")
    if np.any(bid == 0) or np.any(ask == 0):
        raise ValueError("calibration pre-event L5 depths must be positive")
    if any(kind not in CALIBRATION_KINDS for kind in kinds):
        raise ValueError("calibration kind outside QR event family")
    for side, level, kind, size in zip(sides, levels, kinds, sizes, strict=True):
        if kind == "source_only":
            if int(side) != 0 or int(level) != 0 or int(size) != 0:
                raise ValueError("source_only requires side=level=size=0")
        elif (
            int(side) not in SIDES
            or not 1 <= int(level) <= MIN_LEVELS
            or int(size) <= 0
        ):
            raise ValueError("counted calibration event has invalid side/level/size")
    # Within-day input order must be unique and monotone even if callers
    # accidentally interleave dates.
    last_cursor: dict[date, tuple[int, int]] = {}
    for source_date, ordinal, timestamp in zip(
        dates, ordinals, timestamps, strict=True
    ):
        previous = last_cursor.get(source_date)
        if previous is not None and (
            int(ordinal) <= previous[0] or int(timestamp) < previous[1]
        ):
            raise ValueError("calibration rows are not strictly event-ordered")
        last_cursor[source_date] = int(ordinal), int(timestamp)

    depths = np.stack([bid, ask], axis=1)  # row, side, level
    q = config.queue_bins
    upper = np.empty((2, MIN_LEVELS, q), dtype=np.uint64)
    exposure = np.zeros((2, MIN_LEVELS, q), dtype=np.uint64)
    counts = np.zeros((2, MIN_LEVELS, q, len(KINDS)), dtype=np.uint64)
    quantiles = np.arange(1, q + 1, dtype=np.float64) / q
    for si in range(2):
        for li in range(MIN_LEVELS):
            ordered = np.sort(depths[:, si, li])
            indexes = np.maximum(
                0, np.ceil(quantiles * len(ordered)).astype(np.int64) - 1
            )
            upper[si, li] = ordered[indexes]
            # Keep a fixed dense Q dimension even when empirical quantiles tie.
            # Empty bins are allowed, but their integer intervals remain valid
            # and disjoint rather than acquiring lower > upper bounds.
            for bi in range(1, q - 1):
                previous = int(upper[si, li, bi - 1])
                if previous == np.iinfo(np.uint64).max:
                    raise ValueError("queue depth leaves no room for requested bins")
                upper[si, li, bi] = max(int(upper[si, li, bi]), previous + 1)
            upper[si, li, -1] = np.iinfo(np.uint64).max
            bins = np.searchsorted(upper[si, li], depths[:, si, li], side="left")
            for bi in range(q):
                mask = bins == bi
                # Python int avoids uint64 accumulation overflow before checking.
                exposure[si, li, bi] = np.uint64(sum(int(x) for x in duration[mask]))
    for row, (side, level, kind) in enumerate(zip(sides, levels, kinds, strict=True)):
        if kind == "source_only":
            continue
        si, li, ki = _side_index(int(side)), int(level) - 1, KINDS.index(kind)
        bi = int(np.searchsorted(upper[si, li], depths[row, si, li], side="left"))
        counts[si, li, bi, ki] += 1
    jump_s: list[list[list[np.ndarray]]] = []
    jump_p: list[list[list[np.ndarray]]] = []
    for si, side in enumerate(SIDES):
        side_s: list[list[np.ndarray]] = []
        side_p: list[list[np.ndarray]] = []
        for li in range(MIN_LEVELS):
            level_s, level_p = [], []
            for kind in KINDS:
                mask = np.array(
                    [
                        int(s) == side and int(l) == li + 1 and k == kind
                        for s, l, k in zip(sides, levels, kinds, strict=True)
                    ],
                    dtype=bool,
                )
                js, jp = _strictly_positive_distribution(sizes[mask])
                level_s.append(js)
                level_p.append(jp)
            side_s.append(level_s)
            side_p.append(level_p)
        jump_s.append(side_s)
        jump_p.append(side_p)

    return _parameters_from_counts(
        upper,
        exposure,
        counts,
        jump_s,
        jump_p,
        config,
        sum(int(x) for x in duration),
        kinds.count("source_only"),
    )


def parameters_from_table(table_or_path: pa.Table | Path | str) -> QRParameters:
    """Restore the exact dense representation from a published parameter table."""

    table = _as_table(table_or_path)
    _require_schema(table, PARAMETER_SCHEMA, "parameter table")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("empty parameter table")
    levels = max(int(row["level"]) for row in rows)
    bins = max(int(row["queue_bin"]) for row in rows) + 1
    expected = 2 * levels * bins * len(KINDS)
    if len(rows) != expected:
        raise ValueError(
            f"parameter Cartesian product incomplete: {len(rows)} != {expected}"
        )
    upper = np.zeros((2, levels, bins), dtype=np.uint64)
    rates = np.zeros((2, levels, bins, len(KINDS)), dtype=np.float64)
    jump_s: list[list[list[np.ndarray | None]]] = [
        [[None for _ in KINDS] for _ in range(levels)] for _ in SIDES
    ]
    jump_p: list[list[list[np.ndarray | None]]] = [
        [[None for _ in KINDS] for _ in range(levels)] for _ in SIDES
    ]
    seen: set[tuple[int, int, int, int]] = set()
    scalar_fields = (
        "source_only_exposure_ns",
        "source_only_event_count",
        "source_only_intensity_per_second",
        "calibration_queue_bins",
        "rate_prior_events",
        "rate_prior_seconds",
        "development_dates",
    )
    scalar_values = {field: rows[0][field] for field in scalar_fields}
    for row in rows:
        if any(row[field] != scalar_values[field] for field in scalar_fields):
            raise ValueError("parameter provenance/scalar fields vary across rows")
        si = _side_index(int(row["side"]))
        li, bi, ki = (
            int(row["level"]) - 1,
            int(row["queue_bin"]),
            KINDS.index(row["kind"]),
        )
        key = si, li, bi, ki
        if key in seen:
            raise ValueError(f"duplicate parameter cell {key}")
        seen.add(key)
        upper[si, li, bi] = int(row["queue_upper_inclusive"])
        rates[key] = float(row["intensity_per_second"])
        js = np.asarray(row["jump_sizes"], dtype=np.uint64)
        jp = np.asarray(row["jump_probabilities"], dtype=np.float64)
        if len(js) != len(jp) or not len(js) or np.any(js == 0) or np.any(jp < 0):
            raise ValueError(f"invalid jump distribution at {key}")
        if not np.isclose(jp.sum(), 1.0, atol=1e-12):
            raise ValueError(f"jump probabilities do not sum to one at {key}")
        previous_s, previous_p = jump_s[si][li][ki], jump_p[si][li][ki]
        if previous_s is not None and (
            not np.array_equal(previous_s, js) or not np.array_equal(previous_p, jp)
        ):
            raise ValueError("jump distribution varies across queue bins")
        jump_s[si][li][ki], jump_p[si][li][ki] = js, jp
    calibration_config = CalibrationConfig(
        development_dates=tuple(scalar_values["development_dates"]),
        queue_bins=int(scalar_values["calibration_queue_bins"]),
        rate_prior_events=float(scalar_values["rate_prior_events"]),
        rate_prior_seconds=float(scalar_values["rate_prior_seconds"]),
    )
    if calibration_config.queue_bins != bins:
        raise ValueError(
            "calibration queue_bins does not match parameter Cartesian product"
        )
    result = QRParameters(
        upper,
        rates,
        tuple(
            tuple(tuple(x for x in level if x is not None) for level in side)
            for side in jump_s
        ),
        tuple(
            tuple(tuple(x for x in level if x is not None) for level in side)
            for side in jump_p
        ),
        int(scalar_values["source_only_exposure_ns"]),
        int(scalar_values["source_only_event_count"]),
        float(scalar_values["source_only_intensity_per_second"]),
        calibration_config,
        table,
    )
    result.validate()
    return result


def atomic_parquet(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    normalized = table.cast(schema).replace_schema_metadata(schema.metadata)
    pq.write_table(normalized, temporary, compression="zstd", row_group_size=65_536)
    observed = pq.ParquetFile(temporary)
    if observed.metadata.num_rows != len(
        normalized
    ) or not observed.schema_arrow.equals(schema, check_metadata=True):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"atomic Parquet validation failed: {path}")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class QRSimulator:
    """Sequential L5 Queue-Reactive generator with a fixed redraw contract."""

    def __init__(self, parameters: QRParameters, config: SimulationConfig):
        parameters.validate()
        self.parameters = parameters
        self.config = config
        self.rng = np.random.Generator(np.random.PCG64(config.seed))
        self.reference = int(config.initial_reference_price_nano)
        self.bid_prices = np.asarray(config.initial_bid_price_l5, dtype=np.int64)
        self.ask_prices = np.asarray(config.initial_ask_price_l5, dtype=np.int64)
        self.depth = np.asarray(
            [config.initial_bid_depth_l5, config.initial_ask_depth_l5], dtype=np.uint64
        )
        self.next_ordinal = 0
        self.qr_a_timestamp = int(config.start_timestamp_ns)
        self.qr_b_timestamp = int(config.start_timestamp_ns)
        self.raw_ordinal = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "next_ordinal": self.next_ordinal,
            "next_raw_ordinal": self.raw_ordinal,
            "qr_a_timestamp_ns": self.qr_a_timestamp,
            "qr_b_timestamp_ns": self.qr_b_timestamp,
            "reference_price_nano": self.reference,
            "bid_price_l5": self.bid_prices.tolist(),
            "ask_price_l5": self.ask_prices.tolist(),
            "depth": self.depth.tolist(),
            "rng_state": self.rng.bit_generator.state,
            "redraw_contract": REDRAW_CONTRACT,
        }

    def _sample_jump(self, si: int, li: int, ki: int) -> int:
        values = self.parameters.jump_sizes[si][li][ki]
        probs = self.parameters.jump_probabilities[si][li][ki]
        return int(self.rng.choice(values, p=probs))

    def _redraw_all(self) -> None:
        # Fixed contract: after every reference move, independently redraw all
        # ten queues from calibrated positive limit-arrival jumps in
        # side-major, level-major order.  Initialization always uses typed L5.
        for si in range(2):
            for li in range(self.parameters.levels):
                self.depth[si, li] = self._sample_jump(si, li, 0)

    def _prices(self) -> tuple[list[int], list[int]]:
        return self.bid_prices.tolist(), self.ask_prices.tolist()

    def _rates(self) -> np.ndarray:
        selected = np.empty((2, self.parameters.levels, len(KINDS)), dtype=np.float64)
        for si in range(2):
            for li in range(self.parameters.levels):
                bi = int(
                    np.searchsorted(
                        self.parameters.bin_upper[si, li],
                        self.depth[si, li],
                        side="left",
                    )
                )
                selected[si, li] = self.parameters.rates[si, li, bi]
                # The published L5 ladder contains occupied price levels.  At a
                # one-AES non-best queue, removal clocks are inactive until a
                # limit arrival grows it; this prevents zero-size/no-op events
                # and keeps all five displayed levels populated.
                if li > 0 and int(self.depth[si, li]) <= 1:
                    selected[si, li, 1:] = 0.0
        return selected

    def rows(self, clock: RealClockArtifact) -> Iterator[dict[str, Any]]:
        n = self.config.requested_events
        replacement = clock.iter_durations(self.config, self.next_ordinal)
        for ordinal, real_duration in zip(
            range(self.next_ordinal, n), replacement, strict=True
        ):
            rate_cube = self._rates()
            flat = rate_cube.ravel()
            mapped_total = float(flat.sum())
            total = mapped_total + self.parameters.source_only_rate
            if not math.isfinite(total) or total <= 0:
                raise RuntimeError("nonpositive/nonfinite total QR intensity")
            qr_duration = max(
                1, int(math.ceil(float(self.rng.exponential(1.0 / total)) * 1e9))
            )
            real_duration = int(real_duration)
            self.qr_a_timestamp += qr_duration
            self.qr_b_timestamp += real_duration
            pre_depth = self.depth.copy()
            pre_ref = self.reference
            pre_bid_price, pre_ask_price = self._prices()
            shift = 0
            chosen = int(
                self.rng.choice(
                    len(flat) + 1,
                    p=np.append(flat, self.parameters.source_only_rate) / total,
                )
            )
            if chosen == len(flat):
                si = li = -1
                side, kind, requested_size, applied = 0, "source_only", 0, 0
            else:
                si, li, ki = np.unravel_index(chosen, rate_cube.shape)
                side, kind = SIDES[si], KINDS[ki]
                requested_size = self._sample_jump(si, li, ki)
            if kind == "source_only":
                pass
            elif kind == "limit":
                applied = requested_size
                self.depth[si, li] = np.uint64(int(self.depth[si, li]) + applied)
            else:
                available = int(self.depth[si, li])
                if li == 0:
                    applied = min(requested_size, available)
                    remaining = available - applied
                else:
                    # Non-best levels remain positive so every completion has a
                    # fully populated L5 book.
                    applied = min(requested_size, max(0, available - 1))
                    remaining = available - applied
                self.depth[si, li] = np.uint64(remaining)
                if li == 0 and remaining == 0:
                    shift = side  # depleted ask moves up; depleted bid moves down
                    self.reference += shift * self.config.tick_size_nano
                    self.bid_prices += shift * self.config.tick_size_nano
                    self.ask_prices += shift * self.config.tick_size_nano
                    self._redraw_all()
            post_bid_price, post_ask_price = self._prices()
            message_count = {"limit": 1, "cancel": 1, "market": 3, "source_only": 1}[
                kind
            ] + int(shift != 0)
            first_raw_ordinal = self.raw_ordinal
            last_raw_ordinal = self.raw_ordinal + message_count - 1
            self.raw_ordinal += message_count
            self.next_ordinal = ordinal + 1
            yield {
                "simulation_day": self.config.simulation_day,
                "source_date": self.config.source_date,
                "universe": self.config.universe,
                "event_ordinal": ordinal,
                "source_group_id": ordinal,
                "first_raw_ordinal": first_raw_ordinal,
                "last_raw_ordinal": last_raw_ordinal,
                "qr_a_timestamp_ns": self.qr_a_timestamp,
                "qr_a_duration_ns": qr_duration,
                "qr_b_timestamp_ns": self.qr_b_timestamp,
                "qr_b_duration_ns": real_duration,
                "side": side,
                "level": li + 1 if kind != "source_only" else 0,
                "kind": kind,
                "size": requested_size,
                "applied_size": applied,
                "pre_reference_price_nano": pre_ref,
                "post_reference_price_nano": self.reference,
                "reference_shift_ticks": shift,
                "synthetic_m": shift != 0,
                "pre_bid_price_l5": pre_bid_price,
                "pre_ask_price_l5": pre_ask_price,
                "post_bid_price_l5": post_bid_price,
                "post_ask_price_l5": post_ask_price,
                "pre_bid_depth_l5": pre_depth[0, :MIN_LEVELS].tolist(),
                "pre_ask_depth_l5": pre_depth[1, :MIN_LEVELS].tolist(),
                "post_bid_depth_l5": self.depth[0, :MIN_LEVELS].tolist(),
                "post_ask_depth_l5": self.depth[1, :MIN_LEVELS].tolist(),
            }


def event_batches(
    parameters: QRParameters,
    config: SimulationConfig,
    clock: RealClockArtifact,
) -> Iterator[pa.RecordBatch]:
    """Yield bounded-size Arrow batches without materializing a simulation day."""

    yield from _simulator_batches(
        QRSimulator(parameters, config), clock, config.chunk_rows
    )


def _simulator_batches(
    simulator: QRSimulator, clock: RealClockArtifact, chunk_rows: int
) -> Iterator[pa.RecordBatch]:
    """Continue an initialized/restored simulator in deterministic chunks."""

    pending: list[dict[str, Any]] = []
    for row in simulator.rows(clock):
        pending.append(row)
        if len(pending) == chunk_rows:
            yield pa.RecordBatch.from_pylist(pending, schema=EVENT_SCHEMA)
            pending.clear()
    if pending:
        yield pa.RecordBatch.from_pylist(pending, schema=EVENT_SCHEMA)


def message_rows(event: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand one QR source event into its fixed canonical replay messages."""

    kind = str(event["kind"])
    actions = {
        "limit": ("A",),
        "cancel": ("C",),
        "market": ("T", "F", "C"),
        "source_only": ("N",),
    }[kind]
    for index, action in enumerate(actions):
        yield {
            "simulation_day": event["simulation_day"],
            "source_date": event["source_date"],
            "event_ordinal": event["event_ordinal"],
            "source_group_id": event["source_group_id"],
            "raw_ordinal": int(event["first_raw_ordinal"]) + index,
            "message_ordinal": index,
            "qr_a_timestamp_ns": event["qr_a_timestamp_ns"],
            "qr_b_timestamp_ns": event["qr_b_timestamp_ns"],
            "action": action,
            "side": event["side"],
            "level": event["level"],
            "size": event["applied_size"],
            "synthetic": False,
        }
    if bool(event["synthetic_m"]):
        yield {
            "simulation_day": event["simulation_day"],
            "source_date": event["source_date"],
            "event_ordinal": event["event_ordinal"],
            "source_group_id": event["source_group_id"],
            "raw_ordinal": int(event["first_raw_ordinal"]) + len(actions),
            "message_ordinal": len(actions),
            "qr_a_timestamp_ns": event["qr_a_timestamp_ns"],
            "qr_b_timestamp_ns": event["qr_b_timestamp_ns"],
            "action": "M",
            "side": event["side"],
            "level": event["level"],
            "size": 0,
            "synthetic": True,
        }


def adapt_messages(events: pa.Table) -> pa.Table:
    _require_schema(events, EVENT_SCHEMA, "event table")
    rows = [message for event in events.to_pylist() for message in message_rows(event)]
    return pa.Table.from_pylist(rows, schema=MESSAGE_SCHEMA)


def _imbalance(bid: int, ask: int) -> float:
    total = bid + ask
    return (bid - ask) / total if total else 0.0


def _event_state(row: Mapping[str, Any], prefix: str, tick: int) -> dict[str, Any]:
    bid_price = [int(x) for x in row[f"{prefix}_bid_price_l5"]]
    ask_price = [int(x) for x in row[f"{prefix}_ask_price_l5"]]
    bid_depth = [int(x) for x in row[f"{prefix}_bid_depth_l5"]]
    ask_depth = [int(x) for x in row[f"{prefix}_ask_depth_l5"]]
    bid_l5, ask_l5 = sum(bid_depth), sum(ask_depth)
    return {
        "best_bid_nano": bid_price[0],
        "best_ask_nano": ask_price[0],
        "midprice_x2_nano": bid_price[0] + ask_price[0],
        "spread_ticks": (ask_price[0] - bid_price[0]) / tick,
        "bid_depth_l1": bid_depth[0],
        "ask_depth_l1": ask_depth[0],
        "bid_depth_l5": bid_l5,
        "ask_depth_l5": ask_l5,
        "imbalance_l1": _imbalance(bid_depth[0], ask_depth[0]),
        "imbalance_l5": _imbalance(bid_l5, ask_l5),
        "bid_top": dict(zip(bid_price, bid_depth, strict=True)),
        "ask_top": dict(zip(ask_price, ask_depth, strict=True)),
    }


def _transition(
    pre: Mapping[str, Any], post: Mapping[str, Any], tick: int
) -> dict[str, int]:
    result = {
        "bid_l1_abs": abs(post["bid_depth_l1"] - pre["bid_depth_l1"]),
        "ask_l1_abs": abs(post["ask_depth_l1"] - pre["ask_depth_l1"]),
        "bid_l5_abs": abs(post["bid_depth_l5"] - pre["bid_depth_l5"]),
        "ask_l5_abs": abs(post["ask_depth_l5"] - pre["ask_depth_l5"]),
        "mid_abs_half_ticks": abs(post["midprice_x2_nano"] - pre["midprice_x2_nano"])
        // tick,
        "spread_abs_ticks": int(round(abs(post["spread_ticks"] - pre["spread_ticks"]))),
    }
    for side in ("bid", "ask"):
        pre_best, post_best = pre[f"best_{side}_nano"], post[f"best_{side}_nano"]
        pre_map, post_map = pre[f"{side}_top"], post[f"{side}_top"]
        result[f"{side}_depletion"] = int(
            pre_map.get(pre_best, 0) > 0 and post_map.get(pre_best, 0) == 0
        )
        result[f"{side}_appearance"] = int(
            post_map.get(post_best, 0) > 0 and pre_map.get(post_best, 0) == 0
        )
    return result


def _state_columns(prefix: str, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_best_bid_nano": state["best_bid_nano"],
        f"{prefix}_best_ask_nano": state["best_ask_nano"],
        f"{prefix}_midprice_x2_nano": state["midprice_x2_nano"],
        f"{prefix}_spread_ticks": state["spread_ticks"],
        f"{prefix}_bid_depth_l1": state["bid_depth_l1"],
        f"{prefix}_ask_depth_l1": state["ask_depth_l1"],
        f"{prefix}_bid_depth_l5": state["bid_depth_l5"],
        f"{prefix}_ask_depth_l5": state["ask_depth_l5"],
        f"{prefix}_imbalance_l1": state["imbalance_l1"],
        f"{prefix}_imbalance_l5": state["imbalance_l5"],
    }


class _B10Adapter:
    """Bounded exact upstream BlockAccumulator semantics for one clock spec."""

    def __init__(self, config: SimulationConfig, spec: str):
        if spec not in {"qr_a", "qr_b"}:
            raise ValueError("block spec must be qr_a or qr_b")
        self.config, self.spec = config, spec
        bid_price = [int(x) for x in config.initial_bid_price_l5 or ()]
        ask_price = [int(x) for x in config.initial_ask_price_l5 or ()]
        bid_depth = [int(x) for x in config.initial_bid_depth_l5 or ()]
        ask_depth = [int(x) for x in config.initial_ask_depth_l5 or ()]
        bid_l5, ask_l5 = sum(bid_depth), sum(ask_depth)
        self.start_state: dict[str, Any] = {
            "best_bid_nano": bid_price[0],
            "best_ask_nano": ask_price[0],
            "midprice_x2_nano": bid_price[0] + ask_price[0],
            "spread_ticks": (ask_price[0] - bid_price[0]) / config.tick_size_nano,
            "bid_depth_l1": bid_depth[0],
            "ask_depth_l1": ask_depth[0],
            "bid_depth_l5": bid_l5,
            "ask_depth_l5": ask_l5,
            "imbalance_l1": _imbalance(bid_depth[0], ask_depth[0]),
            "imbalance_l5": _imbalance(bid_l5, ask_l5),
            "bid_top": dict(zip(bid_price, bid_depth, strict=True)),
            "ask_top": dict(zip(ask_price, ask_depth, strict=True)),
        }
        self.start_ns: int = int(config.start_timestamp_ns)
        self.first_source: int | None = None
        self.first_raw: int | None = None
        self.count = 0
        self.block_index = 0
        self.sums: Counter[str] = Counter()
        self.depleted_prices: dict[str, set[int]] = {"bid": set(), "ask": set()}

    def add(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        timestamp = int(event[f"{self.spec}_timestamp_ns"])
        post = _event_state(event, "post", self.config.tick_size_nano)
        pre = _event_state(event, "pre", self.config.tick_size_nano)
        metrics = _transition(pre, post, self.config.tick_size_nano)
        if self.first_source is None:
            self.first_source = int(event["source_group_id"])
            self.first_raw = int(event["first_raw_ordinal"])
        self.count += 1
        self.sums.update(metrics)
        for side in ("bid", "ask"):
            if metrics[f"{side}_depletion"]:
                self.depleted_prices[side].add(pre[f"best_{side}_nano"])
        if event["kind"] == "limit":
            side = "bid" if int(event["side"]) == -1 else "ask"
            added_price = int(event[f"post_{side}_price_l5"][int(event["level"]) - 1])
            if (
                added_price in self.depleted_prices[side]
                and post[f"{side}_top"].get(added_price, 0) > 0
            ):
                self.sums["same_price_refill_count"] += 1
                self.depleted_prices[side].discard(added_price)
        family = {
            "limit": "add",
            "cancel": "cancel",
            "market": "displayed_execution",
            "source_only": "other",
        }[event["kind"]]
        self.sums[f"cause_{family}_count"] += 1
        if self.count < self.config.target_b:
            return None
        assert self.start_ns is not None and self.start_state is not None
        start, tick = self.start_state, self.config.tick_size_nano
        dep = self.sums["bid_depletion"] + self.sums["ask_depletion"]
        app = self.sums["bid_appearance"] + self.sums["ask_appearance"]
        row = {
            "dataset": "QR.XNAS.ITCH",
            "symbol": self.config.symbol,
            "source_date": self.config.source_date,
            "target_b": self.config.target_b,
            "block_index": self.block_index,
            "first_source_event_index": self.first_source,
            "last_source_event_index": int(event["source_group_id"]),
            "first_raw_ordinal": self.first_raw,
            "last_raw_ordinal": int(event["last_raw_ordinal"]),
            "realized_source_event_count": self.count,
            "completion_overshoot": 0,
            "block_start_ns": self.start_ns,
            "block_end_ns": timestamp,
            "elapsed_ns": timestamp - self.start_ns,
            "flags_or": 0,
            **_state_columns("start", start),
            **_state_columns("end", post),
            "bid_depth_l1_net_change": post["bid_depth_l1"] - start["bid_depth_l1"],
            "ask_depth_l1_net_change": post["ask_depth_l1"] - start["ask_depth_l1"],
            "bid_depth_l5_net_change": post["bid_depth_l5"] - start["bid_depth_l5"],
            "ask_depth_l5_net_change": post["ask_depth_l5"] - start["ask_depth_l5"],
            "bid_depth_l1_gross_variation": self.sums["bid_l1_abs"],
            "ask_depth_l1_gross_variation": self.sums["ask_l1_abs"],
            "bid_depth_l5_gross_variation": self.sums["bid_l5_abs"],
            "ask_depth_l5_gross_variation": self.sums["ask_l5_abs"],
            "bid_best_depletion_count": self.sums["bid_depletion"],
            "ask_best_depletion_count": self.sums["ask_depletion"],
            "bid_best_appearance_count": self.sums["bid_appearance"],
            "ask_best_appearance_count": self.sums["ask_appearance"],
            "depletion_plus_appearance_count": dep + app,
            "appearance_minus_depletion_count": app - dep,
            "mid_log_return": math.log(
                post["midprice_x2_nano"] / start["midprice_x2_nano"]
            ),
            "mid_change_half_ticks": (
                post["midprice_x2_nano"] - start["midprice_x2_nano"]
            )
            // tick,
            "bid_quote_change_ticks": (post["best_bid_nano"] - start["best_bid_nano"])
            // tick,
            "ask_quote_change_ticks": (post["best_ask_nano"] - start["best_ask_nano"])
            // tick,
            "mid_abs_variation_half_ticks": self.sums["mid_abs_half_ticks"],
            "spread_abs_variation_ticks": self.sums["spread_abs_ticks"],
            "same_price_refill_count": self.sums["same_price_refill_count"],
            "cause_add_count": self.sums["cause_add_count"],
            "cause_cancel_count": self.sums["cause_cancel_count"],
            "cause_replace_count": 0,
            "cause_displayed_execution_count": self.sums[
                "cause_displayed_execution_count"
            ],
            "cause_other_count": self.sums["cause_other_count"],
        }
        self.block_index += 1
        self.start_state, self.start_ns = post, timestamp
        self.first_source = self.first_raw = None
        self.count, self.sums = 0, Counter()
        return row


def block_batches(
    events: Iterable[pa.RecordBatch], config: SimulationConfig, spec: str
) -> Iterator[pa.RecordBatch]:
    adapter = _B10Adapter(config, spec)
    pending: list[dict[str, Any]] = []
    for batch in events:
        if not batch.schema.equals(EVENT_SCHEMA, check_metadata=True):
            raise ValueError("event batch schema mismatch for B10 adapter")
        for event in batch.to_pylist():
            row = adapter.add(event)
            if row is not None:
                pending.append(row)
                if len(pending) >= config.chunk_rows:
                    yield pa.RecordBatch.from_pylist(
                        pending, schema=CANONICAL_BLOCK_SCHEMA
                    )
                    pending.clear()
    if pending:
        yield pa.RecordBatch.from_pylist(pending, schema=CANONICAL_BLOCK_SCHEMA)


def validate_block_support(table: pa.Table, config: SimulationConfig) -> dict[str, Any]:
    if not table.schema.equals(CANONICAL_BLOCK_SCHEMA, check_metadata=True):
        raise ValueError("canonical block schema mismatch")
    expected = config.requested_events // config.target_b
    indexes = table["block_index"].to_numpy(zero_copy_only=False)
    counts = table["realized_source_event_count"].to_numpy(zero_copy_only=False)
    overshoot = table["completion_overshoot"].to_numpy(zero_copy_only=False)
    nulls = sum(column.null_count for column in table.columns)
    mapped = sum(
        table[name].to_numpy(zero_copy_only=False).astype(np.uint64)
        for name in (
            "cause_add_count",
            "cause_cancel_count",
            "cause_displayed_execution_count",
        )
    )
    source_only = (
        table["cause_other_count"].to_numpy(zero_copy_only=False).astype(np.uint64)
    )
    mapped_source_accounting = bool(np.all(mapped + source_only == counts))
    passed = bool(
        table.num_rows == expected
        and np.array_equal(indexes, np.arange(expected, dtype=np.uint32))
        and np.all(counts == config.target_b)
        and np.all(overshoot == 0)
        and nulls == 0
        and mapped_source_accounting
    )
    return {
        "expected_blocks": expected,
        "produced_blocks": table.num_rows,
        "contiguous_block_index": bool(
            np.array_equal(indexes, np.arange(expected, dtype=np.uint32))
        ),
        "exact_b10": bool(np.all(counts == 10) and np.all(overshoot == 0)),
        "mapped_source_only_accounting_passed": mapped_source_accounting,
        "null_count": nulls,
        "passed": passed,
    }


def qa_event_table(events: pa.Table, requested_events: int) -> dict[str, Any]:
    """Run only the four Phase 14 simulator-specific QA gates."""

    _require_schema(events, EVENT_SCHEMA, "event table")
    rows = events.to_pylist()
    groups = [int(row["source_group_id"]) for row in rows]
    exact_n = (
        len(rows) == requested_events
        and groups == list(range(requested_events))
        and (not rows or int(rows[0]["first_raw_ordinal"]) == 0)
    )
    book_valid = True
    continuity = True
    for index, row in enumerate(rows):
        for key in (
            "pre_bid_depth_l5",
            "pre_ask_depth_l5",
            "post_bid_depth_l5",
            "post_ask_depth_l5",
        ):
            book_valid &= len(row[key]) == MIN_LEVELS and all(
                int(x) > 0 for x in row[key]
            )
        book_valid &= max(row["pre_bid_price_l5"]) < min(row["pre_ask_price_l5"])
        book_valid &= max(row["post_bid_price_l5"]) < min(row["post_ask_price_l5"])
        book_valid &= all(
            int(row["pre_bid_price_l5"][i]) > int(row["pre_bid_price_l5"][i + 1])
            and int(row["post_bid_price_l5"][i]) > int(row["post_bid_price_l5"][i + 1])
            and int(row["pre_ask_price_l5"][i]) < int(row["pre_ask_price_l5"][i + 1])
            and int(row["post_ask_price_l5"][i]) < int(row["post_ask_price_l5"][i + 1])
            for i in range(MIN_LEVELS - 1)
        )
        if index:
            previous = rows[index - 1]
            continuity &= (
                int(row["first_raw_ordinal"]) == int(previous["last_raw_ordinal"]) + 1
            )
            continuity &= int(row["qr_a_timestamp_ns"]) > int(
                previous["qr_a_timestamp_ns"]
            )
            continuity &= int(row["qr_b_timestamp_ns"]) >= int(
                previous["qr_b_timestamp_ns"]
            )
            for field in (
                "reference_price_nano",
                "bid_price_l5",
                "ask_price_l5",
                "bid_depth_l5",
                "ask_depth_l5",
            ):
                continuity &= row[f"pre_{field}"] == previous[f"post_{field}"]
    messages = adapt_messages(events)
    message_by_group: dict[int, list[dict[str, Any]]] = {}
    for row in messages.to_pylist():
        message_by_group.setdefault(int(row["source_group_id"]), []).append(row)
    accounting = len(message_by_group) == len(rows)
    for event in rows:
        group = message_by_group.get(int(event["source_group_id"]), [])
        expected = {
            "limit": ["A"],
            "cancel": ["C"],
            "market": ["T", "F", "C"],
            "source_only": ["N"],
        }[event["kind"]]
        if event["synthetic_m"]:
            expected.append("M")
        accounting &= [row["action"] for row in group] == expected
        accounting &= [int(row["raw_ordinal"]) for row in group] == list(
            range(int(event["first_raw_ordinal"]), int(event["last_raw_ordinal"]) + 1)
        )
        accounting &= sum(row["action"] == "M" for row in group) == int(
            event["synthetic_m"]
        )
        accounting &= (
            int(event["applied_size"]) == 0
            if event["kind"] == "source_only"
            else int(event["applied_size"]) > 0
        )
        reference_delta = int(event["post_reference_price_nano"]) - int(
            event["pre_reference_price_nano"]
        )
        bid_delta = int(event["post_bid_price_l5"][0]) - int(
            event["pre_bid_price_l5"][0]
        )
        ask_delta = int(event["post_ask_price_l5"][0]) - int(
            event["pre_ask_price_l5"][0]
        )
        accounting &= reference_delta == bid_delta == ask_delta
        accounting &= (reference_delta == 0) == (
            int(event["reference_shift_ticks"]) == 0
        )
        accounting &= int(event["reference_shift_ticks"]) in {-1, 0, 1}
    return {
        "requested_events": requested_events,
        "produced_events": len(rows),
        "source_groups": len(set(groups)),
        "levels": MIN_LEVELS,
        "exact_n_passed": bool(exact_n),
        "book_validity_passed": bool(book_valid),
        "state_continuity_passed": bool(continuity),
        "event_accounting_passed": bool(accounting),
        "passed": bool(exact_n and book_valid and continuity and accounting),
    }


def qa_hazard_probabilities(
    parameters: QRParameters, events: pa.Table
) -> dict[str, Any]:
    """Verify each observed pre-state defines one finite all-event probability simplex."""

    _require_schema(events, EVENT_SCHEMA, "event table")
    passed = True
    checked = 0
    for row in events.to_pylist():
        depth = np.asarray(
            [row["pre_bid_depth_l5"], row["pre_ask_depth_l5"]], dtype=np.uint64
        )
        weights = []
        for si in range(2):
            for li in range(parameters.levels):
                bi = int(
                    np.searchsorted(
                        parameters.bin_upper[si, li], depth[si, li], side="left"
                    )
                )
                cell = parameters.rates[si, li, bi].copy()
                if li > 0 and int(depth[si, li]) <= 1:
                    cell[1:] = 0.0
                weights.extend(float(value) for value in cell)
        weights.append(parameters.source_only_rate)
        total = float(sum(weights))
        probabilities = np.asarray(weights, dtype=np.float64) / total
        passed &= bool(
            math.isfinite(total)
            and total > 0
            and np.all(np.isfinite(probabilities))
            and np.all(probabilities >= 0)
            and np.isclose(probabilities.sum(), 1.0, atol=1e-12)
        )
        checked += 1
    return {
        "checked_pre_states": checked,
        "finite_nonnegative_sum_one_passed": bool(passed),
        "passed": bool(passed),
    }


def simulate_table(
    parameters: QRParameters,
    config: SimulationConfig,
    clock: RealClockArtifact,
) -> pa.Table:
    batches = list(event_batches(parameters, config, clock))
    return (
        pa.Table.from_batches(batches, schema=EVENT_SCHEMA)
        if batches
        else pa.Table.from_pylist([], schema=EVENT_SCHEMA)
    )


def reproducibility_qa(
    parameters: QRParameters,
    config: SimulationConfig,
    clock: RealClockArtifact,
) -> dict[str, Any]:
    """Perform the required one-time same-seed exact sequence comparison."""

    first = simulate_table(parameters, config, clock)
    second = simulate_table(parameters, config, clock)
    exact = first.equals(second, check_metadata=True)
    return {
        "rows": first.num_rows,
        "seed": config.seed,
        "exact_equality_passed": bool(exact),
    }


def simulate_clock_pair(
    parameters: QRParameters,
    base_config: SimulationConfig,
    clock: RealClockArtifact,
) -> pa.Table:
    """Compatibility name returning the one durable dual-clock content table."""

    return simulate_table(parameters, base_config, clock)


def _validate_block_file(path: Path, config: SimulationConfig) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(CANONICAL_BLOCK_SCHEMA, check_metadata=True):
        raise RuntimeError(f"canonical block schema mismatch: {path}")
    cursor = nulls = 0
    exact = True
    for batch in parquet.iter_batches(batch_size=config.chunk_rows):
        table = pa.Table.from_batches([batch], schema=CANONICAL_BLOCK_SCHEMA)
        indexes = table["block_index"].to_numpy(zero_copy_only=False)
        exact &= np.array_equal(
            indexes, np.arange(cursor, cursor + len(indexes), dtype=np.uint32)
        )
        exact &= bool(np.all(table["realized_source_event_count"].to_numpy() == 10))
        exact &= bool(np.all(table["completion_overshoot"].to_numpy() == 0))
        cause_total = sum(
            table[name].to_numpy(zero_copy_only=False).astype(np.uint64)
            for name in (
                "cause_add_count",
                "cause_cancel_count",
                "cause_replace_count",
                "cause_displayed_execution_count",
                "cause_other_count",
            )
        )
        exact &= bool(np.all(cause_total == 10))
        mapped = sum(
            table[name].to_numpy(zero_copy_only=False).astype(np.uint64)
            for name in (
                "cause_add_count",
                "cause_cancel_count",
                "cause_displayed_execution_count",
            )
        )
        source_only = (
            table["cause_other_count"].to_numpy(zero_copy_only=False).astype(np.uint64)
        )
        exact &= bool(
            np.all(
                mapped + source_only == table["realized_source_event_count"].to_numpy()
            )
        )
        nulls += sum(column.null_count for column in table.columns)
        cursor += len(indexes)
    expected = config.requested_events // 10
    passed = bool(cursor == expected and exact and nulls == 0)
    return {
        "expected_blocks": expected,
        "produced_blocks": cursor,
        "contiguous_block_index": bool(exact),
        "exact_b10": bool(exact),
        "mapped_source_only_accounting_passed": bool(exact),
        "null_count": nulls,
        "schema_passed": True,
        "passed": passed,
    }


def _block_identity_audit(left: Path, right: Path, batch_size: int) -> dict[str, Any]:
    excluded = {"block_start_ns", "block_end_ns", "elapsed_ns"}
    columns = [name for name in CANONICAL_BLOCK_SCHEMA.names if name not in excluded]
    left_batches = pq.ParquetFile(left).iter_batches(
        batch_size=batch_size, columns=columns
    )
    right_batches = pq.ParquetFile(right).iter_batches(
        batch_size=batch_size, columns=columns
    )
    rows = 0
    exact = True
    for a, b in zip_longest(left_batches, right_batches):
        if a is None or b is None:
            exact = False
            break
        exact &= pa.Table.from_batches([a]).equals(
            pa.Table.from_batches([b]), check_metadata=True
        )
        rows += a.num_rows
    return {
        "compared_rows": rows,
        "excluded_clock_fields": sorted(excluded),
        "exact_passed": bool(exact),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ForcedCheckpointInterrupt(RuntimeError):
    """Test-only interruption raised immediately after a durable checkpoint."""


def _config_record(config: SimulationConfig) -> dict[str, Any]:
    value = dict(config.__dict__)
    value["source_date"] = config.source_date.isoformat()
    for key in (
        "initial_bid_price_l5",
        "initial_ask_price_l5",
        "initial_bid_depth_l5",
        "initial_ask_depth_l5",
    ):
        if value[key] is not None:
            value[key] = list(value[key])
    return value


def _clock_lineage(
    clock: RealClockArtifact, config: SimulationConfig
) -> dict[str, Any]:
    # ``durations_for`` is the authoritative exact-coverage check.
    clock.validate_for(config)
    return {
        "artifact_path": clock.artifact_path,
        "manifest_path": clock.manifest_path,
        "artifact_bytes": clock.manifest.get("artifact_bytes"),
        "dataset": config.dataset,
        "symbol": config.symbol,
        "regime": config.regime,
        "source_date": config.source_date.isoformat(),
        "split": config.split,
        "rows": config.requested_events,
        "start_timestamp_ns": config.start_timestamp_ns,
        "manifest": json.loads(
            json.dumps(dict(clock.manifest), sort_keys=True, default=str)
        ),
    }


def _stat_identity(path: Path) -> dict[str, int]:
    observed = path.stat()
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "runtime_version": RUNTIME_VERSION,
        "schema_version": SCHEMA_VERSION,
        "simulator_algorithm": SIMULATOR_ALGORITHM_VERSION,
        "redraw_contract": REDRAW_CONTRACT,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
    }


def _copy_clock_into_recovery(
    clock: RealClockArtifact, staging: Path, config: SimulationConfig
) -> tuple[RealClockArtifact, dict[str, Any]]:
    source_artifact, source_manifest = (
        Path(clock.artifact_path),
        Path(clock.manifest_path),
    )
    if (
        clock.table is not None
        or not source_artifact.is_file()
        or not source_manifest.is_file()
    ):
        raise ValueError(
            "production QR-B clock must be loaded from a physical typed artifact+manifest"
        )
    before = {
        "artifact": _stat_identity(source_artifact),
        "manifest": _stat_identity(source_manifest),
    }
    destination_artifact = staging / "recovery" / "clock.parquet"
    destination_manifest = staging / "recovery" / "clock.complete.json"
    temporary = destination_artifact.with_name(
        f".{destination_artifact.name}.{os.getpid()}.tmp"
    )
    copy_started = last_progress = time.monotonic()
    copied = 0
    _progress(
        staging / "progress.jsonl",
        "clock_stage_start",
        total_bytes=before["artifact"]["bytes"],
        free_disk_bytes=shutil.disk_usage(staging).free,
    )
    with source_artifact.open("rb") as source, temporary.open("xb") as target:
        while payload := source.read(8 * 1024 * 1024):
            target.write(payload)
            copied += len(payload)
            now = time.monotonic()
            if now - last_progress >= config.progress_seconds:
                elapsed = max(now - copy_started, 1e-9)
                throughput = copied / elapsed
                _progress(
                    staging / "progress.jsonl",
                    "clock_stage_progress",
                    bytes=copied,
                    total_bytes=before["artifact"]["bytes"],
                    bytes_per_second=throughput,
                    eta_seconds=(before["artifact"]["bytes"] - copied)
                    / max(throughput, 1e-9),
                    rss_bytes=psutil.Process().memory_info().rss,
                    temp_bytes=temporary.stat().st_size,
                    free_disk_bytes=shutil.disk_usage(staging).free,
                )
                last_progress = now
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, destination_artifact)
    _progress(
        staging / "progress.jsonl",
        "clock_stage_complete",
        bytes=copied,
        elapsed_seconds=time.monotonic() - copy_started,
    )
    atomic_json(destination_manifest, dict(clock.manifest))
    after = {
        "artifact": _stat_identity(source_artifact),
        "manifest": _stat_identity(source_manifest),
    }
    if before != after:
        raise RuntimeError("QR-B clock source changed while staging immutable copy")
    staged = load_real_clock_artifact(destination_artifact, destination_manifest)
    staged.validate_for(config)
    binding = {
        "source_artifact_path": str(source_artifact),
        "source_manifest_path": str(source_manifest),
        "source_stats": before,
        "staged_artifact_path": str(destination_artifact.relative_to(staging)),
        "staged_manifest_path": str(destination_manifest.relative_to(staging)),
        "staged_stats": {
            "artifact": _stat_identity(destination_artifact),
            "manifest": _stat_identity(destination_manifest),
        },
        "manifest": json.loads(
            json.dumps(dict(clock.manifest), sort_keys=True, default=str)
        ),
    }
    return staged, binding


def _load_bound_staged_clock(
    staging: Path, binding: Mapping[str, Any], config: SimulationConfig
) -> RealClockArtifact:
    source_artifact = Path(binding["source_artifact_path"])
    source_manifest = Path(binding["source_manifest_path"])
    staged_artifact = staging / binding["staged_artifact_path"]
    staged_manifest = staging / binding["staged_manifest_path"]
    observed = {
        "source_stats": {
            "artifact": _stat_identity(source_artifact),
            "manifest": _stat_identity(source_manifest),
        },
        "staged_stats": {
            "artifact": _stat_identity(staged_artifact),
            "manifest": _stat_identity(staged_manifest),
        },
    }
    if observed["source_stats"] != binding.get("source_stats") or observed[
        "staged_stats"
    ] != binding.get("staged_stats"):
        raise RuntimeError("QR-B clock source/staged stat identity changed")
    manifest_value = json.loads(staged_manifest.read_text(encoding="utf-8"))
    if manifest_value != binding.get("manifest"):
        raise RuntimeError("QR-B staged manifest content changed")
    clock = load_real_clock_artifact(staged_artifact, staged_manifest)
    clock.validate_for(config)
    return clock


def _progress(log_path: Path, event: str, **fields: Any) -> None:
    row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    line = json.dumps(row, sort_keys=True, default=str)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def _quarantine_staging(staging: Path, reason: str) -> Path:
    target = staging.with_name(
        f"{staging.name}.quarantine-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(staging, target)
    atomic_json(
        target / "quarantine.json",
        {
            "status": "quarantined",
            "reason": reason,
            "quarantined_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return target


def _validate_checkpoint_chunks(staging: Path, checkpoint: Mapping[str, Any]) -> int:
    expected_event_paths = {
        str(staging / item["event_path"]) for item in checkpoint.get("chunks", [])
    }
    expected_message_paths = {
        str(staging / item["message_path"]) for item in checkpoint.get("chunks", [])
    }
    observed_event_paths = {
        str(path)
        for path in (staging / "recovery/content_chunks").glob("part-*.parquet")
    }
    observed_message_paths = {
        str(path)
        for path in (staging / "recovery/message_chunks").glob("part-*.parquet")
    }
    if (
        observed_event_paths != expected_event_paths
        or observed_message_paths != expected_message_paths
    ):
        raise RuntimeError("uncommitted/missing chunk detected; blind append forbidden")
    cursor = 0
    expected_raw = 0
    for expected_index, item in enumerate(checkpoint.get("chunks", [])):
        if int(item.get("index", -1)) != expected_index:
            raise RuntimeError("checkpoint chunk index is not contiguous")
        expected_event = f"recovery/content_chunks/part-{expected_index:06d}.parquet"
        expected_message = f"recovery/message_chunks/part-{expected_index:06d}.parquet"
        if (
            item.get("event_path") != expected_event
            or item.get("message_path") != expected_message
        ):
            raise RuntimeError("checkpoint chunk path is not deterministic")
        event_path = staging / item["event_path"]
        message_path = staging / item["message_path"]
        for path, schema, rows_key, bytes_key in (
            (event_path, EVENT_SCHEMA, "rows", "event_bytes"),
            (message_path, MESSAGE_SCHEMA, "message_rows", "message_bytes"),
        ):
            parquet = pq.ParquetFile(path)
            if (
                not parquet.schema_arrow.equals(schema, check_metadata=True)
                or parquet.metadata.num_rows != int(item[rows_key])
                or path.stat().st_size != int(item[bytes_key])
            ):
                raise RuntimeError(
                    f"checkpoint chunk footer/schema/bytes mismatch: {path}"
                )
        events = pq.ParquetFile(event_path).read(
            columns=[
                "source_group_id",
                "first_raw_ordinal",
                "last_raw_ordinal",
                "kind",
                "synthetic_m",
            ]
        )
        groups = events["source_group_id"].to_numpy(zero_copy_only=False)
        if not np.array_equal(
            groups, np.arange(cursor, cursor + len(groups), dtype=np.uint64)
        ):
            raise RuntimeError("checkpoint chunk source cursor mismatch")
        first_raw = events["first_raw_ordinal"].to_numpy(zero_copy_only=False)
        last_raw = events["last_raw_ordinal"].to_numpy(zero_copy_only=False)
        if len(groups):
            if int(first_raw[0]) != expected_raw or np.any(
                first_raw[1:] != last_raw[:-1] + 1
            ):
                raise RuntimeError("checkpoint chunk raw cursor mismatch")
            expected_raw = int(last_raw[-1]) + 1
        kinds = events["kind"].to_pylist()
        accounting = {
            "source_events": len(groups),
            "raw_records": int(item["message_rows"]),
            "reference_shifts": sum(bool(x) for x in events["synthetic_m"].to_pylist()),
            **{f"kind_{kind}": kinds.count(kind) for kind in CALIBRATION_KINDS},
        }
        if accounting != item.get("event_accounting"):
            raise RuntimeError("checkpoint chunk transition accounting mismatch")
        cursor += len(groups)
    snapshot = checkpoint.get("simulator", {})
    if (
        int(snapshot.get("next_ordinal", -1)) != cursor
        or int(snapshot.get("next_raw_ordinal", -1)) != expected_raw
    ):
        raise RuntimeError("checkpoint simulator/chunk cursor mismatch")
    return cursor


def _reconstruct_from_committed(
    parameters: QRParameters,
    config: SimulationConfig,
    clock: RealClockArtifact,
    staging: Path,
    checkpoint: Mapping[str, Any],
) -> QRSimulator:
    """Ignore persisted RNG trust: replay seed through every committed event."""

    simulator = QRSimulator(parameters, config)
    replay_generator = _simulator_batches(simulator, clock, config.chunk_rows)
    for item in checkpoint.get("chunks", []):
        try:
            replayed = pa.Table.from_batches(
                [next(replay_generator)], schema=EVENT_SCHEMA
            )
        except StopIteration as exc:
            raise RuntimeError(
                "deterministic replay ended before committed chunk"
            ) from exc
        if replayed.num_rows != int(item["rows"]):
            raise RuntimeError("deterministic replay chunk boundary mismatch")
        committed = pq.ParquetFile(staging / item["event_path"]).read()
        if not replayed.equals(committed, check_metadata=True):
            raise RuntimeError(
                f"committed QR content differs from seed replay: chunk={item['index']}"
            )
    persisted = json.loads(
        json.dumps(checkpoint.get("simulator"), sort_keys=True, default=str)
    )
    reconstructed = json.loads(
        json.dumps(simulator.snapshot(), sort_keys=True, default=str)
    )
    if persisted != reconstructed:
        raise RuntimeError(
            "persisted RNG/L5/cursor state differs from deterministic seed replay"
        )
    return simulator


def _quarantine_uncommitted_chunks(
    staging: Path, checkpoint: Mapping[str, Any]
) -> list[str]:
    expected = {
        str(staging / item[key])
        for item in checkpoint.get("chunks", [])
        for key in ("event_path", "message_path")
    }
    candidates = []
    for directory in (
        staging / "recovery/content_chunks",
        staging / "recovery/message_chunks",
    ):
        if directory.exists():
            candidates.extend(path for path in directory.iterdir() if path.is_file())
    uncommitted = [path for path in candidates if str(path) not in expected]
    if not uncommitted:
        return []
    quarantine = staging / "recovery" / "quarantine" / uuid.uuid4().hex
    quarantine.mkdir(parents=True)
    moved = []
    for path in uncommitted:
        target = quarantine / f"{path.parent.name}-{path.name}"
        os.replace(path, target)
        moved.append(str(target.relative_to(staging)))
    atomic_json(
        quarantine / "quarantine.json",
        {
            "status": "quarantined",
            "files": moved,
            "quarantined_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return moved


def _compose_chunks(
    staging: Path,
    checkpoint: Mapping[str, Any],
    key: str,
    output: Path,
    schema: pa.Schema,
) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    rows = 0
    try:
        for item in checkpoint["chunks"]:
            parquet = pq.ParquetFile(staging / item[key])
            for batch in parquet.iter_batches():
                writer.write_batch(batch)
                rows += batch.num_rows
    finally:
        writer.close()
    observed = pq.ParquetFile(temporary)
    if (
        not observed.schema_arrow.equals(schema, check_metadata=True)
        or observed.metadata.num_rows != rows
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"final chunk composition failed: {output}")
    os.replace(temporary, output)


def publish_simulation_day(
    output_dir: Path | str,
    parameters: QRParameters,
    config: SimulationConfig,
    clock: RealClockArtifact,
    *,
    run_reproducibility_qa: bool = False,
    force_interrupt_after_chunks: int | None = None,
) -> dict[str, Any]:
    """Checkpoint, resume, validate and atomically publish one dual-clock QR day."""

    final = Path(output_dir)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite published day: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging"
    detached_recoveries = sorted(final.parent.glob(f".{final.name}.recovery-*"))
    if (
        staging.exists()
        and not (staging / "recovery").exists()
        and len(detached_recoveries) == 1
    ):
        # Recover the narrow publication transaction window after recovery was
        # detached but before the clean staging directory was renamed final.
        os.replace(detached_recoveries[0], staging / "recovery")
        detached_recoveries = []
    if len(detached_recoveries) > 1:
        raise RuntimeError(
            "multiple detached recovery roots require manual quarantine review"
        )
    checkpoint_path = staging / "recovery" / "checkpoint.json"
    parameter_path = staging / "recovery" / "parameters.parquet"
    log_path = staging / "progress.jsonl"
    config_record = _config_record(config)
    lineage = _clock_lineage(clock, config)
    runtime_identity = _runtime_identity()
    started = time.monotonic()

    if staging.exists() and not checkpoint_path.is_file():
        _quarantine_staging(staging, "staging_without_checkpoint")
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        try:
            if (
                checkpoint.get("config") != config_record
                or checkpoint.get("clock_lineage") != lineage
                or checkpoint.get("runtime_identity") != runtime_identity
            ):
                raise RuntimeError("checkpoint config/clock/runtime identity mismatch")
            prior_parameters = pq.ParquetFile(parameter_path).read()
            if not prior_parameters.equals(parameters.table, check_metadata=True):
                raise RuntimeError("checkpoint QR parameters mismatch")
            active_clock = _load_bound_staged_clock(
                staging, checkpoint["clock_binding"], config
            )
            cursor = _validate_checkpoint_chunks(staging, checkpoint)
            simulator = _reconstruct_from_committed(
                parameters, config, active_clock, staging, checkpoint
            )
        except BaseException as exc:
            quarantine = _quarantine_staging(staging, str(exc))
            raise RuntimeError(
                f"incompatible checkpoint quarantined at {quarantine}"
            ) from exc
        _progress(
            log_path,
            "resume",
            cursor=cursor,
            chunks=len(checkpoint["chunks"]),
            requested_events=config.requested_events,
            chunk_rows=config.chunk_rows,
            available_ram_bytes=psutil.virtual_memory().available,
            free_disk_bytes=shutil.disk_usage(staging).free,
        )
        started = time.monotonic() - float(checkpoint.get("elapsed_seconds", 0.0))
        was_resumed = True
    else:
        staging.mkdir(parents=True, exist_ok=False)
        (staging / "recovery" / "content_chunks").mkdir(parents=True)
        (staging / "recovery" / "message_chunks").mkdir(parents=True)
        atomic_parquet(parameter_path, parameters.table, PARAMETER_SCHEMA)
        active_clock, clock_binding = _copy_clock_into_recovery(clock, staging, config)
        simulator = QRSimulator(parameters, config)
        checkpoint = {
            "status": "generating",
            "run_id": uuid.uuid4().hex,
            "config": config_record,
            "clock_lineage": lineage,
            "clock_binding": clock_binding,
            "runtime_identity": runtime_identity,
            "chunks": [],
            "simulator": simulator.snapshot(),
            "elapsed_seconds": 0.0,
        }
        atomic_json(checkpoint_path, checkpoint)
        cursor = 0
        was_resumed = False
        _progress(
            log_path,
            "start",
            cursor=0,
            requested_events=config.requested_events,
            chunk_rows=config.chunk_rows,
            cpu_count=psutil.cpu_count(),
            available_ram_bytes=psutil.virtual_memory().available,
            free_disk_bytes=shutil.disk_usage(staging).free,
        )

    last_progress = time.monotonic()
    try:
        for batch in _simulator_batches(simulator, active_clock, config.chunk_rows):
            index = len(checkpoint["chunks"])
            event_relative = (
                Path("recovery/content_chunks") / f"part-{index:06d}.parquet"
            )
            message_relative = (
                Path("recovery/message_chunks") / f"part-{index:06d}.parquet"
            )
            event_chunk, message_chunk = (
                staging / event_relative,
                staging / message_relative,
            )
            if event_chunk.exists() or message_chunk.exists():
                raise RuntimeError("refusing blind append over an uncommitted chunk")
            event_table = pa.Table.from_batches([batch], schema=EVENT_SCHEMA)
            message_table = adapt_messages(event_table)
            atomic_parquet(event_chunk, event_table, EVENT_SCHEMA)
            atomic_parquet(message_chunk, message_table, MESSAGE_SCHEMA)
            item = {
                "index": index,
                "event_path": str(event_relative),
                "message_path": str(message_relative),
                "rows": event_table.num_rows,
                "message_rows": message_table.num_rows,
                "event_bytes": event_chunk.stat().st_size,
                "message_bytes": message_chunk.stat().st_size,
                "first_source_group_id": int(event_table["source_group_id"][0].as_py()),
                "last_source_group_id": int(event_table["source_group_id"][-1].as_py()),
                "event_accounting": {
                    "source_events": event_table.num_rows,
                    "raw_records": message_table.num_rows,
                    "reference_shifts": sum(
                        bool(x) for x in event_table["synthetic_m"].to_pylist()
                    ),
                    **{
                        f"kind_{kind}": event_table["kind"].to_pylist().count(kind)
                        for kind in CALIBRATION_KINDS
                    },
                },
            }
            checkpoint["chunks"].append(item)
            checkpoint["simulator"] = simulator.snapshot()
            checkpoint["elapsed_seconds"] = time.monotonic() - started
            atomic_json(checkpoint_path, checkpoint)
            cursor += event_table.num_rows
            now = time.monotonic()
            if (
                now - last_progress >= config.progress_seconds
                or cursor == config.requested_events
            ):
                elapsed = max(now - started, 1e-9)
                throughput = cursor / elapsed
                temp_bytes = sum(
                    int(x["event_bytes"]) + int(x["message_bytes"])
                    for x in checkpoint["chunks"]
                )
                _progress(
                    log_path,
                    "progress",
                    chunk=index + 1,
                    rows=cursor,
                    requested_events=config.requested_events,
                    bytes=temp_bytes,
                    elapsed_seconds=elapsed,
                    rows_per_second=throughput,
                    eta_seconds=(config.requested_events - cursor)
                    / max(throughput, 1e-9),
                    rss_bytes=psutil.Process().memory_info().rss,
                    temp_bytes=temp_bytes,
                    free_disk_bytes=shutil.disk_usage(staging).free,
                    checkpoint_cursor=simulator.next_ordinal,
                )
                last_progress = now
            if (
                force_interrupt_after_chunks is not None
                and len(checkpoint["chunks"]) == force_interrupt_after_chunks
            ):
                raise ForcedCheckpointInterrupt(
                    f"forced after durable chunk {force_interrupt_after_chunks}"
                )

        if cursor != config.requested_events:
            raise RuntimeError(
                f"generation cursor {cursor} != requested {config.requested_events}"
            )
        checkpoint["status"] = "generated"
        atomic_json(checkpoint_path, checkpoint)
        event_path, message_path = (
            staging / "events.parquet",
            staging / "messages.parquet",
        )
        _compose_chunks(staging, checkpoint, "event_path", event_path, EVENT_SCHEMA)
        _compose_chunks(
            staging, checkpoint, "message_path", message_path, MESSAGE_SCHEMA
        )
        event_file, message_file = (
            pq.ParquetFile(event_path),
            pq.ParquetFile(message_path),
        )
        if message_file.metadata.num_rows != sum(
            int(item["message_rows"]) for item in checkpoint["chunks"]
        ):
            raise RuntimeError("final message accounting/footer mismatch")

        block_paths = {
            spec: staging
            / "blocks"
            / f"spec={spec}"
            / f"source_date={config.source_date.isoformat()}"
            / "part-00000.parquet"
            for spec in ("qr_a", "qr_b")
        }
        for spec, path in block_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.unlink(missing_ok=True)
            writer = pq.ParquetWriter(
                temporary, CANONICAL_BLOCK_SCHEMA, compression="zstd"
            )
            try:
                for block_batch in block_batches(
                    event_file.iter_batches(batch_size=config.chunk_rows), config, spec
                ):
                    writer.write_batch(block_batch)
            finally:
                writer.close()
            if not pq.ParquetFile(temporary).schema_arrow.equals(
                CANONICAL_BLOCK_SCHEMA, check_metadata=True
            ):
                raise RuntimeError(f"block schema failed before publication: {spec}")
            os.replace(temporary, path)

        # Four source gates plus exact canonical support and clock identity.
        exact_ordinals = book_valid = continuity = accounting = initial_state_valid = (
            hazard_valid
        ) = True
        observed_rows = 0
        previous: dict[str, Any] | None = None
        for batch in event_file.iter_batches(batch_size=config.chunk_rows):
            table = pa.Table.from_batches([batch], schema=EVENT_SCHEMA)
            local = qa_event_table(table, table.num_rows)
            hazard_valid &= qa_hazard_probabilities(parameters, table)["passed"]
            groups = table["source_group_id"].to_numpy(zero_copy_only=False)
            exact_ordinals &= np.array_equal(
                groups, np.arange(observed_rows, observed_rows + len(groups))
            )
            first = table.slice(0, 1).to_pylist()[0]
            if observed_rows == 0:
                initial_state_valid &= (
                    int(first["pre_reference_price_nano"])
                    == config.initial_reference_price_nano
                    and first["pre_bid_price_l5"]
                    == list(config.initial_bid_price_l5 or ())
                    and first["pre_ask_price_l5"]
                    == list(config.initial_ask_price_l5 or ())
                    and first["pre_bid_depth_l5"]
                    == list(config.initial_bid_depth_l5 or ())
                    and first["pre_ask_depth_l5"]
                    == list(config.initial_ask_depth_l5 or ())
                )
            if previous is not None:
                continuity &= (
                    int(first["first_raw_ordinal"])
                    == int(previous["last_raw_ordinal"]) + 1
                )
                for field in (
                    "reference_price_nano",
                    "bid_price_l5",
                    "ask_price_l5",
                    "bid_depth_l5",
                    "ask_depth_l5",
                ):
                    continuity &= first[f"pre_{field}"] == previous[f"post_{field}"]
            previous = table.slice(table.num_rows - 1, 1).to_pylist()[0]
            observed_rows += table.num_rows
            book_valid &= local["book_validity_passed"]
            continuity &= local["state_continuity_passed"]
            accounting &= local["event_accounting_passed"]
        block_support = {
            spec: _validate_block_file(path, config)
            for spec, path in block_paths.items()
        }
        identity = _block_identity_audit(
            block_paths["qr_a"], block_paths["qr_b"], config.chunk_rows
        )
        repro = {"performed": False, "exact_equality_passed": None}
        if run_reproducibility_qa:
            repro = {
                "performed": True,
                **reproducibility_qa(parameters, config, active_clock),
            }
        qa = {
            "requested_events": config.requested_events,
            "produced_events": observed_rows,
            "source_groups": observed_rows,
            "exact_n_passed": bool(
                observed_rows == config.requested_events and exact_ordinals
            ),
            "event_schema_passed": event_file.schema_arrow.equals(
                EVENT_SCHEMA, check_metadata=True
            ),
            "message_schema_passed": message_file.schema_arrow.equals(
                MESSAGE_SCHEMA, check_metadata=True
            ),
            "book_validity_passed": bool(book_valid),
            "state_continuity_passed": bool(continuity),
            "initial_state_identity_passed": bool(initial_state_valid),
            "queue_conditioned_hazard_probability_passed": bool(hazard_valid),
            "event_accounting_passed": bool(accounting),
            "canonical_b10_support_passed": all(
                x["passed"] for x in block_support.values()
            ),
            "qr_a_b_content_identity_passed": identity["exact_passed"],
            "seed_reproducibility_passed": repro["exact_equality_passed"],
        }
        qa["passed"] = bool(
            all(
                qa[key]
                for key in (
                    "exact_n_passed",
                    "event_schema_passed",
                    "message_schema_passed",
                    "book_validity_passed",
                    "state_continuity_passed",
                    "event_accounting_passed",
                    "initial_state_identity_passed",
                    "queue_conditioned_hazard_probability_passed",
                    "canonical_b10_support_passed",
                    "qr_a_b_content_identity_passed",
                )
            )
            and (not run_reproducibility_qa or repro["exact_equality_passed"])
        )
        if not qa["passed"]:
            raise RuntimeError(f"simulation day QA failed: {qa}")
        first_event = event_file.read_row_group(0).slice(0, 1).to_pylist()[0]
        last_group = event_file.read_row_group(event_file.num_row_groups - 1)
        last_event = last_group.slice(last_group.num_rows - 1, 1).to_pylist()[0]
        outputs = {
            "events.parquet": event_path,
            "messages.parquet": message_path,
            **{str(path.relative_to(staging)): path for path in block_paths.values()},
        }
        completion = {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "simulation_day": config.simulation_day,
            "source_date": config.source_date.isoformat(),
            "universe": config.universe,
            "seed": config.seed,
            "redraw_contract": REDRAW_CONTRACT,
            "requested_events": config.requested_events,
            "output_rows": {
                name: pq.ParquetFile(path).metadata.num_rows
                for name, path in outputs.items()
            },
            "output_bytes": {
                name: path.stat().st_size for name, path in outputs.items()
            },
            "key_range": {
                "first": {
                    key: int(first_event[key])
                    for key in (
                        "source_group_id",
                        "qr_a_timestamp_ns",
                        "qr_b_timestamp_ns",
                    )
                },
                "last": {
                    key: int(last_event[key])
                    for key in (
                        "source_group_id",
                        "qr_a_timestamp_ns",
                        "qr_b_timestamp_ns",
                    )
                },
            },
            "clock_lineage": lineage,
            "block_support": block_support,
            "clock_source_identity": checkpoint["clock_binding"]["source_stats"],
            "runtime_identity": runtime_identity,
            "qr_a_b_identity_audit": identity,
            "training_admission": "passed",
            "qa": qa,
            "reproducibility_qa": repro,
            "simulation_accounting": {
                key: sum(
                    int(item["event_accounting"][key]) for item in checkpoint["chunks"]
                )
                for key in (
                    "source_events",
                    "raw_records",
                    "reference_shifts",
                    "kind_limit",
                    "kind_cancel",
                    "kind_market",
                    "kind_source_only",
                )
            },
            "resume": {
                "run_id": checkpoint["run_id"],
                "chunks": len(checkpoint["chunks"]),
                "resumed": was_resumed,
            },
            "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        # Detach recovery atomically before completion.  It remains available
        # for exact restart until the clean staging directory becomes final,
        # while the final durable tree contains one full-L5 event table only.
        detached = staging.with_name(f".{final.name}.recovery-{checkpoint['run_id']}")
        if detached.exists():
            raise RuntimeError(f"detached recovery collision: {detached}")
        os.replace(staging / "recovery", detached)
        atomic_json(staging / "complete.json", completion)
        _progress(
            log_path,
            "complete",
            rows=observed_rows,
            elapsed_seconds=completion["elapsed_seconds"],
        )
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(final.parent)
        shutil.rmtree(detached, ignore_errors=True)
        return completion
    except BaseException as exc:
        if staging.exists():
            quarantined = _quarantine_uncommitted_chunks(staging, checkpoint)
            _progress(
                log_path,
                "interrupted"
                if isinstance(exc, ForcedCheckpointInterrupt)
                else "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                checkpoint_cursor=simulator.next_ordinal,
                quarantined_files=quarantined,
            )
        raise


def _smoke_parameters() -> QRParameters:
    """Small fixed fixture with enough cancellation mass to exercise redraws."""

    rows = []
    calibration_config = CalibrationConfig((date(2026, 7, 1),), queue_bins=1)
    source_only_exposure_ns, source_only_event_count, source_only_rate = (
        1_000_000_000,
        9,
        4.75,
    )
    upper = np.full((2, MIN_LEVELS, 1), np.iinfo(np.uint64).max, dtype=np.uint64)
    rates = np.empty((2, MIN_LEVELS, 1, 3), dtype=np.float64)
    jump_s, jump_p = [], []
    for si, side in enumerate(SIDES):
        side_s, side_p = [], []
        for li in range(MIN_LEVELS):
            level_s, level_p = [], []
            for ki, kind in enumerate(KINDS):
                rate = (4.0, 3.0, 2.0)[ki] / (li + 1)
                rates[si, li, 0, ki] = rate
                js = np.array([1, 2, 4], dtype=np.uint64)
                jp = np.array([0.5, 0.3, 0.2], dtype=np.float64)
                level_s.append(js)
                level_p.append(jp)
                rows.append(
                    {
                        "side": side,
                        "level": li + 1,
                        "queue_bin": 0,
                        "queue_lower_inclusive": 0,
                        "queue_upper_inclusive": np.iinfo(np.uint64).max,
                        "kind": kind,
                        "exposure_ns": 1_000_000_000,
                        "event_count": int(rate),
                        "intensity_per_second": rate,
                        "jump_sizes": js.tolist(),
                        "jump_probabilities": jp.tolist(),
                        "source_only_exposure_ns": source_only_exposure_ns,
                        "source_only_event_count": source_only_event_count,
                        "source_only_intensity_per_second": source_only_rate,
                        "calibration_queue_bins": calibration_config.queue_bins,
                        "rate_prior_events": calibration_config.rate_prior_events,
                        "rate_prior_seconds": calibration_config.rate_prior_seconds,
                        "development_dates": list(calibration_config.development_dates),
                    }
                )
            side_s.append(level_s)
            side_p.append(level_p)
        jump_s.append(side_s)
        jump_p.append(side_p)
    return QRParameters(
        upper,
        rates,
        tuple(tuple(tuple(x for x in level) for level in side) for side in jump_s),
        tuple(tuple(tuple(x for x in level) for level in side) for side in jump_p),
        source_only_exposure_ns,
        source_only_event_count,
        source_only_rate,
        calibration_config,
        pa.Table.from_pylist(rows, schema=PARAMETER_SCHEMA),
    )


def _smoke_initial_state() -> dict[str, Any]:
    tick = 10_000_000
    bid = tuple(99_980_000_000 - level * tick for level in range(5))
    ask = tuple(100_010_000_000 + level * tick for level in range(5))
    return {
        "initial_reference_price_nano": (bid[0] + ask[0]) // 2,
        "initial_midprice_x2_nano": bid[0] + ask[0],
        "tick_size_nano": tick,
        "initial_bid_price_l5": bid,
        "initial_ask_price_l5": ask,
        "initial_bid_depth_l5": (2, 3, 4, 5, 6),
        "initial_ask_depth_l5": (7, 6, 5, 4, 3),
    }


# Explicit scientific names used by phase-local orchestration.
fit_development_qr_parameters = calibrate_qr
generate_qr_event_batches = event_batches
publish_qr_day = publish_simulation_day


def _physical_smoke_clock(clock: RealClockArtifact, root: Path) -> RealClockArtifact:
    if clock.table is None:
        return clock
    root.mkdir(parents=True, exist_ok=False)
    artifact, manifest_path = root / "clock.parquet", root / "complete.json"
    atomic_parquet(artifact, clock.table, REAL_CLOCK_SCHEMA)
    manifest = {**dict(clock.manifest), "artifact_bytes": artifact.stat().st_size}
    atomic_json(manifest_path, manifest)
    return load_real_clock_artifact(artifact, manifest_path)


def smoke_test(output_dir: Path | str | None = None) -> dict[str, Any]:
    """K=5 exact-N, reproducibility, book and accounting smoke (no heavy work)."""

    parameters = _smoke_parameters()
    config_a = SimulationConfig(
        simulation_day="sim_day_smoke",
        source_date=date(2026, 7, 2),
        requested_events=257,
        seed=20260730,
        start_timestamp_ns=0,
        **_smoke_initial_state(),
        split="development",
        chunk_rows=31,
    )
    durations = (np.arange(config_a.requested_events, dtype=np.uint64) % 13) + 1
    clock_table = pa.Table.from_pylist(
        [
            {
                "dataset": config_a.dataset,
                "symbol": config_a.symbol,
                "regime": config_a.regime,
                "source_date": config_a.source_date,
                "split": config_a.split,
                "event_ordinal": i,
                "duration_ns": int(duration),
            }
            for i, duration in enumerate(durations)
        ],
        schema=REAL_CLOCK_SCHEMA,
    )
    clock = real_clock_artifact(
        clock_table,
        artifact_path="fixture/real_clock.parquet",
        manifest_path="fixture/real_clock.complete.json",
        manifest={
            "status": "complete",
            "schema_contract": "qr_real_clock_input_v1",
            "dataset": config_a.dataset,
            "symbol": config_a.symbol,
            "regime": config_a.regime,
            "source_date": config_a.source_date.isoformat(),
            "split": config_a.split,
            "rows": config_a.requested_events,
            "start_timestamp_ns": config_a.start_timestamp_ns,
        },
    )
    first = simulate_table(parameters, config_a, clock)
    second = simulate_table(parameters, config_a, clock)
    qa = qa_event_table(first, config_a.requested_events)
    qa["queue_conditioned_hazard_probability_passed"] = qa_hazard_probabilities(
        parameters, first
    )["passed"]
    qa["seed_reproducibility_passed"] = first.equals(second, check_metadata=True)
    blocks = {
        spec: pa.Table.from_batches(
            list(block_batches(first.to_batches(31), config_a, spec)),
            schema=CANONICAL_BLOCK_SCHEMA,
        )
        for spec in ("qr_a", "qr_b")
    }
    block_qa = {
        spec: validate_block_support(table, config_a) for spec, table in blocks.items()
    }
    nonclock = [
        name
        for name in CANONICAL_BLOCK_SCHEMA.names
        if name not in {"block_start_ns", "block_end_ns", "elapsed_ns"}
    ]
    identity = (
        blocks["qr_a"]
        .select(nonclock)
        .equals(blocks["qr_b"].select(nonclock), check_metadata=True)
    )
    qa["canonical_b10_support_passed"] = all(
        item["passed"] for item in block_qa.values()
    )
    qa["qr_a_b_content_identity_passed"] = bool(identity)
    qa["passed"] = bool(
        qa["passed"]
        and qa["seed_reproducibility_passed"]
        and qa["queue_conditioned_hazard_probability_passed"]
        and qa["canonical_b10_support_passed"]
        and identity
    )
    published = None
    if output_dir is not None:
        output = Path(output_dir)
        physical_clock = _physical_smoke_clock(
            clock,
            output.parent / f".{output.name}.clock-fixture-{uuid.uuid4().hex[:8]}",
        )
        published = publish_simulation_day(
            output,
            parameters,
            config_a,
            physical_clock,
            run_reproducibility_qa=True,
        )
    return {
        "status": "complete" if qa["passed"] else "failed",
        "qa": qa,
        "block_support": block_qa,
        "publication": published,
    }


def resume_smoke_test(output_root: Path | str) -> dict[str, Any]:
    """Force interruption, resume, and prove exact content against a clean run."""

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    parameters = _smoke_parameters()
    config = SimulationConfig(
        simulation_day="sim_day_resume_smoke",
        source_date=date(2026, 7, 2),
        requested_events=257,
        seed=20260730,
        start_timestamp_ns=0,
        **_smoke_initial_state(),
        split="development",
        chunk_rows=31,
    )
    clock_table = pa.Table.from_pylist(
        [
            {
                "dataset": config.dataset,
                "symbol": config.symbol,
                "regime": config.regime,
                "source_date": config.source_date,
                "split": config.split,
                "event_ordinal": i,
                "duration_ns": i % 13 + 1,
            }
            for i in range(config.requested_events)
        ],
        schema=REAL_CLOCK_SCHEMA,
    )
    clock = real_clock_artifact(
        clock_table,
        artifact_path="fixture/real_clock.parquet",
        manifest_path="fixture/real_clock.complete.json",
        manifest={
            "status": "complete",
            "schema_contract": "qr_real_clock_input_v1",
            "dataset": config.dataset,
            "symbol": config.symbol,
            "regime": config.regime,
            "source_date": config.source_date.isoformat(),
            "split": config.split,
            "rows": config.requested_events,
            "start_timestamp_ns": config.start_timestamp_ns,
        },
    )
    clock = _physical_smoke_clock(clock, root / "clock_input")
    baseline, resumed = root / "baseline", root / "resumed"
    publish_simulation_day(baseline, parameters, config, clock)
    interrupted = False
    try:
        publish_simulation_day(
            resumed, parameters, config, clock, force_interrupt_after_chunks=2
        )
    except ForcedCheckpointInterrupt:
        interrupted = True
    if not interrupted or resumed.exists():
        raise AssertionError(
            "forced interruption did not preserve an unpublished staging run"
        )
    checkpoint_path = root / ".resumed.staging" / "recovery" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_cursor = int(checkpoint["simulator"]["next_ordinal"])
    completion = publish_simulation_day(resumed, parameters, config, clock)
    compared: dict[str, bool] = {}
    relative_paths = [
        "events.parquet",
        "messages.parquet",
        f"blocks/spec=qr_a/source_date={config.source_date}/part-00000.parquet",
        f"blocks/spec=qr_b/source_date={config.source_date}/part-00000.parquet",
    ]
    for relative in relative_paths:
        left, right = (
            pq.ParquetFile(baseline / relative).read(),
            pq.ParquetFile(resumed / relative).read(),
        )
        compared[relative] = left.equals(right, check_metadata=True)
    progress_events = [
        json.loads(line)["event"]
        for line in (resumed / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    passed = bool(
        interrupted
        and checkpoint_cursor == 62
        and all(compared.values())
        and completion["resume"]["resumed"] is True
        and not (resumed / "recovery").exists()
        and {"interrupted", "resume", "complete"}.issubset(progress_events)
        and len(list((resumed / "blocks").rglob("events.parquet"))) == 0
    )
    if not passed:
        raise AssertionError("restart exact-content smoke failed")
    return {
        "status": "complete",
        "forced_interruption": True,
        "checkpoint_cursor": checkpoint_cursor,
        "exact_tables": compared,
        "transient_chunks_removed": True,
        "resumed": True,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the bounded K=5 contract smoke")
    smoke.add_argument("--output-dir", type=Path)
    resume_smoke = subparsers.add_parser(
        "resume-smoke", help="force checkpoint interruption and exact resume"
    )
    resume_smoke.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = smoke_test(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["status"] == "complete" else 1
    if args.command == "resume-smoke":
        result = resume_smoke_test(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["status"] == "complete" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
