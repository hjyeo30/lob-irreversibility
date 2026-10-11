"""Phase 14 simulator scientific-unit and aggregation contracts.

This module deliberately does not generate QR events and does not launch model
training. It owns the small, deterministic layer between completed block inputs
and the existing model runner: unit enumeration, fraction selectors, atomic
unit publication, and final aggregate validation.

The physical and logical universes are different.  QR-A and QR-B have different
37-channel inputs because their clocks differ, but their 34-channel inputs must
be byte-for-byte identical after G0 is removed.  Consequently they share one
physical training unit while retaining two explicitly aliased report rows.
"""

from __future__ import annotations

import fcntl
import contextlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from itertools import zip_longest
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


CONTRACT_VERSION = "phase14_simulator_training_v1"
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
SPLIT_DATES = {
    "development": DATES[:9],
    "calibration": DATES[9:12],
    "test": DATES[12:],
}
TEST_DATES = SPLIT_DATES["test"]
FRACTIONS = (10, 25, 50, 100)
FEATURE_COUNTS = (34, 37)
MODELS = ("pooled_mlp", "ordered_tcn")
SEEDS = (20260730, 20260731, 20260732)
SOURCES = ("real", "qr")
DURATION_SPECS = ("observed", "qr_a", "qr_b", "clock_free")
EXPECTED_PHYSICAL_TRAINING = 120
EXPECTED_PHYSICAL_INFERENCE = 360
EXPECTED_LOGICAL_REPORT = 432
EXPECTED_NEW_TRAINING_MAX = 108
EXPECTED_PATHS = {
    "development": 1_751_687,
    "calibration": 671_859,
    "test": 582_767,
}
REQUIRED_METRICS = (
    "nats_path_uncapped",
    "nats_event_uncapped",
    "nats_path_bounded",
    "nats_event_bounded",
)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _jsonable(value: Any) -> Any:
    """Return a stable JSON value, including common scalar/array containers."""

    if hasattr(value, "as_py"):
        return _jsonable(value.as_py())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot enter a content key")
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Durably replace one JSON file without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(value), stream, ensure_ascii=False, indent=2, sort_keys=True
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, order=True)
class TrainingUnit:
    source: str
    feature_count: int
    duration_spec: str
    fraction_percent: int
    model: str
    seed: int
    universe: int = 0
    symbol: str = "NVDA"
    regime: str = "daytime"
    b: int = 10
    k: int = 6

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unsupported source: {self.source}")
        if self.feature_count not in FEATURE_COUNTS:
            raise ValueError(f"unsupported feature count: {self.feature_count}")
        if self.fraction_percent not in FRACTIONS:
            raise ValueError(f"unsupported fraction: {self.fraction_percent}")
        if self.model not in MODELS or self.seed not in SEEDS:
            raise ValueError("model or seed outside the approved Phase 14 universe")
        if (self.universe, self.symbol, self.regime, self.b, self.k) != (
            0,
            "NVDA",
            "daytime",
            10,
            6,
        ):
            raise ValueError("unit expands the fixed Phase 14 v1 scope")
        allowed = (
            {"observed"}
            if self.source == "real"
            else ({"qr_a", "qr_b"} if self.feature_count == 37 else {"clock_free"})
        )
        if self.duration_spec not in allowed:
            raise ValueError(
                f"invalid duration spec for {self.source}/{self.feature_count}: "
                f"{self.duration_spec}"
            )

    @property
    def physical_key(self) -> str:
        return (
            f"phase14/source={self.source}/universe={self.universe}/"
            f"spec={self.duration_spec}/features={self.feature_count}/"
            f"fraction={self.fraction_percent:03d}/model={self.model}/seed={self.seed}"
        )

    @property
    def storage_relative_path(self) -> Path:
        return Path(self.physical_key.removeprefix("phase14/"))

    @property
    def clock_independent_content_key(self) -> str:
        """Readable input identity; QR34 intentionally contains no clock spec."""

        spec = (
            "g0_removed"
            if self.source == "qr" and self.feature_count == 34
            else self.duration_spec
        )
        return (
            f"phase14-input-v1/source={self.source}/universe={self.universe}/"
            f"symbol={self.symbol}/regime={self.regime}/B={self.b}/K={self.k}/"
            f"features={self.feature_count}/fraction={self.fraction_percent:03d}/"
            f"clock={spec}/stride=3/offset=date.toordinal()%3"
        )

    def scientific_config(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "contract": CONTRACT_VERSION,
                "split": {key: list(dates) for key, dates in SPLIT_DATES.items()},
                "path_stride": 3,
                "path_offset_rule": "date.toordinal()%3",
                "fraction_rule": "per-date nested prefix floor(n*fraction/100)",
                "feature_projection": (
                    "continuous_path_37_v1"
                    if self.feature_count == 37
                    else "continuous_path_37_v1_without_g0"
                ),
                "content_key": self.clock_independent_content_key,
                "source_independent_fit": True,
                "calibration_source": self.source,
                "test_dates": list(TEST_DATES),
            }
        )
        return value


def physical_training_units() -> tuple[TrainingUnit, ...]:
    units: list[TrainingUnit] = []
    for feature_count in FEATURE_COUNTS:
        for fraction in FRACTIONS:
            for model in MODELS:
                for seed in SEEDS:
                    units.append(
                        TrainingUnit(
                            "real", feature_count, "observed", fraction, model, seed
                        )
                    )
    for feature_count, specs in ((37, ("qr_a", "qr_b")), (34, ("clock_free",))):
        for spec in specs:
            for fraction in FRACTIONS:
                for model in MODELS:
                    for seed in SEEDS:
                        units.append(
                            TrainingUnit(
                                "qr", feature_count, spec, fraction, model, seed
                            )
                        )
    result = tuple(sorted(units, key=lambda unit: unit.physical_key))
    keys = [unit.physical_key for unit in result]
    if len(result) != EXPECTED_PHYSICAL_TRAINING or len(set(keys)) != len(keys):
        raise AssertionError(
            "Phase 14 physical unit universe is not exactly 120 unique cells"
        )
    return result


def physical_inference_cells(
    units: Sequence[TrainingUnit] | None = None,
) -> tuple[dict[str, Any], ...]:
    selected = physical_training_units() if units is None else tuple(units)
    rows = tuple(
        {
            "physical_unit_key": unit.physical_key,
            "source_date": source_date,
            "source": unit.source,
            "duration_spec": unit.duration_spec,
            "universe": unit.universe,
            "symbol": unit.symbol,
            "regime": unit.regime,
            "b": unit.b,
            "k": unit.k,
            "feature_count": unit.feature_count,
            "fraction_percent": unit.fraction_percent,
            "model": unit.model,
            "seed": unit.seed,
        }
        for unit in selected
        for source_date in TEST_DATES
    )
    if units is None and len(rows) != EXPECTED_PHYSICAL_INFERENCE:
        raise AssertionError("Phase 14 physical inference universe is not 360 cells")
    return rows


def logical_report_cells(
    units: Sequence[TrainingUnit] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for physical in physical_inference_cells(units):
        if physical["source"] == "qr" and physical["feature_count"] == 34:
            for spec in ("qr_a", "qr_b"):
                rows.append(
                    {
                        **physical,
                        "duration_spec": spec,
                        "alias_of": physical["physical_unit_key"],
                        "identical_after_g0_drop": True,
                    }
                )
        else:
            rows.append(
                {
                    **physical,
                    "alias_of": None,
                    "identical_after_g0_drop": False,
                }
            )
    result = tuple(rows)
    if units is None and len(result) != EXPECTED_LOGICAL_REPORT:
        raise AssertionError("Phase 14 logical report universe is not 432 cells")
    return result


def new_training_units(anchor_keys: Iterable[str]) -> tuple[TrainingUnit, ...]:
    """Return units not covered by exact-config anchors; never infer from files."""

    anchors = set(anchor_keys)
    expected_anchor_candidates = {
        unit.physical_key
        for unit in physical_training_units()
        if unit.source == "real" and unit.fraction_percent == 100
    }
    unknown = anchors - expected_anchor_candidates
    if unknown:
        raise ValueError(
            f"anchor keys outside the 12 approved anchor candidates: {sorted(unknown)}"
        )
    result = tuple(
        unit for unit in physical_training_units() if unit.physical_key not in anchors
    )
    if len(anchors) == 12 and len(result) != EXPECTED_NEW_TRAINING_MAX:
        raise AssertionError("exact 12-anchor reuse must leave 108 new training units")
    return result


def validate_anchor_completions(
    completions: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Admit only the 12 real/100% anchors with an exact Phase 14 config.

    Callers adapting an older model-runtime envelope must first translate its
    immutable scientific fields to :meth:`TrainingUnit.scientific_config`.  A
    filename, unit key, or successful status by itself is intentionally
    insufficient evidence for reuse.
    """

    candidates = {
        unit.physical_key: unit
        for unit in physical_training_units()
        if unit.source == "real" and unit.fraction_percent == 100
    }
    admitted: list[str] = []
    for completion in completions:
        key = completion.get("physical_unit_key") or completion.get("unit_id")
        unit = candidates.get(str(key))
        if unit is None:
            raise RuntimeError(f"completion is not an approved Phase 14 anchor: {key}")
        if key in admitted:
            raise RuntimeError(f"duplicate anchor completion: {key}")
        if (
            completion.get("status") != "complete"
            or completion.get("audit", {}).get("passed") is not True
            or completion.get("scientific_config") != unit.scientific_config()
        ):
            raise RuntimeError(f"anchor exact-config audit failed: {key}")
        admitted.append(str(key))
    return tuple(sorted(admitted))


@dataclass(frozen=True)
class PrefixSelector:
    source_date: str
    fraction_percent: int
    available_rows: int
    start: int
    stop: int

    @property
    def selected_rows(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def nested_fraction_selectors(
    paths_by_date: Mapping[str, int],
    *,
    dates: Sequence[str] = SPLIT_DATES["development"],
) -> tuple[PrefixSelector, ...]:
    """Build non-copying, per-date nested prefix selectors.

    Each selector is the half-open interval ``[0, floor(n*f/100))``.  The same
    date-local ordering must be used by every fraction, making nesting explicit
    and preventing calibration/test rows from entering scaler or training data.
    """

    if set(paths_by_date) != set(dates):
        raise ValueError(
            f"development date coverage mismatch observed={sorted(paths_by_date)} "
            f"expected={sorted(dates)}"
        )
    selectors: list[PrefixSelector] = []
    for source_date in dates:
        count = paths_by_date[source_date]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"positive integer path count required for {source_date}")
        prior = -1
        for fraction in FRACTIONS:
            stop = count if fraction == 100 else count * fraction // 100
            if stop < prior or stop > count:
                raise AssertionError("fraction selectors are not nested")
            selectors.append(PrefixSelector(source_date, fraction, count, 0, stop))
            prior = stop
    validate_fraction_selectors(selectors, paths_by_date, dates=dates)
    return tuple(selectors)


def validate_fraction_selectors(
    selectors: Sequence[PrefixSelector | Mapping[str, Any]],
    paths_by_date: Mapping[str, int],
    *,
    dates: Sequence[str] = SPLIT_DATES["development"],
) -> dict[str, Any]:
    normalized = [
        item if isinstance(item, PrefixSelector) else PrefixSelector(**item)
        for item in selectors
    ]
    by_key = {(item.source_date, item.fraction_percent): item for item in normalized}
    expected_keys = {
        (source_date, fraction) for source_date in dates for fraction in FRACTIONS
    }
    if len(normalized) != len(by_key) or set(by_key) != expected_keys:
        raise RuntimeError("fraction selector duplicate/missing cell")
    totals: dict[int, int] = {fraction: 0 for fraction in FRACTIONS}
    for source_date in dates:
        available = int(paths_by_date[source_date])
        previous = 0
        for fraction in FRACTIONS:
            item = by_key[(source_date, fraction)]
            expected_stop = (
                available if fraction == 100 else available * fraction // 100
            )
            if (
                item.available_rows != available
                or item.start != 0
                or item.stop != expected_stop
                or item.stop < previous
            ):
                raise RuntimeError(f"non-canonical fraction selector: {item}")
            totals[fraction] += item.selected_rows
            previous = item.stop
    return {
        "status": "passed",
        "dates": len(dates),
        "selected_rows": totals,
        "nested": True,
    }


@dataclass(frozen=True)
class QR34Batch:
    """One bounded comparison batch of path keys and projected feature rows."""

    path_keys: Sequence[Any]
    feature_rows: Sequence[Any]


def _qr34_rows(
    batches: Iterable[QR34Batch | Mapping[str, Any]],
) -> Iterator[tuple[Any, Any]]:
    for batch in batches:
        if isinstance(batch, QR34Batch):
            keys, features = batch.path_keys, batch.feature_rows
        elif isinstance(batch, Mapping):
            keys, features = batch.get("path_keys"), batch.get("feature_rows")
        else:
            raise TypeError(
                "QR34 batch must be QR34Batch or a path_keys/feature_rows mapping"
            )
        if keys is None or features is None or len(keys) != len(features):
            raise RuntimeError("QR34 batch path-key/feature row count mismatch")
        for key, row in zip(keys, features, strict=True):
            normalized_key = _jsonable(key)
            normalized_row = _jsonable(row)
            try:
                width = len(normalized_row)
            except TypeError as error:
                raise RuntimeError(
                    "each QR34 feature row must be a sequence"
                ) from error
            trailing_width = (
                len(normalized_row[-1])
                if width > 0 and isinstance(normalized_row[-1], list)
                else width
            )
            if trailing_width != 34:
                raise RuntimeError(
                    f"QR34 row trailing width mismatch: {trailing_width}"
                )
            yield normalized_key, normalized_row


def validate_qr34_alias_batches(
    qr_a_batches: Iterable[QR34Batch | Mapping[str, Any]],
    qr_b_batches: Iterable[QR34Batch | Mapping[str, Any]],
    *,
    source_date: str,
    split: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_every_rows: int = 65_536,
) -> dict[str, Any]:
    """Compare clock-free keys and 34-channel arrays directly in bounded memory."""

    if split not in SPLIT_DATES or source_date not in SPLIT_DATES[split]:
        raise ValueError("source date does not belong to the declared split")
    if progress_every_rows <= 0:
        raise ValueError("progress_every_rows must be positive")
    missing = object()
    rows = 0
    last_progress_rows = 0
    first_key: Any = None
    last_key: Any = None
    for left, right in zip_longest(
        _qr34_rows(qr_a_batches), _qr34_rows(qr_b_batches), fillvalue=missing
    ):
        if left is missing or right is missing:
            raise RuntimeError("QR34 alias row count mismatch")
        left_key, left_features = left
        right_key, right_features = right
        if left_key != right_key:
            raise RuntimeError(
                f"QR34 path key mismatch at row={rows}: {left_key!r} != {right_key!r}"
            )
        if left_features != right_features:
            raise RuntimeError(f"QR34 feature mismatch at row={rows} key={left_key!r}")
        if rows == 0:
            first_key = left_key
        last_key = left_key
        rows += 1
        if (
            progress_callback is not None
            and rows - last_progress_rows >= progress_every_rows
        ):
            progress_callback({"source_date": source_date, "rows_processed_date": rows})
            last_progress_rows = rows
    if rows <= 0:
        raise RuntimeError("QR34 alias comparison has no rows")
    if progress_callback is not None and rows != last_progress_rows:
        progress_callback({"source_date": source_date, "rows_processed_date": rows})
    return {
        "status": "passed",
        "source_date": source_date,
        "split": split,
        "rows": rows,
        "first_path_key": first_key,
        "last_path_key": last_key,
        "path_keys_exact": True,
        "feature_rows_exact": True,
        "identical_after_g0_drop": True,
    }


def validate_qr34_alias_arrays(
    *,
    source_date: str,
    split: str,
    qr_a_path_keys: Sequence[Any],
    qr_a_feature_rows: Sequence[Any],
    qr_b_path_keys: Sequence[Any],
    qr_b_feature_rows: Sequence[Any],
) -> dict[str, Any]:
    """Convenience wrapper for one in-memory batch on each side."""

    return validate_qr34_alias_batches(
        (QR34Batch(qr_a_path_keys, qr_a_feature_rows),),
        (QR34Batch(qr_b_path_keys, qr_b_feature_rows),),
        source_date=source_date,
        split=split,
    )


def validate_qr34_alias_set(
    qr_a_by_date: Mapping[str, Iterable[QR34Batch | Mapping[str, Any]]],
    qr_b_by_date: Mapping[str, Iterable[QR34Batch | Mapping[str, Any]]],
    *,
    dates: Sequence[str] = DATES,
    split_dates: Mapping[str, Sequence[str]] = SPLIT_DATES,
    expected_split_rows: Mapping[str, int] | None = EXPECTED_PATHS,
    expected_rows_by_date: Mapping[str, int] | None = None,
    bytes_by_date: Mapping[str, int] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_every_rows: int = 65_536,
) -> dict[str, Any]:
    """Validate complete date coverage and derive one clock-free input key."""

    expected_dates = set(dates)
    if set(qr_a_by_date) != expected_dates or set(qr_b_by_date) != expected_dates:
        raise RuntimeError("QR34 alias date coverage mismatch")
    date_to_split: dict[str, str] = {}
    for split, members in split_dates.items():
        if split not in {"development", "calibration", "test"}:
            raise ValueError(f"unknown split: {split}")
        for source_date in members:
            if source_date in date_to_split:
                raise ValueError(f"date appears in multiple splits: {source_date}")
            date_to_split[source_date] = split
    if set(date_to_split) != expected_dates:
        raise ValueError("split definition does not partition the expected dates")
    if (
        expected_rows_by_date is not None
        and set(expected_rows_by_date) != expected_dates
    ):
        raise ValueError("expected QR34 rows do not cover the audit dates")
    if bytes_by_date is not None and set(bytes_by_date) != expected_dates:
        raise ValueError("QR34 byte inventory does not cover the audit dates")
    audits: list[dict[str, Any]] = []
    split_rows = {split: 0 for split in split_dates}
    completed_rows = 0
    completed_bytes = 0
    total_rows = (
        sum(int(expected_rows_by_date[source_date]) for source_date in dates)
        if expected_rows_by_date is not None
        else None
    )
    total_bytes = (
        sum(int(bytes_by_date[source_date]) for source_date in dates)
        if bytes_by_date is not None
        else None
    )
    for date_index, source_date in enumerate(dates, start=1):
        expected_date_rows = (
            int(expected_rows_by_date[source_date])
            if expected_rows_by_date is not None
            else None
        )
        date_bytes = (
            int(bytes_by_date[source_date]) if bytes_by_date is not None else None
        )

        def report_date_progress(
            value: Mapping[str, Any],
            *,
            _date_index: int = date_index,
            _completed_rows: int = completed_rows,
            _completed_bytes: int = completed_bytes,
            _expected_date_rows: int | None = expected_date_rows,
            _date_bytes: int | None = date_bytes,
        ) -> None:
            if progress_callback is None:
                return
            date_rows = int(value["rows_processed_date"])
            bytes_processed = None
            if _date_bytes is not None and _expected_date_rows is not None:
                bytes_processed = _completed_bytes + min(
                    _date_bytes,
                    int(_date_bytes * date_rows / max(_expected_date_rows, 1)),
                )
            progress_callback(
                {
                    "stage": "qr34_alias_compare",
                    "source_date": source_date,
                    "date_i": _date_index,
                    "date_n": len(dates),
                    "rows_processed": _completed_rows + date_rows,
                    "rows_total": total_rows,
                    "path_pairs_processed": _completed_rows + date_rows,
                    "path_pairs_total": total_rows,
                    "bytes_processed": bytes_processed,
                    "bytes_total": total_bytes,
                }
            )

        audit = validate_qr34_alias_batches(
            qr_a_by_date[source_date],
            qr_b_by_date[source_date],
            source_date=source_date,
            split=date_to_split[source_date],
            progress_callback=report_date_progress,
            progress_every_rows=progress_every_rows,
        )
        if expected_date_rows is not None and int(audit["rows"]) != expected_date_rows:
            raise RuntimeError(
                f"QR34 alias date support mismatch: {source_date} "
                f"observed={audit['rows']} expected={expected_date_rows}"
            )
        completed_rows += int(audit["rows"])
        completed_bytes += date_bytes or 0
        split_rows[date_to_split[source_date]] += int(audit["rows"])
        audits.append(audit)
    if expected_split_rows is not None and split_rows != dict(expected_split_rows):
        raise RuntimeError(
            f"QR34 alias split support mismatch observed={split_rows} "
            f"expected={dict(expected_split_rows)}"
        )
    return {
        "status": "passed",
        "identical_after_g0_drop": True,
        "dates": len(dates),
        "split_rows": split_rows,
        "date_audits": audits,
    }


def _validated_qr_block_paths(
    simulation_root: Path, source_date: str, spec: str
) -> list[Path]:
    day_root = simulation_root / f"source_date={source_date}"
    completion_path = day_root / "complete.json"
    if not completion_path.is_file():
        raise RuntimeError(f"QR day completion is missing: {source_date}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    relative = f"blocks/spec={spec}/source_date={source_date}/part-00000.parquet"
    identity = completion.get("qr_a_b_identity_audit", {})
    support = completion.get("block_support", {})
    if (
        completion.get("status") != "complete"
        or completion.get("source_date") != source_date
        or completion.get("universe") != 0
        or completion.get("training_admission") != "passed"
        or completion.get("qa", {}).get("passed") is not True
        or identity.get("exact_passed") is not True
        or support.get(spec, {}).get("passed") is not True
    ):
        raise RuntimeError(f"QR block lineage/admission failed: {source_date}/{spec}")
    path = day_root / relative
    declared_rows = completion.get("output_rows", {}).get(relative)
    declared_bytes = completion.get("output_bytes", {}).get(relative)
    if (
        not path.is_file()
        or path.stat().st_size != declared_bytes
        or pq.ParquetFile(path).metadata.num_rows != declared_rows
    ):
        raise RuntimeError(f"QR block completion inventory mismatch: {path}")
    return [path]


def _projected_qr34_batches(
    paths: Sequence[Path],
    source_date: str,
    selected_paths: int,
    batch_size: int,
) -> Iterator[QR34Batch]:
    """Stream B10K6 paths with a five-block overlap and project G0 away."""

    from src.scale_runtime import UPSTREAM

    required = list(UPSTREAM.REQUIRED_COLUMNS)
    boundary_count = len(UPSTREAM.BOUNDARY_FIELDS)
    drop = {
        boundary_count + UPSTREAM.INTERVAL_INDEX[name]
        for name in (
            "elapsed_ns",
            "realized_source_event_count",
            "completion_overshoot",
        )
    }
    keep = np.asarray(
        [index for index in range(37) if index not in drop], dtype=np.int64
    )
    if len(keep) != 34:
        raise RuntimeError("QR34 G0 projection is not exactly 34 channels")
    offset = date.fromisoformat(source_date).toordinal() % 3
    next_path_number = 0
    carry: pa.Table | None = None
    prior_last_block: int | None = None

    for path in paths:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(
            batch_size=batch_size, columns=required
        ):
            current = pa.Table.from_batches([record_batch])
            table = current if carry is None else pa.concat_tables([carry, current])
            block_index = np.asarray(table["block_index"], dtype=np.int64)
            if len(block_index) == 0 or np.any(np.diff(block_index) != 1):
                raise RuntimeError(f"QR block stream is not contiguous: {source_date}")
            first_block, last_block = int(block_index[0]), int(block_index[-1])
            if (
                prior_last_block is not None
                and int(np.asarray(current["block_index"])[0]) != prior_last_block + 1
            ):
                raise RuntimeError(
                    f"QR block batches/files are not adjacent: {source_date}"
                )
            prior_last_block = int(np.asarray(current["block_index"])[-1])
            candidate_keys: list[int] = []
            while next_path_number < selected_paths:
                key = offset + 3 * next_path_number
                if key < first_block:
                    raise RuntimeError("QR path cursor fell behind the bounded overlap")
                if key + 5 > last_block:
                    break
                candidate_keys.append(key)
                next_path_number += 1
            if candidate_keys:
                position = np.asarray(candidate_keys, dtype=np.int64) - first_block

                def vector(name: str) -> np.ndarray:
                    return np.asarray(table[name], dtype=np.int64)

                def matrix(names: Sequence[str]) -> np.ndarray:
                    values = np.column_stack(
                        [np.asarray(table[name], dtype=np.float64) for name in names]
                    )
                    if not np.isfinite(values).all():
                        raise RuntimeError(f"non-finite QR block input: {source_date}")
                    return values

                day = UPSTREAM.DayBlocks(
                    source_date=source_date,
                    block_index=block_index,
                    block_start_ns=vector("block_start_ns"),
                    block_end_ns=vector("block_end_ns"),
                    starts=matrix(
                        [f"start_{field}" for field in UPSTREAM.BOUNDARY_FIELDS]
                    ),
                    ends=matrix([f"end_{field}" for field in UPSTREAM.BOUNDARY_FIELDS]),
                    intervals=matrix(UPSTREAM.INTERVAL_FIELDS),
                )
                forward, _, _ = UPSTREAM.path_arrays(day, position, 6)
                yield QR34Batch(
                    [f"{source_date}:B10:K6:{key}" for key in candidate_keys],
                    forward[:, :, keep],
                )
            carry = table.slice(max(0, table.num_rows - 5))
    if next_path_number != selected_paths:
        raise RuntimeError(
            f"QR path support is short for {source_date}: {next_path_number} != {selected_paths}"
        )


def qr34_alias_audit_from_simulation_completions(
    simulation_root: Path,
    path_counts_by_date: Mapping[str, int],
    *,
    batch_size: int = 65_536,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Produce compact 15-day audit from bounded direct path-array equality."""

    if batch_size < 6 or set(path_counts_by_date) != set(DATES):
        raise ValueError("QR34 audit requires batch_size>=6 and exact 15-date support")
    qr_a: dict[str, Iterable[QR34Batch]] = {}
    qr_b: dict[str, Iterable[QR34Batch]] = {}
    bytes_by_date: dict[str, int] = {}
    for source_date in DATES:
        selected = path_counts_by_date[source_date]
        if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
            raise ValueError(f"invalid QR34 selected path support: {source_date}")
        qr_a_paths = _validated_qr_block_paths(simulation_root, source_date, "qr_a")
        qr_b_paths = _validated_qr_block_paths(simulation_root, source_date, "qr_b")
        bytes_by_date[source_date] = sum(
            path.stat().st_size for path in (*qr_a_paths, *qr_b_paths)
        )
        qr_a[source_date] = _projected_qr34_batches(
            qr_a_paths,
            source_date,
            selected,
            batch_size,
        )
        qr_b[source_date] = _projected_qr34_batches(
            qr_b_paths,
            source_date,
            selected,
            batch_size,
        )
    return validate_qr34_alias_set(
        qr_a,
        qr_b,
        expected_rows_by_date=path_counts_by_date,
        bytes_by_date=bytes_by_date,
        progress_callback=progress_callback,
        progress_every_rows=batch_size,
    )


def build_input_envelope(
    unit: TrainingUnit,
    *,
    root: Path,
    inventory: Sequence[Mapping[str, Any]],
    path_counts_by_date: Mapping[str, int],
    qr34_alias_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a unit to exact 15-day input, selectors, and source-specific QA."""

    development = {
        source_date: int(path_counts_by_date[source_date])
        for source_date in SPLIT_DATES["development"]
    }
    envelope = {
        "contract": "phase14_exact_input_envelope_v1",
        "source": unit.source,
        "duration_spec": unit.duration_spec,
        "universe": unit.universe,
        "symbol": unit.symbol,
        "regime": unit.regime,
        "b": unit.b,
        "k": unit.k,
        "feature_count": unit.feature_count,
        "fraction_percent": unit.fraction_percent,
        "dates": list(DATES),
        "split_dates": {key: list(value) for key, value in SPLIT_DATES.items()},
        "root": str(root.resolve()),
        "inventory": sorted(
            [_jsonable(item) for item in inventory], key=lambda item: str(item["path"])
        ),
        "path_counts_by_date": {
            source_date: int(path_counts_by_date[source_date]) for source_date in DATES
        },
        "fraction_selectors": [
            item.as_dict() for item in nested_fraction_selectors(development)
        ],
        "qr34_alias_audit": _jsonable(qr34_alias_audit),
    }
    validate_input_envelope(envelope, unit)
    return envelope


def validate_input_envelope(
    envelope: Mapping[str, Any], unit: TrainingUnit
) -> dict[str, Any]:
    """Re-read an exact unit input envelope; no path-portable reuse is allowed."""

    identity = {
        "contract": "phase14_exact_input_envelope_v1",
        "source": unit.source,
        "duration_spec": unit.duration_spec,
        "universe": unit.universe,
        "symbol": unit.symbol,
        "regime": unit.regime,
        "b": unit.b,
        "k": unit.k,
        "feature_count": unit.feature_count,
        "fraction_percent": unit.fraction_percent,
        "dates": list(DATES),
        "split_dates": {key: list(value) for key, value in SPLIT_DATES.items()},
    }
    expected_fields = set(identity) | {
        "root",
        "inventory",
        "path_counts_by_date",
        "fraction_selectors",
        "qr34_alias_audit",
    }
    if set(envelope) != expected_fields:
        raise RuntimeError("input envelope schema has missing or unexpected fields")
    for name, expected in identity.items():
        if envelope.get(name) != expected:
            raise RuntimeError(
                f"input envelope {name} mismatch: {envelope.get(name)!r} != {expected!r}"
            )
    root_value = envelope.get("root")
    if not isinstance(root_value, str):
        raise RuntimeError("input envelope root is missing")
    root = Path(root_value)
    if not root.is_absolute() or not root.is_dir() or str(root.resolve()) != root_value:
        raise RuntimeError(
            f"input envelope root is not canonical/current: {root_value}"
        )

    counts = envelope.get("path_counts_by_date")
    if not isinstance(counts, Mapping) or set(counts) != set(DATES):
        raise RuntimeError("input path support does not cover exact 15 dates")
    normalized_counts: dict[str, int] = {}
    for source_date in DATES:
        value = counts[source_date]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"invalid path support: {source_date}")
        normalized_counts[source_date] = value
    split_support = {
        split: sum(normalized_counts[source_date] for source_date in dates)
        for split, dates in SPLIT_DATES.items()
    }
    if split_support != EXPECTED_PATHS:
        raise RuntimeError(f"input split support mismatch: {split_support}")
    selectors = envelope.get("fraction_selectors")
    if not isinstance(selectors, list):
        raise RuntimeError("input fraction selectors are missing")
    fraction_audit = validate_fraction_selectors(
        selectors,
        {date: normalized_counts[date] for date in SPLIT_DATES["development"]},
    )
    selected = fraction_audit["selected_rows"].get(unit.fraction_percent)
    if not isinstance(selected, int) or selected <= 0:
        raise RuntimeError("unit fraction has no development support")

    inventory = envelope.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("input inventory is empty")
    if inventory != sorted(inventory, key=lambda item: str(item.get("path"))):
        raise RuntimeError("input inventory is not canonically ordered")
    seen: set[str] = set()
    inventory_dates: set[str] = set()
    total_rows = total_bytes = 0
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "source_date",
            "rows",
            "bytes",
        }:
            raise RuntimeError("input inventory item schema mismatch")
        relative_value = item["path"]
        relative = (
            Path(relative_value) if isinstance(relative_value, str) else Path("..")
        )
        if relative.is_absolute() or ".." in relative.parts or relative_value in seen:
            raise RuntimeError(f"unsafe/duplicate input path: {relative_value!r}")
        source_date, rows, byte_count = item["source_date"], item["rows"], item["bytes"]
        if (
            source_date not in DATES
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise RuntimeError(f"invalid inventory metadata: {relative_value}")
        if f"source_date={source_date}" not in relative.parts:
            raise RuntimeError(f"inventory date partition mismatch: {relative_value}")
        path = root / relative
        if not path.is_file() or path.stat().st_size != byte_count:
            raise RuntimeError(f"input inventory file/size mismatch: {path}")
        if path.suffix != ".parquet" or pq.ParquetFile(path).metadata.num_rows != rows:
            raise RuntimeError(f"input inventory Parquet row mismatch: {path}")
        seen.add(relative_value)
        inventory_dates.add(str(source_date))
        total_rows += rows
        total_bytes += byte_count
    if inventory_dates != set(DATES):
        raise RuntimeError("input inventory lacks exact 15-date coverage")

    alias = envelope.get("qr34_alias_audit")
    if unit.source == "qr" and unit.feature_count == 34:
        date_audits = alias.get("date_audits") if isinstance(alias, Mapping) else None
        if (
            not isinstance(alias, Mapping)
            or alias.get("status") != "passed"
            or alias.get("identical_after_g0_drop") is not True
            or alias.get("dates") != len(DATES)
            or alias.get("split_rows") != EXPECTED_PATHS
            or not isinstance(date_audits, list)
            or [item.get("source_date") for item in date_audits] != list(DATES)
            or any(
                item.get("path_keys_exact") is not True
                or item.get("feature_rows_exact") is not True
                or item.get("identical_after_g0_drop") is not True
                or item.get("rows") != normalized_counts.get(item.get("source_date"))
                or item.get("first_path_key") is None
                or item.get("last_path_key") is None
                for item in date_audits
            )
        ):
            raise RuntimeError("QR34 exact-equality audit is incomplete")
    elif alias is not None:
        raise RuntimeError("QR34 alias audit is prohibited for this unit")
    return {
        "status": "passed",
        "source": unit.source,
        "duration_spec": unit.duration_spec,
        "universe": unit.universe,
        "dates": len(DATES),
        "files": len(inventory),
        "rows": total_rows,
        "bytes": total_bytes,
        "selected_development_paths": selected,
        "split_support": split_support,
        "qr34_alias_exact": unit.source == "qr" and unit.feature_count == 34,
    }


def _finite_metrics(metrics: Mapping[str, Any]) -> None:
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"missing or non-finite metric: {name}")


def _artifact_inventory(staging: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "complete.json":
            artifacts.append(
                {"path": str(path.relative_to(staging)), "bytes": path.stat().st_size}
            )
    return artifacts


def validate_unit_completion(
    unit_root: Path,
    unit: TrainingUnit,
    *,
    expected_input_envelope: Mapping[str, Any] | None = None,
    required_artifacts: Sequence[str] = (),
) -> dict[str, Any]:
    completion_path = unit_root / "complete.json"
    if not completion_path.is_file():
        raise RuntimeError(f"missing unit completion: {unit_root}")
    value = json.loads(completion_path.read_text(encoding="utf-8"))
    published_input = value.get("input")
    if expected_input_envelope is None:
        if not isinstance(published_input, Mapping):
            raise RuntimeError("published exact input envelope is missing")
        expected_input_envelope = published_input
    validate_input_envelope(expected_input_envelope, unit)
    if published_input != _jsonable(expected_input_envelope):
        raise RuntimeError(f"unit exact input envelope mismatch: {unit.physical_key}")
    validate_input_envelope(published_input, unit)
    if (
        value.get("status") != "complete"
        or value.get("contract") != CONTRACT_VERSION
        or value.get("physical_unit_key") != unit.physical_key
        or value.get("content_key") != unit.clock_independent_content_key
        or value.get("scientific_config") != unit.scientific_config()
        or value.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError(f"unit completion envelope mismatch: {unit.physical_key}")
    _finite_metrics(value.get("metrics", {}))
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("unit artifact inventory is missing")
    observed_names: set[str] = set()
    for item in artifacts:
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("unsafe artifact inventory path")
        artifact = unit_root / relative
        if not artifact.is_file() or artifact.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"unit artifact mismatch: {artifact}")
        observed_names.add(relative)
    if not set(required_artifacts).issubset(observed_names):
        raise RuntimeError("required unit artifact is absent")
    return value


def publish_unit_completion(
    staging: Path,
    final: Path,
    unit: TrainingUnit,
    *,
    metrics: Mapping[str, Any],
    audit: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    required_artifacts: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate a run-unique staging tree and atomically expose it as complete."""

    if not staging.is_dir() or staging.resolve() == final.resolve():
        raise ValueError("a distinct existing staging directory is required")
    validate_input_envelope(input_manifest, unit)
    if final.exists():
        return validate_unit_completion(
            final,
            unit,
            expected_input_envelope=input_manifest,
            required_artifacts=required_artifacts,
        )
    if (staging / "complete.json").exists():
        raise RuntimeError("staging completion must be written only by this publisher")
    _finite_metrics(metrics)
    if audit.get("passed") is not True:
        raise RuntimeError("failed scientific audit cannot be published")
    artifacts = _artifact_inventory(staging)
    if not set(required_artifacts).issubset({item["path"] for item in artifacts}):
        raise RuntimeError("required artifact is absent from staging")
    completion = {
        "status": "complete",
        "contract": CONTRACT_VERSION,
        "completed_utc": datetime.now(UTC).isoformat(),
        "physical_unit_key": unit.physical_key,
        "content_key": unit.clock_independent_content_key,
        "scientific_config": unit.scientific_config(),
        "input": _jsonable(input_manifest),
        "metrics": _jsonable(metrics),
        "audit": _jsonable(audit),
        "runtime": _jsonable(runtime),
        "artifacts": artifacts,
    }
    atomic_json(staging / "complete.json", completion)
    validate_unit_completion(
        staging,
        unit,
        expected_input_envelope=input_manifest,
        required_artifacts=required_artifacts,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging, final)
    except OSError:
        # Atomic publication requires a same-filesystem rename; never copy as a
        # fallback because that exposes a partially published scientific unit.
        raise
    return validate_unit_completion(
        final,
        unit,
        expected_input_envelope=input_manifest,
        required_artifacts=required_artifacts,
    )


def validate_aggregate_rows(
    physical_rows: Sequence[Mapping[str, Any]],
    logical_rows: Sequence[Mapping[str, Any]],
    *,
    units: Sequence[TrainingUnit] | None = None,
    expected_test_dates: Sequence[str] = TEST_DATES,
    expected_test_paths_per_unit: int | None = EXPECTED_PATHS["test"],
) -> dict[str, Any]:
    """Validate missing/duplicate/config/finite/support and QR34 aliases."""

    selected_units = physical_training_units() if units is None else tuple(units)
    unit_by_key = {unit.physical_key: unit for unit in selected_units}
    expected_physical = {
        (unit.physical_key, source_date)
        for unit in selected_units
        for source_date in expected_test_dates
    }
    observed_physical: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in physical_rows:
        key = (str(row.get("physical_unit_key")), str(row.get("source_date")))
        if key in observed_physical:
            raise RuntimeError(f"duplicate physical inference row: {key}")
        unit = unit_by_key.get(key[0])
        if unit is None or key[1] not in expected_test_dates:
            raise RuntimeError(f"physical inference row outside universe: {key}")
        exact_fields = {
            "source": unit.source,
            "duration_spec": unit.duration_spec,
            "universe": unit.universe,
            "symbol": unit.symbol,
            "regime": unit.regime,
            "b": unit.b,
            "k": unit.k,
            "feature_count": unit.feature_count,
            "fraction_percent": unit.fraction_percent,
            "model": unit.model,
            "seed": unit.seed,
        }
        for name, expected in exact_fields.items():
            if row.get(name) != expected:
                raise RuntimeError(
                    f"aggregate source/spec config mismatch: {key} {name}="
                    f"{row.get(name)!r} expected={expected!r}"
                )
        if row.get("content_key") != unit.clock_independent_content_key:
            raise RuntimeError(f"aggregate content key mismatch: {key}")
        _finite_metrics(row)
        n_paths = row.get("n_paths")
        n_events = row.get("n_events")
        if (
            isinstance(n_paths, bool)
            or not isinstance(n_paths, int)
            or n_paths <= 0
            or isinstance(n_events, bool)
            or not isinstance(n_events, int)
            or n_events <= 0
        ):
            raise RuntimeError(f"nonpositive aggregate support: {key}")
        observed_physical[key] = row
    if set(observed_physical) != expected_physical:
        missing = expected_physical - set(observed_physical)
        extra = set(observed_physical) - expected_physical
        raise RuntimeError(
            f"physical aggregate coverage mismatch missing={len(missing)} extra={len(extra)}"
        )
    if expected_test_paths_per_unit is not None:
        if expected_test_paths_per_unit <= 0:
            raise ValueError("expected test support must be positive")
        for unit in selected_units:
            observed_paths = sum(
                int(observed_physical[(unit.physical_key, source_date)]["n_paths"])
                for source_date in expected_test_dates
            )
            if observed_paths != expected_test_paths_per_unit:
                raise RuntimeError(
                    f"fixed test support mismatch for {unit.physical_key}: "
                    f"observed={observed_paths} expected={expected_test_paths_per_unit}"
                )

    expected_logical = {
        (
            row["physical_unit_key"],
            row["source_date"],
            row["duration_spec"],
        ): row
        for row in logical_report_cells(selected_units)
        if row["source_date"] in expected_test_dates
    }
    observed_logical: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in logical_rows:
        key = (
            str(row.get("physical_unit_key")),
            str(row.get("source_date")),
            str(row.get("duration_spec")),
        )
        if key in observed_logical:
            raise RuntimeError(f"duplicate logical report row: {key}")
        expected = expected_logical.get(key)
        if expected is None:
            raise RuntimeError(f"logical report row outside universe: {key}")
        physical = observed_physical[(key[0], key[1])]
        for name in (
            "source",
            "duration_spec",
            "universe",
            "symbol",
            "regime",
            "b",
            "k",
            "feature_count",
            "fraction_percent",
            "model",
            "seed",
        ):
            if row.get(name) != expected.get(name):
                raise RuntimeError(
                    f"logical source/spec config mismatch: {key} field={name}"
                )
        for name in (*REQUIRED_METRICS, "n_paths", "n_events", "content_key"):
            if row.get(name) != physical.get(name):
                raise RuntimeError(
                    f"logical alias diverges from physical row: {key} field={name}"
                )
        if row.get("alias_of") != expected["alias_of"]:
            raise RuntimeError(f"logical alias target mismatch: {key}")
        if (
            row.get("identical_after_g0_drop")
            is not expected["identical_after_g0_drop"]
        ):
            raise RuntimeError(f"logical QR34 identity flag mismatch: {key}")
        observed_logical[key] = row
    if set(observed_logical) != set(expected_logical):
        missing = set(expected_logical) - set(observed_logical)
        extra = set(observed_logical) - set(expected_logical)
        raise RuntimeError(
            f"logical aggregate coverage mismatch missing={len(missing)} extra={len(extra)}"
        )
    return {
        "status": "passed",
        "physical_training_units": len(selected_units),
        "physical_inference_cells": len(expected_physical),
        "logical_report_cells": len(expected_logical),
        "finite": True,
        "positive_support": True,
        "qr34_aliases_exact": True,
    }


def materialize_logical_rows(
    physical_rows: Sequence[Mapping[str, Any]],
    *,
    units: Sequence[TrainingUnit] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Expand validated physical metrics into the explicit logical row universe."""

    selected_units = physical_training_units() if units is None else tuple(units)
    physical_by_key = {
        (str(row["physical_unit_key"]), str(row["source_date"])): row
        for row in physical_rows
    }
    if len(physical_by_key) != len(physical_rows):
        raise RuntimeError("duplicate physical rows cannot be materialized")
    result: list[dict[str, Any]] = []
    for descriptor in logical_report_cells(selected_units):
        key = (descriptor["physical_unit_key"], descriptor["source_date"])
        if key not in physical_by_key:
            raise RuntimeError(
                f"physical row missing for logical materialization: {key}"
            )
        result.append({**physical_by_key[key], **descriptor})
    return tuple(result)


def bounded_smoke() -> dict[str, Any]:
    """Exercise contracts with tiny metadata/files; no model or QR computation."""

    units = physical_training_units()
    if len(physical_inference_cells()) != 360 or len(logical_report_cells()) != 432:
        raise AssertionError("unit cardinality smoke failed")
    selectors = nested_fraction_selectors({"d1": 11, "d2": 20}, dates=("d1", "d2"))
    selector_audit = validate_fraction_selectors(
        selectors, {"d1": 11, "d2": 20}, dates=("d1", "d2")
    )
    # Use a canonical project date for the QR34 manifest contract.
    features = [[float(index + column) for column in range(34)] for index in range(3)]
    alias_audit = validate_qr34_alias_arrays(
        source_date=DATES[0],
        split="development",
        qr_a_path_keys=["p0", "p1", "p2"],
        qr_a_feature_rows=features,
        qr_b_path_keys=["p0", "p1", "p2"],
        qr_b_feature_rows=[list(row) for row in features],
    )
    qr34_unit = next(
        unit
        for unit in units
        if unit.source == "qr"
        and unit.feature_count == 34
        and unit.fraction_percent == 10
        and unit.model == "pooled_mlp"
        and unit.seed == SEEDS[0]
    )
    aggregate_physical = tuple(
        {
            **descriptor,
            "content_key": qr34_unit.clock_independent_content_key,
            "n_paths": 2,
            "n_events": 120,
            **{name: 0.01 for name in REQUIRED_METRICS},
        }
        for descriptor in physical_inference_cells((qr34_unit,))
    )
    aggregate_logical = materialize_logical_rows(aggregate_physical, units=(qr34_unit,))
    aggregate_audit = validate_aggregate_rows(
        aggregate_physical,
        aggregate_logical,
        units=(qr34_unit,),
        expected_test_paths_per_unit=6,
    )
    with tempfile.TemporaryDirectory(prefix="phase14-training-smoke-") as directory:
        root = Path(directory)
        input_root = root / "input"
        inventory: list[dict[str, Any]] = []
        path_counts: dict[str, int] = {}
        for split, split_dates in SPLIT_DATES.items():
            quotient, remainder = divmod(EXPECTED_PATHS[split], len(split_dates))
            for index, source_date in enumerate(split_dates):
                path_counts[source_date] = quotient + (1 if index < remainder else 0)
                path = input_root / f"source_date={source_date}" / "part.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({"value": [index]}), path)
                inventory.append(
                    {
                        "path": str(path.relative_to(input_root)),
                        "source_date": source_date,
                        "rows": 1,
                        "bytes": path.stat().st_size,
                    }
                )
        completion_unit = next(unit for unit in units if unit.source == "real")
        input_envelope = build_input_envelope(
            completion_unit,
            root=input_root,
            inventory=inventory,
            path_counts_by_date=path_counts,
        )
