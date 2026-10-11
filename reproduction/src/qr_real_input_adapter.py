"""Restartable local adapter from immutable Databento MBO to Phase 14 QR inputs.

The adapter performs one ordered, bounded-memory DBN replay per symbol-day.  It
publishes the exact Daytime source-event clock for all 15 matched dates and, on
the nine development dates only, the queue-conditioned calibration sufficient
input.  Raw MBO records never leave the approved local data root.

Publication is per day and atomic.  A recovery directory contains only replay
state and typed Parquet parts; it can be resumed after interruption.  Raw input
identity deliberately follows the existing Phase 00 filename/byte inventory
contract and does not introduce a hash or checksum chain.

``enriched-events-day`` is a separate immutable publication mode.  Its exact N,
event ordinal and duration sequence are admitted independently from the existing
``qr_real_input_v2`` clock; the enriched output's own manifest is never the
requested-event authority.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import databento as db
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

try:  # package import from the project root
    from .block_producer import (
        FATAL_BOOK_ERRORS,
        UPSTREAM,
        regime_bounds,
        verify_input_inventory,
    )
    from .qr_simulator_runtime import CALIBRATION_EVENT_SCHEMA, REAL_CLOCK_SCHEMA
    from .simulator_training_runtime import SPLIT_DATES
except ImportError:  # direct module CLI with ``src`` on PYTHONPATH
    from block_producer import (
        FATAL_BOOK_ERRORS,
        UPSTREAM,
        regime_bounds,
        verify_input_inventory,
    )
    from qr_simulator_runtime import CALIBRATION_EVENT_SCHEMA, REAL_CLOCK_SCHEMA
    from simulator_training_runtime import SPLIT_DATES


CONTRACT_VERSION = "qr_real_input_adapter_v2"
ENRICHED_CONTRACT_VERSION = "qr_real_enriched_events_v1"
DATASET = "XNAS.ITCH"
REGIME = "daytime"
SYMBOL = "NVDA"
MIN_LEVELS = 5
TARGET_B = 10
DEFAULT_RAW_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/raw/databento/xnas_itch/mbo/symbol=NVDA"
)
APPROVED_EXTERNAL_DERIVED_ROOT = Path(
    "/data/lob_irreversibility_interpretation_and_execution/derived"
)

L5_PRICE_TYPE = pa.list_(pa.field("element", pa.int64()), list_size=MIN_LEVELS)
L5_DEPTH_TYPE = pa.list_(pa.field("element", pa.uint64()), list_size=MIN_LEVELS)
INITIAL_STATE_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("regime", pa.string(), nullable=False),
        pa.field("source_date", pa.date32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("requested_events", pa.uint64(), nullable=False),
        pa.field("start_timestamp_ns", pa.int64(), nullable=False),
        pa.field("tick_size_nano", pa.int64(), nullable=False),
        pa.field("initial_reference_price_nano", pa.int64(), nullable=False),
        pa.field("initial_midprice_x2_nano", pa.int64(), nullable=False),
        pa.field("bid_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("ask_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("bid_depth_l5", L5_DEPTH_TYPE, nullable=False),
        pa.field("ask_depth_l5", L5_DEPTH_TYPE, nullable=False),
        pa.field("warmup_source_event_index", pa.uint64(), nullable=False),
        pa.field("warmup_raw_ordinal_end", pa.uint64(), nullable=False),
    ],
    metadata={
        b"schema_contract": b"qr_real_day_initial_state_v1",
        b"schema_version": CONTRACT_VERSION.encode(),
        b"price_unit": b"1e-9 currency units",
    },
)

ENRICHED_EVENT_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("regime", pa.string(), nullable=False),
        pa.field("source_date", pa.date32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("event_ordinal", pa.uint64(), nullable=False),
        pa.field("source_group_id", pa.uint64(), nullable=False),
        pa.field("first_raw_ordinal", pa.uint64(), nullable=False),
        pa.field("last_raw_ordinal", pa.uint64(), nullable=False),
        pa.field("message_count", pa.uint32(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("duration_ns", pa.uint64(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("level", pa.uint8(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("size", pa.uint64(), nullable=False),
        pa.field("reference_shift", pa.bool_(), nullable=False),
        pa.field("pre_midprice_x2_nano", pa.int64(), nullable=False),
        pa.field("post_midprice_x2_nano", pa.int64(), nullable=False),
        pa.field("pre_bid_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("pre_ask_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("pre_bid_depth_l5", L5_DEPTH_TYPE, nullable=False),
        pa.field("pre_ask_depth_l5", L5_DEPTH_TYPE, nullable=False),
        pa.field("post_bid_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("post_ask_price_l5", L5_PRICE_TYPE, nullable=False),
        pa.field("post_bid_depth_l5", L5_DEPTH_TYPE, nullable=False),
        pa.field("post_ask_depth_l5", L5_DEPTH_TYPE, nullable=False),
    ],
    metadata={
        b"schema_contract": ENRICHED_CONTRACT_VERSION.encode(),
        b"source_group_identity": b"canonical source_event_index",
        b"reference_shift": b"pre_midprice_x2_nano != post_midprice_x2_nano",
        b"price_unit": b"1e-9 currency units",
    },
)

CANONICAL_B10_TOTAL = (381, 748_649_547, 9_019_011)
CANONICAL_B10_DATE_SUPPORT = {
    "2026-07-02": (31, 63_899_647, 757_502),
    "2026-07-06": (17, 32_031_689, 372_463),
    "2026-07-07": (25, 50_978_178, 603_295),
    "2026-07-08": (26, 49_229_913, 596_894),
    "2026-07-09": (20, 40_111_626, 471_018),
    "2026-07-10": (24, 47_789_910, 569_990),
    "2026-07-13": (25, 48_932_554, 586_570),
    "2026-07-14": (27, 51_979_824, 641_796),
    "2026-07-15": (28, 53_853_778, 655_572),
    "2026-07-16": (28, 54_807_577, 677_403),
    "2026-07-17": (33, 64_652_871, 792_290),
    "2026-07-20": (23, 44_930_330, 545_901),
    "2026-07-21": (20, 39_468_165, 477_521),
    "2026-07-22": (22, 43_320_977, 512_217),
    "2026-07-23": (32, 62_662_508, 758_579),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_jsonable,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ProgressLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_jsonable)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.stream.close()


def split_for(source_date: str) -> str:
    matches = [name for name, dates in SPLIT_DATES.items() if source_date in dates]
    if len(matches) != 1:
        raise ValueError(
            f"source date is not in the fixed Phase 14 split: {source_date}"
        )
    return matches[0]


@dataclass(frozen=True)
class AdapterConfig:
    source_date: str
    input_path: Path
    output_root: Path
    symbol: str = SYMBOL
    tick_size_nano: int = 10_000_000
    checkpoint_source_events: int = 500_000
    part_rows: int = 100_000
    progress_seconds: float = 30.0
    skip_existing: bool = True
    no_resume: bool = False
    log_path: Path | None = None
    max_clock_events: int | None = None

    def __post_init__(self) -> None:
        parsed = date.fromisoformat(self.source_date)
        if parsed.isoformat() != self.source_date:
            raise ValueError("source_date must be canonical ISO date")
        if self.symbol != SYMBOL:
            raise ValueError("Phase 14 v1 symbol is fixed at NVDA")
        split_for(self.source_date)
        if self.tick_size_nano <= 0:
            raise ValueError("tick_size_nano must be positive")
        if self.checkpoint_source_events < 1 or self.part_rows < 1:
            raise ValueError("checkpoint_source_events and part_rows must be positive")
        if not 30.0 <= self.progress_seconds <= 60.0:
            raise ValueError("progress_seconds must be in [30, 60]")
        if self.max_clock_events is not None and self.max_clock_events < 1:
            raise ValueError("max_clock_events must be positive")

    @property
    def split(self) -> str:
        return split_for(self.source_date)

    def readable(self) -> dict[str, Any]:
        path = self.input_path.resolve()
        return {
            "contract_version": CONTRACT_VERSION,
            "dataset": DATASET,
            "symbol": self.symbol,
            "regime": REGIME,
            "source_date": self.source_date,
            "split": self.split,
            "input_path": str(path),
            "input_bytes": path.stat().st_size,
            "output_root": str(self.output_root.resolve()),
            "tick_size_nano": self.tick_size_nano,
            "checkpoint_source_events": self.checkpoint_source_events,
            "part_rows": self.part_rows,
            "max_clock_events": self.max_clock_events,
        }


@dataclass(frozen=True)
class EnrichedAdapterConfig:
    source_date: str
    input_path: Path
    output_root: Path
    block_root: Path
    clock_root: Path
    symbol: str = SYMBOL
    tick_size_nano: int = 10_000_000
    checkpoint_source_events: int = 500_000
    part_rows: int = 100_000
    progress_seconds: float = 30.0
    skip_existing: bool = True
    no_resume: bool = False
    log_path: Path | None = None
    max_events: int | None = None

    def __post_init__(self) -> None:
        if date.fromisoformat(self.source_date).isoformat() != self.source_date:
            raise ValueError("source_date must be canonical ISO date")
        if self.source_date not in sum(SPLIT_DATES.values(), ()):
            raise ValueError("source date is outside the fixed Phase 14 split")
        if not self.output_root.is_absolute():
            raise ValueError(
                "enriched output_root must be an explicit absolute external path"
            )
        project = Path(__file__).resolve().parents[3]
        try:
            self.output_root.resolve().relative_to(project)
        except ValueError:
            pass
        else:
            raise ValueError(
                "enriched output_root must be outside the research workspace"
            )
        try:
            self.output_root.resolve().relative_to(APPROVED_EXTERNAL_DERIVED_ROOT)
        except ValueError as exc:
            raise ValueError(
                "enriched output_root must be under the approved external NVMe derived root"
            ) from exc
        if self.output_root.resolve().name == "qr_real_input_v2":
            raise ValueError(
                "enriched mode must never target the immutable qr_real_input_v2 root"
            )
        if self.clock_root.resolve().name != "qr_real_input_v2":
            raise ValueError(
                "enriched mode requires the immutable qr_real_input_v2 clock root"
            )
        if (
            self.tick_size_nano <= 0
            or self.checkpoint_source_events < 1
            or self.part_rows < 1
        ):
            raise ValueError("invalid enriched replay sizing")
        if not 30.0 <= self.progress_seconds <= 60.0:
            raise ValueError("progress_seconds must be in [30, 60]")
        if self.max_events is not None and self.max_events < 1:
            raise ValueError("max_events must be positive")

    @property
    def split(self) -> str:
        return split_for(self.source_date)

    def readable(self) -> dict[str, Any]:
        path = self.input_path.resolve()
        return {
            "contract_version": ENRICHED_CONTRACT_VERSION,
            "dataset": DATASET,
            "symbol": self.symbol,
            "regime": REGIME,
            "source_date": self.source_date,
            "split": self.split,
            "input_path": str(path),
            "input_bytes": path.stat().st_size,
            "output_root": str(self.output_root.resolve()),
            "block_root": str(self.block_root.resolve()),
            "clock_root": str(self.clock_root.resolve()),
            "tick_size_nano": self.tick_size_nano,
            "checkpoint_source_events": self.checkpoint_source_events,
            "part_rows": self.part_rows,
            "max_events": self.max_events,
        }


def _exact_l5(book: Any) -> dict[str, list[int]] | None:
    bid_price, bid_depth = book.top("B")
    ask_price, ask_depth = book.top("A")
    bp, ap = bid_price[:MIN_LEVELS], ask_price[:MIN_LEVELS]
    bd, ad = bid_depth[:MIN_LEVELS], ask_depth[:MIN_LEVELS]
    if (
        any(value is None for value in bp + ap)
        or any(int(value) <= 0 for value in bd + ad)
        or int(bp[0]) >= int(ap[0])
    ):
        return None
    return {
        "bid_price_l5": [int(value) for value in bp],
        "ask_price_l5": [int(value) for value in ap],
        "bid_depth_l5": [int(value) for value in bd],
        "ask_depth_l5": [int(value) for value in ad],
    }


def map_qr_event(
    records: Sequence[Any],
    source: Mapping[str, Any],
    pre_l5: Mapping[str, Sequence[int]] | None,
) -> tuple[int, int, str, int, str]:
    """Map an exact eligible real source family into one QR calibration event.

    All non-eligible source groups remain explicit ``source_only`` events.  The
    fourth calibrated hazard therefore preserves the full real source-event
    clock and generates a state-unchanged source group; exact N is never reduced
    to the subset that happens to map into the three queue-changing families.
    Level is defined against the actual pre-event L5 price ladder.
    """

    if pre_l5 is None:
        return 0, 0, "source_only", 0, "incomplete_pre_l5"
    family = str(source["event_family"])
    candidate: Any | None = None
    kind = ""
    if family == "add" and len(records) == 1 and records[0].action == "A":
        candidate, kind = records[0], "limit"
    elif family == "cancel" and len(records) == 1 and records[0].action == "C":
        candidate, kind = records[0], "cancel"
    elif family == "displayed_execution":
        cancels = [record for record in records if record.action == "C"]
        if len(cancels) == 1 and int(source["execution_remove_qty"]) > 0:
            candidate, kind = cancels[0], "market"
    if candidate is None or candidate.side not in {"B", "A"}:
        return 0, 0, "source_only", 0, "non_qr_family"
    ladder = pre_l5["bid_price_l5" if candidate.side == "B" else "ask_price_l5"]
    try:
        level = list(ladder).index(int(candidate.price)) + 1
    except ValueError:
        return 0, 0, "source_only", 0, "price_outside_pre_l5"
    size = (
        int(source["execution_remove_qty"]) if kind == "market" else int(candidate.size)
    )
    if size <= 0:
        return 0, 0, "source_only", 0, "nonpositive_size"
    side = -1 if candidate.side == "B" else 1
    return side, level, kind, size, "mapped"


def _write_part(
    staging: Path,
    name: str,
    schema: pa.Schema,
    rows: list[dict[str, Any]],
    next_part: int,
) -> tuple[int, dict[str, Any] | None]:
    if not rows:
        return next_part, None
    directory = staging / "_parts" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"part-{next_part:06d}.parquet"
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=len(rows))
    rows.clear()
    return next_part + 1, {
        "path": str(path.relative_to(staging)),
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
    }


def _compact_parts(
    staging: Path,
    name: str,
    schema: pa.Schema,
    output_name: str,
    batch_rows: int,
) -> dict[str, Any]:
    paths = sorted((staging / "_parts" / name).glob("part-*.parquet"))
    if not paths:
        raise RuntimeError(f"no {name} parts to publish")
    output = staging / output_name
    writer = pq.ParquetWriter(output, schema, compression="zstd")
    rows = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(schema, check_metadata=True):
                raise RuntimeError(f"{name} part schema mismatch: {path}")
            for batch in parquet.iter_batches(batch_size=batch_rows):
                writer.write_batch(batch, row_group_size=batch_rows)
                rows += batch.num_rows
    finally:
        writer.close()
    parquet = pq.ParquetFile(output)
    if parquet.metadata.num_rows != rows or not parquet.schema_arrow.equals(
        schema, check_metadata=True
    ):
        raise RuntimeError(f"compacted {name} validation failed")
    return {"path": output.name, "rows": rows, "bytes": output.stat().st_size}


def _write_initial_state(staging: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    path = staging / "initial_state.parquet"
    table = pa.Table.from_pylist([dict(row)], schema=INITIAL_STATE_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1 or not parquet.schema_arrow.equals(
        INITIAL_STATE_SCHEMA, check_metadata=True
    ):
        raise RuntimeError("initial state validation failed")
    return {"path": path.name, "rows": 1, "bytes": path.stat().st_size}


def _validate_event_file(
    path: Path, schema: pa.Schema, *, calibration: bool = False
) -> dict[str, int]:
    """Validate exact 0..N-1 coverage and typed calibration accounting."""

    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(schema, check_metadata=True):
        raise RuntimeError(f"event artifact schema mismatch: {path}")
    cursor = mapped = source_only = 0
    last_timestamp: int | None = None
    for batch in parquet.iter_batches(batch_size=65_536):
        table = pa.Table.from_batches([batch], schema=schema)
        ordinal = table["event_ordinal"].to_numpy(zero_copy_only=False)
        expected = np.arange(cursor, cursor + len(ordinal), dtype=np.uint64)
        if not np.array_equal(ordinal, expected):
            raise RuntimeError(f"noncontiguous event ordinals in {path}")
        if calibration:
            kinds = table["kind"].to_pylist()
            sides = table["side"].to_numpy(zero_copy_only=False)
            levels = table["level"].to_numpy(zero_copy_only=False)
            sizes = table["size"].to_numpy(zero_copy_only=False)
            timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False)
            for kind, side, level, size, timestamp in zip(
                kinds, sides, levels, sizes, timestamps, strict=True
            ):
                if last_timestamp is not None and int(timestamp) < last_timestamp:
                    raise RuntimeError("calibration timestamps are not monotone")
                last_timestamp = int(timestamp)
                if kind == "source_only":
                    if (int(side), int(level), int(size)) != (0, 0, 0):
                        raise RuntimeError("malformed source_only calibration event")
                    source_only += 1
                elif kind in {"limit", "cancel", "market"}:
                    if (
                        int(side) not in {-1, 1}
                        or not 1 <= int(level) <= 5
                        or int(size) <= 0
                    ):
                        raise RuntimeError("malformed mapped calibration event")
                    mapped += 1
                else:
                    raise RuntimeError(f"unknown calibration event family: {kind}")
        cursor += len(ordinal)
    if cursor != parquet.metadata.num_rows:
        raise RuntimeError(f"event footer mismatch: {path}")
    return {"rows": cursor, "mapped": mapped, "source_only": source_only}


def _canonical_b10_admission(block_root: Path) -> dict[str, Any]:
    total_files = total_bytes = total_rows = 0
    dates: dict[str, Any] = {}
    for source_date, expected in CANONICAL_B10_DATE_SUPPORT.items():
        expected_files, expected_bytes, expected_rows = expected
        paths = sorted(
            (block_root / f"source_date={source_date}" / "target_b=10").glob(
                "part-*.parquet"
            )
        )
        if not paths:
            raise RuntimeError(f"canonical B10 date is absent: {source_date}")
        cursor = 0
        first_group: int | None = None
        previous_last: int | None = None
        realized_sum = 0
        date_bytes = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            date_bytes += path.stat().st_size
            for batch in parquet.iter_batches(
                batch_size=65_536,
                columns=[
                    "block_index",
                    "first_source_event_index",
                    "last_source_event_index",
                    "realized_source_event_count",
                    "completion_overshoot",
                ],
            ):
                table = pa.Table.from_batches([batch])
                indices = table["block_index"].to_numpy(zero_copy_only=False)
                if not np.array_equal(
                    indices, np.arange(cursor, cursor + len(indices), dtype=np.uint32)
                ):
                    raise RuntimeError(
                        f"noncontiguous canonical B10 block_index: {source_date}"
                    )
                firsts = table["first_source_event_index"].to_numpy(
                    zero_copy_only=False
                )
                lasts = table["last_source_event_index"].to_numpy(zero_copy_only=False)
                realized = table["realized_source_event_count"].to_numpy(
                    zero_copy_only=False
                )
                overshoot = table["completion_overshoot"].to_numpy(zero_copy_only=False)
                if len(firsts):
                    if first_group is None:
                        first_group = int(firsts[0])
                    if (
                        previous_last is not None
                        and int(firsts[0]) != previous_last + 1
                    ):
                        raise RuntimeError(
                            f"canonical B10 part boundary gap: {source_date}"
                        )
                    if np.any(firsts[1:] != lasts[:-1] + 1):
                        raise RuntimeError(
                            f"canonical B10 internal source-group gap: {source_date}"
                        )
                    if np.any(lasts - firsts + 1 != realized):
                        raise RuntimeError(
                            f"canonical B10 first/last/count mismatch: {source_date}"
                        )
                    if np.any(overshoot != realized - TARGET_B):
                        raise RuntimeError(
                            f"canonical B10 overshoot mismatch: {source_date}"
                        )
                    previous_last = int(lasts[-1])
                    realized_sum += int(realized.astype(np.uint64).sum())
                cursor += len(indices)
        if (len(paths), date_bytes, cursor) != expected:
            raise RuntimeError(
                f"canonical B10 date support mismatch: {source_date} "
                f"{(len(paths), date_bytes, cursor)}!={expected}"
            )
        dates[source_date] = {
            "files": len(paths),
            "bytes": date_bytes,
            "rows": cursor,
            "first_source_group_id": first_group,
            "last_completed_source_group_id": previous_last,
            "completed_source_groups": realized_sum,
        }
        total_files += len(paths)
        total_bytes += date_bytes
        total_rows += cursor
    actual = (total_files, total_bytes, total_rows)
    if actual != CANONICAL_B10_TOTAL:
        raise RuntimeError(
            f"canonical real B10 inventory mismatch: {actual}!={CANONICAL_B10_TOTAL}"
        )
    return {
        "contract": "phase14_real_b10_admission_v1",
        "target_b": TARGET_B,
        "files": total_files,
        "bytes": total_bytes,
        "rows": total_rows,
        "dates": dates,
        "identity_mode": "filename, bytes, rows and ordered source-group keys; no hash",
    }


def _admit_clock_day(clock_root: Path, source_date: str) -> dict[str, Any]:
    root = clock_root / f"source_date={source_date}"
    clock_path, manifest_path = root / "clock.parquet", root / "complete.json"
    if not clock_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"immutable real clock day is incomplete: {source_date}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(clock_path)
    files = {item.get("path"): item for item in manifest.get("files", [])}
    entry = files.get("clock.parquet")
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_contract") != "qr_real_clock_input_v1"
        or manifest.get("source_date") != source_date
        or entry is None
        or entry.get("bytes") != clock_path.stat().st_size
        or entry.get("rows") != parquet.metadata.num_rows
        or manifest.get("rows") != parquet.metadata.num_rows
        or not parquet.schema_arrow.equals(REAL_CLOCK_SCHEMA, check_metadata=True)
    ):
        raise RuntimeError(f"immutable real clock admission failed: {source_date}")
    audit = _validate_event_file(clock_path, REAL_CLOCK_SCHEMA)
    if audit["rows"] != parquet.metadata.num_rows:
        raise RuntimeError("immutable real clock ordinal coverage mismatch")
    return {
        "root": str(clock_root.resolve()),
        "source_date": source_date,
        "clock_path": str(clock_path.resolve()),
        "clock_bytes": clock_path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "start_timestamp_ns": int(manifest["start_timestamp_ns"]),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_bytes": manifest_path.stat().st_size,
        "identity_mode": "immutable qr_real_input_v2 filename, bytes, rows and exact Arrow schema",
    }


def _validate_enriched_event_file(
    path: Path,
    *,
    expected_rows: int,
    expected_first_group: int,
    initial_timestamp_ns: int,
    block_identity: Mapping[str, Any],
    clock_path: Path | None = None,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(ENRICHED_EVENT_SCHEMA, check_metadata=True):
        raise RuntimeError("enriched event schema mismatch")
    cursor = 0
    previous_timestamp = initial_timestamp_ns
    previous_raw_ordinal: int | None = None
    previous_post_l5: dict[str, list[int]] | None = None
    last_group: int | None = None
    mapped = source_only = reference_shifts = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch], schema=ENRICHED_EVENT_SCHEMA)
        n = table.num_rows
        ordinals = table["event_ordinal"].to_numpy(zero_copy_only=False)
        groups = table["source_group_id"].to_numpy(zero_copy_only=False)
        if not np.array_equal(ordinals, np.arange(cursor, cursor + n, dtype=np.uint64)):
            raise RuntimeError("enriched event ordinals are not exact 0..N-1")
        if not np.array_equal(
            groups,
            np.arange(
                expected_first_group + cursor,
                expected_first_group + cursor + n,
                dtype=np.uint64,
            ),
        ):
            raise RuntimeError(
                "enriched source_group_id is not canonical contiguous source_event_index"
            )
        timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False)
        durations = table["duration_ns"].to_numpy(zero_copy_only=False)
        expected_durations = np.diff(
            np.concatenate(([previous_timestamp], timestamps))
        ).astype(np.int64)
        if np.any(expected_durations < 0) or not np.array_equal(
            durations, expected_durations.astype(np.uint64)
        ):
            raise RuntimeError("enriched timestamp/duration linkage mismatch")
        previous_timestamp = int(timestamps[-1])
        first_raw = table["first_raw_ordinal"].to_numpy(zero_copy_only=False)
        last_raw = table["last_raw_ordinal"].to_numpy(zero_copy_only=False)
        messages = table["message_count"].to_numpy(zero_copy_only=False)
        if (
            np.any(first_raw > last_raw)
            or np.any(messages == 0)
            or np.any(messages != last_raw - first_raw + 1)
        ):
            raise RuntimeError("invalid enriched raw range/message_count")
        if (
            previous_raw_ordinal is not None
            and int(first_raw[0]) != previous_raw_ordinal + 1
        ) or np.any(first_raw[1:] != last_raw[:-1] + 1):
            raise RuntimeError(
                "enriched raw ordinals are not exact-contiguous across source groups"
            )
        previous_raw_ordinal = int(last_raw[-1])
        kinds = table["kind"].to_pylist()
        sides = table["side"].to_numpy(zero_copy_only=False)
        levels = table["level"].to_numpy(zero_copy_only=False)
        sizes = table["size"].to_numpy(zero_copy_only=False)
        refs = table["reference_shift"].to_numpy(zero_copy_only=False)
        pre_mid = table["pre_midprice_x2_nano"].to_numpy(zero_copy_only=False)
        post_mid = table["post_midprice_x2_nano"].to_numpy(zero_copy_only=False)
        if not np.array_equal(refs, pre_mid != post_mid):
            raise RuntimeError(
                "reference_shift is not exact pre/post midpoint_x2 change"
            )
        reference_shifts += int(refs.sum())
        for i, kind in enumerate(kinds):
            if kind == "source_only":
                if (int(sides[i]), int(levels[i]), int(sizes[i])) != (0, 0, 0):
                    raise RuntimeError("malformed enriched source_only mapping")
                source_only += 1
            elif kind in {"limit", "cancel", "market"}:
                if (
                    int(sides[i]) not in {-1, 1}
                    or not 1 <= int(levels[i]) <= 5
                    or int(sizes[i]) <= 0
                ):
                    raise RuntimeError("malformed enriched mapped event")
                mapped += 1
            else:
                raise RuntimeError(f"unknown enriched kind: {kind}")
        for prefix, midpoint in (("pre", pre_mid), ("post", post_mid)):
            bids = table[f"{prefix}_bid_price_l5"].to_pylist()
            asks = table[f"{prefix}_ask_price_l5"].to_pylist()
            bid_depths = table[f"{prefix}_bid_depth_l5"].to_pylist()
            ask_depths = table[f"{prefix}_ask_depth_l5"].to_pylist()
            for i, (bid, ask, bd, ad) in enumerate(
                zip(bids, asks, bid_depths, ask_depths, strict=True)
            ):
                if (
                    len(bid) != 5
                    or len(ask) != 5
                    or len(bd) != 5
                    or len(ad) != 5
                    or not all(int(bid[j]) > int(bid[j + 1]) for j in range(4))
                    or not all(int(ask[j]) < int(ask[j + 1]) for j in range(4))
                    or int(bid[0]) >= int(ask[0])
                    or any(int(x) <= 0 for x in bd + ad)
                    or int(midpoint[i]) != int(bid[0]) + int(ask[0])
                ):
                    raise RuntimeError(
                        f"incomplete/crossed enriched {prefix} L5 at row {cursor + i}"
                    )
        pre_columns = {
            key: table[f"pre_{key}"].to_pylist()
            for key in ("bid_price_l5", "ask_price_l5", "bid_depth_l5", "ask_depth_l5")
        }
        post_columns = {
            key: table[f"post_{key}"].to_pylist()
            for key in ("bid_price_l5", "ask_price_l5", "bid_depth_l5", "ask_depth_l5")
        }
        if previous_post_l5 is not None and any(
            previous_post_l5[key] != pre_columns[key][0] for key in pre_columns
        ):
            raise RuntimeError(
                "enriched post→pre L5 chain breaks across batch boundary"
            )
        for i in range(n - 1):
            if any(
                post_columns[key][i] != pre_columns[key][i + 1] for key in pre_columns
            ):
                raise RuntimeError(
                    f"enriched post→pre L5 chain breaks at row {cursor + i}"
                )
        previous_post_l5 = {key: post_columns[key][-1] for key in post_columns}
        cursor += n
        last_group = int(groups[-1])
    if cursor != expected_rows or cursor != parquet.metadata.num_rows:
        raise RuntimeError("enriched exact event count mismatch")
    if clock_path is None:
        raise RuntimeError(
            "independent immutable clock is required for enriched validation"
        )
    clock = pq.ParquetFile(clock_path)
    compared = 0
    for event_batch, clock_batch in zip_longest(
        parquet.iter_batches(
            batch_size=batch_size, columns=["event_ordinal", "duration_ns"]
        ),
        clock.iter_batches(
            batch_size=batch_size, columns=["event_ordinal", "duration_ns"]
        ),
    ):
        if (
            event_batch is None
            or clock_batch is None
            or event_batch.num_rows != clock_batch.num_rows
        ):
            raise RuntimeError("enriched/clock batch or row coverage mismatch")
        event_table, clock_table = (
            pa.Table.from_batches([event_batch]),
            pa.Table.from_batches([clock_batch]),
        )
        if not event_table.equals(clock_table):
            raise RuntimeError(
                "enriched event ordinal/duration is not ordered-exact to immutable clock"
            )
        compared += event_batch.num_rows
    if compared != expected_rows or compared != clock.metadata.num_rows:
        raise RuntimeError("enriched/clock exact N mismatch")
    block_first = int(block_identity["first_source_group_id"])
    block_last = int(block_identity["last_completed_source_group_id"])
    completed = int(block_identity["completed_source_groups"])
    if (
        block_first != expected_first_group
        or block_last != expected_first_group + completed - 1
    ):
        raise RuntimeError("enriched event↔block first/last identity mismatch")
    assert last_group is not None
    tail = last_group - block_last
    if tail < 0 or tail != expected_rows - completed:
        raise RuntimeError(
            "safe-boundary tail/truncation is not measured from current keys"
        )
    return {
        "rows": cursor,
        "first_source_group_id": expected_first_group,
        "last_source_group_id": last_group,
        "last_completed_source_group_id": block_last,
        "safe_boundary_truncation_count": tail,
        "mapped": mapped,
        "source_only": source_only,
        "reference_shifts": reference_shifts,
        "event_block_key_linkage_passed": True,
    }


def load_initial_state_artifact(
    artifact_path: Path | str, manifest_path: Path | str
) -> dict[str, Any]:
    """Load one typed real initial L5 state after strict manifest admission.

    The returned price ladders are the observed ladders, not prices reconstructed
    from a reference value.  ``initial_midprice_x2_nano`` retains half-tick
    precision for odd sums and spreads of any positive integer tick count.
    """

    artifact, manifest_file = Path(artifact_path), Path(manifest_path)
    if not artifact.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("initial-state artifact and manifest must both exist")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("initial-state manifest is not complete")
    entries = {
        item.get("path"): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    entry = entries.get(artifact.name)
    if entry is None or int(entry.get("bytes", -1)) != artifact.stat().st_size:
        raise ValueError("initial-state artifact byte lineage mismatch")
    parquet = pq.ParquetFile(artifact)
    if parquet.metadata.num_rows != 1 or not parquet.schema_arrow.equals(
        INITIAL_STATE_SCHEMA, check_metadata=True
    ):
        raise ValueError("initial-state physical contract mismatch")
    row = parquet.read().to_pylist()[0]
    for key in ("dataset", "symbol", "regime", "source_date", "split"):
        expected = manifest.get(key)
        if key == "source_date":
            expected = date.fromisoformat(str(expected))
        if row[key] != expected:
            raise ValueError(f"initial-state {key} lineage mismatch")
    if int(row["requested_events"]) != int(manifest.get("rows", -1)):
        raise ValueError("initial-state exact N mismatch")
    if int(row["start_timestamp_ns"]) != int(manifest.get("start_timestamp_ns", -1)):
        raise ValueError("initial-state timestamp mismatch")
    bid, ask = row["bid_price_l5"], row["ask_price_l5"]
    bid_depth, ask_depth = row["bid_depth_l5"], row["ask_depth_l5"]
    if (
        len(bid) != 5
        or len(ask) != 5
        or len(bid_depth) != 5
        or len(ask_depth) != 5
        or not all(bid[index] > bid[index + 1] for index in range(4))
        or not all(ask[index] < ask[index + 1] for index in range(4))
        or int(bid[0]) >= int(ask[0])
        or any(int(value) <= 0 for value in bid_depth + ask_depth)
        or int(row["initial_midprice_x2_nano"]) != int(bid[0]) + int(ask[0])
    ):
        raise ValueError("initial-state L5 is incomplete, unordered, or crossed")
    return row


def simulation_initial_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an admitted initial-state row to ``SimulationConfig`` fields."""

    return {
        "requested_events": int(row["requested_events"]),
        "start_timestamp_ns": int(row["start_timestamp_ns"]),
        "initial_reference_price_nano": int(row["initial_reference_price_nano"]),
        "initial_midprice_x2_nano": int(row["initial_midprice_x2_nano"]),
        "tick_size_nano": int(row["tick_size_nano"]),
        "initial_bid_price_l5": tuple(int(x) for x in row["bid_price_l5"]),
        "initial_ask_price_l5": tuple(int(x) for x in row["ask_price_l5"]),
        "initial_bid_depth_l5": tuple(int(x) for x in row["bid_depth_l5"]),
        "initial_ask_depth_l5": tuple(int(x) for x in row["ask_depth_l5"]),
    }


def _verify_complete(config: AdapterConfig, root: Path) -> dict[str, Any]:
    manifest_path = root / "complete.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing complete record: {root}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config.readable()
    if value.get("status") != "complete" or value.get("config") != expected:
        raise RuntimeError(f"completed adapter config mismatch: {root}")
    expected_schemas = {
        "clock.parquet": REAL_CLOCK_SCHEMA,
        "initial_state.parquet": INITIAL_STATE_SCHEMA,
    }
    if config.split == "development":
        expected_schemas["calibration_events.parquet"] = CALIBRATION_EVENT_SCHEMA
    files = {item["path"]: item for item in value.get("files", [])}
    if set(files) != set(expected_schemas):
        raise RuntimeError(f"completed adapter file set mismatch: {root}")
    for name, schema in expected_schemas.items():
        path, item = root / name, files[name]
        parquet = pq.ParquetFile(path)
        if (
            path.stat().st_size != item["bytes"]
            or parquet.metadata.num_rows != item["rows"]
            or not parquet.schema_arrow.equals(schema, check_metadata=True)
        ):
            raise RuntimeError(f"completed adapter artifact mismatch: {path}")
    if files["clock.parquet"]["rows"] != value.get("rows"):
        raise RuntimeError("clock row count and exact N disagree")
    return value


def run_adapter(config: AdapterConfig) -> dict[str, Any]:
    input_path = config.input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    inventory = verify_input_inventory(input_path)
    source_day = date.fromisoformat(config.source_date)
    bounds = regime_bounds(source_day)[REGIME]
    final = config.output_root.resolve() / f"source_date={config.source_date}"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if config.skip_existing:
            return _verify_complete(config, final)
        raise FileExistsError(f"refusing to overwrite published day: {final}")

    lock_path = final.parent / f".{final.name}.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(
            f"adapter day already running: {config.source_date}"
        ) from exc

    staging = final.parent / f".{final.name}.staging"
    recovery = final.parent / f".{final.name}.recovery"
    staging.mkdir(parents=True, exist_ok=True)
    recovery.mkdir(parents=True, exist_ok=True)
    checkpoint_path = recovery / "checkpoint.pkl"
    checkpoint_config = recovery / "checkpoint_config.json"
    readable = config.readable()
    log_path = config.log_path or (
        Path(__file__).resolve().parents[1]
        / "phases"
        / "14_simulator_arrow_benchmark"
        / "artifacts"
        / "logs"
        / f"real_input_{config.source_date}.jsonl"
    )
    logger = ProgressLogger(log_path)
    logger.log(
        "adapter_preflight",
        **readable,
        available_ram_bytes=psutil.virtual_memory().available,
        swap_used_bytes=psutil.swap_memory().used,
        cpu_count=psutil.cpu_count(),
        disk_free_bytes=shutil.disk_usage(staging).free,
        input_identity_mode=inventory["identity_mode"],
    )

    book = UPSTREAM.BASE.Book()
    counters: Counter[str] = Counter()
    last_sequence: dict[tuple[int, int, int], int] = {}
    raw_cursor = 0
    source_index = 0
    event_ordinal = 0
    initialized = False
    previous_timestamp: int | None = None
    initial_state: dict[str, Any] | None = None
    last_complete_l5: dict[str, list[int]] | None = None
    clock_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    next_parts = {"clock": 0, "calibration": 0}
    part_files: list[dict[str, Any]] = []

    if checkpoint_path.exists() and checkpoint_config.exists() and not config.no_resume:
        prior = json.loads(checkpoint_config.read_text(encoding="utf-8"))
        if prior != readable:
            raise RuntimeError("adapter checkpoint config mismatch")
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        book = saved["book"]
        counters = saved["counters"]
        last_sequence = saved["last_sequence"]
        raw_cursor = saved["raw_cursor"]
        source_index = saved["source_index"]
        event_ordinal = saved["event_ordinal"]
        initialized = saved["initialized"]
        previous_timestamp = saved["previous_timestamp"]
        initial_state = saved["initial_state"]
        last_complete_l5 = saved["last_complete_l5"]
        clock_rows = saved["clock_rows"]
        calibration_rows = saved["calibration_rows"]
        next_parts = saved["next_parts"]
        part_files = saved["part_files"]
        logger.log(
            "adapter_resume",
            raw_cursor=raw_cursor,
            source_events=source_index,
            clock_rows=event_ordinal,
        )
    elif any(staging.iterdir()):
        raise RuntimeError(f"staging exists without a compatible checkpoint: {staging}")

    def flush_parts() -> None:
        nonlocal part_files
        next_parts["clock"], item = _write_part(
            staging, "clock", REAL_CLOCK_SCHEMA, clock_rows, next_parts["clock"]
        )
        if item:
            part_files.append(item)
        if config.split == "development":
            next_parts["calibration"], item = _write_part(
                staging,
                "calibration",
                CALIBRATION_EVENT_SCHEMA,
                calibration_rows,
                next_parts["calibration"],
            )
            if item:
                part_files.append(item)

    def save_checkpoint(cursor_after: int) -> None:
        flush_parts()
        saved = {
            "book": book,
            "counters": counters,
            "last_sequence": last_sequence,
            "raw_cursor": cursor_after,
            "source_index": source_index,
            "event_ordinal": event_ordinal,
            "initialized": initialized,
            "previous_timestamp": previous_timestamp,
            "initial_state": initial_state,
            "last_complete_l5": last_complete_l5,
            "clock_rows": clock_rows,
            "calibration_rows": calibration_rows,
            "next_parts": next_parts,
            "part_files": part_files,
        }
        temporary = checkpoint_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
        _atomic_json(checkpoint_config, readable)
        logger.log(
            "adapter_checkpoint",
            raw_cursor=cursor_after,
            source_events=source_index,
            clock_rows=event_ordinal,
            rss_bytes=psutil.Process().memory_info().rss,
            staging_bytes=sum(
                path.stat().st_size for path in staging.rglob("*") if path.is_file()
            ),
            disk_free_bytes=shutil.disk_usage(staging).free,
        )

    current: list[Any] = []
    current_errors: list[str] = []
    current_pre_l5: dict[str, list[int]] | None = None
    stop = bool(
        config.max_clock_events is not None and event_ordinal >= config.max_clock_events
    )
    started = last_log = time.monotonic()
    next_checkpoint = source_index + config.checkpoint_source_events
    last_seen_ordinal = raw_cursor - 1

    def reset_segment(reason: str) -> None:
        nonlocal initialized, previous_timestamp, last_complete_l5
        initialized = False
        previous_timestamp = None
        last_complete_l5 = None
        counters[f"segment_reset:{reason}"] += 1

    def finalize_source() -> None:
        nonlocal current, current_errors, current_pre_l5, source_index
        nonlocal initialized, previous_timestamp, initial_state, last_complete_l5
        nonlocal event_ordinal, stop
        if not current:
            return
        source = UPSTREAM.BASE.classify_source(
            current, source_index, config.symbol, current_errors
        )
        if source["source_invariant_count"]:
            raise RuntimeError(f"source invariant at source={source_index}")
        stream_key = (
            source["publisher_id"],
            source["channel_id"],
            source["instrument_id"],
        )
        previous = last_sequence.get(stream_key)
        if previous is not None and int(source["sequence"]) <= previous:
            raise RuntimeError(
                f"nonmonotone source sequence at {source_index}: "
                f"{previous}->{source['sequence']}"
            )
        last_sequence[stream_key] = int(source["sequence"])
        post_l5 = _exact_l5(book)
        timestamp = int(source["ts_event_end"])

        if timestamp >= bounds[1]:
            stop = True
        elif timestamp >= bounds[0]:
            boundary_cross = source[
                "event_family"
            ] == "standalone_trade_or_cross" and timestamp in set(bounds)
            if source["event_family"] == "clear" or boundary_cross:
                reset_segment(
                    "clear" if source["event_family"] == "clear" else "boundary_cross"
                )
            elif not initialized:
                if bool(source["has_f_last"]) and post_l5 is not None:
                    initialized = True
                    previous_timestamp = timestamp
                    last_complete_l5 = post_l5
                    mid_x2 = post_l5["bid_price_l5"][0] + post_l5["ask_price_l5"][0]
                    if initial_state is None:
                        initial_state = {
                            "dataset": DATASET,
                            "symbol": config.symbol,
                            "regime": REGIME,
                            "source_date": source_day,
                            "split": config.split,
                            "requested_events": 0,
                            "start_timestamp_ns": timestamp,
                            "tick_size_nano": config.tick_size_nano,
                            "initial_reference_price_nano": mid_x2 // 2,
                            "initial_midprice_x2_nano": mid_x2,
                            **post_l5,
                            "warmup_source_event_index": source_index,
                            "warmup_raw_ordinal_end": int(source["raw_ordinal_end"]),
                        }
                    counters["warmup_completion"] += 1
            else:
                assert previous_timestamp is not None and last_complete_l5 is not None
                duration = timestamp - previous_timestamp
                if duration < 0:
                    raise RuntimeError(f"negative duration at source={source_index}")
                clock_rows.append(
                    {
                        "dataset": DATASET,
                        "symbol": config.symbol,
                        "regime": REGIME,
                        "source_date": source_day,
                        "split": config.split,
                        "event_ordinal": event_ordinal,
                        "duration_ns": duration,
                    }
                )
                if config.split == "development":
                    exposure_l5 = current_pre_l5 or last_complete_l5
                    side, level, kind, size, reason = map_qr_event(
                        current, source, current_pre_l5
                    )
                    calibration_rows.append(
                        {
                            "source_date": source_day,
                            "event_ordinal": event_ordinal,
                            "timestamp_ns": timestamp,
                            "duration_ns": duration,
                            "side": side,
                            "level": level,
                            "kind": kind,
                            "size": size,
                            "pre_bid_depth_l5": exposure_l5["bid_depth_l5"],
                            "pre_ask_depth_l5": exposure_l5["ask_depth_l5"],
                        }
                    )
                    counters[f"calibration:{reason}"] += 1
                    if current_pre_l5 is None:
                        counters["stale_complete_l5_exposure"] += 1
                event_ordinal += 1
                counters["included_source:daytime"] += 1
                counters[f"family:{source['event_family']}"] += 1
                previous_timestamp = timestamp
                if post_l5 is not None:
                    last_complete_l5 = post_l5
                if len(clock_rows) >= config.part_rows:
                    flush_parts()
                if (
                    config.max_clock_events is not None
                    and event_ordinal >= config.max_clock_events
                ):
                    stop = True
        counters["source_events"] += 1
        source_index += 1
        current = []
        current_errors = []
        current_pre_l5 = None

    try:
        store = db.DBNStore.from_file(input_path)
        for ordinal, native in enumerate(store):
            if ordinal < raw_cursor:
                continue
            if stop:
                break
            last_seen_ordinal = ordinal
            raw = UPSTREAM.BASE.Raw.from_record(ordinal, native)
            if current and raw.key != current[0].key:
                finalize_source()
                if stop:
                    break
                if source_index >= next_checkpoint:
                    save_checkpoint(ordinal)
                    next_checkpoint = source_index + config.checkpoint_source_events
            if not current:
                current_pre_l5 = _exact_l5(book)
            errors = book.apply(raw)
            fatal = [error for error in errors if error in FATAL_BOOK_ERRORS]
            if fatal:
                raise RuntimeError(f"book invariant at raw={ordinal}: {fatal}")
            current_errors.extend(errors)
            current.append(raw)
            counters["raw_records"] += 1
            now = time.monotonic()
            if now - last_log >= config.progress_seconds:
                elapsed = now - started
                processed = max(0, ordinal + 1 - raw_cursor)
                logger.log(
                    "adapter_progress",
                    raw_ordinal=ordinal,
                    source_events=source_index,
                    clock_rows=event_ordinal,
                    records_per_second=processed / max(elapsed, 1e-9),
                    eta_seconds=None,
                    rss_bytes=psutil.Process().memory_info().rss,
                    swap_used_bytes=psutil.swap_memory().used,
                    disk_free_bytes=shutil.disk_usage(staging).free,
                )
                last_log = now
        if current and not stop:
            finalize_source()
        if event_ordinal == 0 or initial_state is None:
            raise RuntimeError("no eligible Daytime clock rows were produced")
        save_checkpoint(last_seen_ordinal + 1)
        clock_file = _compact_parts(
            staging, "clock", REAL_CLOCK_SCHEMA, "clock.parquet", config.part_rows
        )
        clock_audit = _validate_event_file(staging / "clock.parquet", REAL_CLOCK_SCHEMA)
        files = [clock_file]
        calibration_audit = {"rows": 0, "mapped": 0, "source_only": 0}
        if config.split == "development":
            calibration_file = _compact_parts(
                staging,
                "calibration",
                CALIBRATION_EVENT_SCHEMA,
                "calibration_events.parquet",
                config.part_rows,
            )
            calibration_audit = _validate_event_file(
                staging / "calibration_events.parquet",
                CALIBRATION_EVENT_SCHEMA,
                calibration=True,
            )
            if calibration_file["rows"] != event_ordinal:
                raise RuntimeError("calibration exposure coverage is not exact N")
            if (
                calibration_audit["mapped"] + calibration_audit["source_only"]
                != event_ordinal
            ):
                raise RuntimeError("mapped/source_only accounting is not exact N")
            files.append(calibration_file)
        initial_state["requested_events"] = event_ordinal
        files.append(_write_initial_state(staging, initial_state))
        if clock_file["rows"] != event_ordinal:
            raise RuntimeError("clock footer is not exact N")
        elapsed = time.monotonic() - started
        complete = {
            "status": "complete",
            "schema_contract": "qr_real_clock_input_v1",
            "adapter_contract": CONTRACT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": DATASET,
            "symbol": config.symbol,
            "regime": REGIME,
            "source_date": config.source_date,
            "split": config.split,
            "rows": event_ordinal,
            "start_timestamp_ns": initial_state["start_timestamp_ns"],
            "artifact_bytes": clock_file["bytes"],
            "sample_limited": config.max_clock_events is not None,
            "config": readable,
            "input": {
                "filename": inventory["filename"],
                "bytes": inventory["bytes"],
                "inventory": inventory["inventory"],
                "job_id": inventory["job_id"],
                "identity_mode": inventory["identity_mode"],
                "checksum_policy": "none; immutable inventory filename and bytes",
            },
            "initial_state": initial_state,
            "files": files,
            "counts": {key: int(value) for key, value in sorted(counters.items())},
            "event_accounting": {
                "included_source_groups": event_ordinal,
                "clock_rows": clock_audit["rows"],
                "calibration_rows": (
                    calibration_audit["rows"] if config.split == "development" else None
                ),
                "mapped_queue_events": (
                    calibration_audit["mapped"]
                    if config.split == "development"
                    else None
                ),
                "source_only_events": (
                    calibration_audit["source_only"]
                    if config.split == "development"
                    else None
                ),
                "mapped_fraction": (
                    calibration_audit["mapped"] / event_ordinal
                    if config.split == "development"
                    else None
                ),
                "identity": (
                    "included_source_groups = mapped_queue_events + source_only_events"
                    if config.split == "development"
                    else "included_source_groups = clock_rows"
                ),
                "passed": bool(
                    clock_audit["rows"] == event_ordinal
                    and (
                        config.split != "development"
                        or calibration_audit["mapped"]
                        + calibration_audit["source_only"]
                        == event_ordinal
                    )
                ),
            },
            "validation": {
                "exact_n": clock_file["rows"] == event_ordinal,
                "clock_schema": "qr_real_clock_input_v1",
                "calibration_schema": (
                    "qr_calibration_events_v1"
                    if config.split == "development"
                    else None
                ),
                "initial_state_schema": "qr_real_day_initial_state_v1",
                "ordered_source_groups": True,
                "nonnegative_durations": True,
                "complete_uncrossed_initial_l5": True,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "final_rss_bytes": psutil.Process().memory_info().rss,
                "log_path": str(log_path),
            },
        }
        _atomic_json(staging / "complete.json", complete)
        _verify_complete(config, staging)
        shutil.rmtree(staging / "_parts")
        os.replace(staging, final)
        shutil.rmtree(recovery)
        logger.log(
            "adapter_published",
            unit=f"{config.symbol}/{config.source_date}",
            elapsed_seconds=elapsed,
            output_rows=event_ordinal,
            output_bytes=sum(item["bytes"] for item in files),
        )
        return complete
    except Exception as exc:
        logger.log("adapter_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        logger.close()
        lock.close()
        lock_path.unlink(missing_ok=True)


def _verify_enriched_complete(
    config: EnrichedAdapterConfig, root: Path
) -> dict[str, Any]:
    manifest_path = root / "complete.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing enriched complete record: {root}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("config") != config.readable():
        raise RuntimeError("completed enriched config/current input identity mismatch")
    inventory = verify_input_inventory(config.input_path.resolve())
    if (
        value.get("input", {}).get("bytes") != inventory["bytes"]
        or value.get("input", {}).get("filename") != inventory["filename"]
    ):
        raise RuntimeError("completed enriched raw input lineage mismatch")
    admission = _canonical_b10_admission(config.block_root.resolve())
    if value.get("canonical_b10_admission") != admission:
        raise RuntimeError("completed enriched canonical B10 lineage mismatch")
    clock = _admit_clock_day(config.clock_root.resolve(), config.source_date)
    if value.get("immutable_clock_admission") != clock:
        raise RuntimeError("completed enriched immutable clock lineage mismatch")
    if int(value.get("rows", -1)) != int(clock["rows"]):
        raise RuntimeError(
            "enriched manifest N disagrees with independent immutable clock"
        )
    files = value.get("files", [])
    if len(files) != 1 or files[0].get("path") != "events.parquet":
        raise RuntimeError("completed enriched file set mismatch")
    event_path = root / "events.parquet"
    parquet = pq.ParquetFile(event_path)
    if event_path.stat().st_size != files[0].get(
        "bytes"
    ) or parquet.metadata.num_rows != files[0].get("rows"):
        raise RuntimeError("completed enriched physical inventory mismatch")
    audit = _validate_enriched_event_file(
        event_path,
        expected_rows=int(clock["rows"]),
        expected_first_group=int(value["first_source_group_id"]),
        initial_timestamp_ns=int(clock["start_timestamp_ns"]),
        block_identity=admission["dates"][config.source_date],
        clock_path=Path(clock["clock_path"]),
    )
    if audit != value.get("validation"):
        raise RuntimeError("completed enriched semantic revalidation mismatch")
    return value


def run_enriched_adapter(config: EnrichedAdapterConfig) -> dict[str, Any]:
    """Replay and publish all included source groups with exact pre/post L5 state."""

    input_path = config.input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    inventory = verify_input_inventory(input_path)
    b10 = _canonical_b10_admission(config.block_root.resolve())
    clock = _admit_clock_day(config.clock_root.resolve(), config.source_date)
    block_identity = b10["dates"][config.source_date]
    source_day = date.fromisoformat(config.source_date)
    bounds = regime_bounds(source_day)[REGIME]
    final = config.output_root.resolve() / f"source_date={config.source_date}"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if config.skip_existing:
            return _verify_enriched_complete(config, final)
        raise FileExistsError(f"refusing to overwrite immutable enriched day: {final}")
    lock_path = final.parent / f".{final.name}.enriched.lock"
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(
            f"enriched adapter day already running: {config.source_date}"
        ) from exc
    run_id = os.urandom(8).hex()
    staging = final.parent / f".{final.name}.{run_id}.staging"
    recovery = final.parent / f".{final.name}.enriched.recovery"
    staging.mkdir(parents=True, exist_ok=False)
    recovery.mkdir(parents=True, exist_ok=True)
    checkpoint_path = recovery / "checkpoint.pkl"
    checkpoint_config = recovery / "checkpoint_config.json"
    readable = config.readable()
    log_path = config.log_path or (
        Path(__file__).resolve().parents[1]
        / "phases"
        / "14_simulator_arrow_benchmark"
        / "artifacts"
        / "logs"
        / f"real_enriched_{config.source_date}.jsonl"
    )
    logger = ProgressLogger(log_path)
    logger.log(
        "enriched_preflight",
        **readable,
        available_ram_bytes=psutil.virtual_memory().available,
        swap_used_bytes=psutil.swap_memory().used,
        cpu_count=psutil.cpu_count(),
        disk_free_bytes=shutil.disk_usage(staging).free,
        canonical_b10_files=b10["files"],
        canonical_b10_bytes=b10["bytes"],
        canonical_b10_rows=b10["rows"],
        input_identity_mode=inventory["identity_mode"],
        immutable_clock_rows=clock["rows"],
        immutable_clock_bytes=clock["clock_bytes"],
    )
    book = UPSTREAM.BASE.Book()
    counters: Counter[str] = Counter()
    last_sequence: dict[tuple[int, int, int], int] = {}
    raw_cursor = source_index = event_ordinal = 0
    initialized = False
    previous_timestamp: int | None = None
    initial_timestamp: int | None = None
    first_included_source_group: int | None = None
    last_included_raw_ordinal: int | None = None
    last_published_post_l5: dict[str, list[int]] | None = None
    rows: list[dict[str, Any]] = []
    next_part = 0
    part_files: list[dict[str, Any]] = []
    if checkpoint_path.exists() and checkpoint_config.exists() and not config.no_resume:
        if json.loads(checkpoint_config.read_text(encoding="utf-8")) != readable:
            raise RuntimeError("enriched checkpoint config mismatch")
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        book = saved["book"]
        counters = saved["counters"]
        last_sequence = saved["last_sequence"]
        raw_cursor = saved["raw_cursor"]
        source_index = saved["source_index"]
        event_ordinal = saved["event_ordinal"]
        initialized = saved["initialized"]
        previous_timestamp = saved["previous_timestamp"]
        initial_timestamp = saved["initial_timestamp"]
        first_included_source_group = saved["first_included_source_group"]
        last_included_raw_ordinal = saved["last_included_raw_ordinal"]
        last_published_post_l5 = saved["last_published_post_l5"]
        rows = saved["rows"]
        next_part = saved["next_part"]
        part_files = saved["part_files"]
        # A resumed run always owns a fresh staging directory. Reuse only the
        # durable parts recorded under recovery, never an orphan staging tree.
        recovery_parts = recovery / "parts"
        (staging / "_parts" / "enriched").mkdir(parents=True, exist_ok=True)
        for path in sorted(recovery_parts.glob("part-*.parquet")):
            destination = staging / "_parts" / "enriched" / path.name
            os.link(path, destination)
        logger.log(
            "enriched_resume",
            raw_cursor=raw_cursor,
            source_events=source_index,
            rows=event_ordinal,
        )
    elif checkpoint_path.exists() or checkpoint_config.exists():
        raise RuntimeError("incomplete enriched checkpoint pair")
    elif any(recovery.iterdir()):
        raise RuntimeError("enriched recovery exists without a compatible checkpoint")

    def flush_parts() -> None:
        nonlocal next_part
        next_part, item = _write_part(
            staging, "enriched", ENRICHED_EVENT_SCHEMA, rows, next_part
        )
        if item:
            part_files.append(item)

    def save_checkpoint(cursor_after: int) -> None:
        flush_parts()
        recovery_parts = recovery / "parts"
        recovery_parts.mkdir(parents=True, exist_ok=True)
        for path in sorted((staging / "_parts" / "enriched").glob("part-*.parquet")):
            destination = recovery_parts / path.name
            if not destination.exists():
                os.link(path, destination)
        saved = {
            "book": book,
            "counters": counters,
            "last_sequence": last_sequence,
            "raw_cursor": cursor_after,
            "source_index": source_index,
            "event_ordinal": event_ordinal,
            "initialized": initialized,
            "previous_timestamp": previous_timestamp,
            "initial_timestamp": initial_timestamp,
            "first_included_source_group": first_included_source_group,
            "last_included_raw_ordinal": last_included_raw_ordinal,
            "last_published_post_l5": last_published_post_l5,
            "rows": rows,
            "next_part": next_part,
            "part_files": part_files,
        }
        temporary = checkpoint_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
        _atomic_json(checkpoint_config, readable)
        logger.log(
            "enriched_checkpoint",
            raw_cursor=cursor_after,
            source_events=source_index,
            rows=event_ordinal,
            rss_bytes=psutil.Process().memory_info().rss,
            disk_free_bytes=shutil.disk_usage(staging).free,
        )

    current: list[Any] = []
    current_errors: list[str] = []
    current_pre_l5: dict[str, list[int]] | None = None
    stop = bool(config.max_events is not None and event_ordinal >= config.max_events)
    started = last_log = time.monotonic()
    next_checkpoint = source_index + config.checkpoint_source_events
    last_seen_ordinal = raw_cursor - 1

    def reset_segment(reason: str, source_family: str, timestamp: int) -> None:
        nonlocal initialized, previous_timestamp, initial_timestamp
        explicit = (reason == "clear" and source_family == "clear") or (
            reason == "boundary_cross"
            and source_family == "standalone_trade_or_cross"
            and timestamp in set(bounds)
        )
        if not explicit:
            raise RuntimeError(
                "segment reset requires an explicit clear/boundary record"
            )
        if event_ordinal > 0:
            raise RuntimeError(
                f"enriched included stream crossed a segment boundary: {reason}"
            )
        initialized = False
        previous_timestamp = None
        initial_timestamp = None
        counters[f"segment_reset:{reason}"] += 1

    def finalize_source() -> None:
        nonlocal current, current_errors, current_pre_l5, source_index, initialized
        nonlocal previous_timestamp, initial_timestamp, first_included_source_group
        nonlocal last_included_raw_ordinal, last_published_post_l5
        nonlocal event_ordinal, stop
        if not current:
            return
        source = UPSTREAM.BASE.classify_source(
            current, source_index, config.symbol, current_errors
        )
        if source["source_invariant_count"]:
            raise RuntimeError(f"source invariant at source={source_index}")
        stream_key = (
            source["publisher_id"],
            source["channel_id"],
            source["instrument_id"],
        )
        previous = last_sequence.get(stream_key)
        if previous is not None and int(source["sequence"]) <= previous:
            raise RuntimeError(
                f"nonmonotone source sequence at {source_index}: {previous}->{source['sequence']}"
            )
        last_sequence[stream_key] = int(source["sequence"])
        post_l5 = _exact_l5(book)
        timestamp = int(source["ts_event_end"])
        if timestamp >= bounds[1]:
            stop = True
        elif timestamp >= bounds[0]:
            boundary_cross = source[
                "event_family"
            ] == "standalone_trade_or_cross" and timestamp in set(bounds)
            if source["event_family"] == "clear" or boundary_cross:
                reset_segment(
                    "clear" if source["event_family"] == "clear" else "boundary_cross",
                    str(source["event_family"]),
                    timestamp,
                )
            elif not initialized:
                if bool(source["has_f_last"]) and post_l5 is not None:
                    initialized = True
                    previous_timestamp = timestamp
                    initial_timestamp = timestamp
                    counters["warmup_completion"] += 1
            else:
                assert previous_timestamp is not None and initial_timestamp is not None
                if current_pre_l5 is None or post_l5 is None:
                    raise RuntimeError(
                        f"incomplete pre/post L5 inside included segment at source={source_index}"
                    )
                duration = timestamp - previous_timestamp
                if duration < 0:
                    raise RuntimeError(f"negative duration at source={source_index}")
                side, level, kind, size, reason = map_qr_event(
                    current, source, current_pre_l5
                )
                pre_mid = int(current_pre_l5["bid_price_l5"][0]) + int(
                    current_pre_l5["ask_price_l5"][0]
                )
                post_mid = int(post_l5["bid_price_l5"][0]) + int(
                    post_l5["ask_price_l5"][0]
                )
                if first_included_source_group is None:
                    first_included_source_group = source_index
                if (
                    last_included_raw_ordinal is not None
                    and int(source["raw_ordinal_start"])
                    != last_included_raw_ordinal + 1
                ):
                    raise RuntimeError(
                        f"producer raw ordinal chain break at source={source_index}"
                    )
                if last_published_post_l5 is not None and any(
                    last_published_post_l5[key] != current_pre_l5[key]
                    for key in last_published_post_l5
                ):
                    raise RuntimeError(
                        f"producer post→pre L5 chain break at source={source_index}"
                    )
                rows.append(
                    {
                        "dataset": DATASET,
                        "symbol": config.symbol,
                        "regime": REGIME,
                        "source_date": source_day,
                        "split": config.split,
                        "event_ordinal": event_ordinal,
                        "source_group_id": source_index,
                        "first_raw_ordinal": int(source["raw_ordinal_start"]),
                        "last_raw_ordinal": int(source["raw_ordinal_end"]),
                        "message_count": len(current),
                        "timestamp_ns": timestamp,
                        "duration_ns": duration,
                        "side": side,
                        "level": level,
                        "kind": kind,
                        "size": size,
                        "reference_shift": pre_mid != post_mid,
                        "pre_midprice_x2_nano": pre_mid,
                        "post_midprice_x2_nano": post_mid,
                        **{
                            f"pre_{key}": value for key, value in current_pre_l5.items()
                        },
                        **{f"post_{key}": value for key, value in post_l5.items()},
                    }
                )
                counters[f"mapping:{reason}"] += 1
                counters[f"family:{source['event_family']}"] += 1
                counters["included_source:daytime"] += 1
                last_included_raw_ordinal = int(source["raw_ordinal_end"])
                last_published_post_l5 = {
                    key: list(value) for key, value in post_l5.items()
                }
                event_ordinal += 1
                previous_timestamp = timestamp
                if len(rows) >= config.part_rows:
                    flush_parts()
                if config.max_events is not None and event_ordinal >= config.max_events:
                    stop = True
        counters["source_events"] += 1
        source_index += 1
        current = []
        current_errors = []
        current_pre_l5 = None

    try:
        for ordinal, native in enumerate(db.DBNStore.from_file(input_path)):
            if ordinal < raw_cursor:
                continue
            if stop:
                break
            last_seen_ordinal = ordinal
            raw = UPSTREAM.BASE.Raw.from_record(ordinal, native)
            if current and raw.key != current[0].key:
                finalize_source()
                if stop:
                    break
                if source_index >= next_checkpoint:
                    save_checkpoint(ordinal)
                    next_checkpoint = source_index + config.checkpoint_source_events
            if not current:
                current_pre_l5 = _exact_l5(book)
            errors = book.apply(raw)
            fatal = [error for error in errors if error in FATAL_BOOK_ERRORS]
            if fatal:
                raise RuntimeError(f"book invariant at raw={ordinal}: {fatal}")
            current_errors.extend(errors)
            current.append(raw)
            counters["raw_records"] += 1
            now = time.monotonic()
            if now - last_log >= config.progress_seconds:
                elapsed = now - started
                logger.log(
                    "enriched_progress",
                    raw_ordinal=ordinal,
                    source_events=source_index,
                    rows=event_ordinal,
                    records_per_second=(ordinal + 1 - raw_cursor) / max(elapsed, 1e-9),
                    rss_bytes=psutil.Process().memory_info().rss,
                    swap_used_bytes=psutil.swap_memory().used,
                    disk_free_bytes=shutil.disk_usage(staging).free,
                )
                last_log = now
        if current and not stop:
            finalize_source()
        if (
            event_ordinal == 0
            or first_included_source_group is None
            or initial_timestamp is None
        ):
            raise RuntimeError("no enriched Daytime events were produced")
        save_checkpoint(last_seen_ordinal + 1)
        event_file = _compact_parts(
            staging,
            "enriched",
            ENRICHED_EVENT_SCHEMA,
            "events.parquet",
            config.part_rows,
        )
        audit = _validate_enriched_event_file(
            staging / "events.parquet",
            expected_rows=int(clock["rows"]),
            expected_first_group=first_included_source_group,
            initial_timestamp_ns=int(clock["start_timestamp_ns"]),
            block_identity=block_identity,
            clock_path=Path(clock["clock_path"]),
        )
        elapsed = time.monotonic() - started
        complete = {
            "status": "complete",
            "schema_contract": ENRICHED_CONTRACT_VERSION,
            "adapter_contract": ENRICHED_CONTRACT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": DATASET,
            "symbol": config.symbol,
            "regime": REGIME,
            "source_date": config.source_date,
            "split": config.split,
            "rows": event_ordinal,
            "first_source_group_id": first_included_source_group,
            "initial_timestamp_ns": initial_timestamp,
            "config": readable,
            "input": {
                "filename": inventory["filename"],
                "bytes": inventory["bytes"],
                "inventory": inventory["inventory"],
                "job_id": inventory["job_id"],
                "identity_mode": inventory["identity_mode"],
                "checksum_policy": "none; immutable filename and bytes",
            },
            "canonical_b10_admission": b10,
            "immutable_clock_admission": clock,
            "files": [event_file],
            "counts": {key: int(value) for key, value in sorted(counters.items())},
            "validation": audit,
            "runtime": {
                "elapsed_seconds": elapsed,
                "final_rss_bytes": psutil.Process().memory_info().rss,
                "log_path": str(log_path),
            },
        }
        _atomic_json(staging / "complete.json", complete)
        _verify_enriched_complete(config, staging)
        shutil.rmtree(staging / "_parts")
        os.replace(staging, final)
        shutil.rmtree(recovery)
        logger.log(
            "enriched_published",
            source_date=config.source_date,
            elapsed_seconds=elapsed,
            output_rows=event_ordinal,
            output_bytes=event_file["bytes"],
        )
        return complete
    except Exception as exc:
        logger.log("enriched_failed", error_type=type(exc).__name__, error=str(exc))
        if staging.exists():
            quarantine = final.parent / f".{final.name}.{run_id}.failed"
            os.replace(staging, quarantine)
        raise
    finally:
        logger.close()
        lock.close()
        lock_path.unlink(missing_ok=True)


def run_contract_smoke() -> dict[str, Any]:
    """Pure synthetic QA for exact Arrow contracts and QR mapping semantics."""

    day = date.fromisoformat(SPLIT_DATES["development"][0])
    l5 = {
        "bid_price_l5": [100, 99, 98, 97, 96],
        "ask_price_l5": [101, 102, 103, 104, 105],
        "bid_depth_l5": [10, 20, 30, 40, 50],
        "ask_depth_l5": [11, 21, 31, 41, 51],
    }

    class Record:
        action = "A"
        side = "B"
        price = 99
        size = 7

    mapped = map_qr_event(
        [Record()], {"event_family": "add", "execution_remove_qty": 0}, l5
    )
    if mapped != (-1, 2, "limit", 7, "mapped"):
        raise RuntimeError(f"synthetic QR mapping mismatch: {mapped}")
    source_only = map_qr_event(
        [Record()], {"event_family": "replace", "execution_remove_qty": 0}, l5
    )
    if source_only != (0, 0, "source_only", 0, "non_qr_family"):
        raise RuntimeError(f"synthetic source-only mapping mismatch: {source_only}")
    calibration = pa.Table.from_pylist(
        [
            {
                "source_date": day,
                "event_ordinal": 0,
                "timestamp_ns": 2,
                "duration_ns": 1,
                "side": mapped[0],
                "level": mapped[1],
                "kind": mapped[2],
                "size": mapped[3],
                "pre_bid_depth_l5": l5["bid_depth_l5"],
                "pre_ask_depth_l5": l5["ask_depth_l5"],
            },
            {
                "source_date": day,
                "event_ordinal": 1,
                "timestamp_ns": 3,
                "duration_ns": 1,
                "side": 0,
                "level": 0,
                "kind": "source_only",
                "size": 0,
                "pre_bid_depth_l5": l5["bid_depth_l5"],
                "pre_ask_depth_l5": l5["ask_depth_l5"],
            },
        ],
        schema=CALIBRATION_EVENT_SCHEMA,
    )
    clock = pa.Table.from_pylist(
        [
            {
                "dataset": DATASET,
                "symbol": SYMBOL,
                "regime": REGIME,
                "source_date": day,
                "split": "development",
                "event_ordinal": 0,
                "duration_ns": 1,
            },
            {
                "dataset": DATASET,
                "symbol": SYMBOL,
                "regime": REGIME,
                "source_date": day,
                "split": "development",
                "event_ordinal": 1,
                "duration_ns": 1,
            },
        ],
        schema=REAL_CLOCK_SCHEMA,
    )
    initial = pa.Table.from_pylist(
        [
            {
                "dataset": DATASET,
                "symbol": SYMBOL,
                "regime": REGIME,
                "source_date": day,
                "split": "development",
                "requested_events": 2,
                "start_timestamp_ns": 1,
                "tick_size_nano": 1,
                "initial_reference_price_nano": 100,
                "initial_midprice_x2_nano": 201,
                **l5,
                "warmup_source_event_index": 0,
                "warmup_raw_ordinal_end": 0,
            }
        ],
        schema=INITIAL_STATE_SCHEMA,
    )
    checks = {
        "calibration_schema_exact": calibration.schema.equals(
            CALIBRATION_EVENT_SCHEMA, check_metadata=True
        ),
        "clock_schema_exact": clock.schema.equals(
            REAL_CLOCK_SCHEMA, check_metadata=True
        ),
        "initial_schema_exact": initial.schema.equals(
            INITIAL_STATE_SCHEMA, check_metadata=True
        ),
        "mapping_exact": True,
        "source_only_mapping_exact": True,
    }
    enriched_rows = [
        {
            "dataset": DATASET,
            "symbol": SYMBOL,
            "regime": REGIME,
            "source_date": day,
            "split": "development",
            "event_ordinal": 0,
            "source_group_id": 100,
            "first_raw_ordinal": 10,
            "last_raw_ordinal": 11,
            "message_count": 2,
            "timestamp_ns": 2,
            "duration_ns": 1,
            "side": -1,
            "level": 2,
            "kind": "limit",
            "size": 7,
            "reference_shift": True,
            "pre_midprice_x2_nano": 201,
            "post_midprice_x2_nano": 203,
            **{f"pre_{key}": value for key, value in l5.items()},
            "post_bid_price_l5": [101, 100, 99, 98, 97],
            "post_ask_price_l5": [102, 103, 104, 105, 106],
            "post_bid_depth_l5": l5["bid_depth_l5"],
            "post_ask_depth_l5": l5["ask_depth_l5"],
        },
        {
            "dataset": DATASET,
            "symbol": SYMBOL,
            "regime": REGIME,
            "source_date": day,
            "split": "development",
            "event_ordinal": 1,
            "source_group_id": 101,
            "first_raw_ordinal": 12,
            "last_raw_ordinal": 12,
            "message_count": 1,
            "timestamp_ns": 4,
            "duration_ns": 2,
            "side": 0,
            "level": 0,
            "kind": "source_only",
            "size": 0,
            "reference_shift": False,
            "pre_midprice_x2_nano": 203,
            "post_midprice_x2_nano": 203,
            "pre_bid_price_l5": [101, 100, 99, 98, 97],
            "pre_ask_price_l5": [102, 103, 104, 105, 106],
            "pre_bid_depth_l5": l5["bid_depth_l5"],
            "pre_ask_depth_l5": l5["ask_depth_l5"],
            "post_bid_price_l5": [101, 100, 99, 98, 97],
            "post_ask_price_l5": [102, 103, 104, 105, 106],
            "post_bid_depth_l5": l5["bid_depth_l5"],
            "post_ask_depth_l5": l5["ask_depth_l5"],
        },
    ]
    block_identity = {
        "first_source_group_id": 100,
        "last_completed_source_group_id": 100,
        "completed_source_groups": 1,
    }

    enriched_clock = pa.Table.from_pylist(
        [
            {
                "dataset": DATASET,
                "symbol": SYMBOL,
                "regime": REGIME,
                "source_date": day,
                "split": "development",
                "event_ordinal": 0,
                "duration_ns": 1,
            },
            {
                "dataset": DATASET,
                "symbol": SYMBOL,
                "regime": REGIME,
                "source_date": day,
                "split": "development",
                "event_ordinal": 1,
                "duration_ns": 2,
            },
        ],
        schema=REAL_CLOCK_SCHEMA,
    )

    def must_fail(
        rows: list[dict[str, Any]],
        identity: Mapping[str, Any],
        label: str,
        *,
        expected_rows: int = 2,
        clock_table: pa.Table = enriched_clock,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="qr_enriched_negative_") as tmp:
            path = Path(tmp) / "events.parquet"
            clock_path = Path(tmp) / "clock.parquet"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=ENRICHED_EVENT_SCHEMA), path
            )
            pq.write_table(clock_table, clock_path)
            try:
                _validate_enriched_event_file(
                    path,
                    expected_rows=expected_rows,
                    expected_first_group=100,
                    initial_timestamp_ns=1,
                    block_identity=identity,
                    clock_path=clock_path,
                    batch_size=1,
                )
            except RuntimeError:
                checks[label] = True
            else:
                checks[label] = False

    with tempfile.TemporaryDirectory(prefix="qr_enriched_positive_") as tmp:
        path = Path(tmp) / "events.parquet"
        clock_path = Path(tmp) / "clock.parquet"
        table = pa.Table.from_pylist(enriched_rows, schema=ENRICHED_EVENT_SCHEMA)
        pq.write_table(table, path)
        pq.write_table(enriched_clock, clock_path)
        audit = _validate_enriched_event_file(
            path,
            expected_rows=2,
            expected_first_group=100,
            initial_timestamp_ns=1,
            block_identity=block_identity,
            clock_path=clock_path,
            batch_size=1,
        )
        checks["enriched_schema_and_semantics"] = (
            audit["safe_boundary_truncation_count"] == 1
        )
    bad_reference = [dict(row) for row in enriched_rows]
    bad_reference[0]["reference_shift"] = False
    must_fail(bad_reference, block_identity, "reject_wrong_reference_shift")
    bad_group = [dict(row) for row in enriched_rows]
    bad_group[1]["source_group_id"] = 102
    must_fail(bad_group, block_identity, "reject_source_group_gap")
    bad_raw = [dict(row) for row in enriched_rows]
    bad_raw[1]["first_raw_ordinal"] = 13
    bad_raw[1]["last_raw_ordinal"] = 13
    must_fail(bad_raw, block_identity, "reject_cross_batch_raw_gap")
    bad_chain = [dict(row) for row in enriched_rows]
    bad_chain[1]["pre_bid_depth_l5"] = [11, 20, 30, 40, 50]
    must_fail(bad_chain, block_identity, "reject_cross_batch_state_chain")
    bad_l5 = [dict(row) for row in enriched_rows]
    bad_l5[0]["post_bid_depth_l5"] = [10, 20, 0, 40, 50]
    must_fail(bad_l5, block_identity, "reject_incomplete_l5")
    bad_block = {**block_identity, "last_completed_source_group_id": 101}
    must_fail(enriched_rows, bad_block, "reject_false_safe_boundary_identity")
    must_fail(enriched_rows[:1], block_identity, "reject_early_termination")
    extra = dict(enriched_rows[1])
    extra.update(
        event_ordinal=2,
        source_group_id=102,
        first_raw_ordinal=13,
        last_raw_ordinal=13,
        timestamp_ns=5,
        duration_ns=1,
    )
    must_fail([*enriched_rows, extra], block_identity, "reject_extra_tail")
    must_fail(
        enriched_rows,
        block_identity,
        "reject_manifest_self_declared_n",
        expected_rows=3,
    )
    try:
        EnrichedAdapterConfig(
            source_date=str(day),
            input_path=Path("/nonexistent/raw.dbn.zst"),
            output_root=Path(__file__).resolve().parents[3] / "forbidden_enriched",
            block_root=Path("/nonexistent/blocks"),
            clock_root=Path("/tmp/qr_real_input_v2"),
        )
    except ValueError:
        checks["reject_workspace_output_root"] = True
    else:
        checks["reject_workspace_output_root"] = False
    try:
        EnrichedAdapterConfig(
            source_date=str(day),
            input_path=Path("/nonexistent/raw.dbn.zst"),
            output_root=APPROVED_EXTERNAL_DERIVED_ROOT / "qr_real_input_v2",
            block_root=Path("/nonexistent/blocks"),
            clock_root=Path("/tmp/qr_real_input_v2"),
        )
    except ValueError:
        checks["reject_legacy_v2_output_root"] = True
    else:
        checks["reject_legacy_v2_output_root"] = False
    if not all(checks.values()):
        raise RuntimeError(f"adapter contract smoke failed: {checks}")
    return {"passed": True, **checks}


def raw_path(raw_root: Path, source_date: str) -> Path:
    compact = source_date.replace("-", "")
    return raw_root / f"date={source_date}" / f"xnas-itch-{compact}.mbo.dbn.zst"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract-smoke")
    one = sub.add_parser("run-day")
    one.add_argument(
        "--source-date", required=True, choices=sum(SPLIT_DATES.values(), ())
    )
    one.add_argument("--input", type=Path)
    one.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    one.add_argument("--output-root", type=Path, required=True)
    one.add_argument("--log-path", type=Path)
    one.add_argument("--tick-size-nano", type=int, default=10_000_000)
    one.add_argument("--checkpoint-source-events", type=int, default=500_000)
    one.add_argument("--part-rows", type=int, default=100_000)
    one.add_argument("--progress-seconds", type=float, default=30.0)
    one.add_argument("--max-clock-events", type=int)
    one.add_argument("--no-resume", action="store_true")
    one.add_argument("--no-skip-existing", action="store_true")
    enriched = sub.add_parser("enriched-events-day")
    enriched.add_argument(
        "--source-date", required=True, choices=sum(SPLIT_DATES.values(), ())
    )
    enriched.add_argument("--input", type=Path)
    enriched.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    enriched.add_argument("--output-root", type=Path, required=True)
    enriched.add_argument("--block-root", type=Path, required=True)
    enriched.add_argument("--clock-root", type=Path, required=True)
    enriched.add_argument("--log-path", type=Path)
    enriched.add_argument("--tick-size-nano", type=int, default=10_000_000)
    enriched.add_argument("--checkpoint-source-events", type=int, default=500_000)
    enriched.add_argument("--part-rows", type=int, default=100_000)
    enriched.add_argument("--progress-seconds", type=float, default=30.0)
    enriched.add_argument("--no-resume", action="store_true")
    enriched.add_argument("--no-skip-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "contract-smoke":
        print(json.dumps(run_contract_smoke(), indent=2, sort_keys=True))
        return
    if args.command == "enriched-events-day":
        input_path = args.input or raw_path(args.raw_root, args.source_date)
        result = run_enriched_adapter(
            EnrichedAdapterConfig(
                source_date=args.source_date,
                input_path=input_path,
                output_root=args.output_root,
                block_root=args.block_root,
                clock_root=args.clock_root,
                tick_size_nano=args.tick_size_nano,
                checkpoint_source_events=args.checkpoint_source_events,
                part_rows=args.part_rows,
                progress_seconds=args.progress_seconds,
                skip_existing=not args.no_skip_existing,
                no_resume=args.no_resume,
                log_path=args.log_path,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=_jsonable))
        return
    input_path = args.input or raw_path(args.raw_root, args.source_date)
    result = run_adapter(
        AdapterConfig(
            source_date=args.source_date,
            input_path=input_path,
            output_root=args.output_root,
            tick_size_nano=args.tick_size_nano,
            checkpoint_source_events=args.checkpoint_source_events,
            part_rows=args.part_rows,
            progress_seconds=args.progress_seconds,
            skip_existing=not args.no_skip_existing,
            no_resume=args.no_resume,
            log_path=args.log_path,
            max_clock_events=args.max_clock_events,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_jsonable))


if __name__ == "__main__":
    main()
