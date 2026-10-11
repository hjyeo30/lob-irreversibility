"""Restartable multi-regime continuous-block producer.

The immutable 2026-07-24 Phase 03C implementation remains the canonical replay
semantics.  This module reuses its Book, source classification, block schema and
feature calculations while adding symbol/regime/segment orchestration.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as wall_time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil


PROJECT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_PROJECT = PROJECT_DIR / "upstream"
UPSTREAM_CODE = (
    UPSTREAM_PROJECT / "phases" / "03c_continuous_path_preflight" / "codebook.py"
)
SPEC = importlib.util.spec_from_file_location(
    "upstream_continuous_blocks", UPSTREAM_CODE
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load immutable upstream code: {UPSTREAM_CODE}")
UPSTREAM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UPSTREAM
SPEC.loader.exec_module(UPSTREAM)

NY = ZoneInfo("America/New_York")
FATAL_BOOK_ERRORS = {
    "duplicate_active_order_id",
    "invalid_add",
    "unknown_cancel",
    "cancel_exceeds_remaining",
    "unknown_modify",
    "invalid_modify",
    "unknown_action",
    "negative_price_level_depth",
}
REGIME_CLOCKS = {
    "premarket": (wall_time(4, 0), wall_time(9, 30)),
    "open": (wall_time(9, 30), wall_time(10, 0)),
    "daytime": (wall_time(10, 0), wall_time(15, 30)),
    "pre_close": (wall_time(15, 30), wall_time(16, 0)),
    "after_hours": (wall_time(16, 0), wall_time(20, 0)),
}
BOUNDARY_FIELDS = (
    "spread_ticks",
    "bid_depth_l1",
    "ask_depth_l1",
    "bid_depth_l5",
    "ask_depth_l5",
    "imbalance_l1",
    "imbalance_l5",
)


def scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True, default=scalar
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=scalar)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.stream.close()


@dataclass(frozen=True)
class ProducerConfig:
    symbol: str
    source_date: str
    input_path: Path
    derived_root: Path
    publish_regimes: dict[str, tuple[int, ...]]
    qa_regimes: dict[str, tuple[int, ...]]
    tick_size_nano: int = 10_000_000
    checkpoint_source_events: int = 1_000_000
    part_rows: int = 100_000
    progress_seconds: float = 30.0
    skip_existing: bool = True
    no_resume: bool = False
    log_path: Path | None = None
    upstream_micro_root: Path | None = None
    upstream_coarse_root: Path | None = None

    def readable(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source_date": self.source_date,
            "input_path": str(self.input_path),
            "input_bytes": self.input_path.stat().st_size,
            "derived_root": str(self.derived_root),
            "publish_regimes": {
                key: list(value) for key, value in self.publish_regimes.items()
            },
            "qa_regimes": {key: list(value) for key, value in self.qa_regimes.items()},
            "tick_size_nano": self.tick_size_nano,
            "checkpoint_source_events": self.checkpoint_source_events,
            "part_rows": self.part_rows,
        }


def regime_bounds(source_date: date) -> dict[str, tuple[int, int]]:
    return {
        name: (
            int(datetime.combine(source_date, start, tzinfo=NY).timestamp() * 1e9),
            int(datetime.combine(source_date, end, tzinfo=NY).timestamp() * 1e9),
        )
        for name, (start, end) in REGIME_CLOCKS.items()
    }


def regime_at(timestamp_ns: int, bounds: dict[str, tuple[int, int]]) -> str | None:
    for name, (start, end) in bounds.items():
        if start <= timestamp_ns < end:
            return name
    return None


def selected_cells(config: ProducerConfig) -> dict[str, tuple[int, ...]]:
    cells: dict[str, tuple[int, ...]] = {}
    for source in (config.publish_regimes, config.qa_regimes):
        for regime, widths in source.items():
            if regime in cells and cells[regime] != widths:
                raise ValueError(f"conflicting widths for regime={regime}")
            cells[regime] = tuple(sorted(set(widths)))
    unknown = set(cells) - set(REGIME_CLOCKS)
    if unknown:
        raise ValueError(f"unknown regimes: {sorted(unknown)}")
    return cells


def verify_input_inventory(input_path: Path) -> dict[str, Any]:
    inventory_path = input_path.parent / "manifest.json"
    if not inventory_path.is_file():
        # The original 15-day NVDA acquisition predates per-date manifests.
        # Phase 00 verified those immutable vendor-native files in place.  Do
        # not synthesize a new manifest or hash chain just to consume them:
        # retain the exact filename and observed byte count, then let the DBN
        # decoder/header checks in ``run`` validate the physical input.
        size = input_path.stat().st_size
        expected_name = (
            f"xnas-itch-{input_path.parent.name.removeprefix('date=').replace('-', '')}"
            ".mbo.dbn.zst"
        )
        if input_path.name != expected_name or size <= 0:
            raise RuntimeError(
                "legacy raw identity mismatch: "
                f"observed=({input_path.name},{size}) expected_name={expected_name}"
            )
        return {
            "inventory": None,
            "filename": input_path.name,
            "bytes": size,
            "job_id": None,
            "identity_mode": "phase00_verified_legacy_filename_bytes",
        }
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_name: str | None = None
    expected_bytes: int | None = None
    if "local_filename" in inventory and "bytes" in inventory:
        expected_name = str(inventory["local_filename"])
        expected_bytes = int(inventory["bytes"])
    elif isinstance(inventory.get("files"), list):
        for item in inventory["files"]:
            if item.get("filename") == input_path.name:
                expected_name = str(item["filename"])
                expected_bytes = int(item["size"])
                break
    if expected_name != input_path.name or expected_bytes != input_path.stat().st_size:
        raise RuntimeError(
            "raw inventory mismatch: "
            f"observed=({input_path.name},{input_path.stat().st_size}) "
            f"expected=({expected_name},{expected_bytes})"
        )
    return {
        "inventory": str(inventory_path),
        "filename": expected_name,
        "bytes": expected_bytes,
        "job_id": inventory.get("job_id"),
        "identity_mode": "vendor_manifest_filename_bytes",
    }


def segment_root(staging: Path, regime: str, segment: int, qa: bool) -> Path:
    base = staging / "_qa" if qa else staging
    return base / f"regime={regime}" / f"segment={segment:04d}"


def write_buffers(
    staging: Path,
    buffers: dict[tuple[str, int, int, bool], list[dict[str, Any]]],
    next_parts: dict[tuple[str, int, int, bool], int],
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for key, rows in buffers.items():
        if not rows:
            continue
        regime, segment, width, qa = key
        directory = segment_root(staging, regime, segment, qa) / f"target_b={width}"
        directory.mkdir(parents=True, exist_ok=True)
        part = next_parts.get(key, 0)
        path = directory / f"part-{part:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=UPSTREAM.BLOCK_SCHEMA)
        pq.write_table(table, path, compression="zstd", row_group_size=100_000)
        next_parts[key] = part + 1
        written.append(
            {
                "path": str(path.relative_to(staging)),
                "rows": table.num_rows,
                "bytes": path.stat().st_size,
                "qa_only": qa,
            }
        )
        rows.clear()
    return written


def _take_edges(paths: list[Path], count: int) -> pa.Table:
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    if total == 0:
        return pa.Table.from_pylist([], schema=UPSTREAM.BLOCK_SCHEMA)
    first_tables: list[pa.Table] = []
    remaining = min(count, total)
    for path in paths:
        if remaining == 0:
            break
        table = pq.ParquetFile(path).read()
        take = min(remaining, table.num_rows)
        first_tables.append(table.slice(0, take))
        remaining -= take
    last_tables: list[pa.Table] = []
    remaining = min(count, total)
    for path in reversed(paths):
        if remaining == 0:
            break
        table = pq.ParquetFile(path).read()
        take = min(remaining, table.num_rows)
        last_tables.append(table.slice(table.num_rows - take, take))
        remaining -= take
    last_tables.reverse()
    return pa.concat_tables(first_tables + last_tables, promote_options="default")


def _tables_match(left: pa.Table, right: pa.Table) -> tuple[bool, str | None]:
    if left.schema.names != right.schema.names or left.num_rows != right.num_rows:
        return False, "shape_or_columns"
    for field in left.schema:
        name = field.name
        a = left[name].combine_chunks().to_numpy(zero_copy_only=False)
        b = right[name].combine_chunks().to_numpy(zero_copy_only=False)
        if pa.types.is_float32(field.type):
            equal = np.allclose(a, b, rtol=0.0, atol=1e-7, equal_nan=False)
        elif pa.types.is_float64(field.type):
            equal = np.array_equal(a, b, equal_nan=False)
        else:
            equal = np.array_equal(a, b)
        if not equal:
            return False, name
    return True, None


def validate_daytime_parity(config: ProducerConfig, staging: Path) -> dict[str, Any]:
    if "daytime" not in config.qa_regimes:
        return {"status": "not_requested"}
    if config.upstream_micro_root is None or config.upstream_coarse_root is None:
        raise ValueError("NVDA Daytime QA requires both upstream block roots")
    results: dict[str, Any] = {}
    for width in config.qa_regimes["daytime"]:
        generated = sorted(
            (staging / "_qa" / "regime=daytime").glob(
                f"segment=*/target_b={width}/*.parquet"
            )
        )
        upstream_root = (
            config.upstream_micro_root
            if width in (2, 5, 10)
            else config.upstream_coarse_root
        )
        upstream = sorted(
            (
                upstream_root
                / "symbol=NVDA"
                / f"source_date={config.source_date}"
                / f"target_b={width}"
            ).glob("*.parquet")
        )
        if not generated or not upstream:
            raise RuntimeError(f"missing parity input for B={width}")
        generated_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in generated)
        upstream_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in upstream)
        if generated_rows != upstream_rows:
            raise RuntimeError(
                f"Daytime row mismatch B={width}: {generated_rows} != {upstream_rows}"
            )
        same, field = _tables_match(
            _take_edges(generated, 128), _take_edges(upstream, 128)
        )
        if not same:
            raise RuntimeError(f"Daytime edge parity mismatch B={width} field={field}")
        results[str(width)] = {
            "rows": generated_rows,
            "first_last_128_equal": True,
        }
    return {"status": "passed", "widths": results}


def validate_files(
    staging: Path,
    files: Iterable[dict[str, Any]],
    config: ProducerConfig,
    counters: Counter[str],
    dropped: Counter[str],
) -> dict[str, Any]:
    rows = 0
    bytes_total = 0
    output_files = 0
    group_expected_index: dict[tuple[str, int, int], int] = {}
    group_last_end: dict[tuple[str, int, int], dict[str, Any]] = {}
    group_realized: Counter[tuple[str, int]] = Counter()
    observed_cells: set[tuple[str, int]] = set()
    ordered_files = sorted(files, key=lambda item: item["path"])
    for item in ordered_files:
        path = staging / item["path"]
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != UPSTREAM.BLOCK_SCHEMA:
            raise RuntimeError(f"output schema mismatch: {path}")
        if parquet.metadata.num_rows != item["rows"]:
            raise RuntimeError(f"footer row mismatch: {path}")
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"file byte mismatch: {path}")
        parts = Path(item["path"]).parts
        offset = 1 if parts[0] == "_qa" else 0
        regime = parts[offset].split("=", 1)[1]
        segment = int(parts[offset + 1].split("=", 1)[1])
        width = int(parts[offset + 2].split("=", 1)[1])
        key = (regime, segment, width)
        observed_cells.add((regime, width))
        for batch in parquet.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch])
            if any(table[name].null_count for name in table.schema.names):
                raise RuntimeError(f"null value in {path}")
            block_index = table["block_index"].to_numpy()
            if (
                any(value.as_py() != "XNAS.ITCH" for value in table["dataset"])
                or any(
                    value.as_py() != config.symbol.upper() for value in table["symbol"]
                )
                or any(
                    str(value.as_py()) != config.source_date
                    for value in table["source_date"]
                )
            ):
                raise RuntimeError(f"row identity/partition mismatch in {path}")
            expected = group_expected_index.get(key, 0)
            wanted = np.arange(
                expected, expected + len(block_index), dtype=block_index.dtype
            )
            if not np.array_equal(block_index, wanted):
                raise RuntimeError(f"non-contiguous block_index in {path}")
            group_expected_index[key] = expected + len(block_index)
            realized = table["realized_source_event_count"].to_numpy()
            overshoot = table["completion_overshoot"].to_numpy()
            target = table["target_b"].to_numpy()
            elapsed = table["elapsed_ns"].to_numpy()
            if (
                np.any(target != width)
                or np.any(realized < width)
                or np.any(overshoot != realized - width)
                or np.any(elapsed < 0)
            ):
                raise RuntimeError(f"block arithmetic invariant in {path}")
            for field in table.schema:
                if pa.types.is_floating(field.type):
                    values = table[field.name].to_numpy()
                    if not np.isfinite(values).all():
                        raise RuntimeError(f"nonfinite {field.name} in {path}")
            starts = {
                name: table[f"start_{name}"].to_numpy() for name in BOUNDARY_FIELDS
            }
            ends = {name: table[f"end_{name}"].to_numpy() for name in BOUNDARY_FIELDS}
            previous = group_last_end.get(key)
            if previous is not None and any(
                previous[name] != starts[name][0] for name in BOUNDARY_FIELDS
            ):
                raise RuntimeError(f"cross-file boundary mismatch in {path}")
            if len(block_index) > 1 and any(
                not np.array_equal(ends[name][:-1], starts[name][1:])
                for name in BOUNDARY_FIELDS
            ):
                raise RuntimeError(f"adjacent boundary mismatch in {path}")
            group_last_end[key] = {name: ends[name][-1] for name in BOUNDARY_FIELDS}
            group_realized[(regime, width)] += int(realized.sum())
        if not item["qa_only"]:
            rows += item["rows"]
            bytes_total += item["bytes"]
            output_files += 1
    expected_cells = {
        (regime, width)
        for source in (config.publish_regimes, config.qa_regimes)
        for regime, widths in source.items()
        for width in widths
    }
    if observed_cells != expected_cells:
        raise RuntimeError(
            f"requested cell mismatch: observed={sorted(observed_cells)} "
            f"expected={sorted(expected_cells)}"
        )
    for regime, width in expected_cells:
        partial = sum(
            count
            for label, count in dropped.items()
            if label.startswith(f"{regime}/") and label.endswith(f"/{width}")
        )
        included = counters[f"included_source:{regime}"]
        if group_realized[(regime, width)] + partial != included:
            raise RuntimeError(
                f"event accounting mismatch {regime}/B{width}: "
                f"used={group_realized[(regime, width)]} partial={partial} "
                f"included={included}"
            )
    return {
        "schema_equal_upstream": True,
        "footer_rows_equal": True,
        "block_arithmetic": True,
        "row_identity_matches_partition": True,
        "finite_nonnull": True,
        "contiguous_block_index": True,
        "adjacent_boundaries": True,
        "event_accounting": True,
        "requested_cells_complete": True,
        "output_files": output_files,
        "output_rows": rows,
        "output_bytes": bytes_total,
    }


def verify_complete_directory(config: ProducerConfig, root: Path) -> dict[str, Any]:
    complete_path = root / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing complete.json: {root}")
    value = json.loads(complete_path.read_text(encoding="utf-8"))
    inventory = verify_input_inventory(config.input_path.resolve())
    expected_requested = {
        "publish_regimes": config.readable()["publish_regimes"],
        "qa_regimes": config.readable()["qa_regimes"],
    }
    if (
        value.get("status") != "complete"
        or value.get("unit") != f"{config.symbol.upper()}/{config.source_date}"
        or value.get("input", {}).get("filename") != inventory["filename"]
        or value.get("input", {}).get("bytes") != inventory["bytes"]
        or value.get("requested") != expected_requested
        or value.get("validation", {}).get("requested_cells_complete") is not True
    ):
        raise RuntimeError(f"complete record mismatch: {root}")
    for item in value.get("files", []):
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or pq.ParquetFile(path).metadata.num_rows != item["rows"]
            or pq.ParquetFile(path).schema_arrow != UPSTREAM.BLOCK_SCHEMA
        ):
            raise RuntimeError(f"completed artifact mismatch: {path}")
    if not value.get("files"):
        raise RuntimeError(f"complete record has no files: {root}")
    return value


def run(config: ProducerConfig) -> dict[str, Any]:
    input_path = config.input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    inventory = verify_input_inventory(input_path)
    source_day = date.fromisoformat(config.source_date)
    symbol = config.symbol.upper()
    cells = selected_cells(config)
    final = (
        config.derived_root.resolve()
        / f"symbol={symbol}"
        / f"source_date={config.source_date}"
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if config.skip_existing:
            return verify_complete_directory(config, final)
        raise FileExistsError(f"unverified final output exists: {final}")

    lock_path = final.parent / f".{final.name}.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"unit already running: {symbol}/{config.source_date}"
        ) from exc

    staging = final.parent / f".{final.name}.staging"
    recovery = final.parent / f".{final.name}.recovery"
    if (staging / "complete.json").is_file():
        value = verify_complete_directory(config, staging)
        os.replace(staging, final)
        if recovery.exists():
            shutil.rmtree(recovery)
        lock_handle.close()
        lock_path.unlink(missing_ok=True)
        return value
    staging.mkdir(parents=True, exist_ok=True)
    recovery.mkdir(parents=True, exist_ok=True)
    checkpoint_path = recovery / "checkpoint.pkl"
    checkpoint_config = recovery / "checkpoint_config.json"
    validation_ready = recovery / "validation_ready.json"
    log_path = config.log_path or (
        PROJECT_DIR
        / "phases"
        / "02_multiregime_blocks"
        / "artifacts"
        / "logs"
        / f"{symbol.lower()}_{config.source_date}.jsonl"
    )
    logger = Logger(log_path)
    readable_config = config.readable()
    logger.log(
        "unit_preflight",
        **readable_config,
        available_ram_bytes=psutil.virtual_memory().available,
        disk_free_bytes=shutil.disk_usage(staging).free,
        cpu_count=psutil.cpu_count(),
    )

    book = UPSTREAM.BASE.Book()
    bounds = regime_bounds(source_day)
    boundary_timestamps = {value for pair in bounds.values() for value in pair}
    accumulators: dict[int, Any] = {}
    active_regime: str | None = None
    segment_numbers: Counter[str] = Counter()
    active_segment = -1
    depleted_prices: dict[str, set[int]] = {"B": set(), "A": set()}
    buffers: dict[tuple[str, int, int, bool], list[dict[str, Any]]] = {}
    next_parts: dict[tuple[str, int, int, bool], int] = {}
    files: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    last_sequence: dict[tuple[int, int, int], int] = {}
    raw_cursor = source_index = 0

    def close_segment() -> None:
        nonlocal accumulators
        if active_regime is None:
            return
        for width, accumulator in accumulators.items():
            dropped[f"{active_regime}/{active_segment}/{width}"] += accumulator.count
        accumulators = {}

    def open_segment(regime: str) -> None:
        nonlocal active_regime, active_segment, accumulators, depleted_prices
        close_segment()
        active_regime = regime
        active_segment = segment_numbers[regime]
        segment_numbers[regime] += 1
        accumulators = {
            width: UPSTREAM.BlockAccumulator(
                width, config.tick_size_nano, symbol, source_day
            )
            for width in cells[regime]
        }
        depleted_prices = {"B": set(), "A": set()}
        counters[f"segment_start:{regime}"] += 1

    def leave_selected_regime() -> None:
        nonlocal active_regime, active_segment, accumulators, depleted_prices
        close_segment()
        active_regime = None
        active_segment = -1
        accumulators = {}
        depleted_prices = {"B": set(), "A": set()}

    if checkpoint_path.exists() and checkpoint_config.exists() and not config.no_resume:
        prior = json.loads(checkpoint_config.read_text(encoding="utf-8"))
        if prior != readable_config:
            raise RuntimeError("checkpoint readable config mismatch")
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        book = saved["book"]
        accumulators = saved["accumulators"]
        active_regime = saved["active_regime"]
        active_segment = saved["active_segment"]
        segment_numbers = saved["segment_numbers"]
        depleted_prices = saved["depleted_prices"]
        buffers = saved["buffers"]
        next_parts = saved["next_parts"]
        files = saved["files"]
        counters = saved["counters"]
        dropped = saved["dropped"]
        last_sequence = saved["last_sequence"]
        raw_cursor = saved["raw_cursor"]
        source_index = saved["source_index"]
        logger.log("unit_resume", raw_cursor=raw_cursor, source_events=source_index)
    elif any(staging.iterdir()):
        raise RuntimeError(f"staging exists without compatible checkpoint: {staging}")

    def save_checkpoint(cursor_after: int) -> None:
        nonlocal files
        files.extend(write_buffers(staging, buffers, next_parts))
        saved = {
            "book": book,
            "accumulators": accumulators,
            "active_regime": active_regime,
            "active_segment": active_segment,
            "segment_numbers": segment_numbers,
            "depleted_prices": depleted_prices,
            "buffers": buffers,
            "next_parts": next_parts,
            "files": files,
            "counters": counters,
            "dropped": dropped,
            "last_sequence": last_sequence,
            "raw_cursor": cursor_after,
            "source_index": source_index,
        }
        temporary = checkpoint_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
        atomic_json(checkpoint_config, readable_config)
        logger.log(
            "checkpoint",
            raw_cursor=cursor_after,
            source_events=source_index,
            blocks=sum(acc.block_index for acc in accumulators.values()),
            rss_bytes=psutil.Process().memory_info().rss,
        )

    current: list[Any] = []
    current_errors: list[str] = []
    current_pre: dict[str, Any] | None = None
    started = last_log = time.monotonic()
    next_checkpoint = source_index + config.checkpoint_source_events

    def finalize_source() -> None:
        nonlocal current, current_errors, current_pre, source_index
        nonlocal active_regime, next_checkpoint
        if not current or current_pre is None:
            return
        source = UPSTREAM.BASE.classify_source(
            current, source_index, symbol, current_errors
        )
        if source["source_invariant_count"]:
            raise RuntimeError(f"source invariant at source={source_index}")
        stream_key = (
            source["publisher_id"],
            source["channel_id"],
            source["instrument_id"],
        )
        previous = last_sequence.get(stream_key)
        if previous is not None and source["sequence"] <= previous:
            raise RuntimeError(
                f"nonmonotone source sequence at {source_index}: "
                f"{previous}->{source['sequence']}"
            )
        last_sequence[stream_key] = source["sequence"]
        post = UPSTREAM.book_state(book, config.tick_size_nano)
        if source["has_f_last"] and post["complete"]:
            if post["best_ask_nano"] - post["best_bid_nano"] <= 0:
                raise RuntimeError(
                    f"locked/crossed completion at source={source_index}"
                )
        timestamp = source["ts_event_end"]
        regime = regime_at(timestamp, bounds)
        selected = regime if regime in cells else None
        transitioned = selected != active_regime
        if transitioned:
            if selected is None:
                leave_selected_regime()
            else:
                open_segment(selected)

        is_boundary_cross = (
            source["event_family"] == "standalone_trade_or_cross"
            and timestamp in boundary_timestamps
        )
        is_clear = source["event_family"] == "clear"
        if (
            active_regime is not None
            and (is_boundary_cross or is_clear)
            and not transitioned
        ):
            counters[
                f"hard_reset:{active_regime}:"
                f"{'clear' if is_clear else 'boundary_cross'}"
            ] += 1
            open_segment(active_regime)
        elif active_regime is not None:
            initialized = all(
                acc.start_state is not None for acc in accumulators.values()
            )
            if not initialized:
                if source["has_f_last"] and post["complete"]:
                    for accumulator in accumulators.values():
                        accumulator.initialize(post, timestamp)
                    counters[f"warmup:{active_regime}"] += 1
            else:
                metrics = UPSTREAM.transition(current_pre, post, config.tick_size_nano)
                same_price_refills = 0
                for side in ("B", "A"):
                    dep_key = "bid_depletion" if side == "B" else "ask_depletion"
                    if metrics[dep_key]:
                        price = current_pre[
                            "best_bid_nano" if side == "B" else "best_ask_nano"
                        ]
                        if price is not None:
                            depleted_prices[side].add(price)
                    for raw in current:
                        if (
                            raw.action == "A"
                            and raw.side == side
                            and raw.price in depleted_prices[side]
                            and book._levels(side).get(raw.price, 0) > 0
                        ):
                            depleted_prices[side].discard(raw.price)
                            same_price_refills += 1
                qa = active_regime in config.qa_regimes
                for width, accumulator in accumulators.items():
                    # A source group may carry F_LAST while one book side is
                    # temporarily empty.  It still contributes to the event
                    # clock, but the incomplete post-state is not an eligible
                    # block boundary.  Suppress only boundary completion; the
                    # next complete F_LAST closes the overshooting block.
                    block_source = source
                    if source["has_f_last"] and not post["complete"]:
                        block_source = {**source, "has_f_last": False}
                        counters[f"incomplete_boundary_deferred:{active_regime}"] += 1
                    row = accumulator.add(
                        block_source, current_pre, post, metrics, same_price_refills
                    )
                    if row is not None:
                        key = (active_regime, active_segment, width, qa)
                        buffers.setdefault(key, []).append(row)
                        if len(buffers[key]) >= config.part_rows:
                            files.extend(write_buffers(staging, buffers, next_parts))
                counters[f"included_source:{active_regime}"] += 1
                counters[f"family:{active_regime}:{source['event_family']}"] += 1
        counters["source_events"] += 1
        source_index += 1
        current = []
        current_errors = []
        current_pre = None

    try:
        store = db.DBNStore.from_file(input_path)
        for ordinal, native in enumerate(store):
            if ordinal < raw_cursor:
                continue
            raw = UPSTREAM.BASE.Raw.from_record(ordinal, native)
            if current and raw.key != current[0].key:
                finalize_source()
                if source_index >= next_checkpoint:
                    save_checkpoint(ordinal)
                    next_checkpoint = source_index + config.checkpoint_source_events
            if not current:
                current_pre = UPSTREAM.book_state(book, config.tick_size_nano)
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
                    "progress",
                    raw_ordinal=ordinal,
                    source_events=source_index,
                    active_regime=active_regime,
                    completed_blocks=sum(int(item["rows"]) for item in files)
                    + sum(len(rows) for rows in buffers.values()),
                    records_per_second=(ordinal + 1 - raw_cursor) / max(elapsed, 1e-9),
                    rss_bytes=psutil.Process().memory_info().rss,
                    disk_free_bytes=shutil.disk_usage(staging).free,
                )
                last_log = now
        finalize_source()
        leave_selected_regime()
        save_checkpoint(counters["raw_records"])
        if validation_ready.is_file():
            ready = json.loads(validation_ready.read_text(encoding="utf-8"))
            parity = ready["parity"]
            validation = ready["validation"]
        else:
            parity = validate_daytime_parity(config, staging)
            validation = validate_files(staging, files, config, counters, dropped)
            atomic_json(validation_ready, {"parity": parity, "validation": validation})
    except Exception as exc:
        logger.log("unit_failed", error_type=type(exc).__name__, error=str(exc))
        logger.close()
        raise

    qa_root = staging / "_qa"
    if qa_root.exists():
        shutil.rmtree(qa_root)
    published_files = [
        {
            "path": item["path"],
            "rows": item["rows"],
            "bytes": item["bytes"],
        }
        for item in files
        if not item["qa_only"]
    ]
    elapsed = time.monotonic() - started
    complete = {
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": f"{symbol}/{config.source_date}",
        "input": {
            "filename": inventory["filename"],
            "bytes": inventory["bytes"],
            "inventory": inventory["inventory"],
            "job_id": inventory["job_id"],
        },
        "requested": {
            "publish_regimes": readable_config["publish_regimes"],
            "qa_regimes": readable_config["qa_regimes"],
        },
        "files": published_files,
        "counts": {
            "raw_records": counters["raw_records"],
            "source_events": counters["source_events"],
            "dropped_partial_source_events": dict(dropped),
            "hard_resets": {
                key: value
                for key, value in counters.items()
                if key.startswith("hard_reset:")
            },
        },
        "validation": {**validation, "daytime_parity": parity},
        "runtime": {
            "elapsed_seconds": elapsed,
            "final_rss_bytes": psutil.Process().memory_info().rss,
        },
    }
    atomic_json(staging / "complete.json", complete)
    os.replace(staging, final)
    shutil.rmtree(recovery)
    logger.log(
        "unit_published",
        unit=complete["unit"],
        elapsed_seconds=elapsed,
        output_rows=validation["output_rows"],
        output_bytes=validation["output_bytes"],
    )
    logger.close()
    lock_handle.close()
    lock_path.unlink(missing_ok=True)
    return complete
