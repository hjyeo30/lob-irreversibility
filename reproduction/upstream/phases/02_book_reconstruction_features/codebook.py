#!/usr/bin/env python3
"""XNAS.ITCH MBO를 source-event와 completion-state Parquet으로 변환한다.

이 변환기는 원본 record 순서를 바꾸지 않는다. 같은 venue sequence의 연속
record를 exchange source event로 접고, F_LAST가 닫는 complete-update
boundary에서만 compact order-book state를 기록한다. F_LAST는 분석 event가
아니라 안전한 state 관찰 지점을 정하는 vendor adapter로만 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import databento as db
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from sortedcontainers import SortedDict

LAST = 128
BAD_TS_RECV = 8
MAYBE_BAD_BOOK = 4
SNAPSHOT = 32
UNDEF_PRICE = 2**63 - 1
NS_PER_SECOND = 1_000_000_000
NY = ZoneInfo("America/New_York")
SCHEMA_VERSION = "replay_v2"
DEFAULT_INPUT = Path(
    "/data/lob_liquidity_time_irreversibility/raw/"
    "databento/xnas_itch/mbo/symbol=NVDA/date=2026-07-23/"
    "xnas-itch-20260723.mbo.dbn.zst"
)
DEFAULT_DERIVED_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/derived/replay_v2"
)
PHASE_DIR = Path(__file__).resolve().parent


def scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(f"[{row['timestamp_utc']}] {event} {fields}", flush=True)

    def close(self) -> None:
        self.stream.close()


def local_clock_fields(ts_ns: int) -> dict[str, Any]:
    local = datetime.fromtimestamp(ts_ns / NS_PER_SECOND, tz=UTC).astimezone(NY)
    seconds = (
        local.hour * 3_600
        + local.minute * 60
        + local.second
        + local.microsecond / 1_000_000
        + (ts_ns % 1_000) / NS_PER_SECOND
    )
    if seconds < 4 * 3_600:
        stratum = "before_premarket"
    elif seconds < 9 * 3_600 + 25 * 60:
        stratum = "premarket"
    elif seconds < 9 * 3_600 + 30 * 60:
        stratum = "opening_auction_buildup"
    elif seconds < 10 * 3_600:
        stratum = "open_transition"
    elif seconds < 15 * 3_600 + 30 * 60:
        stratum = "core_continuous"
    elif seconds < 15 * 3_600 + 50 * 60:
        stratum = "close_transition"
    elif seconds < 16 * 3_600:
        stratum = "closing_auction_buildup"
    else:
        stratum = "after_hours"
    return {
        "local_date": local.date(),
        "utc_offset_seconds": int(local.utcoffset().total_seconds()),
        "seconds_since_0930": seconds - (9 * 3_600 + 30 * 60),
        "minutes_to_1600": (16 * 3_600 - seconds) / 60,
        "clock_stratum": stratum,
    }


SOURCE_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("symbol", pa.string()),
        ("source_date", pa.date32()),
        ("publisher_id", pa.uint16()),
        ("channel_id", pa.uint8()),
        ("instrument_id", pa.uint32()),
        ("source_event_index", pa.uint64()),
        ("raw_ordinal_start", pa.uint64()),
        ("raw_ordinal_end", pa.uint64()),
        ("sequence", pa.uint32()),
        ("ts_event_start", pa.int64()),
        ("ts_event_end", pa.int64()),
        ("ts_recv_start", pa.int64()),
        ("ts_recv_end", pa.int64()),
        ("action_pattern", pa.string()),
        ("event_family", pa.string()),
        ("book_side", pa.string()),
        ("aggressor_side", pa.string()),
        ("raw_record_count", pa.uint16()),
        ("flags_or", pa.uint8()),
        ("has_f_last", pa.bool_()),
        ("completion_group_index", pa.int64()),
        ("completion_position", pa.uint16()),
        ("completion_source_count", pa.uint16()),
        ("first_price_nano", pa.int64()),
        ("last_price_nano", pa.int64()),
        ("old_price_nano", pa.int64()),
        ("new_price_nano", pa.int64()),
        ("old_order_id", pa.uint64()),
        ("new_order_id", pa.uint64()),
        ("add_qty", pa.uint64()),
        ("remove_qty", pa.uint64()),
        ("trade_qty", pa.uint64()),
        ("fill_qty", pa.uint64()),
        ("cancel_qty", pa.uint64()),
        ("execution_remove_qty", pa.uint64()),
        ("replace_remove_qty", pa.uint64()),
        ("standalone_trade_or_cross_qty", pa.uint64()),
        ("local_date", pa.date32()),
        ("utc_offset_seconds", pa.int32()),
        ("seconds_since_0930", pa.float64()),
        ("minutes_to_1600", pa.float64()),
        ("clock_stratum", pa.string()),
        ("source_invariant_count", pa.uint16()),
        ("source_invariant_codes", pa.string()),
    ],
    metadata={
        b"schema_version": SCHEMA_VERSION.encode(),
        b"price_unit": b"1e-9 currency units",
        b"scientific_unit": b"contiguous same-sequence exchange source event",
    },
)

LADDER_PRICE = pa.list_(pa.int64(), 10)
LADDER_SIZE = pa.list_(pa.uint64(), 10)
COMPLETION_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("symbol", pa.string()),
        ("source_date", pa.date32()),
        ("publisher_id", pa.uint16()),
        ("channel_id", pa.uint8()),
        ("instrument_id", pa.uint32()),
        ("completion_group_index", pa.uint64()),
        ("source_event_index_start", pa.uint64()),
        ("source_event_index_end", pa.uint64()),
        ("raw_ordinal_start", pa.uint64()),
        ("raw_ordinal_end", pa.uint64()),
        ("sequence_start", pa.uint32()),
        ("sequence_end", pa.uint32()),
        ("source_event_count", pa.uint16()),
        ("raw_record_count", pa.uint16()),
        ("ts_event_start", pa.int64()),
        ("ts_event_end", pa.int64()),
        ("duration_ns", pa.int64()),
        ("elapsed_from_previous_ns", pa.int64()),
        ("flags_or", pa.uint8()),
        ("bid_prices_nano", LADDER_PRICE),
        ("bid_sizes", LADDER_SIZE),
        ("ask_prices_nano", LADDER_PRICE),
        ("ask_sizes", LADDER_SIZE),
        ("best_bid_nano", pa.int64()),
        ("best_ask_nano", pa.int64()),
        ("spread_nano", pa.int64()),
        ("midprice_x2_nano", pa.int64()),
        ("bid_depth_l1", pa.uint64()),
        ("ask_depth_l1", pa.uint64()),
        ("bid_depth_l5", pa.uint64()),
        ("ask_depth_l5", pa.uint64()),
        ("bid_depth_l10", pa.uint64()),
        ("ask_depth_l10", pa.uint64()),
        ("active_order_count", pa.uint64()),
        ("bid_level_count", pa.uint32()),
        ("ask_level_count", pa.uint32()),
        ("group_add_qty", pa.uint64()),
        ("group_remove_qty", pa.uint64()),
        ("group_trade_qty", pa.uint64()),
        ("group_fill_qty", pa.uint64()),
        ("group_cancel_qty", pa.uint64()),
        ("group_execution_remove_qty", pa.uint64()),
        ("group_replace_remove_qty", pa.uint64()),
        ("add_event_count", pa.uint16()),
        ("cancel_event_count", pa.uint16()),
        ("replace_event_count", pa.uint16()),
        ("displayed_execution_event_count", pa.uint16()),
        ("nonprintable_displayed_execution_event_count", pa.uint16()),
        ("standalone_trade_or_cross_event_count", pa.uint16()),
        ("clear_event_count", pa.uint16()),
        ("other_event_count", pa.uint16()),
        ("locked", pa.bool_()),
        ("crossed", pa.bool_()),
        ("local_date", pa.date32()),
        ("utc_offset_seconds", pa.int32()),
        ("seconds_since_0930", pa.float64()),
        ("minutes_to_1600", pa.float64()),
        ("clock_stratum", pa.string()),
        ("group_invariant_count", pa.uint16()),
        ("group_invariant_codes", pa.string()),
    ],
    metadata={
        b"schema_version": SCHEMA_VERSION.encode(),
        b"price_unit": b"1e-9 currency units",
        b"scientific_unit": b"post-F_LAST complete-update book state",
    },
)


@dataclass(frozen=True)
class Order:
    side: str
    price: int
    size: int


@dataclass
class Raw:
    ordinal: int
    publisher: int
    channel: int
    instrument: int
    sequence: int
    ts_event: int
    ts_recv: int
    action: str
    side: str
    price: int
    size: int
    order_id: int
    flags: int

    @classmethod
    def from_record(cls, ordinal: int, record: Any) -> "Raw":
        return cls(
            ordinal=ordinal,
            publisher=int(record.publisher_id),
            channel=int(record.channel_id),
            instrument=int(record.instrument_id),
            sequence=int(record.sequence),
            ts_event=int(record.ts_event),
            ts_recv=int(record.ts_recv),
            action=str(scalar(record.action)),
            side=str(scalar(record.side)),
            price=int(record.price),
            size=int(record.size),
            order_id=int(record.order_id),
            flags=int(record.flags),
        )

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (
            self.ts_event // (86_400 * NS_PER_SECOND),
            self.publisher,
            self.channel,
            self.instrument,
            self.sequence,
        )


@dataclass
class Book:
    orders: dict[int, Order] = field(default_factory=dict)
    bids: SortedDict[int, int] = field(default_factory=SortedDict)
    asks: SortedDict[int, int] = field(default_factory=SortedDict)

    def _levels(self, side: str) -> SortedDict[int, int]:
        return self.bids if side == "B" else self.asks

    def _level_delta(self, side: str, price: int, delta: int) -> None:
        levels = self._levels(side)
        value = levels.get(price, 0) + delta
        if value < 0:
            raise ValueError("negative_price_level_depth")
        if value == 0:
            levels.pop(price, None)
        else:
            levels[price] = value

    def apply(self, row: Raw) -> list[str]:
        errors: list[str] = []
        if row.action == "R":
            self.orders.clear()
            self.bids.clear()
            self.asks.clear()
            return errors
        if row.action in {"T", "F", "N"}:
            return errors
        if row.action == "A":
            if row.order_id in self.orders:
                return ["duplicate_active_order_id"]
            if row.side not in {"B", "A"} or row.price == UNDEF_PRICE or row.size <= 0:
                return ["invalid_add"]
            self.orders[row.order_id] = Order(row.side, row.price, row.size)
            self._level_delta(row.side, row.price, row.size)
            return errors
        current = self.orders.get(row.order_id)
        if row.action == "C":
            if current is None:
                return ["unknown_cancel"]
            if row.side != current.side:
                errors.append("cancel_side_mismatch")
            if row.price != current.price:
                errors.append("cancel_price_mismatch")
            if row.size > current.size:
                return errors + ["cancel_exceeds_remaining"]
            self._level_delta(current.side, current.price, -row.size)
            remaining = current.size - row.size
            if remaining:
                self.orders[row.order_id] = Order(
                    current.side, current.price, remaining
                )
            else:
                del self.orders[row.order_id]
            return errors
        if row.action == "M":
            if current is None:
                return ["unknown_modify"]
            if row.side not in {"B", "A"} or row.price == UNDEF_PRICE or row.size <= 0:
                return ["invalid_modify"]
            self._level_delta(current.side, current.price, -current.size)
            self.orders[row.order_id] = Order(row.side, row.price, row.size)
            self._level_delta(row.side, row.price, row.size)
            return errors
        return ["unknown_action"]

    def top(self, side: str) -> tuple[list[int | None], list[int]]:
        levels = self._levels(side)
        items = reversed(levels.items()) if side == "B" else iter(levels.items())
        prices: list[int | None] = []
        sizes: list[int] = []
        for price, size in items:
            prices.append(price)
            sizes.append(size)
            if len(prices) == 10:
                break
        prices.extend([None] * (10 - len(prices)))
        sizes.extend([0] * (10 - len(sizes)))
        return prices, sizes


def event_family(actions: list[str]) -> str:
    pattern = "→".join(actions)
    exact = {
        "A": "add",
        "C": "cancel",
        "C→A": "replace",
        "T→F→C": "displayed_execution",
        "F→C": "nonprintable_displayed_execution",
        "T": "standalone_trade_or_cross",
        "R": "clear",
        "M": "modify",
        "N": "none",
    }
    return exact.get(pattern, "other")


def nullable_price(value: int) -> int | None:
    return None if value == UNDEF_PRICE else value


def classify_source(
    records: list[Raw],
    source_event_index: int,
    symbol: str,
    source_errors: list[str],
) -> dict[str, Any]:
    actions = [row.action for row in records]
    family = event_family(actions)
    adds = [row for row in records if row.action == "A"]
    cancels = [row for row in records if row.action == "C"]
    trades = [row for row in records if row.action == "T"]
    fills = [row for row in records if row.action == "F"]
    book_sides = [
        row.side
        for row in records
        if row.action in {"A", "C", "M", "F"} and row.side in {"A", "B"}
    ]
    aggressors = [row.side for row in trades if row.side in {"A", "B"}]
    if family in {"displayed_execution", "nonprintable_displayed_execution"}:
        cancel_qty = 0
        execution_remove_qty = sum(row.size for row in cancels)
    else:
        cancel_qty = sum(row.size for row in cancels) if family == "cancel" else 0
        execution_remove_qty = 0
    replace_remove_qty = sum(row.size for row in cancels) if family == "replace" else 0
    errors = list(dict.fromkeys(source_errors))
    if family == "replace" and len(cancels) == 1 and len(adds) == 1:
        if cancels[0].side != adds[0].side:
            errors.append("replace_side_mismatch")
    if family == "displayed_execution":
        quantities = [
            sum(x.size for x in trades),
            sum(x.size for x in fills),
            sum(x.size for x in cancels),
        ]
        if len(set(quantities)) != 1:
            errors.append("execution_quantity_mismatch")
        if trades and fills and trades[0].side == fills[0].side:
            errors.append("execution_aggressor_resting_side_not_opposite")
    first, last = records[0], records[-1]
    clock = local_clock_fields(last.ts_event)
    return {
        "dataset": "XNAS.ITCH",
        "symbol": symbol,
        "source_date": date.fromtimestamp(first.ts_event / NS_PER_SECOND),
        "publisher_id": first.publisher,
        "channel_id": first.channel,
        "instrument_id": first.instrument,
        "source_event_index": source_event_index,
        "raw_ordinal_start": first.ordinal,
        "raw_ordinal_end": last.ordinal,
        "sequence": first.sequence,
        "ts_event_start": first.ts_event,
        "ts_event_end": last.ts_event,
        "ts_recv_start": first.ts_recv,
        "ts_recv_end": last.ts_recv,
        "action_pattern": "→".join(actions),
        "event_family": family,
        "book_side": book_sides[-1]
        if book_sides and len(set(book_sides)) == 1
        else ("mixed" if book_sides else "N"),
        "aggressor_side": aggressors[0]
        if aggressors and len(set(aggressors)) == 1
        else ("mixed" if aggressors else "N"),
        "raw_record_count": len(records),
        "flags_or": int(np.bitwise_or.reduce([row.flags for row in records])),
        "has_f_last": bool(last.flags & LAST),
        "completion_group_index": None,
        "completion_position": None,
        "completion_source_count": None,
        "first_price_nano": nullable_price(first.price),
        "last_price_nano": nullable_price(last.price),
        "old_price_nano": nullable_price(cancels[0].price) if cancels else None,
        "new_price_nano": nullable_price(adds[-1].price) if adds else None,
        "old_order_id": cancels[0].order_id if cancels else None,
        "new_order_id": adds[-1].order_id if adds else None,
        "add_qty": sum(row.size for row in adds),
        "remove_qty": sum(row.size for row in cancels),
        "trade_qty": sum(row.size for row in trades),
        "fill_qty": sum(row.size for row in fills),
        "cancel_qty": cancel_qty,
        "execution_remove_qty": execution_remove_qty,
        "replace_remove_qty": replace_remove_qty,
        "standalone_trade_or_cross_qty": (
            sum(row.size for row in trades)
            if family == "standalone_trade_or_cross"
            else 0
        ),
        **clock,
        "source_invariant_count": len(errors),
        "source_invariant_codes": "|".join(errors),
    }


def completion_state(
    pending: list[dict[str, Any]],
    index: int,
    book: Book,
    previous_ts: int | None,
) -> dict[str, Any]:
    first, last = pending[0], pending[-1]
    bid_prices, bid_sizes = book.top("B")
    ask_prices, ask_sizes = book.top("A")
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    spread = (
        best_ask - best_bid if best_bid is not None and best_ask is not None else None
    )
    families = Counter(row["event_family"] for row in pending)
    errors = sorted(
        {
            code
            for row in pending
            for code in row["source_invariant_codes"].split("|")
            if code
        }
    )
    crossed = spread is not None and spread < 0
    locked = spread == 0 if spread is not None else False
    if crossed:
        errors.append("crossed_post_completion")
    clock = local_clock_fields(last["ts_event_end"])
    return {
        "dataset": last["dataset"],
        "symbol": last["symbol"],
        "source_date": last["source_date"],
        "publisher_id": last["publisher_id"],
        "channel_id": last["channel_id"],
        "instrument_id": last["instrument_id"],
        "completion_group_index": index,
        "source_event_index_start": first["source_event_index"],
        "source_event_index_end": last["source_event_index"],
        "raw_ordinal_start": first["raw_ordinal_start"],
        "raw_ordinal_end": last["raw_ordinal_end"],
        "sequence_start": first["sequence"],
        "sequence_end": last["sequence"],
        "source_event_count": len(pending),
        "raw_record_count": sum(row["raw_record_count"] for row in pending),
        "ts_event_start": first["ts_event_start"],
        "ts_event_end": last["ts_event_end"],
        "duration_ns": last["ts_event_end"] - first["ts_event_start"],
        "elapsed_from_previous_ns": None
        if previous_ts is None
        else last["ts_event_end"] - previous_ts,
        "flags_or": int(np.bitwise_or.reduce([row["flags_or"] for row in pending])),
        "bid_prices_nano": bid_prices,
        "bid_sizes": bid_sizes,
        "ask_prices_nano": ask_prices,
        "ask_sizes": ask_sizes,
        "best_bid_nano": best_bid,
        "best_ask_nano": best_ask,
        "spread_nano": spread,
        "midprice_x2_nano": best_bid + best_ask
        if best_bid is not None and best_ask is not None
        else None,
        "bid_depth_l1": sum(bid_sizes[:1]),
        "ask_depth_l1": sum(ask_sizes[:1]),
        "bid_depth_l5": sum(bid_sizes[:5]),
        "ask_depth_l5": sum(ask_sizes[:5]),
        "bid_depth_l10": sum(bid_sizes),
        "ask_depth_l10": sum(ask_sizes),
        "active_order_count": len(book.orders),
        "bid_level_count": len(book.bids),
        "ask_level_count": len(book.asks),
        "group_add_qty": sum(row["add_qty"] for row in pending),
        "group_remove_qty": sum(row["remove_qty"] for row in pending),
        "group_trade_qty": sum(row["trade_qty"] for row in pending),
        "group_fill_qty": sum(row["fill_qty"] for row in pending),
        "group_cancel_qty": sum(row["cancel_qty"] for row in pending),
        "group_execution_remove_qty": sum(
            row["execution_remove_qty"] for row in pending
        ),
        "group_replace_remove_qty": sum(row["replace_remove_qty"] for row in pending),
        "add_event_count": families["add"],
        "cancel_event_count": families["cancel"],
        "replace_event_count": families["replace"],
        "displayed_execution_event_count": families["displayed_execution"],
        "nonprintable_displayed_execution_event_count": families[
            "nonprintable_displayed_execution"
        ],
        "standalone_trade_or_cross_event_count": families["standalone_trade_or_cross"],
        "clear_event_count": families["clear"],
        "other_event_count": sum(
            families[name] for name in ("modify", "none", "other")
        ),
        "locked": locked,
        "crossed": crossed,
        **clock,
        "group_invariant_count": len(errors),
        "group_invariant_codes": "|".join(sorted(set(errors))),
    }


def write_part(
    root: Path,
    part: int,
    source_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    content_sha: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows, schema in (
        ("source_events", source_rows, SOURCE_SCHEMA),
        ("completion_states", completion_rows, COMPLETION_SCHEMA),
    ):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"part-{part:05d}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=100_000,
        )
        os.replace(temporary, target)
        result[name] = {
            "path": str(target.relative_to(root)),
            "rows": len(rows),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target) if content_sha else None,
            "content_sha_skipped": not content_sha,
        }
    return result


def code_fingerprint() -> str:
    return sha256_file(Path(__file__))


def preflight(input_path: Path, output_root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(
        output_root.parent if output_root.parent.exists() else output_root.parent.parent
    )
    return {
        "input_exists": input_path.is_file(),
        "input_bytes": input_path.stat().st_size if input_path.exists() else None,
        "memory_total": memory.total,
        "memory_available": memory.available,
        "swap_total": swap.total,
        "swap_used": swap.used,
        "load_average": os.getloadavg(),
        "disk_free": disk.free,
        "cpu_count": psutil.cpu_count(),
    }


def checkpoint_save(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fatal_codes(errors: list[str]) -> list[str]:
    fatal = {
        "duplicate_active_order_id",
        "invalid_add",
        "unknown_cancel",
        "cancel_exceeds_remaining",
        "negative_price_level_depth",
        "unknown_modify",
        "invalid_modify",
        "unknown_action",
    }
    return [code for code in errors if code in fatal]


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    final_root = (
        args.derived_root.resolve()
        / "dataset=XNAS.ITCH"
        / f"symbol={args.symbol}"
        / f"date={args.source_date}"
    )
    staging_root = final_root.parent / f".{final_root.name}.staging"
    if final_root.exists():
        raise FileExistsError(f"final output already exists: {final_root}")
    input_sha = None if args.skip_content_sha else sha256_file(input_path)
    input_identity = (
        {
            "provenance_id": args.input_provenance_id,
            "input_name": input_path.name,
            "input_bytes": input_path.stat().st_size,
        }
        if args.skip_content_sha
        else {"input_sha256": input_sha}
    )
    fingerprint = {
        "input_identity": input_identity,
        "code_sha256": code_fingerprint(),
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol,
        "source_date": args.source_date,
        "part_completion_rows": args.part_completion_rows,
        "content_sha_skipped": args.skip_content_sha,
    }
    run_id = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()[:16]
    staging_root.mkdir(parents=True, exist_ok=True)
    log_path = (
        PHASE_DIR
        / "artifacts"
        / "logs"
        / f"{args.symbol.lower()}_{args.source_date}_{run_id}.jsonl"
    )
    logger = JsonlLogger(log_path)
    logger.log("preflight", run_id=run_id, **preflight(input_path, staging_root))
    checkpoint_path = staging_root / "checkpoint.pkl"
    state_path = staging_root / "checkpoint_state.json"

    book = Book()
    raw_cursor = 0
    source_index = 0
    completion_index = 0
    previous_completion_ts: int | None = None
    part = 0
    parts: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    last_sequence: dict[tuple[int, int, int], int] = {}

    if checkpoint_path.exists() and state_path.exists() and not args.no_resume:
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("stale checkpoint fingerprint mismatch")
        book = saved["book"]
        raw_cursor = saved["raw_cursor"]
        source_index = saved["source_index"]
        completion_index = saved["completion_index"]
        previous_completion_ts = saved["previous_completion_ts"]
        part = saved["part"]
        parts = saved["parts"]
        counters = saved["counters"]
        examples = saved["examples"]
        last_sequence = saved["last_sequence"]
        logger.log(
            "resume",
            raw_cursor=raw_cursor,
            source_index=source_index,
            completion_index=completion_index,
            next_part=part,
        )
    elif any(staging_root.iterdir()):
        allowed = {log_path.name}
        leftovers = [p for p in staging_root.iterdir() if p.name not in allowed]
        if leftovers:
            raise RuntimeError(f"unverified staging output exists: {staging_root}")

    source_buffer: list[dict[str, Any]] = []
    completion_buffer: list[dict[str, Any]] = []
    pending_completion: list[dict[str, Any]] = []
    current_records: list[Raw] = []
    current_errors: list[str] = []
    started = time.monotonic()
    last_log = started

    def remember(codes: list[str], row: Raw) -> None:
        for code in codes:
            counters[f"invariant:{code}"] += 1
            bucket = examples.setdefault(code, [])
            if len(bucket) < 5:
                bucket.append(
                    {
                        "raw_ordinal": row.ordinal,
                        "sequence": row.sequence,
                        "action": row.action,
                        "side": row.side,
                        "price_nano": nullable_price(row.price),
                        "size": row.size,
                        "order_id": row.order_id,
                    }
                )

    def publish_buffers(cursor_after: int) -> None:
        nonlocal part, source_buffer, completion_buffer
        if not completion_buffer:
            return
        part_info = write_part(
            staging_root,
            part,
            source_buffer,
            completion_buffer,
            content_sha=not args.skip_content_sha,
        )
        parts.append({"part": part, **part_info})
        source_buffer = []
        completion_buffer = []
        part += 1
        state = {
            "fingerprint": fingerprint,
            "book": book,
            "raw_cursor": cursor_after,
            "source_index": source_index,
            "completion_index": completion_index,
            "previous_completion_ts": previous_completion_ts,
            "part": part,
            "parts": parts,
            "counters": counters,
            "examples": examples,
            "last_sequence": last_sequence,
        }
        checkpoint_save(checkpoint_path, state)
        atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "raw_cursor": cursor_after,
                "source_index": source_index,
                "completion_index": completion_index,
                "next_part": part,
                "parts": parts,
            },
        )
        logger.log(
            "part_complete",
            part=part - 1,
            raw_cursor=cursor_after,
            source_events=source_index,
            completion_groups=completion_index,
            active_orders=len(book.orders),
            rss_bytes=psutil.Process().memory_info().rss,
            disk_free=shutil.disk_usage(staging_root).free,
        )

    def finalize_source() -> None:
        nonlocal source_index, completion_index, previous_completion_ts
        nonlocal current_records, current_errors, pending_completion
        if not current_records:
            return
        row = classify_source(
            current_records, source_index, args.symbol, current_errors
        )
        stream = (row["publisher_id"], row["channel_id"], row["instrument_id"])
        previous = last_sequence.get(stream)
        if previous is not None:
            if row["sequence"] < previous:
                remember(["sequence_decrease"], current_records[0])
                row["source_invariant_codes"] += (
                    "|" if row["source_invariant_codes"] else ""
                ) + "sequence_decrease"
                row["source_invariant_count"] += 1
            elif row["sequence"] == previous:
                remember(["noncontiguous_sequence_reuse"], current_records[0])
                row["source_invariant_codes"] += (
                    "|" if row["source_invariant_codes"] else ""
                ) + "noncontiguous_sequence_reuse"
                row["source_invariant_count"] += 1
        last_sequence[stream] = row["sequence"]
        counters["source_events"] += 1
        counters[f"family:{row['event_family']}"] += 1
        counters[f"pattern:{row['action_pattern']}"] += 1
        counters["source_invariant_rows"] += row["source_invariant_count"] > 0
        is_leading_boundary = (
            source_index == 0
            and row["event_family"] == "clear"
            and not row["has_f_last"]
            and bool(row["flags_or"] & BAD_TS_RECV)
        )
        if is_leading_boundary:
            counters["leading_boundary_source_events"] += 1
            source_buffer.append(row)
        else:
            pending_completion.append(row)
            if row["has_f_last"]:
                group_count = len(pending_completion)
                for position, pending_row in enumerate(pending_completion):
                    pending_row["completion_group_index"] = completion_index
                    pending_row["completion_position"] = position
                    pending_row["completion_source_count"] = group_count
                source_buffer.extend(pending_completion)
                completion_buffer.append(
                    completion_state(
                        pending_completion,
                        completion_index,
                        book,
                        previous_completion_ts,
                    )
                )
                previous_completion_ts = row["ts_event_end"]
                completion_index += 1
                counters["completion_groups"] += 1
                pending_completion = []
        source_index += 1
        current_records = []
        current_errors = []

    try:
        for ordinal, native in enumerate(db.DBNStore.from_file(input_path)):
            if ordinal < raw_cursor:
                continue
            row = Raw.from_record(ordinal, native)
            if current_records and row.key != current_records[0].key:
                finalize_source()
                if (
                    len(completion_buffer) >= args.part_completion_rows
                    and not pending_completion
                ):
                    publish_buffers(ordinal)
            errors = book.apply(row)
            current_errors.extend(errors)
            remember(errors, row)
            if fatal_codes(errors):
                raise RuntimeError(
                    f"state-threatening replay invariant at raw ordinal {ordinal}: {fatal_codes(errors)}"
                )
            current_records.append(row)
            counters["raw_records"] += 1
            counters[f"action:{row.action}"] += 1
            counters[f"flags:{row.flags}"] += 1
            counters["bad_ts_recv_records"] += bool(row.flags & BAD_TS_RECV)
            counters["maybe_bad_book_records"] += bool(row.flags & MAYBE_BAD_BOOK)
            now = time.monotonic()
            if now - last_log >= args.progress_seconds:
                elapsed = now - started
                rate = (ordinal + 1 - raw_cursor) / elapsed
                logger.log(
                    "progress",
                    raw_ordinal=ordinal,
                    source_events=source_index,
                    completion_groups=completion_index,
                    records_per_second=rate,
                    rss_bytes=psutil.Process().memory_info().rss,
                )
                last_log = now
        finalize_source()
        if pending_completion:
            counters["trailing_incomplete_source_events"] += len(pending_completion)
            raise RuntimeError(
                f"trailing incomplete completion group with {len(pending_completion)} source events"
            )
        publish_buffers(counters["raw_records"])
    except Exception as exc:
        logger.log("failed", error_type=type(exc).__name__, error=str(exc))
        logger.close()
        raise

    elapsed = time.monotonic() - started
    logger.log("replay_complete", elapsed_seconds=elapsed, **dict(counters))
    qa = validate(staging_root, counters)
    files = []
    for path in sorted(staging_root.rglob("*.parquet")):
        files.append(
            {
                "path": str(path.relative_to(staging_root)),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "bytes": path.stat().st_size,
                "sha256": (None if args.skip_content_sha else sha256_file(path)),
                "content_sha_skipped": args.skip_content_sha,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "XNAS.ITCH",
        "schema": "mbo",
        "symbol": args.symbol,
        "source_date": args.source_date,
        "input": {
            "path_role": "immutable external NVMe raw artifact",
            "bytes": input_path.stat().st_size,
            "sha256": input_sha,
            "content_sha_skipped": args.skip_content_sha,
            "provenance_id": args.input_provenance_id,
        },
        "fingerprint": fingerprint,
        "runtime": {
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "databento": getattr(db, "__version__", "unknown"),
            "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__,
            "peak_or_final_rss_bytes": psutil.Process().memory_info().rss,
        },
        "counts": dict(counters),
        "invariant_examples": examples,
        "validation": qa,
        "files": files,
        "execution_log": str(log_path.relative_to(PHASE_DIR)),
    }
    atomic_json(staging_root / "manifest.json", manifest)
    checkpoint_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    os.replace(staging_root, final_root)
    phase_manifest = (
        PHASE_DIR
        / "artifacts"
        / "manifests"
        / f"{args.symbol.lower()}_{args.source_date}_{SCHEMA_VERSION}.json"
    )
    phase_copy = dict(manifest)
    phase_copy["physical_partition"] = str(final_root)
    atomic_json(phase_manifest, phase_copy)
    logger.log(
        "published", final_partition=str(final_root), manifest=str(phase_manifest)
    )
    logger.close()
    return phase_copy


def validate(root: Path, counters: Counter[str]) -> dict[str, Any]:
    source_glob = str(root / "source_events" / "*.parquet")
    completion_glob = str(root / "completion_states" / "*.parquet")
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=2")
    source = connection.execute(
        f"""
        SELECT
          count(*) AS rows,
          min(source_event_index) AS min_index,
          max(source_event_index) AS max_index,
          count(*) FILTER (WHERE source_event_index <> expected_index) AS index_errors,
          count(*) FILTER (WHERE ts_event_end < ts_event_start) AS negative_duration,
          count(*) FILTER (
            WHERE completion_group_index IS NOT NULL
              AND (completion_position IS NULL OR completion_source_count IS NULL)
          ) AS incomplete_link
        FROM (
          SELECT *, row_number() OVER (ORDER BY source_event_index) - 1 AS expected_index
          FROM read_parquet('{source_glob}')
        )
        """
    ).fetchone()
    completion = connection.execute(
        f"""
        SELECT
          count(*) AS rows,
          min(completion_group_index) AS min_index,
          max(completion_group_index) AS max_index,
          count(*) FILTER (WHERE completion_group_index <> expected_index) AS index_errors,
          count(*) FILTER (WHERE duration_ns < 0) AS negative_duration,
          count(*) FILTER (WHERE elapsed_from_previous_ns < 0) AS negative_elapsed,
          count(*) FILTER (WHERE crossed) AS crossed,
          count(*) FILTER (WHERE locked) AS locked
        FROM (
          SELECT *, row_number() OVER (ORDER BY completion_group_index) - 1 AS expected_index
          FROM read_parquet('{completion_glob}')
        )
        """
    ).fetchone()
    link_errors = connection.execute(
        f"""
        SELECT count(*)
        FROM (
          SELECT completion_group_index, count(*) AS n, max(completion_source_count) AS declared
          FROM read_parquet('{source_glob}')
          WHERE completion_group_index IS NOT NULL
          GROUP BY completion_group_index
        )
        WHERE n <> declared
        """
    ).fetchone()[0]
    connection.close()
    result = {
        "source_events": {
            "rows": source[0],
            "min_index": source[1],
            "max_index": source[2],
            "index_errors": source[3],
            "negative_duration": source[4],
            "incomplete_link": source[5],
        },
        "completion_states": {
            "rows": completion[0],
            "min_index": completion[1],
            "max_index": completion[2],
            "index_errors": completion[3],
            "negative_duration": completion[4],
            "negative_elapsed": completion[5],
            "crossed": completion[6],
            "locked": completion[7],
        },
        "completion_link_count_errors": link_errors,
        "count_reconciliation": {
            "source_matches": source[0] == counters["source_events"],
            "completion_matches": completion[0] == counters["completion_groups"],
        },
    }
    hard_failures = (
        source[3]
        + source[4]
        + source[5]
        + completion[3]
        + completion[4]
        + completion[5]
        + link_errors
    )
    if hard_failures or not all(result["count_reconciliation"].values()):
        raise RuntimeError(f"Parquet validation failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--source-date", default="2026-07-23")
    parser.add_argument("--part-completion-rows", type=int, default=500_000)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-content-sha",
        action="store_true",
        help=(
            "Skip raw and derived content SHA calculation. The checkpoint "
            "fingerprint uses input name, bytes, and --input-provenance-id."
        ),
    )
    parser.add_argument(
        "--input-provenance-id",
        default=None,
        help="Stable vendor/publication identity used when content SHA is skipped.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.skip_content_sha and not parsed.input_provenance_id:
        raise SystemExit("--input-provenance-id is required with --skip-content-sha")
    result = run(parsed)
    print(
        json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2)
    )
