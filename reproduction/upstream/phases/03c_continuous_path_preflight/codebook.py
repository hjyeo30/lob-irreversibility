#!/usr/bin/env python3
"""NVDA MBO를 한 번 replay하며 여러 event-count block을 함께 만든다.

Phase 3A의 order episode와 대규모 event-context table을 반복하지 않는다.
한 날짜의 ordered raw stream은 한 process가 순차 처리하고, symbol-day 간
병렬성은 ``run_batch.py``가 담당한다.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import pickle
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as wall_time
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

PHASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PHASE_DIR.parents[1]
PHASE2_CODE = PROJECT_DIR / "phases" / "02_book_reconstruction_features" / "codebook.py"
SPEC = importlib.util.spec_from_file_location("phase2_replay_contract_03c", PHASE2_CODE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load replay contract: {PHASE2_CODE}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

SCHEMA_VERSION = "continuous_blocks_v1"
DATASET = "XNAS.ITCH"
LAST = 128
MAYBE_BAD_BOOK = 4
BAD_TS_RECV = 8
SNAPSHOT = 32
NY = ZoneInfo("America/New_York")
DEFAULT_RAW_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/raw/databento/xnas_itch/mbo/symbol=NVDA"
)
DEFAULT_DERIVED_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/derived/continuous_blocks_v1"
)


def scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.stream.close()


def imbalance(bid: int, ask: int) -> float:
    denominator = bid + ask
    return (bid - ask) / denominator if denominator else 0.0


def book_state(book: Any, tick_size_nano: int) -> dict[str, Any]:
    bid_prices, bid_sizes = book.top("B")
    ask_prices, ask_sizes = book.top("A")
    bid = bid_prices[0]
    ask = ask_prices[0]
    if bid is None or ask is None:
        return {
            "complete": False,
            "best_bid_nano": bid,
            "best_ask_nano": ask,
            "midprice_x2_nano": None,
            "spread_ticks": None,
            "bid_depth_l1": 0,
            "ask_depth_l1": 0,
            "bid_depth_l5": sum(bid_sizes[:5]),
            "ask_depth_l5": sum(ask_sizes[:5]),
            "imbalance_l1": 0.0,
            "imbalance_l5": imbalance(sum(bid_sizes[:5]), sum(ask_sizes[:5])),
            "bid_top": {},
            "ask_top": {},
        }
    spread = ask - bid
    return {
        "complete": True,
        "best_bid_nano": bid,
        "best_ask_nano": ask,
        "midprice_x2_nano": bid + ask,
        "spread_ticks": spread / tick_size_nano,
        "bid_depth_l1": bid_sizes[0],
        "ask_depth_l1": ask_sizes[0],
        "bid_depth_l5": sum(bid_sizes[:5]),
        "ask_depth_l5": sum(ask_sizes[:5]),
        "imbalance_l1": imbalance(bid_sizes[0], ask_sizes[0]),
        "imbalance_l5": imbalance(sum(bid_sizes[:5]), sum(ask_sizes[:5])),
        "bid_top": {
            price: size
            for price, size in zip(bid_prices, bid_sizes)
            if price is not None
        },
        "ask_top": {
            price: size
            for price, size in zip(ask_prices, ask_sizes)
            if price is not None
        },
    }


def state_columns(prefix: str, state: dict[str, Any]) -> dict[str, Any]:
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


def transition(pre: dict[str, Any], post: dict[str, Any], tick: int) -> dict[str, int]:
    if not pre["complete"] or not post["complete"]:
        return {
            name: 0
            for name in (
                "bid_l1_abs",
                "ask_l1_abs",
                "bid_l5_abs",
                "ask_l5_abs",
                "mid_abs_half_ticks",
                "spread_abs_ticks",
                "bid_depletion",
                "ask_depletion",
                "bid_appearance",
                "ask_appearance",
            )
        }
    result = {
        "bid_l1_abs": abs(post["bid_depth_l1"] - pre["bid_depth_l1"]),
        "ask_l1_abs": abs(post["ask_depth_l1"] - pre["ask_depth_l1"]),
        "bid_l5_abs": abs(post["bid_depth_l5"] - pre["bid_depth_l5"]),
        "ask_l5_abs": abs(post["ask_depth_l5"] - pre["ask_depth_l5"]),
        "mid_abs_half_ticks": abs(post["midprice_x2_nano"] - pre["midprice_x2_nano"])
        // tick,
        "spread_abs_ticks": int(round(abs(post["spread_ticks"] - pre["spread_ticks"]))),
        "bid_depletion": 0,
        "ask_depletion": 0,
        "bid_appearance": 0,
        "ask_appearance": 0,
    }
    for side, label in (("bid", "B"), ("ask", "A")):
        pre_best = pre[f"best_{side}_nano"]
        post_best = post[f"best_{side}_nano"]
        pre_map = pre[f"{side}_top"]
        post_map = post[f"{side}_top"]
        result[f"{side}_depletion"] = int(
            pre_best is not None
            and pre_map.get(pre_best, 0) > 0
            and post_map.get(pre_best, 0) == 0
        )
        result[f"{side}_appearance"] = int(
            post_best is not None
            and post_map.get(post_best, 0) > 0
            and pre_map.get(post_best, 0) == 0
        )
    return result


BLOCK_SCHEMA = pa.schema(
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
        b"schema_version": SCHEMA_VERSION.encode(),
        b"scientific_unit": b"completion-closed source-event-count block",
        b"price_unit": b"1e-9 currency units",
    },
)


@dataclass
class BlockAccumulator:
    target: int
    tick: int
    symbol: str
    source_date: date
    block_index: int = 0
    start_state: dict[str, Any] | None = None
    start_ts: int | None = None
    first_source_index: int | None = None
    last_source_index: int | None = None
    first_raw_ordinal: int | None = None
    last_raw_ordinal: int | None = None
    count: int = 0
    flags_or: int = 0
    sums: Counter[str] = field(default_factory=Counter)

    def initialize(self, state: dict[str, Any], ts: int) -> None:
        self.start_state = state
        self.start_ts = ts

    def add(
        self,
        source: dict[str, Any],
        pre: dict[str, Any],
        post: dict[str, Any],
        metrics: dict[str, int],
        same_price_refills: int,
    ) -> dict[str, Any] | None:
        if self.start_state is None or self.start_ts is None:
            return None
        if self.first_source_index is None:
            self.first_source_index = source["source_event_index"]
            self.first_raw_ordinal = source["raw_ordinal_start"]
        self.last_source_index = source["source_event_index"]
        self.last_raw_ordinal = source["raw_ordinal_end"]
        self.count += 1
        self.flags_or |= source["flags_or"]
        for key, value in metrics.items():
            self.sums[key] += value
        self.sums["same_price_refill_count"] += same_price_refills
        family = source["event_family"]
        family_key = (
            family
            if family in {"add", "cancel", "replace", "displayed_execution"}
            else "other"
        )
        self.sums[f"cause_{family_key}_count"] += 1
        if self.count < self.target or not source["has_f_last"]:
            return None
        row = self.finish(post, source["ts_event_end"])
        self.initialize(post, source["ts_event_end"])
        return row

    def finish(self, end: dict[str, Any], end_ts: int) -> dict[str, Any]:
        assert self.start_state is not None and self.start_ts is not None
        start = self.start_state
        if not start["complete"] or not end["complete"]:
            raise RuntimeError("incomplete block boundary")
        dep = self.sums["bid_depletion"] + self.sums["ask_depletion"]
        app = self.sums["bid_appearance"] + self.sums["ask_appearance"]
        mid_start = start["midprice_x2_nano"]
        mid_end = end["midprice_x2_nano"]
        row = {
            "dataset": DATASET,
            "symbol": self.symbol,
            "source_date": self.source_date,
            "target_b": self.target,
            "block_index": self.block_index,
            "first_source_event_index": self.first_source_index,
            "last_source_event_index": self.last_source_index,
            "first_raw_ordinal": self.first_raw_ordinal,
            "last_raw_ordinal": self.last_raw_ordinal,
            "realized_source_event_count": self.count,
            "completion_overshoot": self.count - self.target,
            "block_start_ns": self.start_ts,
            "block_end_ns": end_ts,
            "elapsed_ns": end_ts - self.start_ts,
            "flags_or": self.flags_or,
            **state_columns("start", start),
            **state_columns("end", end),
            "bid_depth_l1_net_change": end["bid_depth_l1"] - start["bid_depth_l1"],
            "ask_depth_l1_net_change": end["ask_depth_l1"] - start["ask_depth_l1"],
            "bid_depth_l5_net_change": end["bid_depth_l5"] - start["bid_depth_l5"],
            "ask_depth_l5_net_change": end["ask_depth_l5"] - start["ask_depth_l5"],
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
            "mid_log_return": math.log(mid_end / mid_start),
            "mid_change_half_ticks": (mid_end - mid_start) // self.tick,
            "bid_quote_change_ticks": (end["best_bid_nano"] - start["best_bid_nano"])
            // self.tick,
            "ask_quote_change_ticks": (end["best_ask_nano"] - start["best_ask_nano"])
            // self.tick,
            "mid_abs_variation_half_ticks": self.sums["mid_abs_half_ticks"],
            "spread_abs_variation_ticks": self.sums["spread_abs_ticks"],
            "same_price_refill_count": self.sums["same_price_refill_count"],
            "cause_add_count": self.sums["cause_add_count"],
            "cause_cancel_count": self.sums["cause_cancel_count"],
            "cause_replace_count": self.sums["cause_replace_count"],
            "cause_displayed_execution_count": self.sums[
                "cause_displayed_execution_count"
            ],
            "cause_other_count": self.sums["cause_other_count"],
        }
        self.block_index += 1
        self.first_source_index = self.last_source_index = None
        self.first_raw_ordinal = self.last_raw_ordinal = None
        self.count = self.flags_or = 0
        self.sums = Counter()
        return row


def write_part(
    root: Path, part: int, buffers: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for target, rows in buffers.items():
        if not rows:
            continue
        directory = root / f"target_b={target}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{part:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=BLOCK_SCHEMA)
        pq.write_table(table, path, compression="zstd", row_group_size=100_000)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "rows": table.num_rows,
                "bytes": path.stat().st_size,
            }
        )
    return files


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_date = date.fromisoformat(args.source_date)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    final_root = (
        args.derived_root.resolve()
        / f"symbol={args.symbol.upper()}"
        / f"source_date={args.source_date}"
    )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        if args.skip_existing:
            return json.loads(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
        raise FileExistsError(final_root)
    lock_path = final_root.parent / f".{final_root.name}.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"partition already running: {final_root}") from exc

    staging = final_root.parent / f".{final_root.name}.staging"
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint_path = staging / "checkpoint.pkl"
    state_path = staging / "checkpoint_state.json"
    log_path = (
        PHASE_DIR
        / "artifacts"
        / "logs"
        / f"{args.symbol.lower()}_{args.source_date}_{args.log_label}.jsonl"
    )
    logger = Logger(log_path)
    core_start = int(
        datetime.combine(source_date, wall_time(10, 0), tzinfo=NY).timestamp() * 1e9
    )
    core_end = int(
        datetime.combine(source_date, wall_time(15, 30), tzinfo=NY).timestamp() * 1e9
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "symbol": args.symbol.upper(),
        "source_date": args.source_date,
        "block_widths": list(args.block_widths),
        "tick_size_nano": args.tick_size_nano,
        "core_start_ns": core_start,
        "core_end_ns": core_end,
    }
    logger.log(
        "preflight",
        **config,
        input_bytes=input_path.stat().st_size,
        memory_available=psutil.virtual_memory().available,
        disk_free=shutil.disk_usage(staging).free,
        cpu_count=psutil.cpu_count(),
    )

    book = BASE.Book()
    accumulators = {
        target: BlockAccumulator(
            target, args.tick_size_nano, args.symbol.upper(), source_date
        )
        for target in args.block_widths
    }
    depleted_prices: dict[str, set[int]] = {"B": set(), "A": set()}
    raw_cursor = source_index = part = 0
    buffers = {target: [] for target in args.block_widths}
    files: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    last_sequence: dict[tuple[int, int, int], int] = {}

    if checkpoint_path.exists() and state_path.exists() and not args.no_resume:
        meta = json.loads(state_path.read_text(encoding="utf-8"))
        if meta["config"] != config:
            raise RuntimeError("checkpoint configuration mismatch")
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        book = saved["book"]
        accumulators = saved["accumulators"]
        depleted_prices = saved["depleted_prices"]
        raw_cursor = saved["raw_cursor"]
        source_index = saved["source_index"]
        part = saved["part"]
        files = saved["files"]
        counters = saved["counters"]
        last_sequence = saved["last_sequence"]
        logger.log(
            "resume", raw_cursor=raw_cursor, source_events=source_index, next_part=part
        )
    elif any(staging.iterdir()):
        raise RuntimeError(f"staging exists without a usable checkpoint: {staging}")

    current: list[Any] = []
    current_errors: list[str] = []
    current_pre: dict[str, Any] | None = None
    last_log = started = time.monotonic()
    next_checkpoint = source_index + args.checkpoint_source_events
    core_initialized = all(acc.start_state is not None for acc in accumulators.values())
    reached_core_end = False

    def save_checkpoint(cursor_after: int) -> None:
        nonlocal buffers, files, part, next_checkpoint
        new_files = write_part(staging, part, buffers)
        if new_files:
            files.extend(new_files)
            part += 1
        buffers = {target: [] for target in args.block_widths}
        saved = {
            "book": book,
            "accumulators": accumulators,
            "depleted_prices": depleted_prices,
            "raw_cursor": cursor_after,
            "source_index": source_index,
            "part": part,
            "files": files,
            "counters": counters,
            "last_sequence": last_sequence,
        }
        temporary = checkpoint_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
        atomic_json(
            state_path, {"config": config, "raw_cursor": cursor_after, "part": part}
        )
        next_checkpoint = source_index + args.checkpoint_source_events
        logger.log(
            "checkpoint",
            raw_cursor=cursor_after,
            source_events=source_index,
            blocks={str(k): v.block_index for k, v in accumulators.items()},
            rss_bytes=psutil.Process().memory_info().rss,
            disk_free=shutil.disk_usage(staging).free,
        )

    def finalize_source() -> None:
        nonlocal current, current_errors, current_pre, source_index
        nonlocal core_initialized, reached_core_end
        if not current or current_pre is None:
            return
        source = BASE.classify_source(
            current, source_index, args.symbol.upper(), current_errors
        )
        stream_key = (
            source["publisher_id"],
            source["channel_id"],
            source["instrument_id"],
        )
        previous = last_sequence.get(stream_key)
        if previous is not None and source["sequence"] <= previous:
            raise RuntimeError(
                f"nonmonotone source sequence at {source_index}: "
                f"{previous} -> {source['sequence']}"
            )
        last_sequence[stream_key] = source["sequence"]
        post = book_state(book, args.tick_size_nano)
        if source["has_f_last"] and post["complete"]:
            spread_nano = post["best_ask_nano"] - post["best_bid_nano"]
            if spread_nano <= 0:
                raise RuntimeError(
                    f"locked/crossed completion state at source {source_index}"
                )
        ts = source["ts_event_end"]
        if ts >= core_end:
            reached_core_end = True
        elif ts >= core_start:
            if not core_initialized:
                if source["has_f_last"] and post["complete"]:
                    for accumulator in accumulators.values():
                        accumulator.initialize(post, ts)
                    core_initialized = True
                    counters["core_warmup_source_events"] += 1
            else:
                metrics = transition(current_pre, post, args.tick_size_nano)
                same_price_refills = 0
                for side in ("B", "A"):
                    dep_key = "bid_depletion" if side == "B" else "ask_depletion"
                    if metrics[dep_key]:
                        price = current_pre[
                            "best_bid_nano" if side == "B" else "best_ask_nano"
                        ]
                        if price is not None:
                            depleted_prices[side].add(price)
                    for row in current:
                        if (
                            row.action == "A"
                            and row.side == side
                            and row.price in depleted_prices[side]
                        ):
                            if book._levels(side).get(row.price, 0) > 0:
                                depleted_prices[side].discard(row.price)
                                same_price_refills += 1
                for target, accumulator in accumulators.items():
                    row = accumulator.add(
                        source, current_pre, post, metrics, same_price_refills
                    )
                    if row is not None:
                        buffers[target].append(row)
                counters["core_source_events"] += 1
                counters[f"family:{source['event_family']}"] += 1
        counters["source_events"] += 1
        counters["source_invariant_rows"] += bool(source["source_invariant_count"])
        source_index += 1
        current = []
        current_errors = []
        current_pre = None

    try:
        store = db.DBNStore.from_file(input_path)
        for ordinal, native in enumerate(store):
            if ordinal < raw_cursor:
                continue
            raw = BASE.Raw.from_record(ordinal, native)
            if current and raw.key != current[0].key:
                finalize_source()
                if reached_core_end:
                    break
                if source_index >= next_checkpoint and current == []:
                    save_checkpoint(ordinal)
            if not current:
                current_pre = book_state(book, args.tick_size_nano)
            errors = book.apply(raw)
            fatal = [
                code
                for code in errors
                if code
                in {
                    "duplicate_active_order_id",
                    "invalid_add",
                    "unknown_cancel",
                    "cancel_exceeds_remaining",
                    "unknown_modify",
                    "invalid_modify",
                    "unknown_action",
                    "negative_price_level_depth",
                }
            ]
            if fatal:
                raise RuntimeError(f"book invariant at raw {ordinal}: {fatal}")
            current_errors.extend(errors)
            current.append(raw)
            counters["raw_records"] += 1
            now = time.monotonic()
            if now - last_log >= args.progress_seconds:
                elapsed = now - started
                logger.log(
                    "progress",
                    raw_ordinal=ordinal,
                    source_events=source_index,
                    core_source_events=counters["core_source_events"],
                    blocks={str(k): v.block_index for k, v in accumulators.items()},
                    records_per_second=max(0, ordinal + 1 - raw_cursor)
                    / max(elapsed, 1e-9),
                    rss_bytes=psutil.Process().memory_info().rss,
                    disk_free=shutil.disk_usage(staging).free,
                )
                last_log = now
        if not reached_core_end:
            finalize_source()
        save_checkpoint(counters["raw_records"])
    except Exception as exc:
        logger.log("failed", error_type=type(exc).__name__, error=str(exc))
        logger.close()
        raise

    elapsed = time.monotonic() - started
    file_rows = Counter()
    for item in files:
        width = Path(item["path"]).parent.name
        file_rows[width] += item["rows"]
    expected = {f"target_b={k}": v.block_index for k, v in accumulators.items()}
    if dict(file_rows) != expected:
        raise RuntimeError(
            f"block row reconciliation failed: {file_rows} != {expected}"
        )
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "input": {"role": "immutable vendor DBN", "bytes": input_path.stat().st_size},
        "counts": dict(counters),
        "blocks": {str(k): v.block_index for k, v in accumulators.items()},
        "dropped_partial_source_events": {
            str(k): v.count for k, v in accumulators.items()
        },
        "files": files,
        "runtime": {
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "databento": getattr(db, "__version__", "unknown"),
            "peak_or_final_rss_bytes": psutil.Process().memory_info().rss,
        },
        "minimal_qa": {
            "source_sequence_monotone": True,
            "completion_book_uncrossed": True,
            "block_row_reconciliation": True,
            "source_invariant_rows": counters["source_invariant_rows"],
        },
        "progress_log": str(log_path.relative_to(PHASE_DIR)),
    }
    atomic_json(staging / "manifest.json", manifest)
    checkpoint_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    os.replace(staging, final_root)
    logger.log(
        "published",
        final_partition=str(final_root),
        elapsed_seconds=elapsed,
        blocks=manifest["blocks"],
    )
    logger.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--source-date", required=True)
    parser.add_argument("--block-widths", type=int, nargs="+", default=[30, 100, 300])
    parser.add_argument("--tick-size-nano", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-source-events", type=int, default=1_000_000)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--log-label", default=SCHEMA_VERSION)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "date": result["config"]["source_date"],
                "blocks": result["blocks"],
            },
            indent=2,
        )
    )
