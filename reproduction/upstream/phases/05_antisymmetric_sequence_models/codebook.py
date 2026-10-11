#!/usr/bin/env python3
"""Phase 5 antisymmetric neural models, profiling, and production units.

Paths are formed and consumed one trading date at a time.  The production
runner implements the approved four-fold walk-forward selection, final fit,
positive scalar calibration, paired test publication, and reversible null.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
    + tuple(field for pair in POSITIVE_PAIRS for field in pair)
)
INTERVAL_INDEX = {field: index for index, field in enumerate(INTERVAL_FIELDS)}
ODD_INDICES = np.asarray(
    [INTERVAL_INDEX[field] for field in ODD_INTERVAL_FIELDS], dtype=np.int64
)
SWAP_PAIRS = tuple(
    (INTERVAL_INDEX[left], INTERVAL_INDEX[right]) for left, right in POSITIVE_PAIRS
)
IDENTITY_FIELDS = (
    "block_index",
    "block_start_ns",
    "block_end_ns",
)
REQUIRED_COLUMNS = (
    IDENTITY_FIELDS
    + tuple(f"start_{field}" for field in BOUNDARY_FIELDS)
    + tuple(f"end_{field}" for field in BOUNDARY_FIELDS)
    + INTERVAL_FIELDS
)
TOKEN_DIM = 2 * len(BOUNDARY_FIELDS) + len(INTERVAL_FIELDS)
MODEL_KINDS = ("pooled_mlp", "ordered_tcn")
ADAMW_CANDIDATES = (
    {"learning_rate": 3e-4, "weight_decay": 0.0},
    {"learning_rate": 3e-4, "weight_decay": 1e-4},
    {"learning_rate": 1e-3, "weight_decay": 0.0},
    {"learning_rate": 1e-3, "weight_decay": 1e-4},
)
NULL_SEED = 20_260_729
APPROVED_PATH_TOTALS = {
    "development": 1_751_687,
    "calibration": 671_859,
    "test": 582_767,
}
NULL_ABSOLUTE_CEILING = 0.00063
STOP_REQUESTED = False
DEFAULT_INPUT_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/"
    "derived/continuous_blocks_micro_v1/symbol=NVDA"
)


@dataclass(frozen=True)
class DaySplit:
    development: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class InnerFold:
    fold: int
    train: tuple[str, ...]
    evaluation: str


def chronological_day_split(dates: Sequence[str]) -> DaySplit:
    ordered = tuple(sorted(set(dates)))
    if len(ordered) != 15:
        raise ValueError(f"expected exactly 15 distinct dates, found {len(ordered)}")
    return DaySplit(ordered[:9], ordered[9:12], ordered[12:15])


def expanding_inner_folds(
    development_dates: Sequence[str],
) -> tuple[InnerFold, ...]:
    ordered = tuple(development_dates)
    if len(ordered) != 9:
        raise ValueError("inner walk-forward requires exactly nine development dates")
    return tuple(
        InnerFold(index - 4, ordered[:index], ordered[index]) for index in range(5, 9)
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def input_inventory(root: Path, source_date: str, target_b: int) -> dict:
    files = partition_paths(root, source_date, target_b)
    rows = 0
    entries = []
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        size = path.stat().st_size
        rows += metadata.num_rows
        entries.append(
            {
                "relative_path": str(path.relative_to(root)),
                "rows": metadata.num_rows,
                "bytes": size,
            }
        )
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(entries),
        "rows": rows,
        "bytes": sum(entry["bytes"] for entry in entries),
        "identity": "relative_path+rows+bytes; no full content hash",
        "fingerprint_sha256": sha256_bytes(canonical),
    }


def emit(log_path: Path, event: str, **values: object) -> None:
    row = {
        "timestamp_utc": utc_now(),
        "event": event,
        "rss_bytes": rss_bytes(),
        **values,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    rendered = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[{row['timestamp_utc']}] {event} {rendered}", flush=True)


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
class ChannelScaler:
    boundary_center: np.ndarray
    boundary_scale: np.ndarray
    interval_center: np.ndarray
    interval_scale: np.ndarray
    sample_tokens: int

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        result = tokens.astype(np.float32, copy=True)
        boundary_count = len(BOUNDARY_FIELDS)
        result[:, :, :boundary_count] = (
            result[:, :, :boundary_count] - self.boundary_center
        ) / self.boundary_scale
        result[:, :, boundary_count : boundary_count + len(INTERVAL_FIELDS)] = (
            result[:, :, boundary_count : boundary_count + len(INTERVAL_FIELDS)]
            - self.interval_center
        ) / self.interval_scale
        result[:, :, -boundary_count:] = (
            result[:, :, -boundary_count:] - self.boundary_center
        ) / self.boundary_scale
        return result

    def to_dict(self) -> dict:
        return {
            "boundary_center": self.boundary_center.tolist(),
            "boundary_scale": self.boundary_scale.tolist(),
            "interval_center": self.interval_center.tolist(),
            "interval_scale": self.interval_scale.tolist(),
            "sample_tokens": self.sample_tokens,
            "method": "development-only median/IQR pooled over positions and orientations",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelScaler":
        return cls(
            np.asarray(value["boundary_center"], dtype=np.float32),
            np.asarray(value["boundary_scale"], dtype=np.float32),
            np.asarray(value["interval_center"], dtype=np.float32),
            np.asarray(value["interval_scale"], dtype=np.float32),
            int(value["sample_tokens"]),
        )


def partition_paths(root: Path, source_date: str, target_b: int) -> list[Path]:
    folder = root / f"source_date={source_date}" / f"target_b={target_b}"
    return sorted(folder.glob("*.parquet"))


def discover_dates(root: Path, target_b: int) -> list[str]:
    return [
        folder.name.split("=", 1)[1]
        for folder in sorted(root.glob("source_date=*"))
        if (folder / f"target_b={target_b}").is_dir()
    ]


def read_day(root: Path, source_date: str, target_b: int) -> DayBlocks:
    paths = partition_paths(root, source_date, target_b)
    if not paths:
        raise FileNotFoundError(
            f"no Parquet input for date={source_date}, B={target_b}"
        )
    tables = [
        pq.ParquetFile(path).read(columns=list(REQUIRED_COLUMNS)) for path in paths
    ]
    table = pa.concat_tables(tables, promote_options="default")
    order = np.asarray(table["block_index"], dtype=np.int64)
    position = np.argsort(order, kind="stable")
    order = order[position]
    if len(order) > 1 and np.any(np.diff(order) != 1):
        raise ValueError(f"{source_date}: non-contiguous block_index")

    def vector(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.int64)[position]

    def matrix(names: Sequence[str]) -> np.ndarray:
        values = np.column_stack(
            [np.asarray(table[name], dtype=np.float64)[position] for name in names]
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{source_date}: non-finite input")
        return values

    starts = matrix([f"start_{field}" for field in BOUNDARY_FIELDS])
    ends = matrix([f"end_{field}" for field in BOUNDARY_FIELDS])
    if len(order) > 1 and not np.allclose(ends[:-1], starts[1:], rtol=0.0, atol=1e-6):
        raise ValueError(f"{source_date}: adjacent boundary mismatch")
    return DayBlocks(
        source_date=source_date,
        block_index=order,
        block_start_ns=vector("block_start_ns"),
        block_end_ns=vector("block_end_ns"),
        starts=starts,
        ends=ends,
        intervals=matrix(INTERVAL_FIELDS),
    )


def reverse_continuous(
    boundaries: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reversed_boundaries = boundaries[:, ::-1, :].copy()
    reversed_intervals = intervals[:, ::-1, :].copy()
    reversed_intervals[:, :, ODD_INDICES] *= -1.0
    for left, right in SWAP_PAIRS:
        saved = reversed_intervals[:, :, left].copy()
        reversed_intervals[:, :, left] = reversed_intervals[:, :, right]
        reversed_intervals[:, :, right] = saved
    return reversed_boundaries, reversed_intervals


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def transform_physical(
    boundaries: np.ndarray, intervals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transformed_boundaries = boundaries.astype(np.float64, copy=True)
    transformed_intervals = intervals.astype(np.float64, copy=True)
    for index, field in enumerate(BOUNDARY_FIELDS):
        if field in BOUNDARY_LOG_FIELDS:
            transformed_boundaries[:, :, index] = np.log1p(
                transformed_boundaries[:, :, index]
            )
    for index, field in enumerate(INTERVAL_FIELDS):
        if field in ODD_INTERVAL_FIELDS:
            if field not in ODD_IDENTITY_FIELDS:
                transformed_intervals[:, :, index] = signed_log1p(
                    transformed_intervals[:, :, index]
                )
        else:
            transformed_intervals[:, :, index] = np.log1p(
                transformed_intervals[:, :, index]
            )
    return transformed_boundaries, transformed_intervals


def make_tokens(boundaries: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if boundaries.shape[1] != intervals.shape[1] + 1:
        raise ValueError("boundary/interval path shape mismatch")
    transformed_boundaries, transformed_intervals = transform_physical(
        boundaries, intervals
    )
    return np.concatenate(
        (
            transformed_boundaries[:, :-1, :],
            transformed_intervals,
            transformed_boundaries[:, 1:, :],
        ),
        axis=2,
    )


def path_arrays(
    day: DayBlocks, starts: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = starts[:, None] + np.arange(k, dtype=np.int64)[None, :]
    intervals = day.intervals[offsets]
    boundaries = np.empty((len(starts), k + 1, len(BOUNDARY_FIELDS)), dtype=np.float64)
    boundaries[:, 0, :] = day.starts[starts]
    boundaries[:, 1:, :] = day.ends[offsets]
    reverse_boundaries, reverse_intervals = reverse_continuous(boundaries, intervals)
    forward = make_tokens(boundaries, intervals)
    reverse = make_tokens(reverse_boundaries, reverse_intervals)
    event_count = intervals[:, :, INTERVAL_INDEX["realized_source_event_count"]].sum(
        axis=1
    )
    return forward, reverse, event_count


def fit_channel_scaler(forward: np.ndarray, reverse: np.ndarray) -> ChannelScaler:
    values = np.concatenate((forward, reverse), axis=0)
    boundary_count = len(BOUNDARY_FIELDS)
    boundary_values = np.concatenate(
        (values[:, :, :boundary_count], values[:, :, -boundary_count:]), axis=1
    ).reshape(-1, boundary_count)
    interval_values = values[
        :, :, boundary_count : boundary_count + len(INTERVAL_FIELDS)
    ].reshape(-1, len(INTERVAL_FIELDS))

    def robust(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(matrix, axis=0)
        lower, upper = np.quantile(matrix, (0.25, 0.75), axis=0)
        scale = (upper - lower) / 1.349
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return center.astype(np.float32), scale.astype(np.float32)

    boundary_center, boundary_scale = robust(boundary_values)
    interval_center, interval_scale = robust(interval_values)
    return ChannelScaler(
        boundary_center,
        boundary_scale,
        interval_center,
        interval_scale,
        int(values.shape[0] * values.shape[1]),
    )


class BlockEncoder(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(TOKEN_DIM, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class PooledMLP(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.encoder = BlockEncoder(width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        encoded = self.encoder(values)
        pooled = torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)
        return self.head(pooled).squeeze(-1)


class TemporalResidual(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.normalization = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        update = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        update = self.dropout(F.gelu(self.normalization(update)))
        return values + update


class OrderedTCN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.encoder = BlockEncoder(width)
        self.temporal = nn.ModuleList(
            [TemporalResidual(width, dilation) for dilation in (1, 2, 4)]
        )
        self.head = nn.Sequential(
            nn.Linear(width, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        encoded = self.encoder(values)
        for layer in self.temporal:
            encoded = layer(encoded)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


def build_model(kind: str) -> nn.Module:
    if kind == "pooled_mlp":
        return PooledMLP()
    if kind == "ordered_tcn":
        return OrderedTCN()
    raise ValueError(f"unknown model: {kind}")


def antisymmetric_score(model: nn.Module, forward: Tensor, reverse: Tensor) -> Tensor:
    return model(forward) - model(reverse)


def iter_batches(
    starts: np.ndarray, batch_size: int, seed: int
) -> Iterator[np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(starts)
    for first in range(0, len(shuffled), batch_size):
        yield shuffled[first : first + batch_size]


def day_path_starts(day: DayBlocks, k: int, stride: int = 3) -> np.ndarray:
    available = len(day.block_index) - k + 1
    offset = date.fromisoformat(day.source_date).toordinal() % stride
    return np.arange(offset, available, stride, dtype=np.int64)


def fit_development_scaler(
    input_root: Path,
    dates: Sequence[str],
    target_b: int,
    k: int,
    paths_per_day: int,
) -> ChannelScaler:
    forwards: list[np.ndarray] = []
    reverses: list[np.ndarray] = []
    for source_date in dates:
        day = read_day(input_root, source_date, target_b)
        starts = day_path_starts(day, k)
        count = min(len(starts), paths_per_day)
        positions = np.linspace(0, len(starts) - 1, count, dtype=np.int64)
        forward, reverse, _ = path_arrays(day, starts[positions], k)
        forwards.append(forward)
        reverses.append(reverse)
    return fit_channel_scaler(
        np.concatenate(forwards, axis=0), np.concatenate(reverses, axis=0)
    )


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return sha256_bytes(encoded)


def reversible_null_swap_mask(
    source_date: str, start_block_index: np.ndarray, null_seed: int
) -> np.ndarray:
    """Deterministic 50/50 pair orientation without global shuffling."""
    date_key = np.uint64(date.fromisoformat(source_date).toordinal())
    values = start_block_index.astype(np.uint64, copy=False)
    date_term = np.uint64(
        (int(date_key) * 0xC2B2AE3D27D4EB4F + int(null_seed)) & ((1 << 64) - 1)
    )
    mixed = values * np.uint64(0x9E3779B185EBCA87) + date_term
    mixed ^= mixed >> np.uint64(33)
    mixed *= np.uint64(0xFF51AFD7ED558CCD)
    mixed ^= mixed >> np.uint64(33)
    return (mixed & np.uint64(1)).astype(bool)


def orient_pair(
    forward: np.ndarray,
    reverse: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mask is None or not np.any(mask):
        return forward, reverse
    oriented_forward = forward.copy()
    oriented_reverse = reverse.copy()
    oriented_forward[mask] = reverse[mask]
    oriented_reverse[mask] = forward[mask]
    return oriented_forward, oriented_reverse


def _torch_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }


def _restore_rng_state(value: dict[str, Any]) -> None:
    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])


def _load_checkpoint(path: Path, spec_hash: str, resume: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"checkpoint exists; use --resume: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("spec_hash") != spec_hash:
        raise RuntimeError(f"stale checkpoint spec_hash mismatch: {path}")
    return value


def _batch_seed(seed: int, epoch: int, date_index: int, null: bool) -> int:
    return seed + 1_000_003 * epoch + 10_007 * date_index + (NULL_SEED if null else 0)


def evaluate_loss(
    model: nn.Module,
    input_root: Path,
    dates: Sequence[str],
    target_b: int,
    k: int,
    scaler: ChannelScaler,
    batch_size: int,
    null_seed: int | None = None,
) -> dict[str, Any]:
    model.eval()
    by_day: dict[str, dict[str, float | int]] = {}
    with torch.no_grad():
        for source_date in dates:
            day = read_day(input_root, source_date, target_b)
            starts = day_path_starts(day, k)
            loss_sum = 0.0
            rows = 0
            for first in range(0, len(starts), batch_size):
                selected = starts[first : first + batch_size]
                forward, reverse, _ = path_arrays(day, selected, k)
                mask = (
                    reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = orient_pair(forward, reverse, mask)
                score = antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)),
                    torch.from_numpy(scaler.transform(reverse)),
                )
                loss_sum += float(F.softplus(-score).sum())
                rows += len(selected)
            by_day[source_date] = {
                "paths": rows,
                "paired_log_loss": loss_sum / rows,
            }
    return {
        "by_day": by_day,
        "day_equal_paired_log_loss": float(
            np.mean([row["paired_log_loss"] for row in by_day.values()])
        ),
    }


def train_stage(
    *,
    model_kind: str,
    seed: int,
    input_root: Path,
    train_dates: Sequence[str],
    evaluation_dates: Sequence[str],
    target_b: int,
    k: int,
    scaler: ChannelScaler,
    candidate: dict[str, float],
    max_epochs: int,
    min_epochs: int,
    patience: int,
    fixed_epochs: int | None,
    batch_size: int,
    checkpoint_batches: int,
    checkpoint_path: Path,
    spec_hash: str,
    resume: bool,
    log_path: Path,
    null_seed: int | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], int, float]:
    """Train one fold/final stage with an exact batch cursor."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    model = build_model(model_kind)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate["weight_decay"],
    )
    state = _load_checkpoint(checkpoint_path, spec_hash, resume)
    epoch = 0
    date_cursor = 0
    batch_cursor = 0
    epoch_loss_sum = 0.0
    epoch_rows = 0
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_model_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    if state is not None:
        if canonical_hash(state.get("scaler")) != canonical_hash(scaler.to_dict()):
            raise RuntimeError(f"checkpoint scaler mismatch: {checkpoint_path}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = int(state["cursor"]["epoch"])
        date_cursor = int(state["cursor"]["date_index"])
        batch_cursor = int(state["cursor"]["batch_index"])
        epoch_loss_sum = float(state["epoch_loss_sum"])
        epoch_rows = int(state["epoch_rows"])
        history = state["history"]
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        best_model_state = state["best_model_state"]
        stale_epochs = int(state["stale_epochs"])
        _restore_rng_state(state["rng_state"])
        emit(
            log_path,
            "checkpoint_resumed",
            checkpoint=str(checkpoint_path),
            epoch=epoch + 1,
            date_index=date_cursor,
            batch_index=batch_cursor,
        )

    def checkpoint(next_epoch: int, next_date: int, next_batch: int) -> None:
        atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 1,
                "spec_hash": spec_hash,
                "scaler": scaler.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cursor": {
                    "epoch": next_epoch,
                    "date_index": next_date,
                    "batch_index": next_batch,
                },
                "epoch_loss_sum": epoch_loss_sum,
                "epoch_rows": epoch_rows,
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
                "stale_epochs": stale_epochs,
                "rng_state": _torch_rng_state(),
            },
        )

    epoch_limit = fixed_epochs if fixed_epochs is not None else max_epochs
    while epoch < epoch_limit:
        model.train()
        started = time.monotonic()
        for day_index in range(date_cursor, len(train_dates)):
            source_date = train_dates[day_index]
            day = read_day(input_root, source_date, target_b)
            starts = day_path_starts(day, k)
            shuffled = np.random.default_rng(
                _batch_seed(seed, epoch, day_index, null_seed is not None)
            ).permutation(starts)
            first_batch = batch_cursor if day_index == date_cursor else 0
            total_batches = math.ceil(len(shuffled) / batch_size)
            for batch_index in range(first_batch, total_batches):
                selected = shuffled[
                    batch_index * batch_size : (batch_index + 1) * batch_size
                ]
                forward, reverse, _ = path_arrays(day, selected, k)
                mask = (
                    reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = orient_pair(forward, reverse, mask)
                score = antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)),
                    torch.from_numpy(scaler.transform(reverse)),
                )
                loss = F.softplus(-score).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss_sum += float(loss.detach()) * len(selected)
                epoch_rows += len(selected)
                next_day = day_index
                next_batch = batch_index + 1
                if next_batch >= total_batches:
                    next_day = day_index + 1
                    next_batch = 0
                if (batch_index + 1) % checkpoint_batches == 0 or STOP_REQUESTED:
                    checkpoint(epoch, next_day, next_batch)
                if STOP_REQUESTED:
                    emit(
                        log_path,
                        "interrupted",
                        epoch=epoch + 1,
                        date_index=next_day,
                        batch_index=next_batch,
                    )
                    raise InterruptedError("stop requested at safe batch boundary")
            batch_cursor = 0
        train_loss = epoch_loss_sum / epoch_rows
        evaluation = (
            evaluate_loss(
                model,
                input_root,
                evaluation_dates,
                target_b,
                k,
                scaler,
                batch_size,
                null_seed=null_seed,
            )
            if evaluation_dates
            else None
        )
        monitor = (
            float(evaluation["day_equal_paired_log_loss"])
            if evaluation is not None
            else train_loss
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_paired_log_loss": train_loss,
                "evaluation": evaluation,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if fixed_epochs is not None or monitor < best_loss - 1e-9:
            best_loss = monitor
            best_epoch = epoch + 1
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        emit(
            log_path,
            "epoch_complete",
            epoch=epoch + 1,
            train_paired_log_loss=round(train_loss, 8),
            evaluation_paired_log_loss=(
                round(monitor, 8) if evaluation is not None else None
            ),
            best_epoch=best_epoch,
            paths=epoch_rows,
        )
        epoch += 1
        date_cursor = 0
        batch_cursor = 0
        epoch_loss_sum = 0.0
        epoch_rows = 0
        checkpoint(epoch, 0, 0)
        if fixed_epochs is None and epoch >= min_epochs and stale_epochs >= patience:
            break
    if best_model_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_model_state)
    return model, history, best_epoch, best_loss


def select_candidate(fold_rows: Sequence[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    if len(fold_rows) != 16:
        raise ValueError("selection requires 4 folds x 4 candidates")
    losses = np.full((4, 4), np.nan, dtype=np.float64)
    epochs = np.zeros((4, 4), dtype=np.int64)
    for row in fold_rows:
        losses[int(row["fold"]) - 1, int(row["candidate_index"])] = float(
            row["best_evaluation_paired_log_loss"]
        )
        epochs[int(row["fold"]) - 1, int(row["candidate_index"])] = int(
            row["best_epoch"]
        )
    if not np.isfinite(losses).all() or np.any(epochs < 1):
        raise ValueError("incomplete fold selection matrix")
    means = losses.mean(axis=0)
    selected = int(np.argmin(means))
    final_epochs = max(6, int(math.ceil(float(np.median(epochs[:, selected])))))
    return selected, {
        "criterion": "mean of four next-day paired-logloss values; folds day-equal",
        "candidate_grid": list(ADAMW_CANDIDATES),
        "fold_losses": losses.tolist(),
        "mean_next_day_paired_log_loss": means.tolist(),
        "best_epochs": epochs.tolist(),
        "selected_candidate_index": selected,
        "selected_candidate": ADAMW_CANDIDATES[selected],
        "final_epoch_rule": "max(6, ceil(median(best_epoch across four folds)))",
        "final_fit_epochs": final_epochs,
    }


def collect_scores_by_day(
    model: nn.Module,
    input_root: Path,
    dates: Sequence[str],
    target_b: int,
    k: int,
    scaler: ChannelScaler,
    batch_size: int,
    null_seed: int | None = None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for source_date in dates:
            day = read_day(input_root, source_date, target_b)
            starts = day_path_starts(day, k)
            pieces: list[np.ndarray] = []
            for first in range(0, len(starts), batch_size):
                selected = starts[first : first + batch_size]
                forward, reverse, _ = path_arrays(day, selected, k)
                mask = (
                    reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = orient_pair(forward, reverse, mask)
                score = antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)),
                    torch.from_numpy(scaler.transform(reverse)),
                )
                pieces.append(score.numpy().astype(np.float64))
            result[source_date] = np.concatenate(pieces)
    return result


def fit_positive_day_equal_calibration(
    scores_by_day: dict[str, np.ndarray],
) -> tuple[float, dict[str, Any]]:
    if not scores_by_day or any(len(scores) == 0 for scores in scores_by_day.values()):
        raise ValueError("calibration requires nonempty scores for every day")

    def derivative(alpha: float) -> float:
        return float(
            np.mean(
                [
                    np.mean(
                        -scores / (1.0 + np.exp(np.clip(alpha * scores, -60.0, 60.0)))
                    )
                    for scores in scores_by_day.values()
                ]
            )
        )

    lower_derivative = derivative(0.0)
    at_lower_boundary = lower_derivative >= 0.0
    at_upper_boundary = False
    upper = 1.0
    if at_lower_boundary:
        alpha = 0.0
        upper_derivative = lower_derivative
        iterations = 0
    else:
        upper_derivative = derivative(upper)
        while upper_derivative < 0.0 and upper < 1_000_000.0:
            upper *= 2.0
            upper_derivative = derivative(upper)
        if upper_derivative < 0.0:
            alpha = upper
            at_upper_boundary = True
            iterations = 0
        else:
            lower = 0.0
            iterations = 0
            for iterations in range(1, 101):
                midpoint = 0.5 * (lower + upper)
                value = derivative(midpoint)
                if value < 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
                if upper - lower <= 1e-10 * max(1.0, upper):
                    break
            alpha = 0.5 * (lower + upper)
    by_day_loss = {
        source_date: float(
            np.logaddexp(0.0, -np.clip(alpha * scores, -80.0, 80.0)).mean()
        )
        for source_date, scores in scores_by_day.items()
    }
    return alpha, {
        "criterion": (
            "alpha>=0 convex minimization of day-equal mean per-day paired "
            "log loss by derivative bracketing and bisection"
        ),
        "alpha": alpha,
        "at_lower_boundary_alpha_zero": at_lower_boundary,
        "at_upper_numerical_boundary": at_upper_boundary,
        "upper_bracket": upper,
        "derivative_at_zero": lower_derivative,
        "derivative_at_upper_bracket": upper_derivative,
        "bisection_iterations": iterations,
        "day_equal_paired_log_loss": float(np.mean(list(by_day_loss.values()))),
        "by_day": {
            source_date: {
                "paths": len(scores),
                "paired_log_loss": by_day_loss[source_date],
                "mean_calibrated_score": float((alpha * scores).mean()),
            }
            for source_date, scores in scores_by_day.items()
        },
    }


def _metric_accumulator() -> dict[str, Any]:
    return {
        "paths": 0,
        "score_sum": 0.0,
        "event_sum": 0.0,
        "seconds_sum": 0.0,
        "loss_sum": 0.0,
        "brier_sum": 0.0,
        "correct_sum": 0.0,
        "saturation_sum": 0.0,
        "reverse_like_sum": 0.0,
        "overlap_sum": 0.0,
        "score_sample": [],
    }


def _update_metrics(
    accumulator: dict[str, Any],
    score: np.ndarray,
    event_count: np.ndarray,
    duration_seconds: np.ndarray,
) -> None:
    clipped = np.clip(score, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-clipped))
    accumulator["paths"] += len(score)
    accumulator["score_sum"] += float(score.sum())
    accumulator["event_sum"] += float(event_count.sum())
    accumulator["seconds_sum"] += float(duration_seconds.sum())
    accumulator["loss_sum"] += float(np.logaddexp(0.0, -clipped).sum())
    accumulator["brier_sum"] += float(np.square(1.0 - probability).sum())
    accumulator["correct_sum"] += float(
        np.count_nonzero(score > 0) + 0.5 * np.count_nonzero(score == 0)
    )
    accumulator["saturation_sum"] += int(
        np.count_nonzero((probability < 0.01) | (probability > 0.99))
    )
    accumulator["reverse_like_sum"] += int(np.count_nonzero(score < 0))
    accumulator["overlap_sum"] += float((1.0 - np.abs(2.0 * probability - 1.0)).sum())
    sampled = sum(len(part) for part in accumulator["score_sample"])
    if sampled < 200_000:
        accumulator["score_sample"].append(score[: 200_000 - sampled].copy())


def _finalize_metrics(accumulator: dict[str, Any], k: int) -> dict[str, Any]:
    rows = accumulator["paths"]
    sample = np.concatenate(accumulator["score_sample"])
    return {
        "paths": rows,
        "nats_path": accumulator["score_sum"] / rows,
        "nats_block": accumulator["score_sum"] / (rows * k),
        "nats_source_event": accumulator["score_sum"] / accumulator["event_sum"],
        "nats_second": accumulator["score_sum"] / accumulator["seconds_sum"],
        "paired_log_loss": accumulator["loss_sum"] / rows,
        "brier": accumulator["brier_sum"] / rows,
        "accuracy": accumulator["correct_sum"] / rows,
        "posterior_saturation_share": accumulator["saturation_sum"] / rows,
        "reverse_like_share": accumulator["reverse_like_sum"] / rows,
        "calibrated_ovl": accumulator["overlap_sum"] / rows,
        "score_q01": float(np.quantile(sample, 0.01)),
        "score_q50": float(np.quantile(sample, 0.50)),
        "score_q99": float(np.quantile(sample, 0.99)),
        "source_event_sum": accumulator["event_sum"],
        "duration_seconds_sum": accumulator["seconds_sum"],
    }


def test_predictions_and_metrics(
    *,
    model: nn.Module,
    input_root: Path,
    dates: Sequence[str],
    target_b: int,
    k: int,
    scaler: ChannelScaler,
    alpha: float,
    batch_size: int,
    prediction_path: Path | None,
    model_kind: str,
    seed: int,
    null_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    writer: pq.ParquetWriter | None = None
    temporary = (
        prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
        if prediction_path is not None
        else None
    )
    pooled = _metric_accumulator()
    by_day: dict[str, dict[str, Any]] = {}
    max_antisymmetry_error = 0.0
    max_probability_pair_error = 0.0
    permutation_error: float | None = None
    model.eval()
    try:
        with torch.no_grad():
            for source_date in dates:
                day = read_day(input_root, source_date, target_b)
                starts = day_path_starts(day, k)
                day_metric = _metric_accumulator()
                for first in range(0, len(starts), batch_size):
                    selected = starts[first : first + batch_size]
                    forward, reverse, event_count = path_arrays(day, selected, k)
                    mask = (
                        reversible_null_swap_mask(source_date, selected, null_seed)
                        if null_seed is not None
                        else None
                    )
                    forward, reverse = orient_pair(forward, reverse, mask)
                    forward_tensor = torch.from_numpy(scaler.transform(forward))
                    reverse_tensor = torch.from_numpy(scaler.transform(reverse))
                    raw_tensor = antisymmetric_score(
                        model, forward_tensor, reverse_tensor
                    )
                    reverse_tensor_score = antisymmetric_score(
                        model, reverse_tensor, forward_tensor
                    )
                    raw_score = raw_tensor.numpy().astype(np.float64)
                    calibrated = alpha * raw_score
                    reverse_calibrated = alpha * reverse_tensor_score.numpy()
                    probability = 1.0 / (
                        1.0 + np.exp(-np.clip(calibrated, -40.0, 40.0))
                    )
                    reverse_probability = 1.0 / (
                        1.0 + np.exp(-np.clip(reverse_calibrated, -40.0, 40.0))
                    )
                    max_antisymmetry_error = max(
                        max_antisymmetry_error,
                        float(np.max(np.abs(calibrated + reverse_calibrated))),
                    )
                    max_probability_pair_error = max(
                        max_probability_pair_error,
                        float(np.max(np.abs(probability + reverse_probability - 1.0))),
                    )
                    if model_kind == "pooled_mlp" and permutation_error is None:
                        permutation = torch.arange(k - 1, -1, -1)
                        permutation_error = float(
                            torch.max(
                                torch.abs(
                                    model(forward_tensor)
                                    - model(forward_tensor[:, permutation, :])
                                )
                            )
                        )
                    path_start_ns = day.block_start_ns[selected]
                    path_end_ns = day.block_end_ns[selected + k - 1]
                    duration_seconds = (path_end_ns - path_start_ns) / 1e9
                    _update_metrics(pooled, calibrated, event_count, duration_seconds)
                    _update_metrics(
                        day_metric, calibrated, event_count, duration_seconds
                    )
                    if prediction_path is not None:
                        count = len(selected)
                        pair_ids = [
                            f"{source_date}:B{target_b}:K{k}:{int(day.block_index[index])}"
                            for index in selected
                        ]
                        table = pa.table(
                            {
                                "pair_id": pair_ids + pair_ids,
                                "source_date": [source_date] * (2 * count),
                                "start_block_index": np.concatenate(
                                    (
                                        day.block_index[selected],
                                        day.block_index[selected],
                                    )
                                ),
                                "path_start_ns": np.tile(path_start_ns, 2),
                                "path_end_ns": np.tile(path_end_ns, 2),
                                "path_event_count": np.tile(
                                    event_count.astype(np.int64), 2
                                ),
                                "duration_seconds": np.tile(duration_seconds, 2),
                                "target_b": np.full(2 * count, target_b, np.int16),
                                "k": np.full(2 * count, k, np.int16),
                                "model": [model_kind] * (2 * count),
                                "seed": np.full(2 * count, seed, np.int64),
                                "orientation": ["forward"] * count
                                + ["reverse"] * count,
                                "label": np.concatenate(
                                    (
                                        np.ones(count, dtype=np.uint8),
                                        np.zeros(count, dtype=np.uint8),
                                    )
                                ),
                                "raw_antisymmetric_score": np.concatenate(
                                    (raw_score, -raw_score)
                                ),
                                "calibrated_antisymmetric_score": np.concatenate(
                                    (calibrated, -calibrated)
                                ),
                                "probability_forward": np.concatenate(
                                    (probability, 1.0 - probability)
                                ),
                            }
                        )
                        if writer is None:
                            prediction_path.parent.mkdir(parents=True, exist_ok=True)
                            writer = pq.ParquetWriter(
                                temporary, table.schema, compression="zstd"
                            )
                        writer.write_table(table)
                by_day[source_date] = _finalize_metrics(day_metric, k)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, prediction_path)
        day_equal_keys = (
            "nats_path",
            "nats_block",
            "nats_source_event",
            "nats_second",
            "paired_log_loss",
            "brier",
            "accuracy",
            "posterior_saturation_share",
            "reverse_like_share",
            "calibrated_ovl",
        )
        metrics = {
            "pooled_path_weighted": _finalize_metrics(pooled, k),
            "by_day": by_day,
            "day_equal": {
                key: float(np.mean([row[key] for row in by_day.values()]))
                for key in day_equal_keys
            },
        }
        audit = {
            "tolerance": 1e-5,
            "score_antisymmetry_max_abs_error": max_antisymmetry_error,
            "probability_pair_max_abs_error": max_probability_pair_error,
            "permutation_max_abs_error": permutation_error,
            "finite_metrics": bool(
                np.isfinite(list(metrics["day_equal"].values())).all()
            ),
        }
        audit["passed"] = bool(
            audit["finite_metrics"]
            and max_antisymmetry_error <= audit["tolerance"]
            and max_probability_pair_error <= audit["tolerance"]
            and (permutation_error is None or permutation_error <= audit["tolerance"])
        )
        return metrics, audit
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _handle_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[{utc_now()}] stop_requested signal={signum}", flush=True)


def _expected_dates() -> tuple[str, ...]:
    return (
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


def preflight_path_totals(
    inventories: dict[str, dict[str, Any]],
    split: DaySplit,
    k: int,
    stride: int,
) -> dict[str, int]:
    def count_for_dates(dates: Sequence[str]) -> int:
        total = 0
        for source_date in dates:
            available = int(inventories[source_date]["rows"]) - k + 1
            offset = date.fromisoformat(source_date).toordinal() % stride
            if available > offset:
                total += (available - 1 - offset) // stride + 1
        return total

    observed = {
        "development": count_for_dates(split.development),
        "calibration": count_for_dates(split.calibration),
        "test": count_for_dates(split.test),
    }
    if observed != APPROVED_PATH_TOTALS:
        raise RuntimeError(
            f"approved path-total preflight mismatch: "
            f"observed={observed} expected={APPROVED_PATH_TOTALS}"
        )
    return observed


def production_spec(args: argparse.Namespace, dates: Sequence[str]) -> dict[str, Any]:
    inventories = {
        source_date: input_inventory(args.input_root, source_date, args.target_b)
        for source_date in dates
    }
    split = chronological_day_split(dates)
    path_totals = preflight_path_totals(inventories, split, args.k, 3)
    code_path = Path(__file__).resolve()
    lock_path = code_path.parents[4] / "uv.lock"
    input_fingerprint = canonical_hash(
        {
            source_date: row["fingerprint_sha256"]
            for source_date, row in inventories.items()
        }
    )
    scientific = {
        "protocol": "phase5_primary_walk_forward_v1",
        "target_b": args.target_b,
        "k": args.k,
        "model": args.model,
        "seed": args.seed,
        "dates": list(dates),
        "split": asdict(split),
        "approved_path_totals": path_totals,
        "start_stride": 3,
        "start_offset_rule": "date_ordinal_modulo_3",
        "candidate_grid": list(ADAMW_CANDIDATES),
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "scaler_paths_per_day": args.scaler_paths_per_day,
        "null_seed": NULL_SEED,
        "code_sha256": file_sha256(code_path),
        "uv_lock_sha256": file_sha256(lock_path),
        "input_fingerprint_sha256": input_fingerprint,
    }
    return {
        **scientific,
        "input_inventory": inventories,
        "spec_hash": canonical_hash(scientific),
    }


def validate_complete_manifest(
    final_dir: Path,
    manifest: dict[str, Any],
    expected_spec_hash: str,
    expected_dates: Sequence[str],
) -> None:
    expected = {
        "status": "complete",
        "spec_hash": expected_spec_hash,
        "target_b": 10,
        "k": 6,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"invalid complete manifest field {key}: {final_dir}")
    if tuple(manifest.get("approved_dates", ())) != tuple(expected_dates):
        raise RuntimeError(f"invalid complete manifest dates: {final_dir}")
    if not manifest.get("input_fingerprint_sha256"):
        raise RuntimeError(f"complete manifest lacks input fingerprint: {final_dir}")
    if manifest.get("null_bias_gate", {}).get("passed") is not True:
        raise RuntimeError(f"complete manifest lacks passed null gate: {final_dir}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(f"complete manifest has no artifact inventory: {final_dir}")
    for name, metadata in artifacts.items():
        path = final_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or file_sha256(path) != metadata.get("sha256")
        ):
            raise RuntimeError(f"complete manifest artifact mismatch: {path}")


def run_unit(args: argparse.Namespace) -> None:
    unit_started = time.monotonic()
    if args.target_b != 10 or args.k != 6:
        raise ValueError("approved production scope is exactly (B,K)=(10,6)")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dates = tuple(discover_dates(args.input_root, args.target_b))
    if dates != _expected_dates():
        raise ValueError(
            f"input dates do not match exact approved tuple: observed={dates}"
        )
    split = chronological_day_split(dates)
    unit_relative = Path("B10_K6") / args.model / f"seed={args.seed}"
    final_dir = args.output_root / unit_relative
    unit_work = args.work_root / unit_relative
    unit_work.mkdir(parents=True, exist_ok=True)
    lock_handle = (unit_work / ".unit.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"unit already has an active writer: {unit_work}") from error
    try:
        spec = production_spec(args, dates)
        spec_hash = spec["spec_hash"]
        if final_dir.exists():
            manifest_path = final_dir / "complete_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"invalid final unit without manifest: {final_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_complete_manifest(final_dir, manifest, spec_hash, dates)
            if args.resume:
                emit(
                    args.log_dir / f"{args.model}_seed{args.seed}.jsonl",
                    "unit_skipped_complete",
                    spec_hash=spec_hash,
                )
                return
            raise FileExistsError(f"completed unit exists; use --resume: {final_dir}")

        staging = unit_work / "staging_output"
        spec_path = unit_work / "spec.json"
        if spec_path.exists():
            prior_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if prior_spec.get("spec_hash") != spec_hash:
                raise RuntimeError(f"stale unit work spec mismatch: {unit_work}")
            if not args.resume:
                raise FileExistsError(f"unit work exists; use --resume: {unit_work}")
        else:
            atomic_write_json(spec_path, spec)
        staging.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{args.model}_seed{args.seed}.jsonl"
        emit(
            log_path,
            "unit_start",
            model=args.model,
            seed=args.seed,
            spec_hash=spec_hash,
            resume=args.resume,
        )

        scaler_path = unit_work / "development_scaler.json"
        if scaler_path.exists():
            scaler = ChannelScaler.from_dict(
                json.loads(scaler_path.read_text(encoding="utf-8"))
            )
        else:
            scaler = fit_development_scaler(
                args.input_root,
                split.development,
                args.target_b,
                args.k,
                args.scaler_paths_per_day,
            )
            atomic_write_json(scaler_path, scaler.to_dict())
        fold_rows: list[dict[str, Any]] = []
        folds_dir = unit_work / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        for fold in expanding_inner_folds(split.development):
            fold_scaler_path = folds_dir / f"fold={fold.fold}_scaler.json"
            if fold_scaler_path.exists():
                fold_scaler = ChannelScaler.from_dict(
                    json.loads(fold_scaler_path.read_text(encoding="utf-8"))
                )
            else:
                fold_scaler = fit_development_scaler(
                    args.input_root,
                    fold.train,
                    args.target_b,
                    args.k,
                    args.scaler_paths_per_day,
                )
                atomic_write_json(fold_scaler_path, fold_scaler.to_dict())
            for candidate_index, candidate in enumerate(ADAMW_CANDIDATES):
                result_path = folds_dir / (
                    f"fold={fold.fold}_candidate={candidate_index}.json"
                )
                stage_spec_hash = canonical_hash(
                    {
                        "unit_spec_hash": spec_hash,
                        "stage": "selection",
                        "fold": asdict(fold),
                        "candidate_index": candidate_index,
                    }
                )
                if result_path.exists():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    if row.get("stage_spec_hash") != stage_spec_hash:
                        raise RuntimeError(f"stale fold result: {result_path}")
                    fold_rows.append(row)
                    continue
                checkpoint_path = result_path.with_suffix(".checkpoint.pt")
                _model, history, best_epoch, best_loss = train_stage(
                    model_kind=args.model,
                    seed=args.seed,
                    input_root=args.input_root,
                    train_dates=fold.train,
                    evaluation_dates=(fold.evaluation,),
                    target_b=args.target_b,
                    k=args.k,
                    scaler=fold_scaler,
                    candidate=candidate,
                    max_epochs=args.max_epochs,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    fixed_epochs=None,
                    batch_size=args.batch_size,
                    checkpoint_batches=args.checkpoint_batches,
                    checkpoint_path=checkpoint_path,
                    spec_hash=stage_spec_hash,
                    resume=args.resume,
                    log_path=log_path,
                )
                row = {
                    "stage_spec_hash": stage_spec_hash,
                    "fold": fold.fold,
                    "train_dates": list(fold.train),
                    "evaluation_date": fold.evaluation,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "best_epoch": best_epoch,
                    "best_evaluation_paired_log_loss": best_loss,
                    "history": history,
                }
                atomic_write_json(result_path, row)
                checkpoint_path.unlink(missing_ok=True)
                fold_rows.append(row)
        selected_index, selection = select_candidate(fold_rows)
        selected_candidate = ADAMW_CANDIDATES[selected_index]
        atomic_write_json(staging / "selection.json", selection)
        emit(
            log_path,
            "selection_complete",
            candidate_index=selected_index,
            final_fit_epochs=selection["final_fit_epochs"],
        )

        final_stage_hash = canonical_hash(
            {
                "unit_spec_hash": spec_hash,
                "stage": "final_fit",
                "candidate_index": selected_index,
                "epochs": selection["final_fit_epochs"],
            }
        )
        final_model, final_history, _, _ = train_stage(
            model_kind=args.model,
            seed=args.seed,
            input_root=args.input_root,
            train_dates=split.development,
            evaluation_dates=(),
            target_b=args.target_b,
            k=args.k,
            scaler=scaler,
            candidate=selected_candidate,
            max_epochs=selection["final_fit_epochs"],
            min_epochs=selection["final_fit_epochs"],
            patience=args.patience,
            fixed_epochs=selection["final_fit_epochs"],
            batch_size=args.batch_size,
            checkpoint_batches=args.checkpoint_batches,
            checkpoint_path=unit_work / "final_fit.checkpoint.pt",
            spec_hash=final_stage_hash,
            resume=args.resume,
            log_path=log_path,
        )
        calibration_scores = collect_scores_by_day(
            final_model,
            args.input_root,
            split.calibration,
            args.target_b,
            args.k,
            scaler,
            args.batch_size,
        )
        alpha, calibration = fit_positive_day_equal_calibration(calibration_scores)
        prediction_path = staging / "test_predictions.parquet"
        test_metrics, audit = test_predictions_and_metrics(
            model=final_model,
            input_root=args.input_root,
            dates=split.test,
            target_b=args.target_b,
            k=args.k,
            scaler=scaler,
            alpha=alpha,
            batch_size=args.batch_size,
            prediction_path=prediction_path,
            model_kind=args.model,
            seed=args.seed,
        )
        if not audit["passed"]:
            raise RuntimeError(f"scientific audit failed: {audit}")
        atomic_torch_save(
            staging / "model.pt",
            {
                "model_state": final_model.state_dict(),
                "model": args.model,
                "seed": args.seed,
                "spec_hash": spec_hash,
            },
        )
        atomic_write_json(staging / "scaler.json", scaler.to_dict())
        atomic_write_json(staging / "calibration.json", calibration)

        null_stage_hash = canonical_hash(
            {
                "unit_spec_hash": spec_hash,
                "stage": "reversible_null",
                "candidate_index": selected_index,
                "epochs": selection["final_fit_epochs"],
                "null_seed": NULL_SEED,
            }
        )
        null_model, null_history, _, _ = train_stage(
            model_kind=args.model,
            seed=args.seed,
            input_root=args.input_root,
            train_dates=split.development,
            evaluation_dates=(),
            target_b=args.target_b,
            k=args.k,
            scaler=scaler,
            candidate=selected_candidate,
            max_epochs=selection["final_fit_epochs"],
            min_epochs=selection["final_fit_epochs"],
            patience=args.patience,
            fixed_epochs=selection["final_fit_epochs"],
            batch_size=args.batch_size,
            checkpoint_batches=args.checkpoint_batches,
            checkpoint_path=unit_work / "reversible_null.checkpoint.pt",
            spec_hash=null_stage_hash,
            resume=args.resume,
            log_path=log_path,
            null_seed=NULL_SEED,
        )
        null_calibration_scores = collect_scores_by_day(
            null_model,
            args.input_root,
            split.calibration,
            args.target_b,
            args.k,
            scaler,
            args.batch_size,
            null_seed=NULL_SEED,
        )
        null_alpha, null_calibration = fit_positive_day_equal_calibration(
            null_calibration_scores
        )
        null_metrics, null_audit = test_predictions_and_metrics(
            model=null_model,
            input_root=args.input_root,
            dates=split.test,
            target_b=args.target_b,
            k=args.k,
            scaler=scaler,
            alpha=null_alpha,
            batch_size=args.batch_size,
            prediction_path=None,
            model_kind=args.model,
            seed=args.seed,
            null_seed=NULL_SEED,
        )
        real_day_equal_d = float(test_metrics["day_equal"]["nats_path"])
        null_day_equal_d = float(null_metrics["day_equal"]["nats_path"])
        null_bias_threshold = max(NULL_ABSOLUTE_CEILING, 0.10 * abs(real_day_equal_d))
        null_bias_gate = {
            "metric": "abs(reversible_null.test.day_equal.nats_path)",
            "observed_abs": abs(null_day_equal_d),
            "threshold": null_bias_threshold,
            "rule": "abs(D_null_day_equal) <= max(0.00063, 0.10*abs(D_real_day_equal))",
            "absolute_ceiling": NULL_ABSOLUTE_CEILING,
            "absolute_ceiling_rationale": (
                "five-times safety ceiling relative to the Phase 4 maximum "
                "reversible-null abs(D) of 1.26e-4 nats/path"
            ),
            "relative_fraction": 0.10,
            "passed": abs(null_day_equal_d) <= null_bias_threshold,
        }
        null_audit["bias_gate"] = null_bias_gate
        scientific_audit_passed = bool(
            null_audit["passed"] and null_bias_gate["passed"]
        )
        metrics = {
            "schema_version": 1,
            "model": args.model,
            "seed": args.seed,
            "target_b": args.target_b,
            "k": args.k,
            "spec_hash": spec_hash,
            "split": asdict(split),
            "selection": selection,
            "calibration": calibration,
            "test": test_metrics,
            "audit": audit,
            "training_history": final_history,
            "reversible_null": {
                "null_seed": NULL_SEED,
                "calibration": null_calibration,
                "test": null_metrics,
                "audit": null_audit,
                "training_history": null_history,
            },
            "runtime": {
                "elapsed_seconds_before_publication": time.monotonic() - unit_started,
                "peak_rss_bytes": rss_bytes(),
                "threads": args.threads,
                "batch_size": args.batch_size,
            },
        }
        atomic_write_json(staging / "metrics.json", metrics)
        if not scientific_audit_passed:
            failure_manifest = {
                "schema_version": 1,
                "status": "diagnostic_failed",
                "created_utc": utc_now(),
                "model": args.model,
                "seed": args.seed,
                "target_b": args.target_b,
                "k": args.k,
                "spec_hash": spec_hash,
                "input_fingerprint_sha256": spec["input_fingerprint_sha256"],
                "null_bias_gate": null_bias_gate,
                "structural_null_audit_passed": null_audit["passed"],
                "note": (
                    "Scientific completion and atomic final publication are "
                    "blocked; diagnostic artifacts remain under unit work."
                ),
            }
            atomic_write_json(
                unit_work / "diagnostic_failed_manifest.json", failure_manifest
            )
            emit(
                log_path,
                "diagnostic_failed",
                null_abs_d=null_bias_gate["observed_abs"],
                null_threshold=null_bias_threshold,
            )
            raise RuntimeError(
                f"reversible-null scientific gate failed: {null_bias_gate}"
            )
        atomic_write_json(staging / "spec.json", spec)
        artifacts = {}
        for path in sorted(staging.iterdir()):
            if path.name == "complete_manifest.json" or not path.is_file():
                continue
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": utc_now(),
            "model": args.model,
            "seed": args.seed,
            "target_b": args.target_b,
            "k": args.k,
            "spec_hash": spec_hash,
            "input_fingerprint_sha256": spec["input_fingerprint_sha256"],
            "approved_dates": list(dates),
            "split": asdict(split),
            "null_bias_gate": null_bias_gate,
            "artifacts": artifacts,
            "test_prediction_rows": (2 * test_metrics["pooled_path_weighted"]["paths"]),
        }
        atomic_write_json(staging / "complete_manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_dir)
        for checkpoint in unit_work.glob("*.checkpoint.pt"):
            checkpoint.unlink()
        emit(
            log_path,
            "unit_complete",
            output=str(final_dir),
            day_equal_nats_path=test_metrics["day_equal"]["nats_path"],
            null_day_equal_nats_path=null_metrics["day_equal"]["nats_path"],
        )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def profile(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.target_b != 10:
        raise ValueError("current approved profiling scope is B=10")
    dates = discover_dates(args.input_root, args.target_b)
    if len(dates) != 15:
        raise ValueError(f"expected 15 input dates, found {len(dates)}")
    source_date = args.source_date or dates[0]
    if source_date not in dates[:9]:
        raise ValueError("profiling date must be one of the development dates")
    emit(
        args.log_path,
        "profile_start",
        model=args.model,
        source_date=source_date,
        target_b=args.target_b,
        k=args.k,
        max_paths=args.max_paths,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threads=args.threads,
        seed=args.seed,
    )
    read_started = time.monotonic()
    day = read_day(args.input_root, source_date, args.target_b)
    available = len(day.block_index) - args.k + 1
    offset = date.fromisoformat(source_date).toordinal() % 3
    starts = np.arange(offset, available, 3, dtype=np.int64)[: args.max_paths]
    if not len(starts):
        raise ValueError("profiling shard contains no paths")
    scaler_count = min(len(starts), args.scaler_paths)
    positions = np.linspace(0, len(starts) - 1, scaler_count, dtype=np.int64)
    scale_forward, scale_reverse, _ = path_arrays(day, starts[positions], args.k)
    scaler = fit_channel_scaler(scale_forward, scale_reverse)
    emit(
        args.log_path,
        "input_ready",
        blocks=len(day.block_index),
        paths=len(starts),
        files=len(partition_paths(args.input_root, source_date, args.target_b)),
        read_elapsed_seconds=round(time.monotonic() - read_started, 3),
        scaler_tokens=scaler.sample_tokens,
    )

    model = build_model(args.model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epoch_rows: list[int] = []
    epoch_seconds: list[float] = []
    epoch_losses: list[float] = []
    total_started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        started = time.monotonic()
        loss_sum = 0.0
        rows = 0
        batches = 0
        for batch_starts in iter_batches(starts, args.batch_size, args.seed + epoch):
            forward, reverse, _ = path_arrays(day, batch_starts, args.k)
            forward_tensor = torch.from_numpy(scaler.transform(forward))
            reverse_tensor = torch.from_numpy(scaler.transform(reverse))
            score = antisymmetric_score(model, forward_tensor, reverse_tensor)
            loss = F.softplus(-score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch_starts)
            loss_sum += float(loss.detach()) * count
            rows += count
            batches += 1
        elapsed = time.monotonic() - started
        epoch_rows.append(rows)
        epoch_seconds.append(elapsed)
        epoch_losses.append(loss_sum / rows)
        emit(
            args.log_path,
            "epoch_complete",
            model=args.model,
            epoch=epoch + 1,
            batches=batches,
            paths=rows,
            paired_log_loss=round(epoch_losses[-1], 8),
            elapsed_seconds=round(elapsed, 3),
            paths_per_second=round(rows / max(elapsed, 1e-9), 1),
        )

    audit_starts = starts[: min(len(starts), 2_048)]
    forward, reverse, _ = path_arrays(day, audit_starts, args.k)
    forward_tensor = torch.from_numpy(scaler.transform(forward))
    reverse_tensor = torch.from_numpy(scaler.transform(reverse))
    model.eval()
    with torch.no_grad():
        score = antisymmetric_score(model, forward_tensor, reverse_tensor)
        reverse_score = antisymmetric_score(model, reverse_tensor, forward_tensor)
        antisymmetry_error = float(torch.max(torch.abs(score + reverse_score)).item())
        permutation_error = None
        if args.model == "pooled_mlp":
            permutation = torch.randperm(args.k)
            original = model(forward_tensor)
            permuted = model(forward_tensor[:, permutation, :])
            permutation_error = float(torch.max(torch.abs(original - permuted)).item())
    finite = bool(torch.isfinite(score).all().item())
    tolerance = 1e-5
    audit_passed = (
        finite
        and antisymmetry_error <= tolerance
        and (permutation_error is None or permutation_error <= tolerance)
    )
    elapsed_total = time.monotonic() - total_started
    result = {
        "schema_version": 1,
        "status": "profile_complete" if audit_passed else "profile_failed",
        "created_utc": utc_now(),
        "scope": "bounded_development_only",
        "source_date": source_date,
        "input_root": str(args.input_root),
        "input_inventory": input_inventory(args.input_root, source_date, args.target_b),
        "target_b": args.target_b,
        "k": args.k,
        "model": args.model,
        "seed": args.seed,
        "token_shape": [args.batch_size, args.k, TOKEN_DIM],
        "parameter_count": parameter_count,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "epochs": args.epochs,
        "paths": len(starts),
        "epoch_paths_per_second": [
            rows / max(seconds, 1e-9)
            for rows, seconds in zip(epoch_rows, epoch_seconds, strict=True)
        ],
        "epoch_paired_log_loss": epoch_losses,
        "training_elapsed_seconds": elapsed_total,
        "peak_rss_bytes": rss_bytes(),
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "torch": torch.__version__,
            "codebook_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "scaler": scaler.to_dict(),
        "audit": {
            "tolerance": tolerance,
            "finite_scores": finite,
            "score_antisymmetry_max_abs_error": antisymmetry_error,
            "permutation_max_abs_error": permutation_error,
            "passed": audit_passed,
        },
        "note": (
            "This is a resource/architecture profile, not a scientific "
            "held-out density-ratio result."
        ),
    }
    atomic_write_json(args.output_path, result)
    emit(
        args.log_path,
        "profile_complete",
        model=args.model,
        status=result["status"],
        parameter_count=parameter_count,
        peak_rss_bytes=result["peak_rss_bytes"],
        training_elapsed_seconds=round(elapsed_total, 3),
        audit_passed=audit_passed,
        output=str(args.output_path),
    )
    if not audit_passed:
        raise RuntimeError("profiling audit failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    profile_parser.add_argument("--target-b", type=int, default=10)
    profile_parser.add_argument("--k", type=int, default=6)
    profile_parser.add_argument("--model", choices=MODEL_KINDS, required=True)
    profile_parser.add_argument("--source-date")
    profile_parser.add_argument("--max-paths", type=int, default=131_072)
    profile_parser.add_argument("--scaler-paths", type=int, default=5_000)
    profile_parser.add_argument("--batch-size", type=int, default=1_024)
    profile_parser.add_argument("--epochs", type=int, default=2)
    profile_parser.add_argument("--threads", type=int, default=3)
    profile_parser.add_argument("--seed", type=int, default=20_260_730)
    profile_parser.add_argument("--learning-rate", type=float, default=1e-3)
    profile_parser.add_argument("--weight-decay", type=float, default=1e-4)
    profile_parser.add_argument("--output-path", type=Path, required=True)
    profile_parser.add_argument("--log-path", type=Path, required=True)

    run_parser = subparsers.add_parser("run-unit")
    run_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--work-root", type=Path, required=True)
    run_parser.add_argument("--log-dir", type=Path, required=True)
    run_parser.add_argument("--target-b", type=int, default=10)
    run_parser.add_argument("--k", type=int, default=6)
    run_parser.add_argument("--model", choices=MODEL_KINDS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--threads", type=int, default=3)
    run_parser.add_argument("--batch-size", type=int, default=1_024)
    run_parser.add_argument("--scaler-paths-per-day", type=int, default=10_000)
    run_parser.add_argument("--max-epochs", type=int, default=30)
    run_parser.add_argument("--min-epochs", type=int, default=6)
    run_parser.add_argument("--patience", type=int, default=3)
    run_parser.add_argument("--checkpoint-batches", type=int, default=50)
    run_parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "profile":
        profile(args)
    elif args.command == "run-unit":
        if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
            raise ValueError("invalid min/max epoch contract")
        if args.patience < 1 or args.checkpoint_batches < 1:
            raise ValueError("patience and checkpoint interval must be positive")
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
        run_unit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
