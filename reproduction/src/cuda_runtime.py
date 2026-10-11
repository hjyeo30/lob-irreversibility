"""CUDA parity and deterministic-resume helpers for immutable Phase 05 models."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import uuid
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_PROJECT = PROJECT_DIR / "upstream"
UPSTREAM_CODE = (
    UPSTREAM_PROJECT / "phases" / "05_antisymmetric_sequence_models" / "codebook.py"
)
SPEC = importlib.util.spec_from_file_location("upstream_phase05_models", UPSTREAM_CODE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load immutable upstream code: {UPSTREAM_CODE}")
UPSTREAM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UPSTREAM
SPEC.loader.exec_module(UPSTREAM)

MODEL_ROOT = (
    UPSTREAM_PROJECT
    / "phases"
    / "05_antisymmetric_sequence_models"
    / "artifacts"
    / "data"
    / "primary_v1"
    / "B10_K6"
)
INPUT_ROOT = Path(
    "/data/lob_liquidity_time_irreversibility/"
    "derived/continuous_blocks_micro_v1/symbol=NVDA"
)
MODELS = ("pooled_mlp", "ordered_tcn")
SEEDS = (20260730, 20260731, 20260732)


def _date_input_bytes(input_root: Path, source_date: str, target_b: int) -> int:
    paths = UPSTREAM.partition_paths(Path(input_root), source_date, target_b)
    if not paths:
        raise FileNotFoundError(
            f"heartbeat input partition absent: {source_date}/B{target_b}"
        )
    return sum(path.stat().st_size for path in paths)


def _dates_input_bytes(
    input_root: Path, dates: tuple[str, ...], target_b: int
) -> dict[str, int]:
    return {day: _date_input_bytes(input_root, day, target_b) for day in dates}


def _filesystem_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _validated_cuda_score_checkpoint(
    date_root: Path,
    config: Mapping[str, Any],
) -> np.ndarray | None:
    score_path = date_root / "scores.npy"
    complete_path = date_root / "complete.json"
    if not score_path.is_file() and not complete_path.is_file():
        return None
    if not score_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"legacy/partial CUDA score checkpoint: {date_root}")
    saved = json.loads(complete_path.read_text())
    values = np.load(score_path, allow_pickle=False)
    if not (
        saved.get("status") == "complete"
        and saved.get("config") == dict(config)
        and saved.get("score_path") == str(score_path.resolve())
        and saved.get("rows") == len(values)
        and saved.get("bytes") == score_path.stat().st_size
        and saved.get("score_filesystem_identity") == _filesystem_identity(score_path)
        and values.ndim == 1
        and np.isfinite(values).all()
    ):
        raise RuntimeError(f"stale/corrupt CUDA score checkpoint: {date_root}")
    return np.asarray(values, dtype=np.float64)


def _validated_cuda_test_checkpoint(
    date_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    prediction = date_root / "test_predictions.parquet"
    metrics_path = date_root / "metrics.json"
    complete_path = date_root / "complete.json"
    present = (prediction.is_file(), metrics_path.is_file(), complete_path.is_file())
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"legacy/partial CUDA test checkpoint: {date_root}")
    saved = json.loads(complete_path.read_text())
    payload = json.loads(metrics_path.read_text())
    if not (
        saved.get("status") == "complete"
        and saved.get("config") == dict(config)
        and saved.get("prediction_rows") == pq.ParquetFile(prediction).metadata.num_rows
        and saved.get("prediction_bytes") == prediction.stat().st_size
        and saved.get("metrics_bytes") == metrics_path.stat().st_size
        and saved.get("prediction_filesystem_identity")
        == _filesystem_identity(prediction)
        and saved.get("metrics_filesystem_identity")
        == _filesystem_identity(metrics_path)
        and payload.get("audit", {}).get("passed") is True
        and saved.get("audit") == payload.get("audit")
    ):
        raise RuntimeError(f"stale/corrupt CUDA test checkpoint: {date_root}")
    return payload


class CudaHeartbeat:
    """Wall-clock heartbeat that remains active inside long CUDA/library calls."""

    def __init__(
        self,
        log_path: Path,
        stage: str,
        *,
        device: torch.device,
        unit_id: str,
        bytes_total: int,
        interval_seconds: float = 30.0,
        checkpoint_path: Path | None = None,
        total_chunks: int | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.log_path = Path(log_path)
        self.stage = stage
        self.device = device
        self.unit_id = unit_id
        self.bytes_total = int(bytes_total)
        if not self.unit_id or self.bytes_total < 0:
            raise ValueError(
                "heartbeat requires exact unit_id and nonnegative bytes_total"
            )
        self.interval_seconds = interval_seconds
        self.checkpoint_path = checkpoint_path
        self.total_chunks = total_chunks
        self.started = time.monotonic()
        self.progress: dict[str, Any] = {
            "chunk_index": 0,
            "rows_processed": 0,
            "bytes_processed": 0,
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def update(self, **values: Any) -> None:
        with self.lock:
            self.progress.update(values)

    @staticmethod
    def _disk_free(path: Path) -> int:
        probe = path if path.is_dir() else path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free

    def snapshot(self, status: str) -> dict[str, Any]:
        with self.lock:
            values = dict(self.progress)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        chunks = int(values.get("chunk_index", 0))
        total = values.get("chunks_total", self.total_chunks)
        rate = chunks / elapsed
        eta = (
            (int(total) - chunks) / rate
            if total is not None and rate > 0 and chunks <= int(total)
            else None
        )
        row: dict[str, Any] = {
            "status": status,
            "stage": self.stage,
            "unit_id": values.get("unit_id", self.unit_id),
            "elapsed_seconds": elapsed,
            "throughput_chunks_per_second": rate,
            "throughput_rows_per_second": int(values.get("rows_processed", 0))
            / elapsed,
            "bytes_processed": int(values.get("bytes_processed", 0)),
            "bytes_total": self.bytes_total,
            "throughput_bytes_per_second": int(values.get("bytes_processed", 0))
            / elapsed,
            "eta_seconds": eta,
            "rss_bytes": psutil.Process().memory_info().rss,
            "temp_disk_free_bytes": self._disk_free(Path(tempfile.gettempdir())),
            "output_disk_free_bytes": self._disk_free(self.log_path),
            **values,
        }
        if self.checkpoint_path is not None:
            row.update(
                {
                    "checkpoint_path": str(self.checkpoint_path),
                    "checkpoint_exists": self.checkpoint_path.is_file(),
                    "checkpoint_bytes": (
                        self.checkpoint_path.stat().st_size
                        if self.checkpoint_path.is_file()
                        else 0
                    ),
                    "checkpoint_disk_free_bytes": self._disk_free(self.checkpoint_path),
                }
            )
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                free, total_vram = torch.cuda.mem_get_info(self.device)
                row.update(
                    {
                        "vram_allocated_bytes": torch.cuda.memory_allocated(
                            self.device
                        ),
                        "vram_reserved_bytes": torch.cuda.memory_reserved(self.device),
                        "vram_free_bytes": free,
                        "vram_total_bytes": total_vram,
                    }
                )
            except (RuntimeError, AssertionError):
                row["vram_available"] = False
        else:
            row["vram_available"] = False
        UPSTREAM.emit(self.log_path, "runtime_heartbeat", **row)
        return row

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.snapshot("running")

    def __enter__(self) -> "CudaHeartbeat":
        self.snapshot("started")
        self.thread = threading.Thread(
            target=self._run, name=f"heartbeat-{self.stage}", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self.snapshot("failed" if exc_type is not None else "complete")


def fit_positive_day_equal_calibration_with_heartbeat(
    scores: Mapping[str, np.ndarray],
    *,
    log_path: Path,
    unit_id: str,
    device: torch.device,
    interval_seconds: float = 30.0,
) -> tuple[float, dict[str, Any]]:
    """Keep the complete blocking calibration optimizer call observable."""
    bytes_total = sum(np.asarray(values).nbytes for values in scores.values())
    with CudaHeartbeat(
        Path(log_path),
        "calibration_optimizer",
        device=device,
        unit_id=unit_id,
        bytes_total=bytes_total,
        interval_seconds=interval_seconds,
        total_chunks=1,
    ) as heartbeat:
        heartbeat.update(
            operation="fit_positive_day_equal_calibration",
            chunk_index=0,
            chunks_total=1,
            rows_processed=sum(len(values) for values in scores.values()),
            bytes_processed=0,
            checkpoint_status="input_scores_durable_in_memory",
        )
        result = UPSTREAM.fit_positive_day_equal_calibration(scores)
        heartbeat.update(
            chunk_index=1,
            bytes_processed=bytes_total,
            checkpoint_status="optimizer_complete",
        )
        return result


class CircularTemporalResidual(nn.Module):
    """Ordered-TCN residual block with pre-registered circular padding."""

    def __init__(self, width: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dilation = dilation
        self.convolution = nn.Conv1d(
            width, width, kernel_size=3, dilation=dilation, padding=0
        )
        self.normalization = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        channels_first = values.transpose(1, 2)
        padded = F.pad(
            channels_first,
            (self.dilation, self.dilation),
            mode="circular",
        )
        update = self.convolution(padded).transpose(1, 2)
        update = self.dropout(F.gelu(self.normalization(update)))
        return values + update


class CircularOrderedTCN(nn.Module):
    def __init__(self, width: int = 32, input_channels: int = 37) -> None:
        super().__init__()
        self.encoder = ConfigurableBlockEncoder(input_channels, width)
        self.temporal = nn.ModuleList(
            [CircularTemporalResidual(width, dilation) for dilation in (1, 2, 4)]
        )
        self.head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        for layer in self.temporal:
            encoded = layer(encoded)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class ConfigurableBlockEncoder(nn.Module):
    def __init__(self, input_channels: int, width: int = 32) -> None:
        super().__init__()
        if input_channels not in (34, 37):
            raise ValueError(f"unsupported model input width: {input_channels}")
        self.network = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ConfigurablePooledMLP(nn.Module):
    def __init__(self, input_channels: int, width: int = 32) -> None:
        super().__init__()
        self.encoder = ConfigurableBlockEncoder(input_channels, width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        return self.head(
            torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)
        ).squeeze(-1)


class ConfigurableOrderedTCN(nn.Module):
    def __init__(self, input_channels: int, width: int = 32) -> None:
        super().__init__()
        self.encoder = ConfigurableBlockEncoder(input_channels, width)
        self.temporal = nn.ModuleList(
            [UPSTREAM.TemporalResidual(width, dilation) for dilation in (1, 2, 4)]
        )
        self.head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        for layer in self.temporal:
            encoded = layer(encoded)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


def _build_model(
    model_kind: str, model_variant: str, input_channels: int = 37
) -> nn.Module:
    if input_channels not in (34, 37):
        raise ValueError(f"unsupported model input width: {input_channels}")
    if model_variant == "standard" and input_channels == 37:
        return UPSTREAM.build_model(model_kind)
    if model_variant == "standard" and model_kind == "pooled_mlp":
        return ConfigurablePooledMLP(input_channels)
    if model_variant == "standard" and model_kind == "ordered_tcn":
        return ConfigurableOrderedTCN(input_channels)
    if model_variant == "circular_padding" and model_kind == "ordered_tcn":
        return CircularOrderedTCN(input_channels=input_channels)
    raise ValueError(f"unsupported model variant: {model_kind}/{model_variant}")


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_or_raise(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def unit_dir(model: str, seed: int) -> Path:
    return MODEL_ROOT / model / f"seed={seed}"


def load_unit(
    model_kind: str, seed: int, device: torch.device
) -> tuple[Any, Any, float]:
    root = unit_dir(model_kind, seed)
    complete = json.loads((root / "complete_manifest.json").read_text())
    if complete.get("status") != "complete":
        raise RuntimeError(f"upstream unit incomplete: {root}")
    scaler = UPSTREAM.ChannelScaler.from_dict(
        json.loads((root / "scaler.json").read_text())
    )
    alpha = float(json.loads((root / "calibration.json").read_text())["alpha"])
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    model = UPSTREAM.build_model(model_kind)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, scaler, alpha


def static_compatibility() -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            root = unit_dir(model, seed)
            loaded, scaler, alpha = load_unit(model, seed, torch.device("cpu"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "parameters": sum(p.numel() for p in loaded.parameters()),
                    "token_dim": UPSTREAM.TOKEN_DIM,
                    "scaler_tokens": scaler.sample_tokens,
                    "alpha": alpha,
                    "model_bytes": (root / "model.pt").stat().st_size,
                    "strict_load": True,
                }
            )
    return rows


def parity(
    *,
    target_device: str,
    source_date: str = "2026-07-21",
    max_paths: int = 131_072,
    batch_size: int = 1_024,
) -> list[dict[str, Any]]:
    configure_determinism()
    device = device_or_raise(target_device)
    day = UPSTREAM.read_day(INPUT_ROOT, source_date, 10)
    starts = UPSTREAM.day_path_starts(day, 6)[:max_paths]
    if len(starts) != max_paths:
        raise RuntimeError(f"parity path set too small: {len(starts)}")
    rows: list[dict[str, Any]] = []
    for model_kind in MODELS:
        for seed in SEEDS:
            cpu_model, scaler, alpha = load_unit(model_kind, seed, torch.device("cpu"))
            target_model, _, _ = load_unit(model_kind, seed, device)
            max_score = max_calibrated = max_antisymmetry = 0.0
            max_permutation = 0.0 if model_kind == "pooled_mlp" else None
            cpu_bounded_sum = target_bounded_sum = 0.0
            cpu_uncapped_sum = target_uncapped_sum = 0.0
            rows_seen = 0
            finite = True
            cpu_seconds = target_seconds = transfer_seconds = 0.0
            for first in range(0, len(starts), batch_size):
                selected = starts[first : first + batch_size]
                forward, reverse, _ = UPSTREAM.path_arrays(day, selected, 6)
                cpu_forward = torch.from_numpy(scaler.transform(forward))
                cpu_reverse = torch.from_numpy(scaler.transform(reverse))
                started = time.monotonic()
                with torch.no_grad():
                    cpu_score = UPSTREAM.antisymmetric_score(
                        cpu_model, cpu_forward, cpu_reverse
                    )
                cpu_seconds += time.monotonic() - started
                started = time.monotonic()
                target_forward = cpu_forward.to(device)
                target_reverse = cpu_reverse.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                transfer_seconds += time.monotonic() - started
                started = time.monotonic()
                with torch.no_grad():
                    target_score = UPSTREAM.antisymmetric_score(
                        target_model, target_forward, target_reverse
                    )
                    reverse_score = UPSTREAM.antisymmetric_score(
                        target_model, target_reverse, target_forward
                    )
                    if model_kind == "pooled_mlp":
                        permutation_score = target_model(
                            target_forward[:, torch.arange(5, -1, -1), :]
                        )
                        original_score = target_model(target_forward)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                target_seconds += time.monotonic() - started
                target_cpu = target_score.detach().cpu()
                finite = bool(
                    finite
                    and torch.isfinite(cpu_score).all()
                    and torch.isfinite(target_cpu).all()
                    and torch.isfinite(reverse_score).all()
                )
                delta = torch.abs(cpu_score - target_cpu)
                max_score = max(max_score, float(delta.max()))
                max_calibrated = max(max_calibrated, float((delta * alpha).max()))
                max_antisymmetry = max(
                    max_antisymmetry,
                    float(torch.abs(target_score + reverse_score).max()),
                )
                if model_kind == "pooled_mlp":
                    assert max_permutation is not None
                    max_permutation = max(
                        max_permutation,
                        float(torch.abs(original_score - permutation_score).max()),
                    )
                cpu_values = cpu_score.numpy().astype(np.float64) * alpha
                target_values = target_cpu.numpy().astype(np.float64) * alpha
                bound = np.log(99.0)
                cpu_uncapped_sum += float(cpu_values.sum())
                target_uncapped_sum += float(target_values.sum())
                cpu_bounded_sum += float(np.clip(cpu_values, -bound, bound).sum())
                target_bounded_sum += float(np.clip(target_values, -bound, bound).sum())
                rows_seen += len(selected)
            cpu_d = cpu_bounded_sum / rows_seen
            target_d = target_bounded_sum / rows_seen
            row = {
                "model": model_kind,
                "seed": seed,
                "device": str(device),
                "paths": rows_seen,
                "max_score_abs_error": max_score,
                "max_calibrated_abs_error": max_calibrated,
                "bounded_nats_path_cpu": cpu_d,
                "bounded_nats_path_target": target_d,
                "bounded_nats_path_abs_error": abs(cpu_d - target_d),
                "uncapped_nats_path_cpu": cpu_uncapped_sum / rows_seen,
                "uncapped_nats_path_target": target_uncapped_sum / rows_seen,
                "antisymmetry_max_abs_error": max_antisymmetry,
                "permutation_max_abs_error": max_permutation,
                "cpu_seconds": cpu_seconds,
                "target_seconds": target_seconds,
                "transfer_seconds": transfer_seconds,
                "finite_scores": finite,
                "path_order_equal": bool(
                    rows_seen == len(starts)
                    and np.all(np.diff(starts.astype(np.int64)) > 0)
                ),
            }
            row["passed"] = bool(
                finite
                and row["path_order_equal"]
                and max_score < 1e-4
                and max_calibrated < 1e-4
                and row["bounded_nats_path_abs_error"] < 5e-6
                and max_antisymmetry < 1e-6
                and (max_permutation is None or max_permutation < 2e-5)
            )
            rows.append(row)
    return rows


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_batches(
    model: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    day: Any,
    starts_batches: list[np.ndarray],
    scaler: Any,
    first: int,
    stop: int,
) -> list[float]:
    history = []
    model.train()
    for index in range(first, stop):
        forward, reverse, _ = UPSTREAM.path_arrays(day, starts_batches[index], 6)
        forward_tensor = torch.from_numpy(scaler.transform(forward)).to(device)
        reverse_tensor = torch.from_numpy(scaler.transform(reverse)).to(device)
        score = UPSTREAM.antisymmetric_score(model, forward_tensor, reverse_tensor)
        loss = F.softplus(-score).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history


def resume_probe(
    checkpoint_path: Path,
    *,
    device_name: str,
    max_paths: int = 131_072,
    batch_size: int = 1_024,
    interrupt_after: int = 50,
) -> dict[str, Any]:
    configure_determinism()
    device = device_or_raise(device_name)
    day = UPSTREAM.read_day(INPUT_ROOT, "2026-07-02", 10)
    starts = UPSTREAM.day_path_starts(day, 6)[:max_paths]
    batches = list(UPSTREAM.iter_batches(starts, batch_size, 20260730))
    config = {
        "model": "ordered_tcn",
        "seed": 20260730,
        "source_date": "2026-07-02",
        "paths": max_paths,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "device": device_name,
    }

    def fresh() -> tuple[Any, Any]:
        random.seed(20260730)
        np.random.seed(20260730)
        torch.manual_seed(20260730)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260730)
        model = UPSTREAM.build_model("ordered_tcn").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        return model, optimizer

    control, control_optimizer = fresh()
    control_history = _train_batches(
        control,
        control_optimizer,
        device,
        day,
        batches,
        load_unit("ordered_tcn", 20260730, torch.device("cpu"))[1],
        0,
        len(batches),
    )

    resumed, resumed_optimizer = fresh()
    scaler = load_unit("ordered_tcn", 20260730, torch.device("cpu"))[1]
    resumed_history = _train_batches(
        resumed, resumed_optimizer, device, day, batches, scaler, 0, interrupt_after
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(
        {
            "config": config,
            "next_batch": interrupt_after,
            "history": resumed_history,
            "model_state": resumed.state_dict(),
            "optimizer_state": resumed_optimizer.state_dict(),
            "rng_state": _rng_state(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["config"] != config:
        raise RuntimeError("resume checkpoint config mismatch")
    resumed = UPSTREAM.build_model("ordered_tcn").to(device)
    resumed.load_state_dict(payload["model_state"], strict=True)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(), lr=1e-3, weight_decay=1e-4
    )
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    _move_optimizer(resumed_optimizer, device)
    _restore_rng(payload["rng_state"])
    resumed_history = payload["history"] + _train_batches(
        resumed,
        resumed_optimizer,
        device,
        day,
        batches,
        scaler,
        payload["next_batch"],
        len(batches),
    )
    parameters_exact = all(
        torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(control.parameters(), resumed.parameters(), strict=True)
    )
    history_exact = control_history == resumed_history
    probe = batches[:1][0]
    forward, reverse, _ = UPSTREAM.path_arrays(day, probe, 6)
    forward_tensor = torch.from_numpy(scaler.transform(forward)).to(device)
    reverse_tensor = torch.from_numpy(scaler.transform(reverse)).to(device)
    control.eval()
    resumed.eval()
    with torch.no_grad():
        control_score = UPSTREAM.antisymmetric_score(
            control, forward_tensor, reverse_tensor
        )
        resumed_score = UPSTREAM.antisymmetric_score(
            resumed, forward_tensor, reverse_tensor
        )
    probe_exact = torch.equal(control_score, resumed_score)
    cursor_equal = bool(
        payload["next_batch"] == interrupt_after
        and len(payload["history"]) == interrupt_after
        and len(resumed_history) == len(batches)
    )
    return {
        "device": device_name,
        "batches": len(batches),
        "interrupt_after": interrupt_after,
        "cursor_equal": cursor_equal,
        "history_exact": history_exact,
        "parameters_exact": parameters_exact,
        "probe_score_exact": probe_exact,
        "passed": bool(
            cursor_equal and history_exact and parameters_exact and probe_exact
        ),
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != len(rows):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row validation failed: {path}")
    os.replace(temporary, path)


def _selected_starts(
    day: Any,
    k: int,
    starts_by_date: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    """Return a validated deterministic subset of canonical path starts.

    The default preserves every existing caller.  Phase 14 uses the optional
    mapping for nested development fractions; calibration and test continue to
    pass ``None`` and therefore use the complete canonical set.
    """
    canonical = UPSTREAM.day_path_starts(day, k)
    if starts_by_date is None:
        return canonical
    if day.source_date not in starts_by_date:
        raise KeyError(f"missing path-start selector for {day.source_date}")
    selected = np.asarray(starts_by_date[day.source_date], dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError(f"empty or non-vector path-start selector: {day.source_date}")
    if len(selected) != len(np.unique(selected)) or np.any(np.diff(selected) <= 0):
        raise ValueError(
            f"path-start selector must be unique and ordered: {day.source_date}"
        )
    positions = np.searchsorted(canonical, selected)
    if np.any(positions >= len(canonical)) or not np.array_equal(
        canonical[positions], selected
    ):
        raise ValueError(
            f"selector is not a subset of canonical starts: {day.source_date}"
        )
    return selected


def _cuda_evaluate_loss(
    model: Any,
    input_root: Path,
    dates: tuple[str, ...],
    target_b: int,
    k: int,
    scaler: Any,
    batch_size: int,
    null_seed: int | None = None,
    starts_by_date: Mapping[str, np.ndarray] | None = None,
    log_path: Path | None = None,
    heartbeat_interval_seconds: float = 30.0,
    heartbeat: CudaHeartbeat | None = None,
    date_bytes: dict[str, int] | None = None,
    heartbeat_bytes_offset: int = 0,
) -> dict[str, Any]:
    if heartbeat is not None:
        return _cuda_evaluate_loss_impl(
            model,
            input_root,
            dates,
            target_b,
            k,
            scaler,
            batch_size,
            null_seed=null_seed,
            starts_by_date=starts_by_date,
            heartbeat=heartbeat,
            date_bytes=date_bytes,
            bytes_offset=heartbeat_bytes_offset,
        )
    if log_path is None:
        return _cuda_evaluate_loss_impl(
            model,
            input_root,
            dates,
            target_b,
            k,
            scaler,
            batch_size,
            null_seed=null_seed,
            starts_by_date=starts_by_date,
            heartbeat=None,
            date_bytes=date_bytes,
        )
    device = next(model.parameters()).device
    date_bytes = _dates_input_bytes(input_root, dates, target_b)
    with CudaHeartbeat(
        Path(log_path),
        "cuda_evaluation",
        device=device,
        unit_id=f"cuda-eval:{Path(input_root).resolve()}:B{target_b}:K{k}:{','.join(dates)}",
        bytes_total=sum(date_bytes.values()),
        interval_seconds=heartbeat_interval_seconds,
        total_chunks=len(dates),
    ) as owned:
        return _cuda_evaluate_loss_impl(
            model,
            input_root,
            dates,
            target_b,
            k,
            scaler,
            batch_size,
            null_seed=null_seed,
            starts_by_date=starts_by_date,
            heartbeat=owned,
            date_bytes=date_bytes,
            bytes_offset=0,
        )


def _cuda_evaluate_loss_impl(
    model: Any,
    input_root: Path,
    dates: tuple[str, ...],
    target_b: int,
    k: int,
    scaler: Any,
    batch_size: int,
    *,
    null_seed: int | None,
    starts_by_date: Mapping[str, np.ndarray] | None,
    heartbeat: CudaHeartbeat | None,
    date_bytes: dict[str, int] | None = None,
    bytes_offset: int = 0,
) -> dict[str, Any]:
    if heartbeat is not None and date_bytes is None:
        date_bytes = _dates_input_bytes(input_root, dates, target_b)
    device = next(model.parameters()).device
    model.eval()
    by_day: dict[str, dict[str, float | int]] = {}
    with torch.no_grad():
        for source_date in dates:
            day = UPSTREAM.read_day(input_root, source_date, target_b)
            starts = _selected_starts(day, k, starts_by_date)
            loss_sum = 0.0
            rows = 0
            for first in range(0, len(starts), batch_size):
                if heartbeat is not None:
                    heartbeat.update(
                        operation="evaluate",
                        chunk_index=len(by_day),
                        chunks_total=len(dates),
                        source_date=source_date,
                        batch_index=first // batch_size,
                        batches_in_date=math.ceil(len(starts) / batch_size),
                        rows_processed=sum(int(row["paths"]) for row in by_day.values())
                        + first,
                        bytes_processed=bytes_offset
                        + sum(date_bytes[day] for day in by_day)
                        + int(date_bytes[source_date] * first / max(len(starts), 1)),
                    )
                selected = starts[first : first + batch_size]
                forward, reverse, _ = UPSTREAM.path_arrays(day, selected, k)
                mask = (
                    UPSTREAM.reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = UPSTREAM.orient_pair(forward, reverse, mask)
                score = UPSTREAM.antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)).to(device),
                    torch.from_numpy(scaler.transform(reverse)).to(device),
                )
                loss_sum += float(F.softplus(-score).sum().detach().cpu())
                rows += len(selected)
            by_day[source_date] = {
                "paths": rows,
                "paired_log_loss": loss_sum / rows,
            }
            if heartbeat is not None:
                heartbeat.update(
                    chunk_index=len(by_day),
                    rows_processed=sum(int(row["paths"]) for row in by_day.values()),
                    bytes_processed=bytes_offset
                    + sum(date_bytes[day] for day in by_day),
                )
    return {
        "by_day": by_day,
        "day_equal_paired_log_loss": float(
            np.mean([row["paired_log_loss"] for row in by_day.values()])
        ),
    }


def phase07_checkpoint_root_relocation(
    observed: Any, expected: dict[str, Any]
) -> dict[str, Any] | None:
    """Return expected only for an exact Phase 07 old-shared to pod-local root move."""
    if not isinstance(observed, dict):
        return None
    scientific = expected.get("unit", {}).get("scientific_config", {})
    if scientific.get("phase14_v5_admission") is not None:
        # Phase 14 V5 has its own explicit shared-runtime admission and must
        # never inherit Phase 07's narrowly scoped root-relocation exception.
        return None
    observed_copy = json.loads(json.dumps(observed, sort_keys=True))
    expected_copy = json.loads(json.dumps(expected, sort_keys=True))
    try:
        old_root = Path(observed_copy["unit"]["input"].pop("root")).resolve()
        new_root = Path(expected_copy["unit"]["input"].pop("root")).resolve()
        old_root.relative_to(Path("/workspace/cache/phase07"))
        new_root.relative_to(Path("/tmp"))
    except (KeyError, TypeError, ValueError):
        return None
    if observed_copy != expected_copy:
        return None
    return expected


def commit_phase07_checkpoint_root_relocation(
    checkpoint_path: Path,
    state: dict[str, Any],
    expected: dict[str, Any],
    *,
    payload_validated: bool,
) -> None:
    """Rewrite only after the caller validates the complete resumable payload."""
    if not payload_validated:
        raise RuntimeError("checkpoint root relocation requires validated payload")
    state["checkpoint_config"] = expected
    UPSTREAM.atomic_torch_save(checkpoint_path, state)


def _cuda_train_stage(
    device: torch.device, **kwargs: Any
) -> tuple[Any, list, int, float]:
    epoch_limit = (
        int(kwargs["fixed_epochs"])
        if kwargs["fixed_epochs"] is not None
        else int(kwargs["max_epochs"])
    )
    train_bytes = _dates_input_bytes(
        Path(kwargs["input_root"]),
        tuple(kwargs["train_dates"]),
        int(kwargs["target_b"]),
    )
    evaluation_bytes = (
        _dates_input_bytes(
            Path(kwargs["input_root"]),
            tuple(kwargs["evaluation_dates"]),
            int(kwargs["target_b"]),
        )
        if kwargs["evaluation_dates"]
        else {}
    )
    bytes_per_epoch = sum(train_bytes.values()) + sum(evaluation_bytes.values())
    checkpoint_path = Path(kwargs["checkpoint_path"])
    heartbeat = CudaHeartbeat(
        Path(kwargs["log_path"]),
        "cuda_training",
        device=device,
        unit_id=f"cuda-train:{checkpoint_path.resolve()}",
        bytes_total=epoch_limit * bytes_per_epoch,
        interval_seconds=float(kwargs.get("heartbeat_interval_seconds", 30.0)),
        checkpoint_path=checkpoint_path,
        total_chunks=epoch_limit,
    )
    with heartbeat:
        kwargs["_heartbeat"] = heartbeat
        kwargs["_heartbeat_train_bytes"] = train_bytes
        kwargs["_heartbeat_evaluation_bytes"] = evaluation_bytes
        kwargs["_heartbeat_bytes_per_epoch"] = bytes_per_epoch
        return _cuda_train_stage_impl(device, **kwargs)


def _cuda_train_stage_impl(
    device: torch.device, **kwargs: Any
) -> tuple[Any, list, int, float]:
    model_kind = kwargs["model_kind"]
    model_variant = str(kwargs.get("model_variant", "standard"))
    input_channels = int(kwargs.get("input_channels", 37))
    seed = int(kwargs["seed"])
    input_root = kwargs["input_root"]
    train_dates = kwargs["train_dates"]
    evaluation_dates = kwargs["evaluation_dates"]
    target_b = int(kwargs["target_b"])
    k = int(kwargs["k"])
    scaler = kwargs["scaler"]
    candidate = kwargs["candidate"]
    max_epochs = int(kwargs["max_epochs"])
    min_epochs = int(kwargs["min_epochs"])
    patience = int(kwargs["patience"])
    fixed_epochs = kwargs["fixed_epochs"]
    batch_size = int(kwargs["batch_size"])
    checkpoint_batches = int(kwargs["checkpoint_batches"])
    checkpoint_path = kwargs["checkpoint_path"]
    spec_hash = kwargs.get("spec_hash")
    checkpoint_config = kwargs.get("checkpoint_config")
    if (spec_hash is None) == (checkpoint_config is None):
        raise ValueError(
            "provide exactly one of spec_hash or readable checkpoint_config"
        )
    resume = bool(kwargs["resume"])
    log_path = kwargs["log_path"]
    null_seed = kwargs.get("null_seed")
    starts_by_date = kwargs.get("starts_by_date")
    heartbeat: CudaHeartbeat = kwargs["_heartbeat"]
    heartbeat_train_bytes: dict[str, int] = kwargs["_heartbeat_train_bytes"]
    heartbeat_evaluation_bytes: dict[str, int] = kwargs["_heartbeat_evaluation_bytes"]
    heartbeat_bytes_per_epoch = int(kwargs["_heartbeat_bytes_per_epoch"])
    runtime_config = {
        "implementation": "phase01_cuda_full_unit_v2",
        "device_type": device.type,
        "device_index": 0 if device.index is None else device.index,
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "amp": False,
        "batch_size": batch_size,
    }
    if model_variant != "standard":
        runtime_config["model_variant"] = model_variant
    if input_channels != 37:
        runtime_config["input_channels"] = input_channels

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32 - 1))
    model = _build_model(model_kind, model_variant, input_channels=input_channels).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate["weight_decay"],
    )
    relocation_pending = False
    if checkpoint_config is None:
        state = UPSTREAM._load_checkpoint(checkpoint_path, spec_hash, resume)
    elif checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"checkpoint exists; use resume: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("checkpoint_config") != checkpoint_config:
            relocated = phase07_checkpoint_root_relocation(
                state.get("checkpoint_config"), checkpoint_config
            )
            if relocated is None:
                raise RuntimeError(
                    f"checkpoint scientific config mismatch: {checkpoint_path}"
                )
            relocation_pending = True
    else:
        state = None
    epoch = date_cursor = batch_cursor = epoch_rows = 0
    epoch_loss_sum = 0.0
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_model_state = None
    stale_epochs = 0
    if state is not None:
        if state.get("runtime_config") != runtime_config:
            raise RuntimeError(f"checkpoint CUDA runtime mismatch: {checkpoint_path}")
        if len(state.get("rng_state", {}).get("cuda", ())) != torch.cuda.device_count():
            raise RuntimeError(
                f"checkpoint CUDA RNG topology mismatch: {checkpoint_path}"
            )
        if state.get("scaler") != scaler.to_dict():
            raise RuntimeError(f"checkpoint scaler mismatch: {checkpoint_path}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        _move_optimizer(optimizer, device)
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
        _restore_rng(state["rng_state"])
        if relocation_pending:
            commit_phase07_checkpoint_root_relocation(
                checkpoint_path,
                state,
                checkpoint_config,
                payload_validated=True,
            )
        UPSTREAM.emit(
            log_path,
            "checkpoint_resumed",
            checkpoint=str(checkpoint_path),
            epoch=epoch + 1,
            date_index=date_cursor,
            batch_index=batch_cursor,
        )
        heartbeat.update(
            chunk_index=epoch,
            chunks_total=(fixed_epochs if fixed_epochs is not None else max_epochs),
            epoch=epoch + 1,
            date_index=date_cursor,
            batch_index=batch_cursor,
            checkpoint_status="resumed",
        )

    def checkpoint(next_epoch: int, next_date: int, next_batch: int) -> None:
        UPSTREAM.atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 1,
                **({"spec_hash": spec_hash} if spec_hash is not None else {}),
                **(
                    {"checkpoint_config": checkpoint_config}
                    if checkpoint_config is not None
                    else {}
                ),
                "runtime_config": runtime_config,
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
                "rng_state": _rng_state(),
            },
        )
        heartbeat.update(
            checkpoint_status="published",
            checkpoint_epoch=next_epoch,
            checkpoint_date_index=next_date,
            checkpoint_batch_index=next_batch,
        )

    epoch_limit = fixed_epochs if fixed_epochs is not None else max_epochs
    while epoch < epoch_limit:
        model.train()
        started = time.monotonic()
        for day_index in range(date_cursor, len(train_dates)):
            source_date = train_dates[day_index]
            day = UPSTREAM.read_day(input_root, source_date, target_b)
            starts = _selected_starts(day, k, starts_by_date)
            shuffled = np.random.default_rng(
                UPSTREAM._batch_seed(seed, epoch, day_index, null_seed is not None)
            ).permutation(starts)
            first_batch = batch_cursor if day_index == date_cursor else 0
            total_batches = math.ceil(len(shuffled) / batch_size)
            for batch_index in range(first_batch, total_batches):
                heartbeat.update(
                    chunk_index=epoch,
                    chunks_total=epoch_limit,
                    operation="train",
                    epoch=epoch + 1,
                    date_index=day_index,
                    source_date=source_date,
                    batch_index=batch_index,
                    batches_in_date=total_batches,
                    rows_processed=epoch_rows,
                    checkpoint_status=(
                        "available" if checkpoint_path.is_file() else "pending"
                    ),
                    bytes_processed=(
                        epoch * heartbeat_bytes_per_epoch
                        + sum(
                            heartbeat_train_bytes[day]
                            for day in train_dates[:day_index]
                        )
                        + int(
                            heartbeat_train_bytes[source_date]
                            * batch_index
                            / max(total_batches, 1)
                        )
                    ),
                )
                selected = shuffled[
                    batch_index * batch_size : (batch_index + 1) * batch_size
                ]
                forward, reverse, _ = UPSTREAM.path_arrays(day, selected, k)
                mask = (
                    UPSTREAM.reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = UPSTREAM.orient_pair(forward, reverse, mask)
                score = UPSTREAM.antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)).to(device),
                    torch.from_numpy(scaler.transform(reverse)).to(device),
                )
                loss = F.softplus(-score).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss_sum += float(loss.detach().cpu()) * len(selected)
                epoch_rows += len(selected)
                heartbeat.update(rows_processed=epoch_rows, batch_index=batch_index + 1)
                next_day, next_batch = day_index, batch_index + 1
                if next_batch >= total_batches:
                    next_day, next_batch = day_index + 1, 0
                if (
                    batch_index + 1
                ) % checkpoint_batches == 0 or UPSTREAM.STOP_REQUESTED:
                    checkpoint(epoch, next_day, next_batch)
                if UPSTREAM.STOP_REQUESTED:
                    UPSTREAM.emit(
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
            _cuda_evaluate_loss(
                model,
                input_root,
                evaluation_dates,
                target_b,
                k,
                scaler,
                batch_size,
                null_seed=null_seed,
                starts_by_date=starts_by_date,
                heartbeat=heartbeat,
                date_bytes=heartbeat_evaluation_bytes,
                heartbeat_bytes_offset=(
                    epoch * heartbeat_bytes_per_epoch
                    + sum(heartbeat_train_bytes.values())
                ),
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
        UPSTREAM.emit(
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
        heartbeat.update(
            chunk_index=epoch,
            chunks_total=epoch_limit,
            operation="epoch_complete",
            epoch=epoch,
            rows_processed=epoch_rows,
        )
        date_cursor = batch_cursor = epoch_rows = 0
        epoch_loss_sum = 0.0
        checkpoint(epoch, 0, 0)
        if fixed_epochs is None and epoch >= min_epochs and stale_epochs >= patience:
            break
    if best_model_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_model_state)
    return model, history, best_epoch, best_loss


def _cuda_collect_scores_by_day(
    model: Any,
    input_root: Path,
    dates: tuple[str, ...],
    target_b: int,
    k: int,
    scaler: Any,
    batch_size: int,
    null_seed: int | None = None,
    starts_by_date: Mapping[str, np.ndarray] | None = None,
    log_path: Path | None = None,
    heartbeat_interval_seconds: float = 30.0,
    checkpoint_root: Path | None = None,
    unit_id: str | None = None,
    model_identity: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if checkpoint_root is not None:
        if log_path is None or unit_id is None or model_identity is None:
            raise ValueError(
                "calibration checkpoints require log_path, exact unit_id/model identity"
            )
        date_bytes = _dates_input_bytes(input_root, dates, target_b)
        device = next(model.parameters()).device
        result: dict[str, np.ndarray] = {}
        with CudaHeartbeat(
            Path(log_path),
            "cuda_inference_scores",
            device=device,
            unit_id=unit_id,
            bytes_total=sum(date_bytes.values()),
            interval_seconds=heartbeat_interval_seconds,
            total_chunks=len(dates),
        ) as heartbeat:
            for date_index, source_date in enumerate(dates):
                date_root = Path(checkpoint_root) / f"source_date={source_date}"
                score_path = date_root / "scores.npy"
                complete_path = date_root / "complete.json"
                paths = UPSTREAM.partition_paths(
                    Path(input_root), source_date, target_b
                )
                config = {
                    "contract": "cuda_calibration_score_date_v1",
                    "code_identity": "cuda_collect_scores_by_day_v1",
                    "unit_id": f"{unit_id}/date={source_date}",
                    "source_date": source_date,
                    "target_b": target_b,
                    "k": k,
                    "null_seed": null_seed,
                    "batch_size": batch_size,
                    "model_class": type(model).__name__,
                    "scaler": scaler.to_dict(),
                    "model_identity": dict(model_identity),
                    "inputs": [
                        {
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "rows": pq.ParquetFile(path).metadata.num_rows,
                        }
                        for path in paths
                    ],
                }
                try:
                    restored = _validated_cuda_score_checkpoint(date_root, config)
                except RuntimeError:
                    quarantine = Path(checkpoint_root) / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(
                        date_root,
                        quarantine / f"{date_root.name}.static.{uuid.uuid4().hex}",
                    )
                    date_root.mkdir(parents=True, exist_ok=True)
                    restored = None
                if restored is None:
                    restored = _cuda_collect_scores_by_day_impl(
                        model,
                        input_root,
                        (source_date,),
                        target_b,
                        k,
                        scaler,
                        batch_size,
                        null_seed=null_seed,
                        starts_by_date=starts_by_date,
                        heartbeat=heartbeat,
                        date_bytes={source_date: date_bytes[source_date]},
                    )[source_date]
                    date_root.mkdir(parents=True, exist_ok=True)
                    temporary = score_path.with_name(
                        f".{score_path.name}.{os.getpid()}.tmp.npy"
                    )
                    np.save(temporary, restored, allow_pickle=False)
                    os.replace(temporary, score_path)
                    UPSTREAM.atomic_write_json(
                        complete_path,
                        {
                            "status": "complete",
                            "config": config,
                            "score_path": str(score_path.resolve()),
                            "rows": len(restored),
                            "bytes": score_path.stat().st_size,
                            "score_filesystem_identity": _filesystem_identity(
                                score_path
                            ),
                        },
                    )
                    if _validated_cuda_score_checkpoint(date_root, config) is None:
                        raise RuntimeError(
                            f"CUDA score checkpoint publication audit failed: {date_root}"
                        )
                    checkpoint_status = "published_verified"
                else:
                    checkpoint_status = "verified_skip"
                result[source_date] = restored
                heartbeat.update(
                    unit_id=config["unit_id"],
                    chunk_index=date_index + 1,
                    rows_processed=sum(len(values) for values in result.values()),
                    bytes_processed=sum(date_bytes[day] for day in result),
                    checkpoint_status=checkpoint_status,
                    checkpoint_path=str(complete_path.resolve()),
                    checkpoint_exists=True,
                    checkpoint_bytes=complete_path.stat().st_size,
                )
        return result
    if log_path is not None:
        device = next(model.parameters()).device
        date_bytes = _dates_input_bytes(input_root, dates, target_b)
        with CudaHeartbeat(
            Path(log_path),
            "cuda_inference_scores",
            device=device,
            unit_id=(
                unit_id
                or f"cuda-calibration:{Path(input_root).resolve()}:B{target_b}:K{k}:{','.join(dates)}"
            ),
            bytes_total=sum(date_bytes.values()),
            interval_seconds=heartbeat_interval_seconds,
            total_chunks=len(dates),
        ) as heartbeat:
            return _cuda_collect_scores_by_day_impl(
                model,
                input_root,
                dates,
                target_b,
                k,
                scaler,
                batch_size,
                null_seed=null_seed,
                starts_by_date=starts_by_date,
                heartbeat=heartbeat,
                date_bytes=date_bytes,
            )
    return _cuda_collect_scores_by_day_impl(
        model,
        input_root,
        dates,
        target_b,
        k,
        scaler,
        batch_size,
        null_seed=null_seed,
        starts_by_date=starts_by_date,
        heartbeat=None,
    )


def _cuda_collect_scores_by_day_impl(
    model: Any,
    input_root: Path,
    dates: tuple[str, ...],
    target_b: int,
    k: int,
    scaler: Any,
    batch_size: int,
    *,
    null_seed: int | None,
    starts_by_date: Mapping[str, np.ndarray] | None,
    heartbeat: CudaHeartbeat | None,
    date_bytes: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    result: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for source_date in dates:
            day = UPSTREAM.read_day(input_root, source_date, target_b)
            starts = _selected_starts(day, k, starts_by_date)
            pieces = []
            for first in range(0, len(starts), batch_size):
                if heartbeat is not None:
                    heartbeat.update(
                        operation="collect_scores",
                        chunk_index=len(result),
                        chunks_total=len(dates),
                        source_date=source_date,
                        batch_index=first // batch_size,
                        batches_in_date=math.ceil(len(starts) / batch_size),
                        rows_processed=sum(len(values) for values in result.values())
                        + first,
                        bytes_processed=sum(date_bytes[day] for day in result)
                        + int(date_bytes[source_date] * first / max(len(starts), 1)),
                    )
                selected = starts[first : first + batch_size]
                forward, reverse, _ = UPSTREAM.path_arrays(day, selected, k)
                mask = (
                    UPSTREAM.reversible_null_swap_mask(source_date, selected, null_seed)
                    if null_seed is not None
                    else None
                )
                forward, reverse = UPSTREAM.orient_pair(forward, reverse, mask)
                score = UPSTREAM.antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(forward)).to(device),
                    torch.from_numpy(scaler.transform(reverse)).to(device),
                )
                pieces.append(score.detach().cpu().numpy().astype(np.float64))
            result[source_date] = np.concatenate(pieces)
            if heartbeat is not None:
                heartbeat.update(
                    chunk_index=len(result),
                    rows_processed=sum(len(values) for values in result.values()),
                    bytes_processed=sum(date_bytes[day] for day in result),
                )
    return result


def _atomic_concat_prediction_parts(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer = None
    rows = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                # Preserve producer row-group boundaries.  Pair validation is
                # streaming and does not depend on this layout, but retaining
                # the logical groups makes future audits stable and inspectable.
                writer.write_table(table, row_group_size=table.num_rows)
                rows += table.num_rows
        if writer is None:
            raise RuntimeError("empty CUDA prediction checkpoint reduction")
        writer.close()
        writer = None
        if pq.ParquetFile(temporary).metadata.num_rows != rows:
            raise RuntimeError("CUDA prediction checkpoint reduction row mismatch")
        os.replace(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _parquet_value_exact(left: Path, right: Path) -> bool:
    left_file, right_file = pq.ParquetFile(left), pq.ParquetFile(right)
    if (
        left_file.metadata.num_rows != right_file.metadata.num_rows
        or not left_file.schema_arrow.equals(right_file.schema_arrow)
    ):
        return False
    for left_batch, right_batch in zip_longest(
        left_file.iter_batches(batch_size=65_536),
        right_file.iter_batches(batch_size=65_536),
    ):
        if (
            left_batch is None
            or right_batch is None
            or not left_batch.equals(right_batch)
        ):
            return False
    return True


def _merge_cuda_date_metrics(
    date_roots: list[Path], k: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    pooled = UPSTREAM._metric_accumulator()
    bounded = UPSTREAM._metric_accumulator()
    by_day: dict[str, Any] = {}
    bounded_by_day: dict[str, Any] = {}
    audits = []
    for root in date_roots:
        payload = json.loads((root / "metrics.json").read_text())
        metrics, audit = payload["metrics"], payload["audit"]
        by_day.update(metrics["by_day"])
        bounded_by_day.update(metrics["bounded_log99"]["by_day"])
        audits.append(audit)
        columns = [
            "orientation",
            "calibrated_antisymmetric_score",
            "path_event_count",
            "duration_seconds",
        ]
        for batch in pq.ParquetFile(root / "test_predictions.parquet").iter_batches(
            batch_size=65_536, columns=columns
        ):
            table = pa.Table.from_batches([batch])
            forward = table.filter(pc.equal(table["orientation"], "forward"))
            scores = np.asarray(
                forward["calibrated_antisymmetric_score"], dtype=np.float64
            )
            events = np.asarray(forward["path_event_count"], dtype=np.float64)
            duration = np.asarray(forward["duration_seconds"], dtype=np.float64)
            UPSTREAM._update_metrics(pooled, scores, events, duration)
            UPSTREAM._update_metrics(
                bounded, np.clip(scores, -np.log(99.0), np.log(99.0)), events, duration
            )
    keys = (
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
        "pooled_path_weighted": UPSTREAM._finalize_metrics(pooled, k),
        "by_day": by_day,
        "day_equal": {
            key: float(np.mean([row[key] for row in by_day.values()])) for key in keys
        },
        "bounded_log99": {
            "pooled_path_weighted": UPSTREAM._finalize_metrics(bounded, k),
            "by_day": bounded_by_day,
            "day_equal": {
                key: float(np.mean([row[key] for row in bounded_by_day.values()]))
                for key in keys
            },
        },
    }
    audit = {
        "score_tolerance": 1e-6,
        "pooled_permutation_tolerance": 2e-5,
        "score_antisymmetry_max_abs_error": max(
            row["score_antisymmetry_max_abs_error"] for row in audits
        ),
        "probability_pair_max_abs_error": max(
            row["probability_pair_max_abs_error"] for row in audits
        ),
        "permutation_max_abs_error": max(
            (
                row["permutation_max_abs_error"]
                for row in audits
                if row["permutation_max_abs_error"] is not None
            ),
            default=None,
        ),
        "finite_metrics": all(row["finite_metrics"] for row in audits),
        "raw_and_calibrated_finite": all(
            row["raw_and_calibrated_finite"] for row in audits
        ),
        "reverse_involution_exact": all(
            row["reverse_involution_exact"] for row in audits
        ),
        "pair_identity_order_exact": all(
            row["pair_identity_order_exact"] for row in audits
        ),
    }
    audit["passed"] = all(row.get("passed") is True for row in audits)
    return metrics, audit


def _cuda_test_predictions_and_metrics(
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log_path = kwargs.pop("log_path", None)
    interval = float(kwargs.pop("heartbeat_interval_seconds", 30.0))
    checkpoint_root = kwargs.pop("checkpoint_root", None)
    unit_id = kwargs.pop("unit_id", None)
    model_identity = kwargs.pop("model_identity", None)
    if checkpoint_root is not None:
        if log_path is None or unit_id is None or model_identity is None:
            raise ValueError(
                "test checkpoints require log_path, exact unit_id/model identity"
            )
        dates = tuple(kwargs["dates"])
        input_root = Path(kwargs["input_root"])
        target_b = int(kwargs["target_b"])
        k = int(kwargs["k"])
        date_bytes = _dates_input_bytes(input_root, dates, target_b)
        model = kwargs["model"]
        date_roots = []
        with CudaHeartbeat(
            Path(log_path),
            "cuda_test_inference",
            device=next(model.parameters()).device,
            unit_id=unit_id,
            bytes_total=sum(date_bytes.values()),
            interval_seconds=interval,
            total_chunks=len(dates),
        ) as heartbeat:
            for date_index, source_date in enumerate(dates):
                root = Path(checkpoint_root) / f"source_date={source_date}"
                prediction = root / "test_predictions.parquet"
                metrics_path = root / "metrics.json"
                complete_path = root / "complete.json"
                inputs = UPSTREAM.partition_paths(input_root, source_date, target_b)
                config = {
                    "contract": "cuda_test_prediction_date_v1",
                    "code_identity": "cuda_test_predictions_and_metrics_v1",
                    "unit_id": f"{unit_id}/date={source_date}",
                    "source_date": source_date,
                    "target_b": target_b,
                    "k": k,
                    "model_kind": kwargs["model_kind"],
                    "seed": int(kwargs["seed"]),
                    "alpha": float(kwargs["alpha"]),
                    "batch_size": int(kwargs["batch_size"]),
                    "null_seed": kwargs.get("null_seed"),
                    "scaler": kwargs["scaler"].to_dict(),
                    "model_identity": dict(model_identity),
                    "inputs": [
                        {
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "rows": pq.ParquetFile(path).metadata.num_rows,
                        }
                        for path in inputs
                    ],
                }
                try:
                    restored_payload = _validated_cuda_test_checkpoint(root, config)
                except RuntimeError:
                    quarantine = Path(checkpoint_root) / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(
                        root, quarantine / f"{root.name}.static.{uuid.uuid4().hex}"
                    )
                    root.mkdir(parents=True, exist_ok=True)
                    restored_payload = None
                restored = restored_payload is not None
                if not restored:
                    root.mkdir(parents=True, exist_ok=True)
                    one_kwargs = {
                        **kwargs,
                        "dates": (source_date,),
                        "prediction_path": prediction,
                    }
                    metrics, audit = _cuda_test_predictions_and_metrics_impl(
                        **one_kwargs,
                        heartbeat=heartbeat,
                        date_bytes={source_date: date_bytes[source_date]},
                    )
                    UPSTREAM.atomic_write_json(
                        metrics_path, {"metrics": metrics, "audit": audit}
                    )
                    UPSTREAM.atomic_write_json(
                        complete_path,
                        {
                            "status": "complete",
                            "config": config,
                            "prediction_rows": pq.ParquetFile(
                                prediction
                            ).metadata.num_rows,
                            "prediction_bytes": prediction.stat().st_size,
                            "metrics_bytes": metrics_path.stat().st_size,
                            "audit": audit,
                            "prediction_filesystem_identity": _filesystem_identity(
                                prediction
                            ),
                            "metrics_filesystem_identity": _filesystem_identity(
                                metrics_path
                            ),
                        },
                    )
                    if _validated_cuda_test_checkpoint(root, config) is None:
                        raise RuntimeError(
                            f"CUDA test checkpoint publication audit failed: {root}"
                        )
                date_roots.append(root)
                heartbeat.update(
                    unit_id=config["unit_id"],
                    chunk_index=date_index + 1,
                    bytes_processed=sum(
                        date_bytes[day] for day in dates[: date_index + 1]
                    ),
                    checkpoint_status="verified_skip"
                    if restored
                    else "published_verified",
                    checkpoint_path=str(complete_path.resolve()),
                    checkpoint_exists=True,
                    checkpoint_bytes=complete_path.stat().st_size,
                )
        if kwargs["prediction_path"] is not None:
            _atomic_concat_prediction_parts(
                [root / "test_predictions.parquet" for root in date_roots],
                Path(kwargs["prediction_path"]),
            )
        return _merge_cuda_date_metrics(date_roots, k)
    if log_path is None:
        return _cuda_test_predictions_and_metrics_impl(**kwargs, heartbeat=None)
    model = kwargs["model"]
    date_bytes = _dates_input_bytes(
        kwargs["input_root"], kwargs["dates"], int(kwargs["target_b"])
    )
    with CudaHeartbeat(
        Path(log_path),
        "cuda_test_inference",
        device=next(model.parameters()).device,
        unit_id=(
            f"cuda-test:{Path(kwargs['input_root']).resolve()}:"
            f"B{int(kwargs['target_b'])}:K{int(kwargs['k'])}:"
            f"{kwargs['model_kind']}:{int(kwargs['seed'])}"
        ),
        bytes_total=sum(date_bytes.values()),
        interval_seconds=interval,
        total_chunks=len(kwargs["dates"]),
    ) as heartbeat:
        return _cuda_test_predictions_and_metrics_impl(
            **kwargs, heartbeat=heartbeat, date_bytes=date_bytes
        )


def _cuda_test_predictions_and_metrics_impl(
    *,
    heartbeat: CudaHeartbeat | None,
    date_bytes: dict[str, int] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = kwargs["model"]
    input_root = kwargs["input_root"]
    dates = kwargs["dates"]
    target_b = int(kwargs["target_b"])
    k = int(kwargs["k"])
    scaler = kwargs["scaler"]
    alpha = float(kwargs["alpha"])
    batch_size = int(kwargs["batch_size"])
    prediction_path = kwargs["prediction_path"]
    model_kind = kwargs["model_kind"]
    seed = int(kwargs["seed"])
    null_seed = kwargs.get("null_seed")
    starts_by_date = kwargs.get("starts_by_date")
    device = next(model.parameters()).device
    writer = None
    temporary = (
        prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
        if prediction_path is not None
        else None
    )
    pooled = UPSTREAM._metric_accumulator()
    bounded_pooled = UPSTREAM._metric_accumulator()
    by_day: dict[str, dict[str, Any]] = {}
    bounded_by_day: dict[str, dict[str, Any]] = {}
    max_antisymmetry_error = max_probability_pair_error = 0.0
    reverse_involution_exact = True
    pair_identity_order_exact = True
    raw_and_calibrated_finite = True
    permutation_error = None
    model.eval()
    try:
        with torch.no_grad():
            for source_date in dates:
                day = UPSTREAM.read_day(input_root, source_date, target_b)
                starts = _selected_starts(day, k, starts_by_date)
                day_metric = UPSTREAM._metric_accumulator()
                bounded_day_metric = UPSTREAM._metric_accumulator()
                for first in range(0, len(starts), batch_size):
                    if heartbeat is not None:
                        completed_rows = sum(
                            int(row["paths"]) for row in by_day.values()
                        )
                        heartbeat.update(
                            operation="test_predictions",
                            chunk_index=len(by_day),
                            chunks_total=len(dates),
                            source_date=source_date,
                            batch_index=first // batch_size,
                            batches_in_date=math.ceil(len(starts) / batch_size),
                            rows_processed=completed_rows + first,
                            checkpoint_status="streaming_temporary_parquet",
                            bytes_processed=sum(date_bytes[day] for day in by_day)
                            + int(
                                date_bytes[source_date] * first / max(len(starts), 1)
                            ),
                        )
                    selected = starts[first : first + batch_size]
                    forward, reverse, event_count = UPSTREAM.path_arrays(
                        day, selected, k
                    )
                    offsets = selected[:, None] + np.arange(k, dtype=np.int64)[None, :]
                    intervals = day.intervals[offsets]
                    boundaries = np.empty(
                        (
                            len(selected),
                            k + 1,
                            len(UPSTREAM.BOUNDARY_FIELDS),
                        ),
                        dtype=np.float64,
                    )
                    boundaries[:, 0, :] = day.starts[selected]
                    boundaries[:, 1:, :] = day.ends[offsets]
                    reversed_once = UPSTREAM.reverse_continuous(boundaries, intervals)
                    reversed_twice = UPSTREAM.reverse_continuous(*reversed_once)
                    reverse_involution_exact = bool(
                        reverse_involution_exact
                        and np.array_equal(reversed_twice[0], boundaries)
                        and np.array_equal(reversed_twice[1], intervals)
                    )
                    pair_identity_order_exact = bool(
                        pair_identity_order_exact
                        and np.array_equal(selected, np.sort(selected))
                        and len(selected) == len(np.unique(selected))
                        and forward.shape == reverse.shape
                    )
                    mask = (
                        UPSTREAM.reversible_null_swap_mask(
                            source_date, selected, null_seed
                        )
                        if null_seed is not None
                        else None
                    )
                    forward, reverse = UPSTREAM.orient_pair(forward, reverse, mask)
                    forward_tensor = torch.from_numpy(scaler.transform(forward)).to(
                        device
                    )
                    reverse_tensor = torch.from_numpy(scaler.transform(reverse)).to(
                        device
                    )
                    raw_tensor = UPSTREAM.antisymmetric_score(
                        model, forward_tensor, reverse_tensor
                    )
                    reverse_tensor_score = UPSTREAM.antisymmetric_score(
                        model, reverse_tensor, forward_tensor
                    )
                    raw_score = raw_tensor.detach().cpu().numpy().astype(np.float64)
                    reverse_raw_score = (
                        reverse_tensor_score.detach().cpu().numpy().astype(np.float64)
                    )
                    calibrated = alpha * raw_score
                    reverse_calibrated = alpha * reverse_raw_score
                    finite = bool(
                        np.isfinite(forward).all()
                        and np.isfinite(reverse).all()
                        and np.isfinite(raw_score).all()
                        and np.isfinite(reverse_raw_score).all()
                        and np.isfinite(calibrated).all()
                        and np.isfinite(reverse_calibrated).all()
                    )
                    raw_and_calibrated_finite = bool(
                        raw_and_calibrated_finite and finite
                    )
                    if not finite:
                        raise RuntimeError(
                            f"nonfinite token/score date={source_date} batch={first}"
                        )
                    probability = 1.0 / (
                        1.0 + np.exp(-np.clip(calibrated, -40.0, 40.0))
                    )
                    reverse_probability = 1.0 / (
                        1.0 + np.exp(-np.clip(reverse_calibrated, -40.0, 40.0))
                    )
                    max_antisymmetry_error = max(
                        max_antisymmetry_error,
                        float(np.max(np.abs(raw_score + reverse_raw_score))),
                    )
                    max_probability_pair_error = max(
                        max_probability_pair_error,
                        float(np.max(np.abs(probability + reverse_probability - 1.0))),
                    )
                    if model_kind == "pooled_mlp" and permutation_error is None:
                        permutation = torch.arange(k - 1, -1, -1, device=device)
                        permutation_error = float(
                            torch.max(
                                torch.abs(
                                    model(forward_tensor)
                                    - model(forward_tensor[:, permutation, :])
                                )
                            )
                            .detach()
                            .cpu()
                        )
                    path_start_ns = day.block_start_ns[selected]
                    path_end_ns = day.block_end_ns[selected + k - 1]
                    duration_seconds = (path_end_ns - path_start_ns) / 1e9
                    UPSTREAM._update_metrics(
                        pooled, calibrated, event_count, duration_seconds
                    )
                    UPSTREAM._update_metrics(
                        day_metric, calibrated, event_count, duration_seconds
                    )
                    bounded = np.clip(calibrated, -np.log(99.0), np.log(99.0))
                    UPSTREAM._update_metrics(
                        bounded_pooled, bounded, event_count, duration_seconds
                    )
                    UPSTREAM._update_metrics(
                        bounded_day_metric, bounded, event_count, duration_seconds
                    )
                    if prediction_path is not None:
                        count = len(selected)
                        pair_ids = [
                            f"{source_date}:B{target_b}:K{k}:"
                            f"{int(day.block_index[index])}"
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
                by_day[source_date] = UPSTREAM._finalize_metrics(day_metric, k)
                bounded_by_day[source_date] = UPSTREAM._finalize_metrics(
                    bounded_day_metric, k
                )
                if heartbeat is not None:
                    heartbeat.update(
                        chunk_index=len(by_day),
                        rows_processed=sum(
                            int(row["paths"]) for row in by_day.values()
                        ),
                        bytes_processed=sum(date_bytes[day] for day in by_day),
                    )
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, prediction_path)
            if heartbeat is not None:
                heartbeat.update(checkpoint_status="prediction_parquet_published")
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
            "pooled_path_weighted": UPSTREAM._finalize_metrics(pooled, k),
            "by_day": by_day,
            "day_equal": {
                key: float(np.mean([row[key] for row in by_day.values()]))
                for key in day_equal_keys
            },
            "bounded_log99": {
                "pooled_path_weighted": UPSTREAM._finalize_metrics(bounded_pooled, k),
                "by_day": bounded_by_day,
                "day_equal": {
                    key: float(np.mean([row[key] for row in bounded_by_day.values()]))
                    for key in day_equal_keys
                },
            },
        }
        audit = {
            "score_tolerance": 1e-6,
            "pooled_permutation_tolerance": 2e-5,
            "score_antisymmetry_max_abs_error": max_antisymmetry_error,
            "probability_pair_max_abs_error": max_probability_pair_error,
            "permutation_max_abs_error": permutation_error,
            "finite_metrics": bool(
                np.isfinite(list(metrics["day_equal"].values())).all()
                and np.isfinite(
                    list(metrics["bounded_log99"]["day_equal"].values())
                ).all()
            ),
            "raw_and_calibrated_finite": raw_and_calibrated_finite,
            "reverse_involution_exact": reverse_involution_exact,
            "pair_identity_order_exact": pair_identity_order_exact,
        }
        audit["passed"] = bool(
            audit["finite_metrics"]
            and audit["raw_and_calibrated_finite"]
            and audit["reverse_involution_exact"]
            and audit["pair_identity_order_exact"]
            and max_antisymmetry_error <= audit["score_tolerance"]
            and max_probability_pair_error <= audit["score_tolerance"]
            and (
                permutation_error is None
                or permutation_error <= audit["pooled_permutation_tolerance"]
            )
        )
        return metrics, audit
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _prediction_identity_equal(left: Path, right: Path) -> bool:
    columns = (
        "pair_id",
        "source_date",
        "start_block_index",
        "path_start_ns",
        "path_end_ns",
        "path_event_count",
        "target_b",
        "k",
        "orientation",
        "label",
    )
    left_file = pq.ParquetFile(left)
    right_file = pq.ParquetFile(right)
    if left_file.metadata.num_rows != right_file.metadata.num_rows:
        return False
    left_batches = left_file.iter_batches(batch_size=65_536, columns=columns)
    right_batches = right_file.iter_batches(batch_size=65_536, columns=columns)
    for left_batch, right_batch in zip_longest(left_batches, right_batches):
        if left_batch is None or right_batch is None:
            return False
        if left_batch.num_rows != right_batch.num_rows:
            return False
        for name in columns:
            if not left_batch.column(name).equals(right_batch.column(name)):
                return False
    return True


def full_unit_benchmark(
    *,
    device_name: str,
    output_root: Path,
    work_root: Path,
    log_dir: Path,
    model_kind: str = "ordered_tcn",
    seed: int = 20260730,
    hourly_cost_usd: float = 0.69,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the immutable full scientific unit with CUDA-only runtime overrides."""
    configure_determinism()
    device = device_or_raise(device_name)
    if device.type != "cuda":
        raise ValueError("the qualification benchmark requires CUDA")
    original = {
        "evaluate_loss": UPSTREAM.evaluate_loss,
        "train_stage": UPSTREAM.train_stage,
        "collect_scores_by_day": UPSTREAM.collect_scores_by_day,
        "test_predictions_and_metrics": UPSTREAM.test_predictions_and_metrics,
    }
    UPSTREAM.evaluate_loss = lambda *values, **named: _cuda_evaluate_loss(
        *values, **named, log_path=log_dir / "benchmark.jsonl"
    )
    UPSTREAM.train_stage = lambda **values: _cuda_train_stage(device, **values)
    UPSTREAM.collect_scores_by_day = (
        lambda *values, **named: _cuda_collect_scores_by_day(
            *values, **named, log_path=log_dir / "benchmark.jsonl"
        )
    )
    UPSTREAM.test_predictions_and_metrics = (
        lambda **named: _cuda_test_predictions_and_metrics(
            **named, log_path=log_dir / "benchmark.jsonl"
        )
    )
    args = SimpleNamespace(
        input_root=INPUT_ROOT,
        output_root=output_root,
        work_root=work_root,
        log_dir=log_dir,
        target_b=10,
        k=6,
        model=model_kind,
        seed=seed,
        threads=1,
        batch_size=1_024,
        scaler_paths_per_day=10_000,
        max_epochs=30,
        min_epochs=6,
        patience=3,
        checkpoint_batches=50,
        resume=resume,
    )
    final_dir = output_root / "B10_K6" / model_kind / f"seed={seed}"
    unit_work = work_root / "B10_K6" / model_kind / f"seed={seed}"
    unit_work.mkdir(parents=True, exist_ok=True)
    runtime_path = unit_work / "benchmark_runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {
            "attempts": 0,
            "cumulative_elapsed_seconds": 0.0,
            "cumulative_cost_usd": 0.0,
        }
    )
    preexisting_complete = (final_dir / "complete_manifest.json").is_file()
    started = time.monotonic()
    succeeded = False
    try:
        UPSTREAM.run_unit(args)
        succeeded = True
    finally:
        elapsed = time.monotonic() - started
        if not preexisting_complete:
            runtime["attempts"] = int(runtime["attempts"]) + 1
            runtime["cumulative_elapsed_seconds"] = (
                float(runtime["cumulative_elapsed_seconds"]) + elapsed
            )
            runtime["cumulative_cost_usd"] = (
                float(runtime["cumulative_cost_usd"])
                + elapsed * hourly_cost_usd / 3600.0
            )
        runtime["last_attempt_seconds"] = elapsed
        runtime["last_status"] = "complete" if succeeded else "interrupted_or_failed"
        UPSTREAM.atomic_write_json(runtime_path, runtime)
        for name, value in original.items():
            setattr(UPSTREAM, name, value)
    manifest = json.loads(
        (final_dir / "complete_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((final_dir / "metrics.json").read_text(encoding="utf-8"))
    reference_dir = unit_dir(model_kind, seed)
    reference_metrics = json.loads(
        (reference_dir / "metrics.json").read_text(encoding="utf-8")
    )
    selection_equal = bool(
        metrics["selection"]["selected_candidate_index"]
        == reference_metrics["selection"]["selected_candidate_index"]
        and metrics["selection"]["final_fit_epochs"]
        == reference_metrics["selection"]["final_fit_epochs"]
    )
    prediction_order_equal = _prediction_identity_equal(
        final_dir / "test_predictions.parquet",
        reference_dir / "test_predictions.parquet",
    )
    expected_prediction_rows = 2 * UPSTREAM.APPROVED_PATH_TOTALS["test"]
    output_bytes = sum(
        path.stat().st_size for path in final_dir.iterdir() if path.is_file()
    )
    passed = bool(
        not preexisting_complete
        and manifest.get("status") == "complete"
        and manifest.get("test_prediction_rows") == expected_prediction_rows
        and manifest.get("null_bias_gate", {}).get("passed") is True
        and metrics.get("audit", {}).get("passed") is True
        and metrics.get("reversible_null", {}).get("audit", {}).get("passed") is True
        and selection_equal
        and prediction_order_equal
        and np.isfinite(
            [
                metrics["test"]["day_equal"]["nats_path"],
                metrics["reversible_null"]["test"]["day_equal"]["nats_path"],
            ]
        ).all()
    )
    row = {
        "device": device_name,
        "model": model_kind,
        "seed": seed,
        "execution_state": "skipped_complete" if preexisting_complete else "executed",
        "attempts": int(runtime["attempts"]),
        "elapsed_seconds": float(runtime["cumulative_elapsed_seconds"]),
        "hourly_cost_usd": hourly_cost_usd,
        "measured_cost_usd": float(runtime["cumulative_cost_usd"]),
        "development_paths": UPSTREAM.APPROVED_PATH_TOTALS["development"],
        "calibration_paths": UPSTREAM.APPROVED_PATH_TOTALS["calibration"],
        "test_paths": UPSTREAM.APPROVED_PATH_TOTALS["test"],
        "day_equal_nats_path": metrics["test"]["day_equal"]["nats_path"],
        "null_day_equal_nats_path": metrics["reversible_null"]["test"]["day_equal"][
            "nats_path"
        ],
        "null_bias_gate_passed": manifest["null_bias_gate"]["passed"],
        "scientific_audit_passed": metrics["audit"]["passed"],
        "null_audit_passed": metrics["reversible_null"]["audit"]["passed"],
        "reference_selection_equal": selection_equal,
        "reference_prediction_order_equal": prediction_order_equal,
        "test_prediction_rows": manifest["test_prediction_rows"],
        "output_bytes_before_cleanup": output_bytes,
        "passed": passed,
    }
    # This qualification deliberately does not retain a duplicate scientific
    # model artifact or its upstream-style hash manifest.  The compact,
    # validated benchmark row is the durable Phase 01 output.
    shutil.rmtree(final_dir)
    shutil.rmtree(unit_work)
    return row
