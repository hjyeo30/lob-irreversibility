"""Paper-facing 34-channel and circular-padding controls.

This file extracts the scientific definitions from the production control
runner. Provider scheduling, billing, recovery, and artifact transport are not
part of the review artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


BOUNDARY_WIDTH = 7
FULL_INTERVAL_WIDTH = 23
G0_CHANNELS = ("elapsed_ns", "realized_source_event_count", "completion_overshoot")


def robust_center_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Development-only median and normal-IQR scale used in the paper."""
    center = np.median(matrix, axis=0)
    lower, upper = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = (upper - lower) / 1.349
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


@dataclass(frozen=True)
class Reduced34Scaler:
    """Scaler fitted after removing the three activity-clock channels."""

    boundary_center: np.ndarray
    boundary_scale: np.ndarray
    interval_indices: np.ndarray
    interval_center: np.ndarray
    interval_scale: np.ndarray
    sample_tokens: int

    CONTRACT = "phase13_development_34_channel_scaler_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "boundary_center", np.asarray(self.boundary_center, dtype=np.float32)
        )
        object.__setattr__(
            self, "boundary_scale", np.asarray(self.boundary_scale, dtype=np.float32)
        )
        object.__setattr__(
            self, "interval_indices", np.asarray(self.interval_indices, dtype=np.int64)
        )
        object.__setattr__(
            self, "interval_center", np.asarray(self.interval_center, dtype=np.float32)
        )
        object.__setattr__(
            self, "interval_scale", np.asarray(self.interval_scale, dtype=np.float32)
        )
        if self.boundary_center.shape != (7,) or self.boundary_scale.shape != (7,):
            raise ValueError("boundary scaler width must be seven")
        if (
            self.interval_indices.shape != (20,)
            or len(np.unique(self.interval_indices)) != 20
        ):
            raise ValueError("active interval width must be twenty")
        if self.interval_center.shape != (20,) or self.interval_scale.shape != (20,):
            raise ValueError("interval scaler width must be twenty")

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != 37:
            raise ValueError("input tokens must have 37 channels")
        start = (values[..., :7] - self.boundary_center) / self.boundary_scale
        interval = (
            values[..., 7 + self.interval_indices] - self.interval_center
        ) / self.interval_scale
        end = (values[..., -7:] - self.boundary_center) / self.boundary_scale
        result = np.concatenate((start, interval, end), axis=-1).astype(
            np.float32, copy=False
        )
        if result.shape[-1] != 34:
            raise RuntimeError("reduced representation is not 34-wide")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.CONTRACT,
            "feature_count": 34,
            "removed_channels": list(G0_CHANNELS),
            "boundary_center": self.boundary_center.tolist(),
            "boundary_scale": self.boundary_scale.tolist(),
            "interval_indices": self.interval_indices.tolist(),
            "interval_center": self.interval_center.tolist(),
            "interval_scale": self.interval_scale.tolist(),
            "sample_tokens": int(self.sample_tokens),
            "method": "development-only median/IQR after exact raw G0 removal",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Reduced34Scaler":
        if (
            value.get("contract") != cls.CONTRACT
            or value.get("feature_count") != 34
            or tuple(value.get("removed_channels", ())) != G0_CHANNELS
        ):
            raise ValueError("34-channel scaler contract mismatch")
        return cls(
            np.asarray(value["boundary_center"]),
            np.asarray(value["boundary_scale"]),
            np.asarray(value["interval_indices"]),
            np.asarray(value["interval_center"]),
            np.asarray(value["interval_scale"]),
            int(value["sample_tokens"]),
        )


def fit_reduced34_scaler(
    token_batches: Sequence[np.ndarray], interval_fields: Sequence[str]
) -> Reduced34Scaler:
    """Fit on forward and reverse development tokens supplied by the caller."""
    interval_fields = tuple(interval_fields)
    if len(interval_fields) != FULL_INTERVAL_WIDTH:
        raise ValueError("the full representation must have 23 interval channels")
    indices = np.asarray(
        [
            index
            for index, name in enumerate(interval_fields)
            if name not in G0_CHANNELS
        ],
        dtype=np.int64,
    )
    if len(indices) != 20 or any(name not in interval_fields for name in G0_CHANNELS):
        raise ValueError("activity-clock registry mismatch")
    batches = [np.asarray(batch) for batch in token_batches]
    if not batches or any(
        batch.ndim != 3 or batch.shape[-1] != 37 for batch in batches
    ):
        raise ValueError("expected non-empty batches with shape (paths, blocks, 37)")
    boundary = np.concatenate(
        [
            np.concatenate((batch[..., :7], batch[..., -7:]), axis=1).reshape(-1, 7)
            for batch in batches
        ]
    )
    interval = np.concatenate(
        [batch[..., 7 + indices].reshape(-1, 20) for batch in batches]
    )
    boundary_center, boundary_scale = robust_center_scale(boundary)
    interval_center, interval_scale = robust_center_scale(interval)
    sample_tokens = sum(int(batch.shape[0] * batch.shape[1]) for batch in batches)
    return Reduced34Scaler(
        boundary_center,
        boundary_scale,
        indices,
        interval_center,
        interval_scale,
        sample_tokens,
    )


def build_control_model(
    model: str, feature_count: int = 37, *, circular: bool = False
) -> Any:
    """Build the exact standard or circular model from the shared runtime."""
    from src.cuda_runtime import _build_model

    variant = "circular_padding" if circular else "standard"
    return _build_model(model, variant, input_channels=feature_count)


def smoke() -> None:
    from src.cuda_runtime import UPSTREAM

    rng = np.random.default_rng(20260810)
    batch = rng.normal(size=(32, 6, 37)).astype(np.float32)
    scaler = fit_reduced34_scaler([batch], UPSTREAM.INTERVAL_FIELDS)
    assert scaler.transform(batch).shape == (32, 6, 34)
    assert build_control_model("pooled_mlp", 34)(
        __import__("torch").from_numpy(scaler.transform(batch))
    ).shape == (32,)
    assert build_control_model("ordered_tcn", 37, circular=True)(
        __import__("torch").from_numpy(batch)
    ).shape == (32,)


if __name__ == "__main__":
    smoke()
    print("short-term controls smoke passed")
