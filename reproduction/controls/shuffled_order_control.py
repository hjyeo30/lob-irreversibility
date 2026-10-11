"""Deterministic shuffled-block control used in the paper.

Every path receives a non-identity permutation drawn uniformly from the
``K! - 1`` allowed permutations. The reverse member is constructed only after
permutation, as ``R(P_pi(W))``. The pooled anchor is reused only after an exact
permutation-invariance gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


K = 6
SEEDS = (20260730, 20260731, 20260732)
POOLED_PERMUTATION_TOLERANCE = 2e-5


def canonical_key(path_key: Any) -> bytes:
    if isinstance(path_key, bytes):
        return path_key
    if isinstance(path_key, np.generic):
        path_key = path_key.item()
    if isinstance(path_key, (str, int)):
        return str(path_key).encode("utf-8")
    return json.dumps(path_key, sort_keys=True, separators=(",", ":")).encode("utf-8")


def draw_nonidentity(generator: np.random.Generator, k: int) -> np.ndarray:
    identity = np.arange(k, dtype=np.int64)
    while True:
        candidate = generator.permutation(k).astype(np.int64, copy=False)
        if not np.array_equal(candidate, identity):
            return candidate


def deterministic_nonidentity_permutation(
    path_key: Any, seed: int, k: int = K
) -> np.ndarray:
    material = (
        seed.to_bytes(8, "little", signed=False) + b"\0" + canonical_key(path_key)
    )
    digest = hashlib.sha256(material).digest()
    generator = np.random.default_rng(
        int.from_bytes(digest[:16], "little", signed=False)
    )
    return draw_nonidentity(generator, k)


def reverse_paths(forward: np.ndarray) -> np.ndarray:
    """Apply physical reversal using the shared 37-channel registry."""
    from src.cuda_runtime import UPSTREAM

    if forward.ndim != 3 or forward.shape[-1] != 37:
        raise ValueError("forward paths must have shape (paths, blocks, 37)")
    boundary = np.concatenate((forward[:, :1, :7], forward[:, :, -7:]), axis=1)
    interval = forward[:, :, 7:-7]
    reverse_boundary, reverse_interval = UPSTREAM.reverse_continuous(boundary, interval)
    return UPSTREAM.make_tokens(reverse_boundary, reverse_interval).reshape(
        forward.shape
    )


@dataclass(frozen=True)
class PairedShuffledBatch:
    forward: np.ndarray
    reverse: np.ndarray
    permutations: np.ndarray
    path_keys: tuple[Any, ...]
    seed: int


def paired_shuffle(
    forward: np.ndarray, path_keys: Sequence[Any], seed: int
) -> PairedShuffledBatch:
    values = np.asarray(forward)
    if values.ndim != 3 or values.shape[1:] != (K, 37):
        raise ValueError("expected shape (paths, 6, 37)")
    if len(path_keys) != len(values):
        raise ValueError("path key count mismatch")
    permutations = np.stack(
        [deterministic_nonidentity_permutation(key, seed, K) for key in path_keys]
    )
    shuffled = np.take_along_axis(values, permutations[:, :, None], axis=1)
    return PairedShuffledBatch(
        forward=shuffled,
        reverse=reverse_paths(shuffled),
        permutations=permutations,
        path_keys=tuple(path_keys),
        seed=int(seed),
    )


def pooled_prediction_equality_gate(
    original: np.ndarray,
    shuffled: np.ndarray,
    *,
    tolerance: float = POOLED_PERMUTATION_TOLERANCE,
) -> dict[str, float | bool]:
    left, right = (
        np.asarray(original, dtype=np.float64),
        np.asarray(shuffled, dtype=np.float64),
    )
    if left.shape != right.shape or left.size == 0:
        raise ValueError("pooled predictions must have the same non-empty shape")
    maximum = float(np.max(np.abs(left - right)))
    passed = bool(math.isfinite(maximum) and maximum <= tolerance)
    if not passed:
        raise RuntimeError(f"pooled permutation gate failed: {maximum} > {tolerance}")
    return {"passed": True, "max_abs_error": maximum, "tolerance": float(tolerance)}


def smoke() -> None:
    rng = np.random.default_rng(20260810)
    # The physical-reversal helper applies log1p to size/duration channels, so
    # the fixture must respect their non-negative domain.
    paths = (0.1 + rng.random(size=(128, K, 37))).astype(np.float64)
    paired = paired_shuffle(paths, list(range(len(paths))), SEEDS[0])
    identity = np.arange(K)
    assert not any(np.array_equal(row, identity) for row in paired.permutations)
    assert np.array_equal(
        np.sort(paired.forward, axis=1),
        np.sort(paths, axis=1),
    )
    assert np.allclose(reverse_paths(paired.forward), paired.reverse)
    pooled_prediction_equality_gate(np.ones(128), np.ones(128))


if __name__ == "__main__":
    smoke()
    print("shuffled-order control smoke passed")
