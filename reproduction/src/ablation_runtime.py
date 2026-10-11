"""Shared Phase 06 feature-ablation calculation runtime.

The phase-local controller owns admission.  This module owns the exact feature
registry, immutable scaler use, masked model input, restartable scientific unit,
and compact Arrow schemas.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.cuda_runtime import (
    _cuda_collect_scores_by_day,
    _cuda_test_predictions_and_metrics,
    _cuda_train_stage,
    configure_determinism,
    device_or_raise,
    fit_positive_day_equal_calibration_with_heartbeat,
)
from src.scale_runtime import (
    DATES,
    EXPECTED_PATHS,
    UPSTREAM,
    atomic_json,
    input_preflight,
)

MODELS = ("pooled_mlp", "ordered_tcn")
SCREEN_SEED = 20260730
EXPANSION_SEEDS = (20260731, 20260732)
ALL_SEEDS = (SCREEN_SEED, *EXPANSION_SEEDS)
FULL_ALIAS = "full_gall"
FEATURE_ALIASES = {
    "ladder_g0_gs_gf_gl": "drop_gp",
    "ladder_g0_gs_gf_gl_gp": FULL_ALIAS,
}
FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    "only_gs": ("GS",),
    "only_gf": ("GF",),
    "only_gl": ("GL",),
    "only_gp": ("GP",),
    "drop_gs": ("G0", "GF", "GL", "GP"),
    "drop_gf": ("G0", "GS", "GL", "GP"),
    "drop_gl": ("G0", "GS", "GF", "GP"),
    "drop_gp": ("G0", "GS", "GF", "GL"),
    "ladder_g0": ("G0",),
    "ladder_g0_gs": ("G0", "GS"),
    "ladder_g0_gs_gf": ("G0", "GS", "GF"),
}
DIRECT_FEATURE_CONFIG_CHANNELS: dict[str, tuple[str, ...]] = {}
BOUNDARY = tuple(UPSTREAM.BOUNDARY_FIELDS)
INTERVAL = tuple(UPSTREAM.INTERVAL_FIELDS)
CHANNELS = tuple(
    [f"start_{name}" for name in BOUNDARY]
    + list(INTERVAL)
    + [f"end_{name}" for name in BOUNDARY]
)
GROUP_CHANNELS: dict[str, tuple[str, ...]] = {
    "G0": ("elapsed_ns", "realized_source_event_count", "completion_overshoot"),
    "GS": tuple(
        [f"start_{name}" for name in BOUNDARY] + [f"end_{name}" for name in BOUNDARY]
    ),
    "GF": (
        "bid_depth_l1_net_change",
        "ask_depth_l1_net_change",
        "bid_depth_l5_net_change",
        "ask_depth_l5_net_change",
        "bid_depth_l1_gross_variation",
        "ask_depth_l1_gross_variation",
        "bid_depth_l5_gross_variation",
        "ask_depth_l5_gross_variation",
    ),
    "GL": (
        "appearance_minus_depletion_count",
        "depletion_plus_appearance_count",
        "bid_best_depletion_count",
        "bid_best_appearance_count",
        "ask_best_depletion_count",
        "ask_best_appearance_count",
    ),
    "GP": (
        "mid_log_return",
        "mid_change_half_ticks",
        "bid_quote_change_ticks",
        "ask_quote_change_ticks",
        "mid_abs_variation_half_ticks",
        "spread_abs_variation_ticks",
    ),
}


def validate_feature_registry() -> dict[str, Any]:
    occurrences = {name: 0 for name in CHANNELS}
    for names in GROUP_CHANNELS.values():
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate channel inside feature group")
        for name in names:
            if name not in occurrences:
                raise RuntimeError(f"unknown grouped channel: {name}")
            occurrences[name] += 1
    invalid = {name: count for name, count in occurrences.items() if count != 1}
    if len(CHANNELS) != 37 or invalid:
        raise RuntimeError(f"feature partition is not exact: {invalid}")
    masks = {name: active_mask(name) for name in FEATURE_CONFIGS}
    serialized = {name: tuple(np.flatnonzero(mask)) for name, mask in masks.items()}
    if len(set(serialized.values())) != 11:
        raise RuntimeError("canonical feature masks are not 11 unique masks")
    return {
        "channels": len(CHANNELS),
        "group_sizes": {name: len(values) for name, values in GROUP_CHANNELS.items()},
        "unique_new_masks": len(set(serialized.values())),
        "aliases": FEATURE_ALIASES,
    }


def canonical_feature_config(value: str) -> str:
    canonical = FEATURE_ALIASES.get(value, value)
    if canonical == FULL_ALIAS or (
        canonical not in FEATURE_CONFIGS
        and canonical not in DIRECT_FEATURE_CONFIG_CHANNELS
    ):
        raise ValueError(f"not a new Phase 06 feature config: {value}")
    return canonical


def register_direct_feature_configs(
    values: dict[str, tuple[str, ...]],
) -> None:
    """Register phase-local direct masks without changing the Phase 06 registry."""
    for config, names in values.items():
        if (
            config in FEATURE_CONFIGS
            or config in FEATURE_ALIASES
            or config == FULL_ALIAS
        ):
            raise ValueError(f"direct feature config collides with Phase 06: {config}")
        if (
            not names
            or len(names) != len(set(names))
            or any(name not in CHANNELS for name in names)
        ):
            raise ValueError(f"invalid direct feature config: {config}")
        prior = DIRECT_FEATURE_CONFIG_CHANNELS.get(config)
        normalized = tuple(names)
        if prior is not None and prior != normalized:
            raise RuntimeError(f"direct feature config changed in process: {config}")
        DIRECT_FEATURE_CONFIG_CHANNELS[config] = normalized


def active_mask(feature_config: str) -> np.ndarray:
    canonical = FEATURE_ALIASES.get(feature_config, feature_config)
    if canonical in DIRECT_FEATURE_CONFIG_CHANNELS:
        active = set(DIRECT_FEATURE_CONFIG_CHANNELS[canonical])
    else:
        groups = (
            ("G0", "GS", "GF", "GL", "GP")
            if canonical == FULL_ALIAS
            else FEATURE_CONFIGS[canonical]
        )
        active = {channel for group in groups for channel in GROUP_CHANNELS[group]}
    return np.asarray([name in active for name in CHANNELS], dtype=np.bool_)


class MaskedScaler:
    """Read-only full scaler followed by a fixed zero mask."""

    def __init__(self, base: Any, feature_config: str) -> None:
        self.base = base
        self.feature_config = canonical_feature_config(feature_config)
        self.mask = active_mask(self.feature_config)

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(self.base.transform(values), dtype=np.float32)
        if transformed.shape[-1] != 37:
            raise RuntimeError(f"unexpected transformed width: {transformed.shape}")
        transformed[..., ~self.mask] = 0.0
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "feature_config": self.feature_config,
            "active_channel_indices": np.flatnonzero(self.mask).tolist(),
        }


@dataclass(frozen=True)
class AblationUnitConfig:
    feature_config: str
    model: str
    seed: int
    device: str
    input_root: Path
    anchor_root: Path
    output_root: Path
    work_root: Path
    log_root: Path
    fold_scaler_root: Path
    resume_qualification_path: Path
    hourly_cost_usd: float = 0.69
    batch_size: int = 1_024
    scaler_paths_per_day: int = 10_000
    max_epochs: int = 30
    min_epochs: int = 6
    patience: int = 3
    checkpoint_batches: int = 50
    resume: bool = True

    @property
    def canonical_config(self) -> str:
        return canonical_feature_config(self.feature_config)

    @property
    def unit_id(self) -> str:
        return f"{self.canonical_config}/{self.model}/seed={self.seed}"

    def scientific_config(self) -> dict[str, Any]:
        excluded = {
            "device",
            "input_root",
            "anchor_root",
            "output_root",
            "work_root",
            "log_root",
            "fold_scaler_root",
            "resume_qualification_path",
            "hourly_cost_usd",
            "resume",
        }
        value = {key: item for key, item in asdict(self).items() if key not in excluded}
        active_groups = (
            []
            if self.canonical_config in DIRECT_FEATURE_CONFIG_CHANNELS
            else list(FEATURE_CONFIGS[self.canonical_config])
        )
        value.update(
            {
                "feature_config": self.canonical_config,
                "active_groups": active_groups,
                "active_channels": [
                    name
                    for name, keep in zip(
                        CHANNELS, active_mask(self.canonical_config), strict=True
                    )
                    if keep
                ],
                "B": 10,
                "K": 6,
                "dates": list(DATES),
                "split": {
                    "development": list(DATES[:9]),
                    "calibration": list(DATES[9:12]),
                    "test": list(DATES[12:]),
                },
                "path_stride": 3,
                "path_offset_rule": "date.toordinal()%3",
                "feature_projection": "continuous_path_37_v1",
                "mask_order": "physical_reverse_then_full_scaler_then_zero_inactive",
                "model_implementation": "immutable_upstream_phase05_v1",
                "adamw_candidates": list(UPSTREAM.ADAMW_CANDIDATES),
                "reversible_null_seed": UPSTREAM.NULL_SEED,
                "fp32": True,
                "tf32": False,
                "amp": False,
            }
        )
        return value


def _anchor_scaler_paths(anchor_root: Path) -> list[Path]:
    return [
        anchor_root / model / f"seed={seed}" / "scaler.json"
        for model in MODELS
        for seed in ALL_SEEDS
    ]


def load_immutable_full_scaler(anchor_root: Path) -> tuple[Any, dict[str, Any]]:
    paths = _anchor_scaler_paths(anchor_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"immutable full scaler missing: {missing}")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError("six immutable full scalers are not exact equal")
    return UPSTREAM.ChannelScaler.from_dict(values[0]), {
        "canonical_path": str(paths[0]),
        "compared_paths": [str(path) for path in paths],
        "exact_equality": True,
        "reuse": True,
    }


def ensure_fold_scalers(
    config: AblationUnitConfig, *, verify_regeneration: bool = False
) -> dict[int, Any]:
    config.fold_scaler_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.fold_scaler_root / ".writer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        split = UPSTREAM.chronological_day_split(DATES)
        folds = tuple(UPSTREAM.expanding_inner_folds(split.development))
        expected_paths = {
            fold.fold: config.fold_scaler_root / f"fold={fold.fold}.json"
            for fold in folds
        }
        if (
            all(path.is_file() for path in expected_paths.values())
            and not verify_regeneration
        ):
            return {
                fold: UPSTREAM.ChannelScaler.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                for fold, path in expected_paths.items()
            }
        result: dict[int, Any] = {}
        for fold in folds:
            path = config.fold_scaler_root / f"fold={fold.fold}.json"
            computed = UPSTREAM.fit_development_scaler(
                config.input_root, fold.train, 10, 6, 10_000
            ).to_dict()
            if path.is_file():
                if json.loads(path.read_text(encoding="utf-8")) != computed:
                    raise RuntimeError(f"shared fold scaler mismatch: {path}")
            else:
                atomic_json(path, computed)
            result[fold.fold] = UPSTREAM.ChannelScaler.from_dict(computed)
        return result


def ablation_input_preflight(
    config: AblationUnitConfig, *, deep: bool = False
) -> dict[str, Any]:
    from src.scale_runtime import ScaleUnitConfig

    proxy = ScaleUnitConfig(
        10,
        6,
        "pooled_mlp",
        SCREEN_SEED,
        "cuda",
        config.input_root,
        config.output_root,
        config.work_root,
        config.log_root,
    )
    observed = input_preflight(proxy, deep=deep)
    if observed["path_counts"] != EXPECTED_PATHS[(10, 6)]:
        raise RuntimeError("Phase 06 exact path count mismatch")
    _, scaler_reference = load_immutable_full_scaler(config.anchor_root)
    return {
        **observed,
        "scaler_reference": scaler_reference,
        "feature_registry": validate_feature_registry(),
        "resume_qualification": validate_resume_qualification(
            config.resume_qualification_path
        ),
    }


def validate_resume_qualification(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume qualification missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit") != "resume_probe_cuda"
        or value.get("checks") != 1
        or not value.get("artifacts")
    ):
        raise RuntimeError("Phase 01 exact resume qualification did not pass")
    artifact_root = path.parents[2]
    for item in value["artifacts"]:
        artifact = artifact_root / item["filename"]
        if not artifact.is_file() or artifact.stat().st_size != item["bytes"]:
            raise RuntimeError(f"resume qualification artifact mismatch: {artifact}")
    return {"path": str(path), "status": "complete", "checks": 1, "exact": True}


def synthetic_bounded_smoke() -> dict[str, Any]:
    class IdentityScaler:
        def transform(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.float32).copy()

        def to_dict(self) -> dict[str, Any]:
            return {"kind": "identity_synthetic"}

    rng = np.random.default_rng(20260730)
    peak = 0
    checked = 0
    for feature_config in FEATURE_CONFIGS:
        scaler = MaskedScaler(IdentityScaler(), feature_config)
        for _ in range(4):
            batch = rng.normal(size=(32, 6, 37)).astype(np.float32)
            transformed = scaler.transform(batch)
            peak = max(peak, batch.nbytes + transformed.nbytes)
            if np.any(transformed[..., ~scaler.mask] != 0.0):
                raise RuntimeError("synthetic inactive channel is nonzero")
            checked += len(batch)
    row = {
        "stage": "synthetic",
        "symbol": "NVDA",
        "source_date": "2026-07-21",
        "train_regime": "daytime",
        "test_regime": "daytime",
        "B": 10,
        "K": 6,
        "model": "pooled_mlp",
        "feature_config": "full_gall",
        "seed": SCREEN_SEED,
        "score_contract": "bounded_log99",
        "run_unit_id": "synthetic",
        "n_paths": 1,
        "realized_source_events": 60,
        "elapsed_seconds": 1.0,
        "nats_path": 0.1,
        "nats_block": 0.1 / 6,
        "nats_event": 0.1 / 60,
        "nats_second": 0.1,
        "paired_log_loss": 0.6,
        "calibration_alpha": 1.0,
        "calibrated_ovl": 0.8,
        "posterior_saturation_share": 0.0,
        "reverse_like_share": 0.4,
        "null_nats_path": None,
        "null_score_contract": "unavailable_upstream_no_null_predictions",
    }
    with tempfile.TemporaryDirectory(prefix="phase06-smoke-") as temporary:
        output = Path(temporary) / "unit_metrics.parquet"
        atomic_parquet(output, [row], UNIT_METRICS_SCHEMA)
        if pq.ParquetFile(output).metadata.num_rows != 1:
            raise RuntimeError("synthetic atomic Parquet row mismatch")
    return {
        "status": "complete",
        "registry": validate_feature_registry(),
        "batches": checked // 32,
        "paths": checked,
        "peak_live_array_bytes": peak,
        "atomic_parquet_rows": 1,
    }


def _strict(schema: pa.Schema, nullable: set[str] | None = None) -> pa.Schema:
    allowed = nullable or set()
    return pa.schema(
        [
            pa.field(field.name, field.type, nullable=field.name in allowed)
            for field in schema
        ]
    )


UNIT_METRICS_SCHEMA = _strict(
    pa.schema(
        [
            ("stage", pa.string()),
            ("symbol", pa.string()),
            ("source_date", pa.string()),
            ("train_regime", pa.string()),
            ("test_regime", pa.string()),
            ("B", pa.int16()),
            ("K", pa.int16()),
            ("model", pa.string()),
            ("feature_config", pa.string()),
            ("seed", pa.int64()),
            ("score_contract", pa.string()),
            ("run_unit_id", pa.string()),
            ("n_paths", pa.int64()),
            ("realized_source_events", pa.int64()),
            ("elapsed_seconds", pa.float64()),
            ("nats_path", pa.float64()),
            ("nats_block", pa.float64()),
            ("nats_event", pa.float64()),
            ("nats_second", pa.float64()),
            ("paired_log_loss", pa.float64()),
            ("calibration_alpha", pa.float64()),
            ("calibrated_ovl", pa.float64()),
            ("posterior_saturation_share", pa.float64()),
            ("reverse_like_share", pa.float64()),
            ("null_nats_path", pa.float64()),
            ("null_score_contract", pa.string()),
        ]
    ),
    {"null_nats_path"},
)

RATE_METRICS = ("nats_path", "nats_block", "nats_event", "nats_second")
STRING_LIST = pa.list_(pa.field("element", pa.string()))
PAIRED_CONTRAST_SCHEMA = _strict(
    pa.schema(
        [
            ("stage", pa.string()),
            ("symbol", pa.string()),
            ("source_date", pa.string()),
            ("train_regime", pa.string()),
            ("test_regime", pa.string()),
            ("B", pa.int16()),
            ("K", pa.int16()),
            ("feature_config", pa.string()),
            ("seed", pa.int64()),
            ("score_contract", pa.string()),
            *[
                (f"{prefix}_{metric}", pa.float64())
                for metric in RATE_METRICS
                for prefix in ("pooled", "ordered", "ordered_minus_pooled")
            ],
        ]
    )
)
ABLATION_CONTRAST_SCHEMA = _strict(
    pa.schema(
        [
            ("stage", pa.string()),
            ("symbol", pa.string()),
            ("source_date", pa.string()),
            ("train_regime", pa.string()),
            ("test_regime", pa.string()),
            ("B", pa.int16()),
            ("K", pa.int16()),
            ("model", pa.string()),
            ("feature_config", pa.string()),
            ("seed", pa.int64()),
            ("score_contract", pa.string()),
            ("analysis_labels", STRING_LIST),
            *[
                (f"{prefix}_{metric}", pa.float64())
                for metric in RATE_METRICS
                for prefix in (
                    "config",
                    "full",
                    "config_minus_full",
                    "full_minus_config",
                )
            ],
        ]
    )
)
SCREEN_DECISION_SCHEMA = _strict(
    pa.schema(
        [
            ("canonical_feature_config", pa.string()),
            ("analysis_labels", STRING_LIST),
            ("screen_seed", pa.int64()),
            ("both_units_valid", pa.bool_()),
            ("any_model_config_full_same_sign_3d", pa.bool_()),
            ("temporal_gap_change_same_sign_3d", pa.bool_()),
            ("support_not_worse", pa.bool_()),
            ("bounded_uncapped_gap_sign_match", pa.bool_()),
            ("exploratory_expand", pa.bool_()),
            ("decision_basis", pa.string()),
        ]
    )
)
PREDICTION_SCHEMA = _strict(
    pa.schema(
        [
            ("pair_id", pa.string()),
            ("source_date", pa.string()),
            ("start_block_index", pa.uint32()),
            ("path_start_ns", pa.int64()),
            ("path_end_ns", pa.int64()),
            ("path_event_count", pa.int64()),
            ("duration_seconds", pa.float64()),
            ("target_b", pa.int16()),
            ("k", pa.int16()),
            ("model", pa.string()),
            ("feature_config", pa.string()),
            ("seed", pa.int64()),
            ("orientation", pa.string()),
            ("label", pa.uint8()),
            ("raw_antisymmetric_score", pa.float64()),
            ("calibrated_antisymmetric_score", pa.float64()),
            ("probability_forward", pa.float64()),
        ]
    )
)
SUPPORT_AUDIT_SCHEMA = _strict(
    pa.schema(
        [
            ("stage", pa.string()),
            ("symbol", pa.string()),
            ("train_regime", pa.string()),
            ("test_regime", pa.string()),
            ("B", pa.int16()),
            ("K", pa.int16()),
            ("model", pa.string()),
            ("feature_config", pa.string()),
            ("seed", pa.int64()),
            ("status", pa.string()),
            ("structural_audit_passed", pa.bool_()),
            ("null_gate_passed", pa.bool_()),
            ("resume_exact", pa.bool_()),
            ("mask_exact", pa.bool_()),
            ("completion_passed", pa.bool_()),
            ("score_antisymmetry_max_abs_error", pa.float64()),
            ("prediction_rows", pa.int64()),
            ("permutation_applicable", pa.bool_()),
            ("permutation_max_abs_error", pa.float64()),
        ]
    ),
    {"permutation_max_abs_error"},
)
INPUT_LINEAGE_SCHEMA = _strict(
    pa.schema(
        [
            ("lineage_role", pa.string()),
            ("upstream_path", pa.string()),
            ("object_count", pa.int64()),
            ("bytes", pa.int64()),
            ("start_date", pa.string()),
            ("end_date", pa.string()),
            ("validation_status", pa.string()),
            ("rows", pa.int64()),
        ]
    ),
    {"rows"},
)


def atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary, compression="zstd")
    observed = pq.ParquetFile(temporary)
    if observed.metadata.num_rows != len(rows) or not observed.schema_arrow.equals(
        schema, check_metadata=True
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"atomic parquet validation failed: {path}")
    os.replace(temporary, path)


def _add_prediction_feature_config(path: Path, feature_config: str) -> None:
    source = pq.ParquetFile(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.feature.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for index in range(source.num_row_groups):
            table = source.read_row_group(index)
            if "feature_config" in table.column_names:
                raise RuntimeError("prediction already has feature_config")
            insert_at = table.column_names.index("model") + 1
            table = table.add_column(
                insert_at,
                "feature_config",
                pa.array([feature_config] * len(table), type=pa.string()),
            )
            table = table.select(PREDICTION_SCHEMA.names).cast(PREDICTION_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if pq.ParquetFile(temporary).metadata.num_rows != source.metadata.num_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("prediction feature-config rewrite row mismatch")
    os.replace(temporary, path)


def validate_ablation_complete(
    final: Path, config: AblationUnitConfig
) -> dict[str, Any]:
    complete_path = final / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing complete record: {final}")
    value = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config") != config.scientific_config()
        or value.get("audit", {}).get("passed") is not True
        or value.get("audit", {}).get("resume_exact") is not True
    ):
        raise RuntimeError(f"complete scientific contract mismatch: {final}")
    required = {
        "selection.json",
        "scaler_reference.json",
        "calibration.json",
        "model.pt",
        "metrics.json",
        "test_predictions.parquet",
    }
    artifacts = value.get("artifacts", [])
    if {item.get("filename") for item in artifacts} != required:
        raise RuntimeError(f"artifact inventory mismatch: {final}")
    for item in artifacts:
        path = final / item["filename"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"artifact bytes mismatch: {path}")
    prediction = pq.ParquetFile(final / "test_predictions.parquet")
    if prediction.metadata.num_rows != 2 * EXPECTED_PATHS[(10, 6)]["test"]:
        raise RuntimeError("ablation prediction row mismatch")
    required_prediction = {
        "pair_id",
        "source_date",
        "start_block_index",
        "path_start_ns",
        "path_end_ns",
        "path_event_count",
        "duration_seconds",
        "target_b",
        "k",
        "model",
        "feature_config",
        "seed",
        "orientation",
        "label",
        "raw_antisymmetric_score",
        "calibrated_antisymmetric_score",
        "probability_forward",
    }
    if set(
        prediction.schema_arrow.names
    ) != required_prediction or not prediction.schema_arrow.equals(
        PREDICTION_SCHEMA, check_metadata=True
    ):
        raise RuntimeError("ablation prediction exact schema mismatch")
    for index in range(prediction.num_row_groups):
        table = prediction.read_row_group(index)
        count = len(table)
        if count % 2:
            raise RuntimeError("odd prediction row group")
        half = count // 2
        if (
            table["pair_id"].slice(0, half).to_pylist()
            != table["pair_id"].slice(half, half).to_pylist()
            or table["orientation"].slice(0, half).to_pylist() != ["forward"] * half
            or table["orientation"].slice(half, half).to_pylist() != ["reverse"] * half
            or table["feature_config"].to_pylist() != [config.canonical_config] * count
        ):
            raise RuntimeError("prediction pair/config identity mismatch")
        raw = table["raw_antisymmetric_score"].to_numpy()
        calibrated = table["calibrated_antisymmetric_score"].to_numpy()
        probability = table["probability_forward"].to_numpy()
        if (
            not np.array_equal(raw[:half], -raw[half:])
            or not np.array_equal(calibrated[:half], -calibrated[half:])
            or not np.allclose(
                probability[:half] + probability[half:], 1.0, rtol=0.0, atol=1e-7
            )
        ):
            raise RuntimeError("prediction antisymmetry mismatch")
    return value


def run_ablation_unit(config: AblationUnitConfig) -> dict[str, Any]:
    validate_feature_registry()
    if config.model not in MODELS or config.seed not in ALL_SEEDS:
        raise ValueError("model/seed outside Phase 06 contract")
    configure_determinism()
    device = device_or_raise(config.device)
    if device.type != "cuda":
        raise ValueError("Phase 06 scientific units require qualified CUDA")
    preflight = ablation_input_preflight(config)
    split = UPSTREAM.chronological_day_split(DATES)
    final = config.output_root / config.unit_id
    work = config.work_root / config.unit_id
    if final.exists():
        if config.resume:
            return validate_ablation_complete(final, config)
        raise FileExistsError(final)
    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / ".unit.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"unit already has writer: {config.unit_id}") from error
    started = time.monotonic()
    runtime_path = work / "runtime.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {"attempts": 0, "elapsed_seconds": 0.0, "measured_cost_usd": 0.0}
    )
    succeeded = False
    try:
        readable = config.scientific_config()
        exact_config = {"scientific_config": readable, "input": preflight}
        spec_path = work / "config.json"
        if spec_path.is_file():
            if json.loads(spec_path.read_text(encoding="utf-8")) != exact_config:
                raise RuntimeError(f"work config mismatch: {work}")
            if not config.resume:
                raise FileExistsError(f"work exists; use resume: {work}")
        else:
            atomic_json(spec_path, exact_config)
        staging = work / "staging"
        if (staging / "complete.json").is_file():
            validate_ablation_complete(staging, config)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            validated = validate_ablation_complete(final, config)
            shutil.rmtree(work)
            return validated
        staging.mkdir(parents=True, exist_ok=True)
        log_path = (
            config.log_root
            / f"{config.canonical_config}_{config.model}_seed{config.seed}.jsonl"
        )
        UPSTREAM.emit(
            log_path,
            "unit_start",
            unit_id=config.unit_id,
            input_files=preflight["files"],
            input_bytes=preflight["bytes"],
        )

        base_scaler, scaler_reference = load_immutable_full_scaler(config.anchor_root)
        scaler = MaskedScaler(base_scaler, config.canonical_config)
        atomic_json(staging / "scaler_reference.json", scaler_reference)
        fold_scalers = ensure_fold_scalers(config)
        fold_rows: list[dict[str, Any]] = []
        folds_root = work / "folds"
        folds_root.mkdir(parents=True, exist_ok=True)
        for fold in UPSTREAM.expanding_inner_folds(split.development):
            fold_scaler = MaskedScaler(fold_scalers[fold.fold], config.canonical_config)
            for candidate_index, candidate in enumerate(UPSTREAM.ADAMW_CANDIDATES):
                result_path = (
                    folds_root / f"fold={fold.fold}_candidate={candidate_index}.json"
                )
                stage_config = {
                    "unit": exact_config,
                    "stage": "selection",
                    "fold": asdict(fold),
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                }
                normalized_stage = json.loads(json.dumps(stage_config, sort_keys=True))
                if result_path.is_file():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    if row.get("checkpoint_config") != normalized_stage:
                        raise RuntimeError(f"stale fold result: {result_path}")
                    fold_rows.append(row)
                    continue
                checkpoint = result_path.with_suffix(".checkpoint.pt")
                _, history, best_epoch, best_loss = _cuda_train_stage(
                    device,
                    model_kind=config.model,
                    seed=config.seed,
                    input_root=config.input_root,
                    train_dates=fold.train,
                    evaluation_dates=(fold.evaluation,),
                    target_b=10,
                    k=6,
                    scaler=fold_scaler,
                    candidate=candidate,
                    max_epochs=config.max_epochs,
                    min_epochs=config.min_epochs,
                    patience=config.patience,
                    fixed_epochs=None,
                    batch_size=config.batch_size,
                    checkpoint_batches=config.checkpoint_batches,
                    checkpoint_path=checkpoint,
                    checkpoint_config=stage_config,
                    resume=config.resume,
                    log_path=log_path,
                )
                row = {
                    "checkpoint_config": stage_config,
                    "fold": fold.fold,
                    "train_dates": list(fold.train),
                    "evaluation_date": fold.evaluation,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "best_epoch": best_epoch,
                    "best_evaluation_paired_log_loss": best_loss,
                    "history": history,
                }
                atomic_json(result_path, row)
                checkpoint.unlink(missing_ok=True)
                fold_rows.append(row)

        selected_index, selection = UPSTREAM.select_candidate(fold_rows)
        selected_candidate = UPSTREAM.ADAMW_CANDIDATES[selected_index]
        atomic_json(staging / "selection.json", selection)
        final_stage_config = {
            "unit": exact_config,
            "stage": "final_fit",
            "candidate_index": selected_index,
            "epochs": selection["final_fit_epochs"],
        }
        final_model, final_history, _, _ = _cuda_train_stage(
            device,
            model_kind=config.model,
            seed=config.seed,
            input_root=config.input_root,
            train_dates=split.development,
            evaluation_dates=(),
            target_b=10,
            k=6,
            scaler=scaler,
            candidate=selected_candidate,
            max_epochs=selection["final_fit_epochs"],
            min_epochs=selection["final_fit_epochs"],
            patience=config.patience,
            fixed_epochs=selection["final_fit_epochs"],
            batch_size=config.batch_size,
            checkpoint_batches=config.checkpoint_batches,
            checkpoint_path=work / "final_fit.checkpoint.pt",
            checkpoint_config=final_stage_config,
            resume=config.resume,
            log_path=log_path,
        )
        calibration_scores = _cuda_collect_scores_by_day(
            final_model,
            config.input_root,
            split.calibration,
            10,
            6,
            scaler,
            config.batch_size,
            log_path=log_path,
            checkpoint_root=work / "calibration_scores",
            unit_id=f"{config.unit_id}/calibration_scores",
            model_identity={
                "checkpoint_path": str((work / "final_fit.checkpoint.pt").resolve()),
                "checkpoint_bytes": (work / "final_fit.checkpoint.pt").stat().st_size,
                "stage_config": final_stage_config,
            },
        )
        alpha, calibration = fit_positive_day_equal_calibration_with_heartbeat(
            calibration_scores,
            log_path=log_path,
            unit_id=f"{config.unit_id}/calibration",
            device=device,
        )
        prediction_path = staging / "test_predictions.parquet"
        test_metrics, audit = _cuda_test_predictions_and_metrics(
            model=final_model,
            input_root=config.input_root,
            dates=split.test,
            target_b=10,
            k=6,
            scaler=scaler,
            alpha=alpha,
            batch_size=config.batch_size,
            prediction_path=prediction_path,
            model_kind=config.model,
            seed=config.seed,
            log_path=log_path,
            checkpoint_root=work / "test_predictions",
            unit_id=f"{config.unit_id}/test_predictions",
            model_identity={
                "checkpoint_path": str((work / "final_fit.checkpoint.pt").resolve()),
                "checkpoint_bytes": (work / "final_fit.checkpoint.pt").stat().st_size,
                "stage_config": final_stage_config,
            },
        )
        if not audit["passed"]:
            raise RuntimeError(f"test scientific audit failed: {audit}")
        _add_prediction_feature_config(prediction_path, config.canonical_config)
        UPSTREAM.atomic_torch_save(
            staging / "model.pt",
            {
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in final_model.state_dict().items()
                },
                "model": config.model,
                "feature_config": config.canonical_config,
                "seed": config.seed,
                "B": 10,
                "K": 6,
            },
        )
        atomic_json(staging / "calibration.json", calibration)

        null_stage_config = {
            "unit": exact_config,
            "stage": "reversible_null",
            "candidate_index": selected_index,
            "epochs": selection["final_fit_epochs"],
            "null_seed": UPSTREAM.NULL_SEED,
        }
        null_model, null_history, _, _ = _cuda_train_stage(
            device,
            model_kind=config.model,
            seed=config.seed,
            input_root=config.input_root,
            train_dates=split.development,
            evaluation_dates=(),
            target_b=10,
            k=6,
            scaler=scaler,
            candidate=selected_candidate,
            max_epochs=selection["final_fit_epochs"],
            min_epochs=selection["final_fit_epochs"],
            patience=config.patience,
            fixed_epochs=selection["final_fit_epochs"],
            batch_size=config.batch_size,
            checkpoint_batches=config.checkpoint_batches,
            checkpoint_path=work / "reversible_null.checkpoint.pt",
            checkpoint_config=null_stage_config,
            resume=config.resume,
            log_path=log_path,
            null_seed=UPSTREAM.NULL_SEED,
        )
        null_scores = _cuda_collect_scores_by_day(
            null_model,
            config.input_root,
            split.calibration,
            10,
            6,
            scaler,
            config.batch_size,
            null_seed=UPSTREAM.NULL_SEED,
            log_path=log_path,
            checkpoint_root=work / "reversible_null_calibration_scores",
            unit_id=f"{config.unit_id}/reversible_null_calibration_scores",
            model_identity={
                "checkpoint_path": str(
                    (work / "reversible_null.checkpoint.pt").resolve()
                ),
                "checkpoint_bytes": (work / "reversible_null.checkpoint.pt")
                .stat()
                .st_size,
                "stage_config": null_stage_config,
            },
        )
        null_alpha, null_calibration = (
            fit_positive_day_equal_calibration_with_heartbeat(
                null_scores,
                log_path=log_path,
                unit_id=f"{config.unit_id}/reversible_null_calibration",
                device=device,
            )
        )
        null_metrics, null_audit = _cuda_test_predictions_and_metrics(
            model=null_model,
            input_root=config.input_root,
            dates=split.test,
            target_b=10,
            k=6,
            scaler=scaler,
            alpha=null_alpha,
            batch_size=config.batch_size,
            prediction_path=None,
            model_kind=config.model,
            seed=config.seed,
            null_seed=UPSTREAM.NULL_SEED,
            log_path=log_path,
            checkpoint_root=work / "reversible_null_test_predictions",
            unit_id=f"{config.unit_id}/reversible_null_test_predictions",
            model_identity={
                "checkpoint_path": str(
                    (work / "reversible_null.checkpoint.pt").resolve()
                ),
                "checkpoint_bytes": (work / "reversible_null.checkpoint.pt")
                .stat()
                .st_size,
                "stage_config": null_stage_config,
            },
        )
        real_d = float(test_metrics["day_equal"]["nats_path"])
        null_d = float(null_metrics["day_equal"]["nats_path"])
        threshold = max(UPSTREAM.NULL_ABSOLUTE_CEILING, 0.10 * abs(real_d))
        null_bias = {
            "observed_abs": abs(null_d),
            "threshold": threshold,
            "passed": abs(null_d) <= threshold,
        }
        if not null_audit["passed"] or not null_bias["passed"]:
            raise RuntimeError(f"reversible-null gate failed: {null_audit} {null_bias}")
        metrics = {
            "B": 10,
            "K": 6,
            "feature_config": config.canonical_config,
            "model": config.model,
            "seed": config.seed,
            "split": {
                "development": list(split.development),
                "calibration": list(split.calibration),
                "test": list(split.test),
            },
            "selection": selection,
            "calibration": calibration,
            "test": test_metrics,
            "audit": audit,
            "training_history": final_history,
            "reversible_null": {
                "calibration": null_calibration,
                "test": null_metrics,
                "audit": null_audit,
                "bias_gate": null_bias,
                "training_history": null_history,
            },
        }
        atomic_json(staging / "metrics.json", metrics)
        artifacts = []
        for path in sorted(staging.iterdir()):
            if path.is_file():
                item = {"filename": path.name, "bytes": path.stat().st_size}
                if path.suffix == ".parquet":
                    item["rows"] = pq.ParquetFile(path).metadata.num_rows
                artifacts.append(item)
        elapsed = time.monotonic() - started
        cumulative_elapsed = float(runtime["elapsed_seconds"]) + elapsed
        cumulative_cost = (
            float(runtime["measured_cost_usd"])
            + elapsed * config.hourly_cost_usd / 3600.0
        )
        complete = {
            "status": "complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "unit_id": config.unit_id,
            "scientific_config": readable,
            "input": preflight,
            "artifacts": artifacts,
            "audit": {
                "test": audit,
                "reversible_null": null_audit,
                "null_bias": null_bias,
                "mask_exact": True,
                "resume_exact": bool(preflight["resume_qualification"]["exact"]),
                "passed": True,
            },
            "runtime": {
                "attempts": int(runtime["attempts"]) + 1,
                "elapsed_seconds": cumulative_elapsed,
                "hourly_cost_usd": config.hourly_cost_usd,
                "measured_cost_usd": cumulative_cost,
            },
        }
        atomic_json(staging / "complete.json", complete)
        validate_ablation_complete(staging, config)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        succeeded = True
        validated = validate_ablation_complete(final, config)
        shutil.rmtree(work)
        return validated
    finally:
        elapsed = time.monotonic() - started
        if not succeeded and work.exists():
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": float(runtime["elapsed_seconds"]) + elapsed,
                    "measured_cost_usd": float(runtime["measured_cost_usd"])
                    + elapsed * config.hourly_cost_usd / 3600.0,
                },
            )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
