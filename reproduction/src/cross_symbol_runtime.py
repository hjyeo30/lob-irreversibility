"""Phase 07 cross-symbol input, unit, gate, and model-run primitives."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import numpy as np

from src.scale_runtime import (
    CANONICAL_BLOCK_SCHEMA,
    DATES,
    MODELS,
    SEEDS,
    ScaleUnitConfig,
    atomic_json,
    run_scale_unit,
    validate_complete,
)

PILOT_SYMBOLS = ("QQQ", "SOFI", "SAIA")
EXPANSION_SYMBOLS = ("AAPL", "TSLA", "COST", "PEP", "CROX", "PODD", "SPY", "IWM")
NON_NVDA_SYMBOLS = PILOT_SYMBOLS + EXPANSION_SYMBOLS
ALL_SYMBOLS = ("NVDA",) + NON_NVDA_SYMBOLS
PILOT_SEED = 20260730
NULL_ABSOLUTE_GATE = 0.00063
LOCAL_CACHE_ROOT = Path("/tmp")


@dataclass(frozen=True, order=True)
class CrossUnit:
    symbol: str
    model: str
    seed: int

    def __post_init__(self) -> None:
        if self.symbol not in NON_NVDA_SYMBOLS:
            raise ValueError(f"symbol outside Phase 07 new-unit scope: {self.symbol}")
        if self.model not in MODELS or self.seed not in SEEDS:
            raise ValueError("model or seed outside approved Phase 07 scope")

    @property
    def logical_id(self) -> str:
        return (
            f"phase07/{self.symbol}/daytime/daytime/B10/K6/"
            f"{self.model}/all37/{self.seed}"
        )

    @property
    def storage_relative_path(self) -> Path:
        return Path(f"symbol={self.symbol}") / self.model / f"seed={self.seed}"


def pilot_units() -> tuple[CrossUnit, ...]:
    return tuple(
        CrossUnit(symbol, model, PILOT_SEED)
        for symbol in PILOT_SYMBOLS
        for model in MODELS
    )


def production_units() -> tuple[CrossUnit, ...]:
    pilot_replicates = tuple(
        CrossUnit(symbol, model, seed)
        for symbol in PILOT_SYMBOLS
        for model in MODELS
        for seed in SEEDS
        if seed != PILOT_SEED
    )
    expansion = tuple(
        CrossUnit(symbol, model, seed)
        for symbol in EXPANSION_SYMBOLS
        for model in MODELS
        for seed in SEEDS
    )
    return pilot_replicates + expansion


def all_new_units() -> tuple[CrossUnit, ...]:
    return pilot_units() + production_units()


def symbol_root(input_base: Path, symbol: str) -> Path:
    return input_base / f"symbol={symbol}"


def partition_paths(input_base: Path, symbol: str, source_date: str) -> list[Path]:
    return sorted(
        symbol_root(input_base, symbol).glob(
            f"source_date={source_date}/regime=daytime/segment=*/target_b=10/*.parquet"
        )
    )


def _segment_rows(paths: Sequence[Path]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for path in paths:
        segment = next(
            (part for part in path.parts if part.startswith("segment=")), None
        )
        if segment is None:
            raise RuntimeError(f"missing segment partition: {path}")
        rows[segment] = rows.get(segment, 0) + pq.ParquetFile(path).metadata.num_rows
    return rows


def _path_count(rows: int, source_date: str, k: int = 6) -> int:
    available = rows - k + 1
    offset = date.fromisoformat(source_date).toordinal() % 3
    if available <= offset:
        return 0
    return (available - 1 - offset) // 3 + 1


def preflight_symbol(input_base: Path, symbol: str) -> dict[str, Any]:
    root = symbol_root(input_base, symbol)
    rows_by_date: dict[str, int] = {}
    paths_by_date: dict[str, int] = {}
    segments_by_date: dict[str, list[str]] = {}
    total_files = total_bytes = 0
    for source_date in DATES:
        complete_path = root / f"source_date={source_date}" / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        requested = complete.get("requested", {}).get("publish_regimes", {})
        validation = complete.get("validation", {})
        if (
            complete.get("status") != "complete"
            or 10 not in requested.get("daytime", [])
            or validation.get("requested_cells_complete") is not True
            or validation.get("schema_equal_upstream") is not True
            or validation.get("footer_rows_equal") is not True
            or validation.get("contiguous_block_index") is not True
            or validation.get("adjacent_boundaries") is not True
            or validation.get("block_arithmetic") is not True
            or validation.get("event_accounting") is not True
            or validation.get("finite_nonnull") is not True
            or validation.get("row_identity_matches_partition") is not True
        ):
            raise RuntimeError(f"Phase 02 completion gate failed: {complete_path}")
        paths = partition_paths(input_base, symbol, source_date)
        if not paths:
            raise FileNotFoundError(
                f"missing daytime B10 parquet: {symbol} {source_date}"
            )
        segments = _segment_rows(paths)
        if len(segments) != 1:
            raise RuntimeError(
                f"current exact runner requires one continuous segment: "
                f"{symbol} {source_date} observed={sorted(segments)}"
            )
        rows = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(
                CANONICAL_BLOCK_SCHEMA, check_metadata=True
            ):
                raise RuntimeError(f"canonical 61-column schema mismatch: {path}")
            rows += parquet.metadata.num_rows
            total_files += 1
            total_bytes += path.stat().st_size
        manifest_files = {
            str(item["path"]): (int(item["rows"]), int(item["bytes"]))
            for item in complete.get("files", [])
            if str(item.get("path", "")).startswith("regime=daytime/segment=")
            and "/target_b=10/" in str(item.get("path", ""))
        }
        for path in paths:
            relative = str(path.relative_to(complete_path.parent))
            observed = (pq.ParquetFile(path).metadata.num_rows, path.stat().st_size)
            if manifest_files.get(relative) != observed:
                raise RuntimeError(f"Phase 02 file inventory mismatch: {path}")
        observed_relatives = {
            str(path.relative_to(complete_path.parent)) for path in paths
        }
        if set(manifest_files) != observed_relatives:
            raise RuntimeError(f"Phase 02 B10 manifest set mismatch: {complete_path}")
        count = sum(_path_count(value, source_date) for value in segments.values())
        if count <= 0:
            raise RuntimeError(f"no admissible K6 paths: {symbol} {source_date}")
        rows_by_date[source_date] = rows
        paths_by_date[source_date] = count
        segments_by_date[source_date] = sorted(segments)
    split_counts = {
        "development": sum(paths_by_date[item] for item in DATES[:9]),
        "calibration": sum(paths_by_date[item] for item in DATES[9:12]),
        "test": sum(paths_by_date[item] for item in DATES[12:]),
    }
    return {
        "symbol": symbol,
        "root": str(root),
        "dates": list(DATES),
        "files": total_files,
        "bytes": total_bytes,
        "rows_by_date": rows_by_date,
        "paths_by_date": paths_by_date,
        "path_counts": split_counts,
        "segments_by_date": segments_by_date,
        "status": "passed",
    }


def preflight_all(input_base: Path) -> dict[str, Any]:
    records = [preflight_symbol(input_base, symbol) for symbol in NON_NVDA_SYMBOLS]
    return {
        "contract": "phase07_input_gate_v1",
        "input_base": str(input_base),
        "symbols": records,
        "status": "passed",
    }


def require_local_cache(cache_base: Path) -> Path:
    resolved = cache_base.resolve()
    try:
        resolved.relative_to(LOCAL_CACHE_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Phase 07 cache must resolve under local /tmp, observed={resolved}"
        ) from error
    return resolved


def copy_ephemeral_partition(sources: Sequence[Path], target: Path) -> None:
    """Race-safe immutable copy publication, including concurrent identical staging."""
    target.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for index, source in enumerate(sources):
        name = f"part-{index:05d}.parquet"
        expected_names.add(name)
        destination = target / name
        if (
            destination.is_file()
            and destination.stat().st_size == source.stat().st_size
        ):
            continue
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source, temporary)
            if temporary.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"ephemeral cache copy mismatch: {source}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    unexpected = [
        item for item in target.glob("*.parquet") if item.name not in expected_names
    ]
    if unexpected:
        raise RuntimeError(f"stale ephemeral cache files: {unexpected[:3]}")


def stage_ephemeral_cache(input_base: Path, cache_base: Path, symbol: str) -> Path:
    """Copy verified B10 files into the upstream runner's flat ephemeral layout."""
    preflight_symbol(input_base, symbol)
    target_root = require_local_cache(cache_base) / f"symbol={symbol}"
    for source_date in DATES:
        target = target_root / f"source_date={source_date}" / "target_b=10"
        sources = partition_paths(input_base, symbol, source_date)
        copy_ephemeral_partition(sources, target)
    return target_root


def scale_config(
    unit: CrossUnit,
    *,
    cache_root: Path,
    artifacts: Path,
    path_counts: dict[str, int],
    device: str,
    hourly_cost_usd: float,
    resume: bool,
) -> ScaleUnitConfig:
    relative = unit.storage_relative_path
    return ScaleUnitConfig(
        b=10,
        k=6,
        model=unit.model,
        seed=unit.seed,
        device=device,
        input_root=cache_root,
        output_root=artifacts / "units",
        work_root=artifacts / "work",
        log_root=artifacts / "logs" / relative,
        hourly_cost_usd=hourly_cost_usd,
        resume=resume,
        logical_unit_id=unit.logical_id,
        storage_relative_path=str(relative),
        expected_paths_override=dict(path_counts),
        prediction_symbol=unit.symbol,
        prediction_segment="0000",
    )


def validate_fixed_null_gate(complete: dict[str, Any], final: Path) -> None:
    metrics_path = final / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    observed = abs(float(metrics["reversible_null"]["test"]["day_equal"]["nats_path"]))
    if not np.isfinite(observed) or observed > NULL_ABSOLUTE_GATE:
        raise RuntimeError(
            f"fixed reversible-null gate failed observed={observed} "
            f"threshold={NULL_ABSOLUTE_GATE}"
        )
    if complete.get("audit", {}).get("passed") is not True:
        raise RuntimeError(f"scientific audit failed: {final}")


def run_cross_unit(
    unit: CrossUnit,
    *,
    input_base: Path,
    cache_base: Path,
    artifacts: Path,
    device: str,
    hourly_cost_usd: float,
    resume: bool = True,
) -> dict[str, Any]:
    source = preflight_symbol(input_base, unit.symbol)
    unit_cache_base = cache_base / unit.storage_relative_path
    cache_root = stage_ephemeral_cache(input_base, unit_cache_base, unit.symbol)
    config = scale_config(
        unit,
        cache_root=cache_root,
        artifacts=artifacts,
        path_counts=source["path_counts"],
        device=device,
        hourly_cost_usd=hourly_cost_usd,
        resume=resume,
    )
    complete = run_scale_unit(config)
    final = artifacts / "units" / unit.storage_relative_path
    validate_complete(final, config)
    validate_fixed_null_gate(complete, final)
    return complete


def unit_complete(artifacts: Path, unit: CrossUnit) -> bool:
    path = artifacts / "units" / unit.storage_relative_path / "complete.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "complete" and value.get("unit_id") == unit.logical_id


def validate_cross_complete(
    artifacts: Path, unit: CrossUnit, source: dict[str, Any]
) -> dict[str, Any]:
    final = artifacts / "units" / unit.storage_relative_path
    complete_path = final / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing cross-symbol complete: {final}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    scientific = complete.get("scientific_config", {})
    expected_split = {
        "development": list(DATES[:9]),
        "calibration": list(DATES[9:12]),
        "test": list(DATES[12:]),
    }
    if (
        complete.get("status") != "complete"
        or complete.get("unit_id") != unit.logical_id
        or scientific.get("b") != 10
        or scientific.get("k") != 6
        or scientific.get("model") != unit.model
        or scientific.get("seed") != unit.seed
        or scientific.get("dates") != list(DATES)
        or scientific.get("split") != expected_split
        or scientific.get("expected_path_counts") != source["path_counts"]
        or scientific.get("prediction_symbol") != unit.symbol
        or scientific.get("prediction_segment") != "0000"
        or complete.get("input", {}).get("path_counts") != source["path_counts"]
        or complete.get("audit", {}).get("passed") is not True
        or complete.get("audit", {}).get("resume", {}).get("checkpoint_config_exact")
        is not True
        or complete.get("audit", {})
        .get("resume", {})
        .get("cursor_history_parameters_enforced")
        is not True
        or complete.get("audit", {})
        .get("resume", {})
        .get("phase01_forced_resume_probe_required_by_controller")
        is not True
        or int(complete.get("runtime", {}).get("attempts", 0)) < 1
    ):
        raise RuntimeError(f"cross-symbol scientific complete mismatch: {final}")
    required = {
        "selection.json",
        "scaler.json",
        "calibration.json",
        "model.pt",
        "metrics.json",
        "test_predictions.parquet",
        "null_test_predictions.parquet",
    }
    inventory = {item.get("filename"): item for item in complete.get("artifacts", [])}
    if set(inventory) != required:
        raise RuntimeError(f"cross-symbol artifact inventory mismatch: {final}")
    for name, item in inventory.items():
        path = final / name
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"cross-symbol artifact size mismatch: {path}")
    selection = json.loads((final / "selection.json").read_text(encoding="utf-8"))
    losses = np.asarray(selection.get("fold_losses"), dtype=np.float64)
    if (
        losses.shape != (4, 4)
        or not np.isfinite(losses).all()
        or int(selection.get("selected_candidate_index", -1)) not in range(4)
        or int(selection.get("final_fit_epochs", 0)) < 6
    ):
        raise RuntimeError(f"selection contract mismatch: {final}")
    calibration = json.loads((final / "calibration.json").read_text(encoding="utf-8"))
    alpha = float(calibration.get("alpha", float("nan")))
    if (
        not np.isfinite(alpha)
        or alpha < 0
        or not np.isfinite(
            float(calibration.get("day_equal_paired_log_loss", float("nan")))
        )
        or set(calibration.get("by_day", {})) != set(DATES[9:12])
        or not calibration.get("criterion")
        or not (
            calibration.get("at_lower_boundary_alpha_zero") is True
            or int(calibration.get("bisection_iterations", 0)) > 0
        )
    ):
        raise RuntimeError(f"calibration convergence mismatch: {final}")
    prediction = pq.ParquetFile(final / "test_predictions.parquet")
    null_prediction = pq.ParquetFile(final / "null_test_predictions.parquet")
    expected_rows = 2 * int(source["path_counts"]["test"])
    if (
        prediction.metadata.num_rows != expected_rows
        or null_prediction.metadata.num_rows != expected_rows
        or not {
            "pair_id",
            "orientation",
            "label",
            "source_date",
            "start_block_index",
            "symbol",
            "segment",
            "direction",
            "bounded_log99_score",
        }.issubset(prediction.schema_arrow.names)
        or not set(prediction.schema_arrow.names).issubset(
            set(null_prediction.schema_arrow.names)
        )
    ):
        raise RuntimeError(f"prediction coverage/schema mismatch: {final}")
    validate_fixed_null_gate(complete, final)
    return complete


def require_pilot_gate(artifacts: Path, input_base: Path) -> None:
    missing = [
        unit.logical_id for unit in pilot_units() if not unit_complete(artifacts, unit)
    ]
    if missing:
        raise RuntimeError(f"pilot gate incomplete: {missing}")
    for unit in pilot_units():
        source = preflight_symbol(input_base, unit.symbol)
        validate_cross_complete(artifacts, unit, source)
