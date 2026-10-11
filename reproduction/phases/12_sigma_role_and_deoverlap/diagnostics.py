#!/usr/bin/env python3
"""Leakage-safe diagnostics for the Phase 11 baseline forecast and sigma roles."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
PHASE11_ARTIFACTS = HERE.parent / "11_lgbm_multihorizon_return" / "artifacts"
OUTPUT_ROOT = HERE / "artifacts" / "units" / "diagnostics"
TABLES_DIR = HERE / "artifacts" / "tables"
RUN_ID = "phase12_sigma_role_v1"
SYMBOLS = ("NVDA", "QQQ", "SOFI", "SAIA")
DATES = (
    "2026-07-16",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
)
HORIZONS = (2, 4, 8, 16, 32, 64)
FOLDS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "F1": {"refit": DATES[:3], "test": DATES[3]},
    "F2": {"refit": DATES[:4], "test": DATES[4]},
    "F3": {"refit": DATES[:5], "test": DATES[5]},
}
KEYS = ("symbol", "source_date", "segment_id", "path_end_block_index")
SIGMA_COLUMNS = {
    "O": "r2_ordered_sigma",
    "P": "r2_pooled_sigma",
    "G": "r2_temporal_gap",
    "relative_gap_ratio": "r3_relative_gap_ratio",
    "log_magnitude_ratio": "r3_log_magnitude_ratio",
}
CONDITIONAL_SIGMAS = ("O", "P", "G")
TABLE_NAMES = (
    "bin_thresholds",
    "baseline_calibration",
    "sigma_1d",
    "conditional_role",
    "tail_gate",
)
METRICS = (
    "mean_prediction",
    "mean_return",
    "mean_residual",
    "mae",
    "mean_aligned_return",
    "mean_aligned_residual",
    "mean_absolute_return",
    "move_probability",
    "directional_hit_rate",
    "baseline_mse",
    "zero_mse",
    "mean_forecast_value",
    "relative_forecast_value",
)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    if pq.ParquetFile(temporary).metadata.num_rows != table.num_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row validation failed: {path}")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def return_path(root: Path, symbol: str, source_date: str) -> Path:
    return (
        root
        / "data"
        / "return_rows"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def model_dir(root: Path, symbol: str, fold: str, horizon: int) -> Path:
    return (
        root
        / "models"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / "features=R1_baseline"
    )


def prediction_path(root: Path, symbol: str, fold: str, horizon: int) -> Path:
    return (
        root
        / "predictions"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / "features=R1_baseline.parquet"
    )


def unit_dir(root: Path, symbol: str, fold: str, horizon: int) -> Path:
    return root / f"symbol={symbol}" / f"fold={fold}" / f"horizon={horizon}"


def valid_unit_complete(directory: Path, symbol: str, fold: str, horizon: int) -> bool:
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    try:
        complete = json.loads(marker.read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or complete.get("run_id") != RUN_ID:
            return False
        if complete.get("unit_id") != f"diagnostics__{symbol}__{fold}__h{horizon}":
            return False
        rows = complete.get("table_rows", {})
        return all(
            (directory / f"{name}.parquet").is_file()
            and pq.ParquetFile(directory / f"{name}.parquet").metadata.num_rows
            == rows.get(name)
            and rows.get(name, 0) > 0
            for name in TABLE_NAMES
        )
    except Exception:
        return False


def load_complete(
    root: Path, symbol: str, fold: str, horizon: int
) -> tuple[dict[str, Any], Path]:
    directory = model_dir(root, symbol, fold, horizon)
    complete_path = directory / "complete.json"
    model_path = directory / "model.txt"
    if not complete_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"missing Phase 11 R1 model: {symbol} {fold} h={horizon}"
        )
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    config = complete.get("config", {})
    expected = {
        "status": "complete",
        "contract": "phase11_lgbm_unit_v1",
        "feature_count": 29,
    }
    if any(complete.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"invalid Phase 11 completion: {symbol} {fold} h={horizon}")
    if (
        config.get("symbol"),
        config.get("fold"),
        config.get("horizon"),
        config.get("feature_set"),
    ) != (symbol, fold, horizon, "R1_baseline"):
        raise RuntimeError("Phase 11 completion config mismatch")
    features = complete.get("features")
    if (
        not isinstance(features, list)
        or len(features) != 29
        or len(set(features)) != 29
    ):
        raise RuntimeError("invalid Phase 11 feature order")
    return complete, model_path


def load_return_date(
    root: Path,
    symbol: str,
    source_date: str,
    horizon: int,
    features: list[str],
    include_features: bool,
) -> dict[str, np.ndarray]:
    target_name = f"target_return_bps_h{horizon}"
    columns = list(KEYS) + list(SIGMA_COLUMNS.values()) + [target_name]
    if include_features:
        columns += features
    table = pq.read_table(return_path(root, symbol, source_date), columns=columns)
    target_array = table[target_name].combine_chunks()
    valid = np.asarray(pc.invert(pc.is_null(target_array)), dtype=bool)
    result: dict[str, np.ndarray] = {
        name: np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))[
            valid
        ]
        for name in KEYS
    }
    result["target"] = target_array.to_numpy(zero_copy_only=False)[valid].astype(
        np.float64
    )
    for short_name, column in SIGMA_COLUMNS.items():
        result[short_name] = np.asarray(table[column], dtype=np.float64)[valid]
    if include_features:
        result["x"] = np.column_stack(
            [np.asarray(table[name], dtype=np.float64)[valid] for name in features]
        )
    numeric = [result["target"], *(result[name] for name in SIGMA_COLUMNS)]
    if include_features:
        numeric.append(result["x"])
    if len(result["target"]) == 0 or any(
        not np.isfinite(values).all() for values in numeric
    ):
        raise RuntimeError(f"invalid return rows: {symbol} {source_date} h={horizon}")
    if len(np.unique(result["path_end_block_index"])) != len(result["target"]):
        raise RuntimeError("duplicate return-row key")
    return result


def date_equal_weights(dates: Iterable[str], sizes: Iterable[int]) -> np.ndarray:
    date_list = list(dates)
    size_list = list(sizes)
    if (
        not date_list
        or len(date_list) != len(size_list)
        or any(size <= 0 for size in size_list)
    ):
        raise ValueError("invalid date-equal weight inputs")
    pieces = [
        np.full(size, 1.0 / (len(size_list) * size), dtype=np.float64)
        for size in size_list
    ]
    result = np.concatenate(pieces)
    if not np.isclose(result.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("date-equal weights do not sum to one")
    return result


def weighted_thresholds(
    values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]
) -> np.ndarray:
    if (
        len(values) == 0
        or values.shape != weights.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("invalid weighted quantile input")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative /= cumulative[-1]
    output = []
    for probability in probabilities:
        if not 0.0 < probability < 1.0:
            raise ValueError("quantile probability must lie inside (0,1)")
        index = min(
            int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1
        )
        output.append(float(ordered_values[index]))
    return np.asarray(output, dtype=np.float64)


def active_edges(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # A threshold at the reference maximum would create an empty upper bin. Equal
    # values are never jittered or split by row order.
    return np.asarray(
        [value for value in np.unique(thresholds) if value < np.max(values)],
        dtype=np.float64,
    )


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="left").astype(np.int16) + 1


def metric_values(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int | None]:
    if len(target) == 0:
        return {
            **{name: None for name in METRICS},
            "directional_hit_n": 0,
            "directional_support_n": 0,
        }
    residual = target - prediction
    aligned = np.sign(prediction) * target
    aligned_residual = np.sign(prediction) * residual
    forecast_value = target * target - residual * residual
    direction_support = target != 0.0
    direction_hits = direction_support & (np.sign(prediction) == np.sign(target))
    zero_mse = float(np.mean(target * target))
    return {
        "mean_prediction": float(np.mean(prediction)),
        "mean_return": float(np.mean(target)),
        "mean_residual": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "mean_aligned_return": float(np.mean(aligned)),
        "mean_aligned_residual": float(np.mean(aligned_residual)),
        "mean_absolute_return": float(np.mean(np.abs(target))),
        "move_probability": float(np.mean(target != 0.0)),
        "directional_hit_rate": float(
            np.sum(direction_hits) / np.sum(direction_support)
        )
        if np.any(direction_support)
        else None,
        "directional_hit_n": int(np.sum(direction_hits)),
        "directional_support_n": int(np.sum(direction_support)),
        "baseline_mse": float(np.mean(residual * residual)),
        "zero_mse": zero_mse,
        "mean_forecast_value": float(np.mean(forecast_value)),
        "relative_forecast_value": float(np.mean(forecast_value) / zero_mse)
        if zero_mse > 0.0
        else None,
    }


def base_row(
    symbol: str, fold: str, test_date: str, horizon: int, mask: np.ndarray, total: int
) -> dict[str, Any]:
    count = int(np.sum(mask))
    return {
        "aggregation": "test_date",
        "symbol": symbol,
        "fold": fold,
        "test_date": test_date,
        "horizon": horizon,
        "n": count,
        "share": float(count / total),
        "dates_present": 1 if count else 0,
        "min_date_n": count,
        "max_date_n": count,
    }


def add_metrics(
    row: dict[str, Any], prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> None:
    row.update(metric_values(prediction[mask], target[mask]))


def threshold_rows(
    symbol: str,
    fold: str,
    horizon: int,
    test_date: str,
    refit_dates: tuple[str, ...],
    variable: str,
    probabilities: tuple[float, ...],
    thresholds: np.ndarray,
    edges: np.ndarray,
    refit_rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "fold": fold,
            "test_date": test_date,
            "horizon": horizon,
            "refit_dates": ",".join(refit_dates),
            "variable": variable,
            "intended_bins": len(probabilities) + 1,
            "effective_bins": len(edges) + 1,
            "probability": probability,
            "threshold": float(threshold),
            "is_active_edge": bool(np.any(edges == threshold)),
            "refit_rows": refit_rows,
            "method": "date_equal_weighted_empirical_refit_only",
        }
        for probability, threshold in zip(probabilities, thresholds, strict=True)
    ]


def run_unit(args: argparse.Namespace) -> None:
    symbol, fold, horizon = args.symbol, args.fold, args.horizon
    phase11 = Path(args.phase11_artifacts)
    output = unit_dir(Path(args.output_root), symbol, fold, horizon)
    if valid_unit_complete(output, symbol, fold, horizon):
        print(
            json.dumps(
                {
                    "status": "skip_complete",
                    "symbol": symbol,
                    "fold": fold,
                    "horizon": horizon,
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    complete, model_path = load_complete(phase11, symbol, fold, horizon)
    features = list(complete["features"])
    spec = FOLDS[fold]
    refit_dates = tuple(spec["refit"])  # type: ignore[arg-type]
    test_date = str(spec["test"])
    refit_parts = [
        load_return_date(phase11, symbol, day, horizon, features, True)
        for day in refit_dates
    ]
    refit_sizes = [len(part["target"]) for part in refit_parts]
    weights = date_equal_weights(refit_dates, refit_sizes)
    booster = lgb.Booster(
        model_file=str(model_path), params={"num_threads": args.threads}
    )
    refit_prediction_parts = [
        np.asarray(booster.predict(part["x"]), dtype=np.float64) for part in refit_parts
    ]
    refit_prediction = np.concatenate(refit_prediction_parts)
    if not np.isfinite(refit_prediction).all() or len(refit_prediction) != sum(
        refit_sizes
    ):
        raise RuntimeError("invalid refit prediction")
    refit_signals = {
        name: np.concatenate([part[name] for part in refit_parts])
        for name in SIGMA_COLUMNS
    }

    test = load_return_date(phase11, symbol, test_date, horizon, features, False)
    saved = pq.read_table(prediction_path(phase11, symbol, fold, horizon))
    if saved.num_rows != len(test["target"]):
        raise RuntimeError("saved test prediction row mismatch")
    for key in KEYS:
        if not np.array_equal(np.asarray(saved[key]), test[key]):
            raise RuntimeError(f"saved test prediction key mismatch: {key}")
    prediction = np.asarray(saved["prediction"], dtype=np.float64)
    saved_target = np.asarray(saved["target_return_bps"], dtype=np.float64)
    if (
        not np.array_equal(saved_target, test["target"])
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError("saved test prediction target/value mismatch")
    target = test["target"]
    total = len(target)

    threshold_output: list[dict[str, Any]] = []
    decile_probabilities = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))
    quintile_probabilities = (0.2, 0.4, 0.6, 0.8)
    prediction_deciles = weighted_thresholds(
        refit_prediction, weights, decile_probabilities
    )
    prediction_decile_edges = active_edges(refit_prediction, prediction_deciles)
    prediction_quintiles = weighted_thresholds(
        refit_prediction, weights, quintile_probabilities
    )
    prediction_quintile_edges = active_edges(refit_prediction, prediction_quintiles)
    threshold_output += threshold_rows(
        symbol,
        fold,
        horizon,
        test_date,
        refit_dates,
        "baseline_prediction_decile",
        decile_probabilities,
        prediction_deciles,
        prediction_decile_edges,
        len(refit_prediction),
    )
    threshold_output += threshold_rows(
        symbol,
        fold,
        horizon,
        test_date,
        refit_dates,
        "baseline_prediction_quintile",
        quintile_probabilities,
        prediction_quintiles,
        prediction_quintile_edges,
        len(refit_prediction),
    )

    signal_edges: dict[str, np.ndarray] = {}
    signal_q80: dict[str, float] = {}
    for name, values in refit_signals.items():
        thresholds = weighted_thresholds(values, weights, quintile_probabilities)
        edges = active_edges(values, thresholds)
        signal_edges[name] = edges
        signal_q80[name] = float(thresholds[-1])
        threshold_output += threshold_rows(
            symbol,
            fold,
            horizon,
            test_date,
            refit_dates,
            name,
            quintile_probabilities,
            thresholds,
            edges,
            len(values),
        )
    absolute_prediction = np.abs(refit_prediction)
    strong_threshold = float(
        weighted_thresholds(absolute_prediction, weights, (0.8,))[0]
    )
    threshold_output += threshold_rows(
        symbol,
        fold,
        horizon,
        test_date,
        refit_dates,
        "absolute_baseline_prediction",
        (0.8,),
        np.asarray([strong_threshold]),
        active_edges(absolute_prediction, np.asarray([strong_threshold])),
        len(refit_prediction),
    )

    calibration_rows: list[dict[str, Any]] = []
    prediction_decile_bin = assign_bins(prediction, prediction_decile_edges)
    for prediction_bin in range(1, len(prediction_decile_edges) + 2):
        mask = prediction_decile_bin == prediction_bin
        row = base_row(symbol, fold, test_date, horizon, mask, total)
        row["prediction_bin"] = prediction_bin
        row["effective_prediction_bins"] = len(prediction_decile_edges) + 1
        add_metrics(row, prediction, target, mask)
        calibration_rows.append(row)

    sigma_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    prediction_quintile_bin = assign_bins(prediction, prediction_quintile_edges)
    strong = np.abs(prediction) >= strong_threshold
    for name in SIGMA_COLUMNS:
        sigma_bin_values = assign_bins(test[name], signal_edges[name])
        effective_sigma_bins = len(signal_edges[name]) + 1
        for sigma_bin in range(1, effective_sigma_bins + 1):
            mask = sigma_bin_values == sigma_bin
            row = base_row(symbol, fold, test_date, horizon, mask, total)
            row.update(
                {
                    "sigma_name": name,
                    "sigma_bin": sigma_bin,
                    "effective_sigma_bins": effective_sigma_bins,
                }
            )
            add_metrics(row, prediction, target, mask)
            sigma_rows.append(row)
        if name not in CONDITIONAL_SIGMAS:
            continue
        for prediction_bin in range(1, len(prediction_quintile_edges) + 2):
            for sigma_bin in range(1, effective_sigma_bins + 1):
                mask = (prediction_quintile_bin == prediction_bin) & (
                    sigma_bin_values == sigma_bin
                )
                row = base_row(symbol, fold, test_date, horizon, mask, total)
                row.update(
                    {
                        "sigma_name": name,
                        "prediction_bin": prediction_bin,
                        "sigma_bin": sigma_bin,
                        "effective_prediction_bins": len(prediction_quintile_edges) + 1,
                        "effective_sigma_bins": effective_sigma_bins,
                    }
                )
                add_metrics(row, prediction, target, mask)
                conditional_rows.append(row)
        high = test[name] >= signal_q80[name]
        for strong_value in (False, True):
            for high_value in (False, True):
                mask = (strong == strong_value) & (high == high_value)
                row = base_row(symbol, fold, test_date, horizon, mask, total)
                row.update(
                    {
                        "sigma_name": name,
                        "strong_baseline": strong_value,
                        "sigma_high": high_value,
                        "strong_threshold": strong_threshold,
                        "sigma_high_threshold": signal_q80[name],
                        "actual_strong_share": float(np.mean(strong)),
                        "actual_sigma_high_share": float(np.mean(high)),
                    }
                )
                add_metrics(row, prediction, target, mask)
                tail_rows.append(row)

    tables = {
        "bin_thresholds": pa.Table.from_pylist(threshold_output),
        "baseline_calibration": pa.Table.from_pylist(calibration_rows),
        "sigma_1d": pa.Table.from_pylist(sigma_rows),
        "conditional_role": pa.Table.from_pylist(conditional_rows),
        "tail_gate": pa.Table.from_pylist(tail_rows),
    }
    for name, table in tables.items():
        if table.num_rows == 0:
            raise RuntimeError(f"empty diagnostic table: {name}")
        atomic_parquet(output / f"{name}.parquet", table)
    table_rows = {name: table.num_rows for name, table in tables.items()}
    atomic_json(
        output / "complete.json",
        {
            "status": "complete",
            "run_id": RUN_ID,
            "unit_id": f"diagnostics__{symbol}__{fold}__h{horizon}",
            "symbol": symbol,
            "fold": fold,
            "horizon": horizon,
            "refit_dates": list(refit_dates),
            "test_date": test_date,
            "refit_rows": len(refit_prediction),
            "test_rows": total,
            "table_rows": table_rows,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    if not valid_unit_complete(output, symbol, fold, horizon):
        raise RuntimeError("published diagnostic unit failed validation")
    print(
        json.dumps(
            {
                "status": "complete",
                "symbol": symbol,
                "fold": fold,
                "horizon": horizon,
                "refit_rows": len(refit_prediction),
                "test_rows": total,
                "output": str(output),
                "table_rows": table_rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def date_equal_rows(table: pa.Table, dimensions: tuple[str, ...]) -> pa.Table:
    rows = table.to_pylist()
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in ("symbol", "horizon", *dimensions))
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        nonempty = [row for row in group if row["n"] > 0]
        result = {
            "aggregation": "date_equal",
            "symbol": key[0],
            "fold": "ALL",
            "test_date": "ALL",
            "horizon": key[1],
        }
        result.update(
            {name: value for name, value in zip(dimensions, key[2:], strict=True)}
        )
        result["n"] = int(sum(row["n"] for row in group))
        # Structural bins removed by refit ties have zero realized share. Cell
        # moments below remain equal-weighted over dates on which the cell exists.
        result["share"] = float(sum(row["share"] for row in group) / 3.0)
        result["dates_present"] = len(nonempty)
        result["date_coverage"] = float(len(nonempty) / 3.0)
        result["min_date_n"] = int(
            min([row["n"] for row in group] + ([0] if len(group) < 3 else []))
        )
        result["max_date_n"] = int(max((row["n"] for row in group), default=0))
        for metric in METRICS:
            values = [row[metric] for row in nonempty if row[metric] is not None]
            result[metric] = (
                float(np.mean(values))
                if len(values) == len(nonempty) and values
                else None
            )
        result["directional_hit_n"] = int(
            sum(row["directional_hit_n"] for row in group)
        )
        result["directional_support_n"] = int(
            sum(row["directional_support_n"] for row in group)
        )
        # The contract calls for a ratio of date-equal losses, not an average
        # of the three date-specific ratios.
        zero_mse = result["zero_mse"]
        baseline_mse = result["baseline_mse"]
        result["mean_forecast_value"] = (
            float(zero_mse - baseline_mse)
            if zero_mse is not None and baseline_mse is not None
            else None
        )
        result["relative_forecast_value"] = (
            float(1.0 - baseline_mse / zero_mse)
            if zero_mse is not None and zero_mse > 0.0 and baseline_mse is not None
            else None
        )
        # Dimension metadata is fixed only when all folds agree; otherwise the
        # fold-specific rows retain the exact effective-bin information.
        for name in group[0]:
            if (
                name in result
                or name in METRICS
                or name
                in {
                    "aggregation",
                    "symbol",
                    "fold",
                    "test_date",
                    "horizon",
                    "n",
                    "share",
                    "dates_present",
                    "date_coverage",
                    "min_date_n",
                    "max_date_n",
                    "directional_hit_n",
                    "directional_support_n",
                }
            ):
                continue
            values = [row.get(name) for row in group]
            result[name] = (
                values[0] if all(value == values[0] for value in values) else None
            )
            if name.startswith("effective_"):
                present_values = [int(value) for value in values if value is not None]
                result[f"{name}_min"] = min(present_values) if present_values else None
                result[f"{name}_max"] = max(present_values) if present_values else None
        aggregates.append(result)
    # ``Table.from_pylist`` takes its field set from the first row. Normalize
    # keys so aggregate-only coverage/range fields are not silently discarded.
    output_rows = rows + aggregates
    field_names = set().union(*(row.keys() for row in output_rows))
    for row in output_rows:
        for name in field_names:
            row.setdefault(name, None)
    return pa.Table.from_pylist(output_rows)


def aggregate(args: argparse.Namespace) -> None:
    unit_root = Path(args.unit_root)
    tables_dir = Path(args.tables_dir)
    if not args.allow_partial:
        missing = [
            f"{symbol}/{fold}/h{horizon}"
            for symbol in SYMBOLS
            for fold in FOLDS
            for horizon in HORIZONS
            if not valid_unit_complete(
                unit_dir(unit_root, symbol, fold, horizon), symbol, fold, horizon
            )
        ]
        if missing:
            raise RuntimeError(
                f"diagnostic aggregate requires 72 verified units; missing/stale={len(missing)} first={missing[:5]}"
            )
    dimensions = {
        "baseline_calibration": ("prediction_bin",),
        "sigma_1d": ("sigma_name", "sigma_bin"),
        "conditional_role": ("sigma_name", "prediction_bin", "sigma_bin"),
        "tail_gate": ("sigma_name", "strong_baseline", "sigma_high"),
    }
    for name in TABLE_NAMES:
        paths = sorted(unit_root.glob(f"symbol=*/fold=*/horizon=*/{name}.parquet"))
        if not paths:
            raise FileNotFoundError(f"no unit tables found for {name}: {unit_root}")
        combined = pa.concat_tables(
            [pq.read_table(path) for path in paths], promote_options="default"
        )
        if name != "bin_thresholds":
            combined = date_equal_rows(combined, dimensions[name])
        atomic_parquet(tables_dir / f"{name}.parquet", combined)
    print(
        json.dumps(
            {
                "status": "complete",
                "unit_root": str(unit_root),
                "tables_dir": str(tables_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    unit = subparsers.add_parser("run-unit")
    unit.add_argument("--symbol", required=True, choices=SYMBOLS)
    unit.add_argument("--fold", required=True, choices=tuple(FOLDS))
    unit.add_argument("--horizon", required=True, type=int, choices=HORIZONS)
    unit.add_argument("--phase11-artifacts", default=str(PHASE11_ARTIFACTS))
    unit.add_argument("--output-root", default=str(OUTPUT_ROOT))
    unit.add_argument("--threads", type=int, default=4)
    unit.set_defaults(function=run_unit)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--unit-root", default=str(OUTPUT_ROOT))
    aggregate_parser.add_argument("--tables-dir", default=str(TABLES_DIR))
    aggregate_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="development-only partial aggregation",
    )
    aggregate_parser.set_defaults(function=aggregate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
