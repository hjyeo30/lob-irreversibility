#!/usr/bin/env python3
"""Phase 11: frozen-sigma LightGBM multi-horizon return forecasts."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parents[1]
PHASE10 = PROJECT / "phases/10_predictive_bridge/artifacts"
ARTIFACTS = HERE / "artifacts"
RUN_ID = "phase11_lgbm_v1"
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
FEATURE_SETS = ("R1_baseline", "R2_core_sigma", "R3_sigma_transforms")
OWNERS = {"local": ("QQQ",), "m3": ("NVDA",), "m4": ("SOFI", "SAIA")}
SEEDS = (20260730, 20260731, 20260732)
K = 6
RANDOM_STATE = 20260801
KEYS = ("symbol", "source_date", "segment_id", "path_end_block_index")
FOLDS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "F1": {
        "train": DATES[:2],
        "validation": DATES[2],
        "refit": DATES[:3],
        "test": DATES[3],
    },
    "F2": {
        "train": DATES[:3],
        "validation": DATES[3],
        "refit": DATES[:4],
        "test": DATES[4],
    },
    "F3": {
        "train": DATES[:4],
        "validation": DATES[4],
        "refit": DATES[:5],
        "test": DATES[5],
    },
}
BLOCK_ROOTS = {
    "NVDA": Path(
        "/data/lob_liquidity_time_irreversibility/derived/continuous_blocks_micro_v1/symbol=NVDA"
    ),
    "QQQ": Path(
        "/data/lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=QQQ"
    ),
    "SOFI": Path(
        "/data/lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=SOFI"
    ),
    "SAIA": Path(
        "/data/lob_irreversibility_interpretation_and_execution/derived/continuous_blocks_v1/symbol=SAIA"
    ),
}
BLOCK_COLUMNS = (
    "block_index",
    "end_midprice_x2_nano",
    "end_spread_ticks",
    "block_end_ns",
    "realized_source_event_count",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps({"utc": utc_now(), "event": event, **values}, sort_keys=True),
        flush=True,
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temp, compression="zstd")
    if pq.ParquetFile(temp).metadata.num_rows != table.num_rows:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row validation failed: {path}")
    os.replace(temp, path)


def append_log(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc_now(), **value}, sort_keys=True) + "\n")
        handle.flush()


def block_paths(symbol: str, source_date: str) -> list[Path]:
    root = BLOCK_ROOTS[symbol]
    direct = sorted(
        (root / f"source_date={source_date}" / "target_b=10").glob("*.parquet")
    )
    nested = sorted(
        (
            root
            / f"source_date={source_date}"
            / "regime=daytime"
            / "segment=0000"
            / "target_b=10"
        ).glob("*.parquet")
    )
    paths = direct or nested
    if not paths:
        raise FileNotFoundError(f"B10 blocks not found: {symbol} {source_date}")
    return paths


def read_blocks(symbol: str, source_date: str) -> dict[str, np.ndarray]:
    tables = [
        pq.ParquetFile(path).read(columns=list(BLOCK_COLUMNS))
        for path in block_paths(symbol, source_date)
    ]
    table = pa.concat_tables(tables, promote_options="default")
    index = np.asarray(table["block_index"], dtype=np.int64)
    order = np.argsort(index, kind="stable")
    index = index[order]
    if len(index) == 0 or index[0] != 0 or np.any(np.diff(index) != 1):
        raise RuntimeError(f"non-contiguous block index: {symbol} {source_date}")
    result = {"block_index": index}
    for name in BLOCK_COLUMNS[1:]:
        result[name] = np.asarray(table[name])[order]
    return result


def forecast_path(symbol: str, source_date: str) -> Path:
    return (
        PHASE10
        / "data/forecast_rows"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def score_path(symbol: str, source_date: str) -> Path:
    return (
        PHASE10
        / "data/scores"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def return_path(artifacts: Path, symbol: str, source_date: str) -> Path:
    return (
        artifacts
        / "data/return_rows"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def baseline_columns(names: Iterable[str]) -> list[str]:
    result = [
        name for name in names if name.startswith("m0_") or name.startswith("m1_")
    ]
    if len(result) != 29:
        raise RuntimeError(f"expected 29 baseline features, observed {len(result)}")
    return result


def feature_columns(table: pa.Table) -> dict[str, list[str]]:
    r1 = baseline_columns(table.column_names)
    r2_extra = ["r2_ordered_sigma", "r2_pooled_sigma", "r2_temporal_gap"]
    r3_extra = [
        "r3_relative_gap_ratio",
        "r3_log_magnitude_ratio",
        "r3_ordered_spread_normalized",
        "r3_pooled_spread_normalized",
        "r3_gap_spread_normalized",
        "r3_delta_ordered",
        "r3_delta_pooled",
        "r3_delta_gap",
    ]
    groups = {
        "R1_baseline": r1,
        "R2_core_sigma": r1 + r2_extra,
        "R3_sigma_transforms": r1 + r2_extra + r3_extra,
    }
    expected = {"R1_baseline": 29, "R2_core_sigma": 32, "R3_sigma_transforms": 40}
    for key, value in groups.items():
        if len(value) != expected[key] or any(
            name not in table.column_names for name in value
        ):
            raise RuntimeError(f"feature contract mismatch: {key}")
    return groups


def build_return_unit(
    symbol: str, source_date: str, artifacts: Path, replace: bool = False
) -> dict[str, Any]:
    if symbol not in SYMBOLS or source_date not in DATES:
        raise ValueError("symbol/date outside contract")
    output = return_path(artifacts, symbol, source_date)
    marker = output.with_suffix(".complete.json")
    if output.is_file() and marker.is_file() and not replace:
        saved = json.loads(marker.read_text())
        if (
            saved.get("run_id") == RUN_ID
            and saved.get("rows") == pq.ParquetFile(output).metadata.num_rows
        ):
            emit(
                "return_unit_skip",
                symbol=symbol,
                source_date=source_date,
                rows=saved["rows"],
            )
            return saved
        raise RuntimeError(f"stale return output exists: {output}")
    started = time.monotonic()
    forecast = pq.read_table(forecast_path(symbol, source_date))
    score = pq.read_table(score_path(symbol, source_date))
    if forecast.num_rows < 2 or score.num_rows != forecast.num_rows + 2:
        raise RuntimeError("Phase 10 forecast/score coverage mismatch")
    forecast_end = np.asarray(forecast["path_end_block_index"], dtype=np.int64)
    score_end = np.asarray(score["path_end_block_index"], dtype=np.int64)
    if not np.array_equal(forecast_end, score_end[: forecast.num_rows]):
        raise RuntimeError("forecast/score key mismatch")
    if len(np.unique(forecast_end)) != len(forecast_end) or np.any(
        np.diff(forecast_end) != 3
    ):
        raise RuntimeError("forecast key uniqueness/stride mismatch")
    blocks = read_blocks(symbol, source_date)
    if forecast_end[-1] >= len(blocks["block_index"]):
        raise RuntimeError("forecast cutoff outside block table")
    baseline = baseline_columns(forecast.column_names)
    ordered = np.median(
        np.column_stack(
            [
                np.asarray(score[f"ordered_tcn_seed_{seed}"], dtype=np.float64)[
                    : forecast.num_rows
                ]
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    pooled = np.median(
        np.column_stack(
            [
                np.asarray(score[f"pooled_mlp_seed_{seed}"], dtype=np.float64)[
                    : forecast.num_rows
                ]
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    gap = ordered - pooled
    spread = blocks["end_spread_ticks"].astype(np.float64)[forecast_end]
    if not np.isfinite(spread).all() or np.any(spread < 1.0):
        raise RuntimeError("invalid raw tick spread")
    # Delta is formed on the full Phase 10 sampled universe, before horizon filtering.
    delta_ordered = np.empty_like(ordered)
    delta_ordered[0] = np.nan
    delta_ordered[1:] = np.diff(ordered)
    delta_pooled = np.empty_like(pooled)
    delta_pooled[0] = np.nan
    delta_pooled[1:] = np.diff(pooled)
    delta_gap = np.empty_like(gap)
    delta_gap[0] = np.nan
    delta_gap[1:] = np.diff(gap)
    keep = np.arange(1, forecast.num_rows, dtype=np.int64)
    ordered, pooled, gap, spread = ordered[keep], pooled[keep], gap[keep], spread[keep]
    eps = 1e-6
    data: dict[str, Any] = {name: forecast[name].take(pa.array(keep)) for name in KEYS}
    for name in baseline:
        data[name] = forecast[name].take(pa.array(keep))
    data.update(
        {
            "r2_ordered_sigma": ordered,
            "r2_pooled_sigma": pooled,
            "r2_temporal_gap": gap,
            "r3_relative_gap_ratio": gap / (np.abs(ordered) + np.abs(pooled) + eps),
            "r3_log_magnitude_ratio": np.clip(
                np.log((np.abs(ordered) + eps) / (np.abs(pooled) + eps)), -5.0, 5.0
            ),
            "r3_ordered_spread_normalized": ordered / spread,
            "r3_pooled_spread_normalized": pooled / spread,
            "r3_gap_spread_normalized": gap / spread,
            "r3_delta_ordered": delta_ordered[keep],
            "r3_delta_pooled": delta_pooled[keep],
            "r3_delta_gap": delta_gap[keep],
        }
    )
    cutoff = forecast_end[keep]
    midpoint = blocks["end_midprice_x2_nano"].astype(np.float64)
    block_end = blocks["block_end_ns"].astype(np.int64)
    events = blocks["realized_source_event_count"].astype(np.int64)
    cumulative_events = np.concatenate(([0], np.cumsum(events, dtype=np.int64)))
    support: dict[str, Any] = {}
    for horizon in HORIZONS:
        future = cutoff + horizon
        valid = future < len(midpoint)
        values = np.zeros(len(cutoff), dtype=np.float64)
        event_values = np.zeros(len(cutoff), dtype=np.int64)
        second_values = np.zeros(len(cutoff), dtype=np.float64)
        valid_pos = np.flatnonzero(valid)
        current_mid = midpoint[cutoff[valid]]
        if np.any(current_mid <= 0) or not np.isfinite(current_mid).all():
            raise RuntimeError("invalid cutoff midpoint")
        values[valid] = 1e4 * (midpoint[future[valid]] - current_mid) / current_mid
        event_values[valid] = (
            cumulative_events[future[valid] + 1] - cumulative_events[cutoff[valid] + 1]
        )
        second_values[valid] = (
            block_end[future[valid]] - block_end[cutoff[valid]]
        ) / 1e9
        if (
            not np.isfinite(values[valid]).all()
            or not np.isfinite(second_values[valid]).all()
            or np.any(second_values[valid] < 0)
        ):
            raise RuntimeError(f"invalid target/support h={horizon}")
        invalid = ~valid
        data[f"target_return_bps_h{horizon}"] = pa.array(values, mask=invalid)
        data[f"future_event_count_h{horizon}"] = pa.array(event_values, mask=invalid)
        data[f"future_seconds_h{horizon}"] = pa.array(second_values, mask=invalid)
        support[str(horizon)] = {
            "rows": int(valid.sum()),
            "excluded_tail": int(invalid.sum()),
            "event_mean": float(event_values[valid].mean()),
            "event_median": float(np.median(event_values[valid])),
            "seconds_mean": float(second_values[valid].mean()),
            "seconds_median": float(np.median(second_values[valid])),
        }
    table = pa.table(data)
    groups = feature_columns(table)
    for name in groups["R3_sigma_transforms"]:
        values = np.asarray(table[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite feature: {name}")
    atomic_parquet(output, table)
    complete = {
        "status": "complete",
        "run_id": RUN_ID,
        "contract": "phase11_return_rows_v1",
        "symbol": symbol,
        "source_date": source_date,
        "rows": table.num_rows,
        "source_forecast_rows": forecast.num_rows,
        "first_delta_row_excluded": 1,
        "feature_counts": {key: len(value) for key, value in groups.items()},
        "support": support,
        "output": str(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": utc_now(),
    }
    atomic_json(marker, complete)
    emit(
        "return_unit_complete",
        symbol=symbol,
        source_date=source_date,
        rows=table.num_rows,
        seconds=complete["elapsed_seconds"],
    )
    return complete


def build_all(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts)
    units = [(symbol, source_date) for symbol in SYMBOLS for source_date in DATES]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_return_unit, symbol, source_date, artifacts, args.replace
            ): (symbol, source_date)
            for symbol, source_date in units
        }
        for future in as_completed(futures):
            future.result()


def day_equal_weights(days: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(days, return_counts=True)
    total = len(days)
    lookup = {
        day: total / (len(unique) * count)
        for day, count in zip(unique, counts, strict=True)
    }
    result = np.asarray([lookup[day] for day in days], dtype=np.float64)
    for day in unique:
        if not np.isclose(
            result[days == day].sum(), total / len(unique), rtol=1e-12, atol=1e-9
        ):
            raise RuntimeError("day equal-weight contract failed")
    return result


def load_date(
    artifacts: Path, symbol: str, source_date: str, horizon: int, features: list[str]
) -> dict[str, np.ndarray]:
    columns = (
        list(KEYS)
        + features
        + [
            f"target_return_bps_h{horizon}",
            f"future_event_count_h{horizon}",
            f"future_seconds_h{horizon}",
        ]
    )
    table = pq.read_table(return_path(artifacts, symbol, source_date), columns=columns)
    target_array = table[f"target_return_bps_h{horizon}"].combine_chunks()
    valid = np.asarray(pc.invert(pc.is_null(target_array)), dtype=bool)
    target = target_array.to_numpy(zero_copy_only=False)[valid].astype(np.float64)
    x = np.column_stack(
        [np.asarray(table[name], dtype=np.float64)[valid] for name in features]
    )
    keys = {name: np.asarray(table[name])[valid] for name in KEYS}
    future_events = (
        table[f"future_event_count_h{horizon}"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)[valid]
        .astype(np.int64)
    )
    future_seconds = (
        table[f"future_seconds_h{horizon}"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)[valid]
        .astype(np.float64)
    )
    if (
        len(target) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(target).all()
        or not np.isfinite(future_seconds).all()
    ):
        raise RuntimeError(f"invalid model input: {symbol} {source_date} h={horizon}")
    if len(np.unique(keys["path_end_block_index"])) != len(target):
        raise RuntimeError("duplicate prediction key")
    return {
        "x": x,
        "y": target,
        "events": future_events,
        "seconds": future_seconds,
        **keys,
    }


def concatenate(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([item[key] for item in items]) for key in items[0]}


def safe_metric(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def regression_metrics(
    y: np.ndarray, prediction: np.ndarray
) -> dict[str, float | None]:
    residual = y - prediction
    mse = float(np.mean(residual * residual))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(y * y))
    pearson = (
        float(pearsonr(y, prediction).statistic)
        if len(y) > 1 and np.std(y) > 0 and np.std(prediction) > 0
        else math.nan
    )
    spearman = (
        float(spearmanr(y, prediction).statistic)
        if len(y) > 1 and len(np.unique(y)) > 1 and len(np.unique(prediction)) > 1
        else math.nan
    )
    return {
        "mse": mse,
        "mae": mae,
        "pearson": safe_metric(pearson),
        "spearman": safe_metric(spearman),
        "oos_r2_zero": safe_metric(
            1.0 - float(np.sum(residual * residual)) / denominator
        )
        if denominator > 0
        else None,
        "bias_prediction_minus_realized": float(np.mean(prediction) - np.mean(y)),
    }


def unit_paths(
    artifacts: Path, symbol: str, fold: str, horizon: int, feature_set: str
) -> dict[str, Path]:
    model_dir = (
        artifacts
        / "models"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / f"features={feature_set}"
    )
    return {
        "model": model_dir / "model.txt",
        "metric": model_dir / "metric.parquet",
        "complete": model_dir / "complete.json",
        "prediction": artifacts
        / "predictions"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / f"features={feature_set}.parquet",
    }


def unit_config(
    symbol: str, fold: str, horizon: int, feature_set: str, threads: int
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "symbol": symbol,
        "fold": fold,
        "horizon": horizon,
        "feature_set": feature_set,
        "objective": "regression_l2",
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "n_estimators_max": 2000,
        "early_stopping_rounds": 100,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": RANDOM_STATE,
        "threads": threads,
    }


def current_input_counts(artifacts: Path, symbol: str, horizon: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for source_date in DATES:
        path = return_path(artifacts, symbol, source_date)
        marker = path.with_suffix(".complete.json")
        if not path.is_file() or not marker.is_file():
            raise RuntimeError(f"missing return input: {symbol} {source_date}")
        complete = json.loads(marker.read_text())
        if (
            complete.get("run_id") != RUN_ID
            or complete.get("rows") != pq.ParquetFile(path).metadata.num_rows
        ):
            raise RuntimeError(f"stale return input: {symbol} {source_date}")
        result[source_date] = int(complete["support"][str(horizon)]["rows"])
    return result


def validate_complete(
    paths: dict[str, Path], config: dict[str, Any], artifacts: Path
) -> bool:
    if not all(
        paths[name].is_file() for name in ("model", "metric", "prediction", "complete")
    ):
        return False
    try:
        complete = json.loads(paths["complete"].read_text())
        if complete.get("status") != "complete" or complete.get("config") != config:
            return False
        if complete.get("input_rows_by_date") != current_input_counts(
            artifacts, config["symbol"], config["horizon"]
        ):
            return False
        if paths["model"].stat().st_size == 0:
            return False
        if pq.ParquetFile(paths["metric"]).metadata.num_rows != 1:
            return False
        if (
            pq.ParquetFile(paths["prediction"]).metadata.num_rows
            != complete["prediction_rows"]
        ):
            return False
        prediction = pq.read_table(
            paths["prediction"],
            columns=list(KEYS) + ["target_return_bps", "prediction"],
        )
        if (
            pc.any(pc.invert(pc.is_finite(prediction["prediction"]))).as_py()
            or pc.any(pc.invert(pc.is_finite(prediction["target_return_bps"]))).as_py()
        ):
            return False
        metric = pq.read_table(paths["metric"]).to_pylist()[0]
        if any(
            metric[name] != config[name]
            for name in ("run_id", "symbol", "fold", "horizon", "feature_set")
        ):
            return False
        target_values = np.asarray(prediction["target_return_bps"], dtype=np.float64)
        prediction_values = np.asarray(prediction["prediction"], dtype=np.float64)
        recomputed = regression_metrics(target_values, prediction_values)
        for name in ("mse", "mae", "oos_r2_zero", "bias_prediction_minus_realized"):
            if metric[name] is None or recomputed[name] is None:
                if metric[name] != recomputed[name]:
                    return False
            elif not np.isclose(metric[name], recomputed[name], rtol=1e-12, atol=1e-15):
                return False
        return (
            len(np.unique(np.asarray(prediction["path_end_block_index"])))
            == prediction.num_rows
        )
    except Exception:
        return False


def lgbm_parameters(train_rows: int, threads: int, n_estimators: int) -> dict[str, Any]:
    return {
        "objective": "regression_l2",
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "n_estimators": n_estimators,
        "min_child_samples": max(200, min(20_000, math.floor(0.005 * train_rows))),
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "deterministic": True,
        "force_col_wise": True,
        "random_state": RANDOM_STATE,
        "n_jobs": threads,
        "verbosity": -1,
    }


def fit_unit(
    symbol: str,
    fold: str,
    horizon: int,
    feature_set: str,
    artifacts: Path,
    threads: int,
    replace: bool = False,
) -> dict[str, Any]:
    if (
        symbol not in SYMBOLS
        or fold not in FOLDS
        or horizon not in HORIZONS
        or feature_set not in FEATURE_SETS
    ):
        raise ValueError("unit outside contract")
    paths = unit_paths(artifacts, symbol, fold, horizon, feature_set)
    config = unit_config(symbol, fold, horizon, feature_set, threads)
    if validate_complete(paths, config, artifacts) and not replace:
        complete = json.loads(paths["complete"].read_text())
        emit(
            "fit_unit_skip",
            symbol=symbol,
            fold=fold,
            horizon=horizon,
            feature_set=feature_set,
        )
        return complete
    if replace:
        for path in paths.values():
            path.unlink(missing_ok=True)
    elif paths["complete"].exists():
        raise RuntimeError(f"stale completed unit exists: {paths['complete']}")
    else:
        # A crash between per-file atomic renames leaves an incomplete unit. Only
        # this exact unit's partial finals are replaced; no shared output is touched.
        for name in ("model", "metric", "prediction"):
            paths[name].unlink(missing_ok=True)
    started = time.monotonic()
    probe = pq.read_table(return_path(artifacts, symbol, DATES[0]))
    features = feature_columns(probe)[feature_set]
    spec = FOLDS[fold]
    train_dates = tuple(spec["train"])  # type: ignore[arg-type]
    validation_date = str(spec["validation"])
    refit_dates = tuple(spec["refit"])  # type: ignore[arg-type]
    test_date = str(spec["test"])
    needed = sorted(
        set(train_dates + (validation_date,) + refit_dates + (test_date,)),
        key=DATES.index,
    )
    days = {day: load_date(artifacts, symbol, day, horizon, features) for day in needed}
    train = concatenate([days[day] for day in train_dates])
    validation = days[validation_date]
    refit = concatenate([days[day] for day in refit_dates])
    test = days[test_date]
    train_weights = day_equal_weights(train["source_date"])
    refit_weights = day_equal_weights(refit["source_date"])
    min_leaf = max(200, min(20_000, math.floor(0.005 * len(train["y"]))))
    estimator = lgb.LGBMRegressor(**lgbm_parameters(len(train["y"]), threads, 2000))
    estimator.fit(
        train["x"],
        train["y"],
        sample_weight=train_weights,
        eval_set=[(validation["x"], validation["y"])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(50)],
    )
    best_iteration = int(estimator.best_iteration_ or 2000)
    validation_prediction = estimator.predict(
        validation["x"], num_iteration=best_iteration
    )
    refit_estimator = lgb.LGBMRegressor(
        **lgbm_parameters(len(train["y"]), threads, best_iteration)
    )
    # Keep min_data_in_leaf fixed from the early-stop training rows.
    refit_estimator.set_params(min_child_samples=min_leaf)
    refit_estimator.fit(refit["x"], refit["y"], sample_weight=refit_weights)
    prediction = np.asarray(refit_estimator.predict(test["x"]), dtype=np.float64)
    if len(prediction) != len(test["y"]) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid test prediction")
    metrics = regression_metrics(test["y"], prediction)
    validation_mse = float(np.mean((validation["y"] - validation_prediction) ** 2))
    rss = psutil.Process().memory_info().rss
    metric_row = {
        "run_id": RUN_ID,
        "symbol": symbol,
        "fold": fold,
        "horizon": horizon,
        "feature_set": feature_set,
        "test_date": test_date,
        "train_dates": ",".join(train_dates),
        "validation_date": validation_date,
        "refit_dates": ",".join(refit_dates),
        "feature_count": len(features),
        "train_rows": len(train["y"]),
        "validation_rows": len(validation["y"]),
        "refit_rows": len(refit["y"]),
        "test_rows": len(test["y"]),
        "min_data_in_leaf": min_leaf,
        "best_iteration": best_iteration,
        "validation_mse": validation_mse,
        **metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": rss,
        "host": os.uname().nodename,
        "threads": threads,
    }
    prediction_table = pa.table(
        {
            **{name: test[name] for name in KEYS},
            "horizon": np.repeat(horizon, len(prediction)),
            "feature_set": np.repeat(feature_set, len(prediction)),
            "target_return_bps": test["y"],
            "prediction": prediction,
            "future_event_count": test["events"],
            "future_seconds": test["seconds"],
        }
    )
    metric_table = pa.Table.from_pylist([metric_row])
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model_temp = paths["model"].with_name(
        f".{paths['model'].name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    refit_estimator.booster_.save_model(model_temp)
    if not model_temp.is_file() or model_temp.stat().st_size == 0:
        raise RuntimeError("empty LightGBM model")
    atomic_parquet(paths["prediction"], prediction_table)
    atomic_parquet(paths["metric"], metric_table)
    os.replace(model_temp, paths["model"])
    input_rows_by_date = current_input_counts(artifacts, symbol, horizon)
    complete = {
        "status": "complete",
        "contract": "phase11_lgbm_unit_v1",
        "config": config,
        "features": features,
        "feature_count": len(features),
        "train_rows": len(train["y"]),
        "validation_rows": len(validation["y"]),
        "refit_rows": len(refit["y"]),
        "input_rows_by_date": input_rows_by_date,
        "prediction_rows": len(prediction),
        "best_iteration": best_iteration,
        "model_bytes": paths["model"].stat().st_size,
        "prediction_bytes": paths["prediction"].stat().st_size,
        "metric_bytes": paths["metric"].stat().st_size,
        "elapsed_seconds": metric_row["elapsed_seconds"],
        "completed_utc": utc_now(),
    }
    atomic_json(paths["complete"], complete)
    if not validate_complete(paths, config, artifacts):
        raise RuntimeError("published unit validation failed")
    emit(
        "fit_unit_complete",
        symbol=symbol,
        fold=fold,
        horizon=horizon,
        feature_set=feature_set,
        rows=len(prediction),
        best_iteration=best_iteration,
        seconds=metric_row["elapsed_seconds"],
        rss_bytes=rss,
    )
    return complete


def approved_queue(symbols: Iterable[str]) -> list[tuple[str, str, int, str]]:
    queue = [
        (symbol, fold, horizon, feature_set)
        for symbol in symbols
        for fold in FOLDS
        for horizon in HORIZONS
        for feature_set in FEATURE_SETS
    ]
    if (
        len(queue) != len(set(queue))
        or len(
            approved_queue_all := [
                (s, f, h, r)
                for s in SYMBOLS
                for f in FOLDS
                for h in HORIZONS
                for r in FEATURE_SETS
            ]
        )
        != 216
    ):
        raise RuntimeError("queue cardinality contract failed")
    if len(set(approved_queue_all)) != 216:
        raise RuntimeError("duplicate full queue")
    return queue


def resource_snapshot(path: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(path)
    return {
        "cpu_count_logical": psutil.cpu_count(),
        "load_1m": os.getloadavg()[0],
        "memory_available_bytes": memory.available,
        "swap_used_bytes": swap.used,
        "disk_free_bytes": disk.free,
    }


def run_queue(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts).resolve()
    symbols = tuple(args.symbols.split(","))
    if not symbols or any(symbol not in SYMBOLS for symbol in symbols):
        raise ValueError("invalid symbols")
    if args.host_lane not in OWNERS or tuple(symbols) != OWNERS[args.host_lane]:
        raise ValueError(f"host lane ownership mismatch: {args.host_lane} {symbols}")
    log = artifacts / "logs" / f"run_{RUN_ID}_{args.host_lane}.jsonl"
    manifest = artifacts / "manifests" / f"run_{RUN_ID}_{args.host_lane}.json"
    snapshot = resource_snapshot(artifacts)
    atomic_json(
        manifest,
        {
            "run_id": RUN_ID,
            "host_lane": args.host_lane,
            "symbols": symbols,
            "workers": args.workers,
            "threads": args.threads,
            "lightgbm_version": lgb.__version__,
            "resource": snapshot,
            "started_utc": utc_now(),
        },
    )
    append_log(log, {"event": "run_start", **json.loads(manifest.read_text())})
    queue = approved_queue(symbols)
    pending: list[tuple[str, str, int, str]] = []
    for unit in queue:
        paths = unit_paths(artifacts, *unit)
        if not validate_complete(paths, unit_config(*unit, args.threads), artifacts):
            pending.append(unit)
    completed_at_start = len(queue) - len(pending)
    emit(
        "queue_start",
        host_lane=args.host_lane,
        total=len(queue),
        complete=completed_at_start,
        pending=len(pending),
        **snapshot,
    )
    running: dict[subprocess.Popen[str], tuple[str, str, int, str, Any]] = {}
    completed = completed_at_start
    failed = 0
    started = time.monotonic()
    last_status = 0.0
    while pending or running:
        while pending and len(running) < args.workers:
            unit = pending.pop(0)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "fit-unit",
                "--artifacts",
                str(artifacts),
                "--symbol",
                unit[0],
                "--fold",
                unit[1],
                "--horizon",
                str(unit[2]),
                "--feature-set",
                unit[3],
                "--threads",
                str(args.threads),
            ]
            unit_log = (
                artifacts
                / "logs"
                / "units"
                / f"{unit[0]}_{unit[1]}_h{unit[2]}_{unit[3]}.log"
            )
            unit_log.parent.mkdir(parents=True, exist_ok=True)
            handle = unit_log.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            running[process] = (*unit, handle)
            append_log(log, {"event": "unit_start", "unit": unit, "pid": process.pid})
        time.sleep(1.0)
        for process, value in list(running.items()):
            if process.poll() is None:
                continue
            unit = value[:4]
            handle = value[4]
            handle.close()
            del running[process]
            if process.returncode == 0:
                completed += 1
                append_log(log, {"event": "unit_complete", "unit": unit})
            else:
                failed += 1
                append_log(
                    log,
                    {
                        "event": "unit_failed",
                        "unit": unit,
                        "returncode": process.returncode,
                    },
                )
                for peer, peer_value in list(running.items()):
                    peer.terminate()
                for peer, peer_value in list(running.items()):
                    try:
                        peer.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        peer.kill()
                        peer.wait()
                    peer_value[4].close()
                running.clear()
                raise RuntimeError(f"unit failed: {unit}; inspect unit log")
        if time.monotonic() - last_status >= args.status_seconds:
            elapsed = time.monotonic() - started
            done_this_run = completed - completed_at_start
            throughput = done_this_run / elapsed * 3600 if elapsed > 0 else 0.0
            eta = (
                (len(pending) + len(running)) / (throughput / 3600)
                if throughput > 0
                else None
            )
            status = {
                "event": "queue_status",
                "complete": completed,
                "running": len(running),
                "pending": len(pending),
                "failed": failed,
                "units_per_hour": throughput,
                "eta_seconds": eta,
                **resource_snapshot(artifacts),
            }
            append_log(log, status)
            emit(**status)
            last_status = time.monotonic()
    append_log(
        log,
        {
            "event": "run_complete",
            "complete": completed,
            "failed": failed,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def aggregate(args: argparse.Namespace) -> None:
    artifacts = Path(args.artifacts)
    metrics: list[pa.Table] = []
    completions: list[dict[str, Any]] = []
    for unit in approved_queue(SYMBOLS):
        paths = unit_paths(artifacts, *unit)
        config = unit_config(*unit, args.threads_by_owner[unit[0]])
        if not validate_complete(paths, config, artifacts):
            raise RuntimeError(f"missing/invalid unit: {unit}")
        metrics.append(pq.read_table(paths["metric"]))
        completions.append(json.loads(paths["complete"].read_text()))
    fold_metrics = pa.concat_tables(metrics, promote_options="default")
    if fold_metrics.num_rows != 216:
        raise RuntimeError("fold metric count mismatch")
    metric_rows = fold_metrics.to_pylist()
    by_key = {
        (row["symbol"], row["fold"], row["horizon"], row["feature_set"]): row
        for row in metric_rows
    }
    if len(by_key) != 216:
        raise RuntimeError("duplicate unit metric")
    increments: list[dict[str, Any]] = []
    comparisons = (
        ("R1_baseline", "R2_core_sigma"),
        ("R2_core_sigma", "R3_sigma_transforms"),
        ("R1_baseline", "R3_sigma_transforms"),
    )
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for horizon in HORIZONS:
                prediction_tables = {
                    feature_set: pq.read_table(
                        unit_paths(artifacts, symbol, fold, horizon, feature_set)[
                            "prediction"
                        ]
                    )
                    for feature_set in FEATURE_SETS
                }
                reference = prediction_tables["R1_baseline"]
                for feature_set, table in prediction_tables.items():
                    for key in KEYS:
                        if not np.array_equal(
                            np.asarray(reference[key]), np.asarray(table[key])
                        ):
                            raise RuntimeError(
                                f"paired prediction key mismatch: {symbol} {fold} h={horizon} {feature_set}"
                            )
                    if not np.array_equal(
                        np.asarray(reference["target_return_bps"]),
                        np.asarray(table["target_return_bps"]),
                    ):
                        raise RuntimeError("paired target mismatch")
                for left, right in comparisons:
                    a = by_key[(symbol, fold, horizon, left)]
                    b = by_key[(symbol, fold, horizon, right)]
                    relative = (
                        (a["mse"] - b["mse"]) / a["mse"] if a["mse"] > 0 else None
                    )
                    increments.append(
                        {
                            "run_id": RUN_ID,
                            "symbol": symbol,
                            "fold": fold,
                            "test_date": a["test_date"],
                            "horizon": horizon,
                            "left_feature_set": left,
                            "right_feature_set": right,
                            "mse_improvement_left_minus_right": a["mse"] - b["mse"],
                            "relative_mse_reduction": relative,
                        }
                    )
    audit_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for source_date in DATES:
            marker = return_path(artifacts, symbol, source_date).with_suffix(
                ".complete.json"
            )
            if not marker.is_file():
                raise RuntimeError(f"missing return completion: {symbol} {source_date}")
            complete = json.loads(marker.read_text())
            return_file = return_path(artifacts, symbol, source_date)
            if (
                complete.get("status") != "complete"
                or complete.get("run_id") != RUN_ID
                or not return_file.is_file()
            ):
                raise RuntimeError(f"invalid return completion: {symbol} {source_date}")
            return_table = pq.read_table(return_file)
            if return_table.num_rows != complete.get("rows"):
                raise RuntimeError(f"return row count mismatch: {symbol} {source_date}")
            for name in feature_columns(return_table)["R3_sigma_transforms"]:
                if pc.any(pc.invert(pc.is_finite(return_table[name]))).as_py():
                    raise RuntimeError(
                        f"nonfinite return feature: {symbol} {source_date} {name}"
                    )
            for horizon in HORIZONS:
                support = complete["support"][str(horizon)]
                target = return_table[f"target_return_bps_h{horizon}"]
                valid = pc.drop_null(target)
                if (
                    len(valid) != support["rows"]
                    or pc.any(pc.invert(pc.is_finite(valid))).as_py()
                ):
                    raise RuntimeError(
                        f"return target support mismatch: {symbol} {source_date} h={horizon}"
                    )
                audit_rows.append(
                    {
                        "run_id": RUN_ID,
                        "symbol": symbol,
                        "source_date": source_date,
                        "horizon": horizon,
                        **support,
                    }
                )
    resource_rows = []
    for row, complete in zip(metric_rows, completions, strict=True):
        resource_rows.append(
            {
                "run_id": RUN_ID,
                "symbol": row["symbol"],
                "fold": row["fold"],
                "horizon": row["horizon"],
                "feature_set": row["feature_set"],
                "host": row["host"],
                "threads": row["threads"],
                "elapsed_seconds": row["elapsed_seconds"],
                "postfit_process_rss_bytes": row["peak_process_rss_bytes"],
                "rss_measurement": "after_fit_before_publication",
                "best_iteration": row["best_iteration"],
                "model_bytes": complete["model_bytes"],
                "prediction_bytes": complete["prediction_bytes"],
            }
        )
    tables = artifacts / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    # An old marker must never advertise a partially refreshed aggregate bundle.
    (tables / "complete.json").unlink(missing_ok=True)
    atomic_parquet(tables / "fold_metrics.parquet", fold_metrics)
    atomic_parquet(
        tables / "paired_increments.parquet", pa.Table.from_pylist(increments)
    )
    atomic_parquet(tables / "feature_audits.parquet", pa.Table.from_pylist(audit_rows))
    atomic_parquet(
        tables / "resource_summary.parquet", pa.Table.from_pylist(resource_rows)
    )
    # Date-equal compact loss, then ratios of those date-equal losses.
    compact: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            model_summary: dict[str, dict[str, Any]] = {}
            for feature_set in FEATURE_SETS:
                rows = [by_key[(symbol, fold, horizon, feature_set)] for fold in FOLDS]
                summary = {
                    metric: (
                        float(np.mean([row[metric] for row in rows]))
                        if all(row[metric] is not None for row in rows)
                        else None
                    )
                    for metric in (
                        "mse",
                        "mae",
                        "pearson",
                        "spearman",
                        "oos_r2_zero",
                        "bias_prediction_minus_realized",
                    )
                }
                model_summary[feature_set] = summary
            base = model_summary["R1_baseline"]["mse"]
            core = model_summary["R2_core_sigma"]["mse"]
            total = model_summary["R3_sigma_transforms"]["mse"]
            for feature_set in FEATURE_SETS:
                compact.append(
                    {
                        "run_id": RUN_ID,
                        "symbol": symbol,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        **model_summary[feature_set],
                        "relative_mse_reduction_vs_r1": (
                            (base - model_summary[feature_set]["mse"]) / base
                            if base and model_summary[feature_set]["mse"] is not None
                            else None
                        ),
                        "relative_mse_reduction_vs_r2": (
                            (core - model_summary[feature_set]["mse"]) / core
                            if feature_set == "R3_sigma_transforms"
                            and core
                            and model_summary[feature_set]["mse"] is not None
                            else None
                        ),
                    }
                )
    atomic_parquet(tables / "compact_summary.parquet", pa.Table.from_pylist(compact))
    atomic_json(
        tables / "complete.json",
        {
            "status": "complete",
            "run_id": RUN_ID,
            "contract": "phase11_compact_v1",
            "return_units": 24,
            "model_units": 216,
            "missing_units": 0,
            "duplicate_units": 0,
            "feature_counts": {
                "R1_baseline": 29,
                "R2_core_sigma": 32,
                "R3_sigma_transforms": 40,
            },
            "paired_prediction_key_mismatches": 0,
            "nonfinite_predictions": 0,
            "tables": {
                path.name: pq.ParquetFile(path).metadata.num_rows
                for path in tables.glob("*.parquet")
            },
            "completed_utc": utc_now(),
        },
    )
    emit(
        "aggregate_complete",
        model_units=216,
        return_units=24,
        compact_rows=len(compact),
    )


def determinism_check(args: argparse.Namespace) -> None:
    source = Path(args.artifacts)
    roots = [Path(args.temp_root) / "a", Path(args.temp_root) / "b"]
    for root in roots:
        for symbol in (args.symbol,):
            target = root / "data/return_rows" / f"symbol={symbol}"
            target.mkdir(parents=True, exist_ok=True)
            for source_date in DATES:
                source_file = return_path(source, symbol, source_date)
                os.symlink(source_file, target / source_file.name)
                source_marker = source_file.with_suffix(".complete.json")
                os.symlink(source_marker, target / source_marker.name)
        fit_unit(
            args.symbol, args.fold, args.horizon, args.feature_set, root, args.threads
        )
    pa_ = unit_paths(roots[0], args.symbol, args.fold, args.horizon, args.feature_set)
    pb_ = unit_paths(roots[1], args.symbol, args.fold, args.horizon, args.feature_set)
    a = pq.read_table(pa_["prediction"])
    b = pq.read_table(pb_["prediction"])
    if not np.array_equal(np.asarray(a["prediction"]), np.asarray(b["prediction"])):
        raise RuntimeError("deterministic rerun prediction mismatch")
    ca = json.loads(pa_["complete"].read_text())
    cb = json.loads(pb_["complete"].read_text())
    if ca["best_iteration"] != cb["best_iteration"]:
        raise RuntimeError("deterministic rerun best-iteration mismatch")
    emit(
        "determinism_check_complete",
        symbol=args.symbol,
        fold=args.fold,
        horizon=args.horizon,
        feature_set=args.feature_set,
        rows=a.num_rows,
        best_iteration=ca["best_iteration"],
    )


def parser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser()
    commands = parser_.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-return-unit")
    build.add_argument("--symbol", required=True, choices=SYMBOLS)
    build.add_argument("--source-date", required=True, choices=DATES)
    build.add_argument("--artifacts", default=str(ARTIFACTS))
    build.add_argument("--replace", action="store_true")
    build.set_defaults(
        function=lambda args: build_return_unit(
            args.symbol, args.source_date, Path(args.artifacts), args.replace
        )
    )
    build_many = commands.add_parser("build-all")
    build_many.add_argument("--artifacts", default=str(ARTIFACTS))
    build_many.add_argument("--workers", type=int, default=4)
    build_many.add_argument("--replace", action="store_true")
    build_many.set_defaults(function=build_all)
    fit = commands.add_parser("fit-unit")
    fit.add_argument("--symbol", required=True, choices=SYMBOLS)
    fit.add_argument("--fold", required=True, choices=tuple(FOLDS))
    fit.add_argument("--horizon", required=True, type=int, choices=HORIZONS)
    fit.add_argument("--feature-set", required=True, choices=FEATURE_SETS)
    fit.add_argument("--artifacts", default=str(ARTIFACTS))
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--replace", action="store_true")
    fit.set_defaults(
        function=lambda args: fit_unit(
            args.symbol,
            args.fold,
            args.horizon,
            args.feature_set,
            Path(args.artifacts),
            args.threads,
            args.replace,
        )
    )
    queue = commands.add_parser("run-queue")
    queue.add_argument("--symbols", required=True)
    queue.add_argument("--host-lane", required=True)
    queue.add_argument("--artifacts", default=str(ARTIFACTS))
    queue.add_argument("--workers", type=int, default=1)
    queue.add_argument("--threads", type=int, default=8)
    queue.add_argument("--status-seconds", type=float, default=60.0)
    queue.set_defaults(function=run_queue)
    collect = commands.add_parser("aggregate")
    collect.add_argument("--artifacts", default=str(ARTIFACTS))
    collect.add_argument("--local-threads", type=int, default=8)
    collect.add_argument("--m3-threads", type=int, default=10)
    collect.add_argument("--m4-threads", type=int, default=4)
    collect.set_defaults(
        function=lambda args: aggregate(
            argparse.Namespace(
                artifacts=args.artifacts,
                threads_by_owner={
                    "QQQ": args.local_threads,
                    "NVDA": args.m3_threads,
                    "SOFI": args.m4_threads,
                    "SAIA": args.m4_threads,
                },
            )
        )
    )
    deterministic = commands.add_parser("determinism-check")
    deterministic.add_argument("--artifacts", default=str(ARTIFACTS))
    deterministic.add_argument("--temp-root", required=True)
    deterministic.add_argument("--symbol", default="SAIA", choices=SYMBOLS)
    deterministic.add_argument("--fold", default="F1", choices=tuple(FOLDS))
    deterministic.add_argument("--horizon", type=int, default=2, choices=HORIZONS)
    deterministic.add_argument(
        "--feature-set", default="R1_baseline", choices=FEATURE_SETS
    )
    deterministic.add_argument("--threads", type=int, default=4)
    deterministic.set_defaults(function=determinism_check)
    return parser_


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
