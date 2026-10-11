#!/usr/bin/env python3
"""Leakage-safe Phase 12 nuisance features and matched forecast units."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
PHASE11_DIR = HERE.parent / "11_lgbm_multihorizon_return"
DEFAULT_ARTIFACTS = HERE / "artifacts"
RUN_ID = "phase12_sigma_role_v1"
NUISANCE_TREES = 300
KEYS = ("symbol", "source_date", "segment_id", "path_end_block_index")


def _load_phase11() -> Any:
    path = PHASE11_DIR / "codebook.py"
    spec = importlib.util.spec_from_file_location("phase11_lgbm_codebook", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Phase 11 codebook: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P11 = _load_phase11()
SYMBOLS = tuple(P11.SYMBOLS)
DATES = tuple(P11.DATES)
HORIZONS = tuple(P11.HORIZONS)
FOLDS = P11.FOLDS
SEEDS = tuple(P11.SEEDS)

SIGNED_SPECS: dict[str, tuple[str, ...]] = {
    "D1_p_perp": ("p_perp_x",),
    "D2_p_o_perp": ("p_perp_x", "o_perp_xp"),
    "odd": ("ordered_odd", "pooled_odd", "gap_odd"),
    "even": ("ordered_even", "pooled_even", "gap_even"),
    "lag2": ("gap_innovation_lag2",),
    "ewma": ("gap_innovation_ewma_hl4",),
}
ROLE_SPECS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "even": SIGNED_SPECS["even"],
    "odd": SIGNED_SPECS["odd"],
}
DERIVED_COLUMNS = tuple(
    dict.fromkeys(
        SIGNED_SPECS["odd"]
        + SIGNED_SPECS["even"]
        + SIGNED_SPECS["lag2"]
        + SIGNED_SPECS["ewma"]
        + SIGNED_SPECS["D2_p_o_perp"]
    )
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


def derived_path(artifacts: Path, symbol: str, fold: str, stage: str) -> Path:
    return (
        artifacts
        / "data/derived_features"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"stage={stage}.parquet"
    )


def nuisance_paths(
    artifacts: Path, symbol: str, fold: str, stage: str
) -> dict[str, Path]:
    root = (
        artifacts / "nuisance" / f"symbol={symbol}" / f"fold={fold}" / f"stage={stage}"
    )
    return {"metric": root / "metric.parquet", "complete": root / "complete.json"}


def model_paths(
    artifacts: Path, target: str, symbol: str, fold: str, horizon: int, spec: str
) -> dict[str, Path]:
    root = (
        artifacts
        / "models"
        / f"target={target}"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / f"spec={spec}"
    )
    prediction = (
        artifacts
        / "predictions"
        / f"target={target}"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
        / f"spec={spec}.parquet"
    )
    return {
        "model": root / "model.txt",
        "metric": root / "metric.parquet",
        "complete": root / "complete.json",
        "prediction": prediction,
    }


def _return_table(
    symbol: str, source_date: str, columns: list[str] | None = None
) -> pa.Table:
    return pq.read_table(
        P11.return_path(P11.ARTIFACTS, symbol, source_date), columns=columns
    )


def _key_table(table: pa.Table) -> pa.Table:
    return table.select(list(KEYS))


def _keys_equal(left: pa.Table, right: pa.Table) -> bool:
    if left.num_rows != right.num_rows:
        return False
    return all(pc.all(pc.equal(left[name], right[name])).as_py() for name in KEYS)


def _base_arrays(symbol: str, dates: tuple[str, ...]) -> dict[str, Any]:
    probe = _return_table(symbol, dates[0])
    x_names = P11.feature_columns(probe)["R1_baseline"]
    tables = [
        _return_table(
            symbol, day, list(KEYS) + x_names + ["r2_ordered_sigma", "r2_pooled_sigma"]
        )
        for day in dates
    ]
    table = pa.concat_tables(tables, promote_options="default")
    x = np.column_stack([np.asarray(table[name], dtype=np.float64) for name in x_names])
    p = np.asarray(table["r2_pooled_sigma"], dtype=np.float64)
    o = np.asarray(table["r2_ordered_sigma"], dtype=np.float64)
    days = np.asarray(table["source_date"], dtype=str)
    if not np.isfinite(x).all() or not np.isfinite(p).all() or not np.isfinite(o).all():
        raise RuntimeError("nonfinite nuisance input")
    return {"table": table, "x": x, "p": p, "o": o, "days": days, "x_names": x_names}


def _score_medians(table: pa.Table, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    ordered = np.median(
        np.column_stack(
            [
                np.asarray(table[f"{prefix}ordered_tcn_seed_{seed}"], dtype=np.float64)
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    pooled = np.median(
        np.column_stack(
            [
                np.asarray(table[f"{prefix}pooled_mlp_seed_{seed}"], dtype=np.float64)
                for seed in SEEDS
            ]
        ),
        axis=1,
    )
    return ordered, pooled


def _rolling_gap(gap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lag-2 and same-parity past-only EWMA innovations on a full score stream."""
    lag2 = np.full(len(gap), np.nan, dtype=np.float64)
    ewma = np.full(len(gap), np.nan, dtype=np.float64)
    if len(gap) > 2:
        lag2[2:] = gap[2:] - gap[:-2]
    alpha = 1.0 - math.exp(math.log(0.5) / 4.0)
    for parity in (0, 1):
        state = 0.0
        count = 0
        for index in range(parity, len(gap), 2):
            if count >= 4:
                ewma[index] = gap[index] - state
            state = (
                gap[index] if count == 0 else alpha * gap[index] + (1.0 - alpha) * state
            )
            count += 1
    return lag2, ewma


def _base_derived_table(
    artifacts: Path, symbol: str, dates: tuple[str, ...]
) -> pa.Table:
    pieces: list[pa.Table] = []
    for source_date in dates:
        returns = _return_table(symbol, source_date, list(KEYS))
        original = pq.read_table(P11.score_path(symbol, source_date))
        reflected_path = (
            artifacts
            / "data/reflected_scores"
            / f"symbol={symbol}"
            / f"source_date={source_date}.parquet"
        )
        if not reflected_path.is_file():
            raise RuntimeError(f"missing reflected score input: {reflected_path}")
        reflected = pq.read_table(reflected_path)
        if not _keys_equal(_key_table(original), _key_table(reflected)):
            raise RuntimeError(
                f"original/reflected score key mismatch: {symbol} {source_date}"
            )
        ordered, pooled = _score_medians(original)
        reflected_ordered, reflected_pooled = _score_medians(reflected, "reflected_")
        gap = ordered - pooled
        reflected_gap = reflected_ordered - reflected_pooled
        lag2, ewma = _rolling_gap(gap)
        # Phase 11 removed the first score row for delta support and two tail rows.
        if original.num_rows != returns.num_rows + 3:
            raise RuntimeError(
                f"score/return coverage mismatch: {symbol} {source_date}"
            )
        aligned = original.slice(1, returns.num_rows)
        if not _keys_equal(_key_table(aligned), _key_table(returns)):
            raise RuntimeError(
                f"score/return alignment mismatch: {symbol} {source_date}"
            )
        selected = slice(1, 1 + returns.num_rows)
        pieces.append(
            pa.table(
                {
                    **{name: returns[name] for name in KEYS},
                    "ordered_odd": (ordered - reflected_ordered)[selected] / 2.0,
                    "pooled_odd": (pooled - reflected_pooled)[selected] / 2.0,
                    "gap_odd": (gap - reflected_gap)[selected] / 2.0,
                    "ordered_even": (ordered + reflected_ordered)[selected] / 2.0,
                    "pooled_even": (pooled + reflected_pooled)[selected] / 2.0,
                    "gap_even": (gap + reflected_gap)[selected] / 2.0,
                    "gap_innovation_lag2": pa.array(
                        lag2[selected], mask=np.isnan(lag2[selected])
                    ),
                    "gap_innovation_ewma_hl4": pa.array(
                        ewma[selected], mask=np.isnan(ewma[selected])
                    ),
                }
            )
        )
    return pa.concat_tables(pieces, promote_options="default")


def _lgbm_common(train_rows: int, threads: int, n_estimators: int) -> dict[str, Any]:
    params = P11.lgbm_parameters(train_rows, threads, n_estimators)
    params["verbosity"] = -1
    return params


def _fit_nuisance(
    x: np.ndarray, y: np.ndarray, days: np.ndarray, threads: int
) -> lgb.LGBMRegressor:
    if len(y) == 0 or len(np.unique(days)) == 0:
        raise RuntimeError("empty nuisance training data")
    model = lgb.LGBMRegressor(**_lgbm_common(len(y), threads, NUISANCE_TREES))
    model.fit(x, y, sample_weight=P11.day_equal_weights(days))
    return model


def _crossfit_nuisance(
    base: dict[str, Any],
    base_dates: tuple[str, ...],
    endpoint_date: str,
    include_p: bool,
    threads: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    days = base["days"]
    x0 = base["x"]
    x = np.column_stack([x0, base["p"]]) if include_p else x0
    y = base["o"] if include_p else base["p"]
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    metrics: list[dict[str, Any]] = []
    for held_date in base_dates:
        train_mask = np.isin(days, base_dates) & (days != held_date)
        held_mask = days == held_date
        if not train_mask.any() or not held_mask.any():
            raise RuntimeError(f"invalid date-LOO split: {held_date}")
        model = _fit_nuisance(x[train_mask], y[train_mask], days[train_mask], threads)
        prediction[held_mask] = model.booster_.predict(x[held_mask])
        metrics.append(
            _nuisance_metric(y[held_mask], prediction[held_mask], held_date, "date_loo")
        )
    endpoint_mask = days == endpoint_date
    train_mask = np.isin(days, base_dates)
    if not endpoint_mask.any():
        raise RuntimeError(f"missing nuisance endpoint: {endpoint_date}")
    model = _fit_nuisance(x[train_mask], y[train_mask], days[train_mask], threads)
    prediction[endpoint_mask] = model.booster_.predict(x[endpoint_mask])
    metrics.append(
        _nuisance_metric(
            y[endpoint_mask], prediction[endpoint_mask], endpoint_date, "prior_all"
        )
    )
    required = train_mask | endpoint_mask
    if not np.isfinite(prediction[required]).all():
        raise RuntimeError("nonfinite nuisance prediction")
    return prediction[required], metrics


def _nuisance_metric(
    y: np.ndarray, prediction: np.ndarray, source_date: str, role: str
) -> dict[str, Any]:
    error = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "source_date": source_date,
        "prediction_role": role,
        "rows": len(y),
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "r2_mean": float(1.0 - np.sum(error * error) / denominator)
        if denominator > 0
        else None,
        "residual_mean": float(np.mean(error)),
    }


def build_nuisance(
    symbol: str,
    fold: str,
    stage: str,
    artifacts: Path,
    threads: int,
    replace: bool = False,
) -> dict[str, Any]:
    if (
        symbol not in SYMBOLS
        or fold not in FOLDS
        or stage not in ("selection", "refit")
    ):
        raise ValueError("nuisance unit outside contract")
    spec = FOLDS[fold]
    if stage == "selection":
        base_dates = tuple(spec["train"])
        endpoint = str(spec["validation"])
    else:
        base_dates = tuple(spec["refit"])
        endpoint = str(spec["test"])
    dates = base_dates + (endpoint,)
    output = derived_path(artifacts, symbol, fold, stage)
    paths = nuisance_paths(artifacts, symbol, fold, stage)
    config = {
        "run_id": RUN_ID,
        "symbol": symbol,
        "fold": fold,
        "stage": stage,
        "base_dates": list(base_dates),
        "endpoint_date": endpoint,
        "nuisance_trees": NUISANCE_TREES,
        "threads": threads,
    }
    if not replace and _valid_nuisance(output, paths, config):
        emit("nuisance_skip", symbol=symbol, fold=fold, stage=stage)
        return json.loads(paths["complete"].read_text())
    started = time.monotonic()
    base = _base_arrays(symbol, dates)
    p_prediction, p_metrics = _crossfit_nuisance(
        base, base_dates, endpoint, False, threads
    )
    o_prediction, o_metrics = _crossfit_nuisance(
        base, base_dates, endpoint, True, threads
    )
    p_perp = base["p"] - p_prediction
    o_perp = base["o"] - o_prediction
    # _crossfit_nuisance returns rows in the same concatenated base-date + endpoint order.
    required = np.isin(base["days"], dates)
    keys = _key_table(base["table"].filter(pa.array(required)))
    values = _base_derived_table(artifacts, symbol, dates)
    if not _keys_equal(_key_table(values), keys):
        raise RuntimeError("derived/nuisance key mismatch")
    values = values.append_column("p_perp_x", pa.array(p_perp)).append_column(
        "o_perp_xp", pa.array(o_perp)
    )
    if not np.isfinite(p_perp).all() or not np.isfinite(o_perp).all():
        raise RuntimeError("nonfinite nuisance residual")
    metric_rows = []
    for target_name, rows in (("P_given_X", p_metrics), ("O_given_XP", o_metrics)):
        metric_rows.extend(
            {
                "run_id": RUN_ID,
                "symbol": symbol,
                "fold": fold,
                "stage": stage,
                "nuisance_target": target_name,
                **row,
            }
            for row in rows
        )
    atomic_parquet(output, values)
    atomic_parquet(paths["metric"], pa.Table.from_pylist(metric_rows))
    complete = {
        "status": "complete",
        "config": config,
        "rows": values.num_rows,
        "metric_rows": len(metric_rows),
        "completed_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(paths["complete"], complete)
    emit(
        "nuisance_complete", symbol=symbol, fold=fold, stage=stage, rows=values.num_rows
    )
    return complete


def _valid_nuisance(
    output: Path, paths: dict[str, Path], config: dict[str, Any]
) -> bool:
    try:
        if (
            not output.is_file()
            or not paths["metric"].is_file()
            or not paths["complete"].is_file()
        ):
            return False
        complete = json.loads(paths["complete"].read_text())
        if complete.get("status") != "complete" or complete.get("config") != config:
            return False
        schema = pq.read_schema(output)
        return (
            all(name in schema.names for name in KEYS + DERIVED_COLUMNS)
            and pq.ParquetFile(output).metadata.num_rows == complete["rows"]
            and pq.ParquetFile(paths["metric"]).metadata.num_rows
            == complete["metric_rows"]
        )
    except Exception:
        return False


def _spec_columns(target: str, spec: str) -> tuple[str, ...]:
    choices = SIGNED_SPECS if target == "signed" else ROLE_SPECS
    if spec not in choices:
        raise ValueError(f"invalid {target} spec: {spec}")
    return choices[spec]


def _load_model_date(
    artifacts: Path,
    symbol: str,
    fold: str,
    stage: str,
    source_date: str,
    horizon: int,
    target_kind: str,
    spec: str,
) -> dict[str, np.ndarray]:
    raw = _return_table(symbol, source_date)
    base_names = P11.feature_columns(raw)["R1_baseline"]
    extra_names = _spec_columns(target_kind, spec)
    derived = pq.read_table(derived_path(artifacts, symbol, fold, stage))
    derived = derived.filter(pc.equal(derived["source_date"], source_date))
    if not _keys_equal(_key_table(raw), _key_table(derived)):
        raise RuntimeError(
            f"derived/base key mismatch: {symbol} {fold} {stage} {source_date}"
        )
    missing = [name for name in extra_names if name not in derived.column_names]
    if missing:
        raise RuntimeError(f"missing derived features: {missing}")
    target_array = raw[f"target_return_bps_h{horizon}"].combine_chunks()
    valid = np.asarray(pc.invert(pc.is_null(target_array)), dtype=bool)
    extra_values = []
    for name in extra_names:
        array = derived[name].combine_chunks()
        finite = np.asarray(pc.fill_null(pc.is_finite(array), False), dtype=bool)
        valid &= finite
        extra_values.append(array.to_numpy(zero_copy_only=False))
    y_signed = target_array.to_numpy(zero_copy_only=False)[valid].astype(np.float64)
    if target_kind == "signed":
        y = y_signed
    elif target_kind == "absolute":
        y = np.abs(y_signed)
    else:
        y = (y_signed != 0.0).astype(np.float64)
    columns = [np.asarray(raw[name], dtype=np.float64)[valid] for name in base_names]
    columns.extend(
        np.asarray(values, dtype=np.float64)[valid] for values in extra_values
    )
    x = np.column_stack(columns)
    keys = {name: np.asarray(raw[name])[valid] for name in KEYS}
    events = (
        raw[f"future_event_count_h{horizon}"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)[valid]
        .astype(np.int64)
    )
    seconds = (
        raw[f"future_seconds_h{horizon}"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)[valid]
        .astype(np.float64)
    )
    if (
        not len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(seconds).all()
    ):
        raise RuntimeError("invalid model rows")
    return {
        "x": x,
        "y": y,
        "signed_y": y_signed,
        "events": events,
        "seconds": seconds,
        **keys,
    }


def _concat(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([item[name] for item in items]) for name in items[0]}


def _forecast_metrics(
    y: np.ndarray, prediction: np.ndarray, target: str
) -> dict[str, Any]:
    error = y - prediction
    mse = float(np.mean(error * error))
    result: dict[str, Any] = {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "bias_prediction_minus_realized": float(np.mean(prediction) - np.mean(y)),
    }
    if target == "move":
        clipped = np.clip(prediction, 1e-15, 1.0 - 1e-15)
        result.update(
            {
                "brier": mse,
                "log_loss": float(
                    -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
                ),
            }
        )
    else:
        denominator = float(np.sum(y * y))
        result["oos_r2_zero"] = (
            float(1.0 - np.sum(error * error) / denominator)
            if denominator > 0
            else None
        )
    return result


def _unit_config(
    symbol: str, fold: str, horizon: int, target: str, spec: str, threads: int
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "symbol": symbol,
        "fold": fold,
        "horizon": horizon,
        "target": target,
        "spec": spec,
        "threads": threads,
        "extra_features": list(_spec_columns(target, spec)),
        "selection_metric": "brier" if target == "move" else "l2",
    }


def _valid_model(paths: dict[str, Path], config: dict[str, Any]) -> bool:
    try:
        if not all(path.is_file() for path in paths.values()):
            return False
        complete = json.loads(paths["complete"].read_text())
        if (
            complete.get("status") != "complete"
            or complete.get("config") != config
            or paths["model"].stat().st_size == 0
        ):
            return False
        prediction = pq.read_table(paths["prediction"])
        if (
            prediction.num_rows != complete["prediction_rows"]
            or pq.ParquetFile(paths["metric"]).metadata.num_rows != 1
        ):
            return False
        metric = pq.read_table(paths["metric"]).to_pylist()[0]
        if any(
            metric.get(name) != config.get(name)
            for name in ("run_id", "symbol", "fold", "horizon", "target", "spec")
        ):
            return False
        indices = np.asarray(prediction["path_end_block_index"], dtype=np.int64)
        return (
            len(np.unique(indices)) == prediction.num_rows
            and pc.all(pc.is_finite(prediction["prediction"])).as_py()
            and pc.all(pc.is_finite(prediction["target"])).as_py()
        )
    except Exception:
        return False


def fit_unit(
    symbol: str,
    fold: str,
    horizon: int,
    target: str,
    spec: str,
    artifacts: Path,
    threads: int,
    replace: bool = False,
) -> dict[str, Any]:
    if (
        symbol not in SYMBOLS
        or fold not in FOLDS
        or horizon not in HORIZONS
        or target not in ("signed", "absolute", "move")
    ):
        raise ValueError("model unit outside contract")
    _spec_columns(target, spec)
    paths = model_paths(artifacts, target, symbol, fold, horizon, spec)
    config = _unit_config(symbol, fold, horizon, target, spec, threads)
    if not replace and _valid_model(paths, config):
        emit(
            "fit_unit_skip",
            symbol=symbol,
            fold=fold,
            horizon=horizon,
            target=target,
            spec=spec,
        )
        return json.loads(paths["complete"].read_text())
    if replace:
        for path in paths.values():
            path.unlink(missing_ok=True)
    elif paths["complete"].exists():
        raise RuntimeError(f"stale completed unit: {paths['complete']}")
    else:
        for name in ("model", "metric", "prediction"):
            paths[name].unlink(missing_ok=True)
    started = time.monotonic()
    fold_spec = FOLDS[fold]
    train_dates = tuple(fold_spec["train"])
    validation_date = str(fold_spec["validation"])
    refit_dates = tuple(fold_spec["refit"])
    test_date = str(fold_spec["test"])
    selection = {
        day: _load_model_date(
            artifacts, symbol, fold, "selection", day, horizon, target, spec
        )
        for day in train_dates + (validation_date,)
    }
    refit_stage = {
        day: _load_model_date(
            artifacts, symbol, fold, "refit", day, horizon, target, spec
        )
        for day in refit_dates + (test_date,)
    }
    train = _concat([selection[day] for day in train_dates])
    validation = selection[validation_date]
    refit = _concat([refit_stage[day] for day in refit_dates])
    test = refit_stage[test_date]
    objective = "binary" if target == "move" else "regression_l2"
    params = _lgbm_common(len(train["y"]), threads, 2000)
    params.update({"objective": objective})
    if target == "move":
        # For a binary probability prediction LightGBM's native L2 metric is
        # exactly the Brier score, without a Python callback on every tree.
        params["metric"] = "l2"
    estimator = lgb.LGBMRegressor(**params)
    estimator.fit(
        train["x"],
        train["y"],
        sample_weight=P11.day_equal_weights(train["source_date"]),
        eval_set=[(validation["x"], validation["y"])],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(estimator.best_iteration_ or 2000)
    validation_prediction = np.asarray(
        estimator.booster_.predict(validation["x"], num_iteration=best_iteration),
        dtype=np.float64,
    )
    refit_params = _lgbm_common(len(train["y"]), threads, best_iteration)
    refit_params.update({"objective": objective})
    refit_estimator = lgb.LGBMRegressor(**refit_params)
    # Match Phase 11: keep min_data_in_leaf selected from the early-stop training rows.
    refit_estimator.set_params(min_child_samples=params["min_child_samples"])
    refit_estimator.fit(
        refit["x"],
        refit["y"],
        sample_weight=P11.day_equal_weights(refit["source_date"]),
    )
    prediction = np.asarray(
        refit_estimator.booster_.predict(test["x"]), dtype=np.float64
    )
    if target == "move" and (np.any(prediction < 0.0) or np.any(prediction > 1.0)):
        raise RuntimeError("binary prediction outside [0,1]")
    metrics = _forecast_metrics(test["y"], prediction, target)
    validation_metrics = _forecast_metrics(
        validation["y"], validation_prediction, target
    )
    metric_row = {
        **config,
        "test_date": test_date,
        "train_dates": ",".join(train_dates),
        "validation_date": validation_date,
        "refit_dates": ",".join(refit_dates),
        "feature_count": train["x"].shape[1],
        "train_rows": len(train["y"]),
        "validation_rows": len(validation["y"]),
        "refit_rows": len(refit["y"]),
        "test_rows": len(test["y"]),
        "best_iteration": best_iteration,
        "validation_primary_loss": validation_metrics[
            "brier" if target == "move" else "mse"
        ],
        **metrics,
        "elapsed_seconds": time.monotonic() - started,
        "host": os.uname().nodename,
    }
    prediction_table = pa.table(
        {
            **{name: test[name] for name in KEYS},
            "horizon": np.repeat(horizon, len(prediction)),
            "target_kind": np.repeat(target, len(prediction)),
            "spec": np.repeat(spec, len(prediction)),
            "target": test["y"],
            "signed_return_bps": test["signed_y"],
            "prediction": prediction,
            "future_event_count": test["events"],
            "future_seconds": test["seconds"],
        }
    )
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model_temp = paths["model"].with_name(
        f".{paths['model'].name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    refit_estimator.booster_.save_model(model_temp)
    atomic_parquet(paths["prediction"], prediction_table)
    atomic_parquet(paths["metric"], pa.Table.from_pylist([metric_row]))
    os.replace(model_temp, paths["model"])
    complete = {
        "status": "complete",
        "config": config,
        "prediction_rows": len(prediction),
        "best_iteration": best_iteration,
        "completed_utc": utc_now(),
    }
    atomic_json(paths["complete"], complete)
    emit(
        "fit_unit_complete",
        symbol=symbol,
        fold=fold,
        horizon=horizon,
        target=target,
        spec=spec,
        rows=len(prediction),
    )
    return complete


def _matched_baseline(
    artifacts: Path,
    symbol: str,
    fold: str,
    horizon: int,
    target: str,
    candidate: pa.Table,
) -> tuple[float, int]:
    if target == "signed":
        path = P11.unit_paths(P11.ARTIFACTS, symbol, fold, horizon, "R1_baseline")[
            "prediction"
        ]
        baseline = pq.read_table(
            path, columns=list(KEYS) + ["target_return_bps", "prediction"]
        )
        baseline_target = "target_return_bps"
    else:
        path = model_paths(artifacts, target, symbol, fold, horizon, "baseline")[
            "prediction"
        ]
        baseline = pq.read_table(path, columns=list(KEYS) + ["target", "prediction"])
        baseline_target = "target"
    base_index = np.asarray(baseline["path_end_block_index"], dtype=np.int64)
    candidate_index = np.asarray(candidate["path_end_block_index"], dtype=np.int64)
    positions = np.searchsorted(base_index, candidate_index)
    if np.any(positions >= len(base_index)) or not np.array_equal(
        base_index[positions], candidate_index
    ):
        raise RuntimeError("candidate keys absent from baseline")
    for name in KEYS[:-1]:
        if not np.array_equal(
            np.asarray(baseline[name])[positions], np.asarray(candidate[name])
        ):
            raise RuntimeError(f"matched baseline key mismatch: {name}")
    target_values = np.asarray(candidate["target"], dtype=np.float64)
    if not np.allclose(
        np.asarray(baseline[baseline_target], dtype=np.float64)[positions],
        target_values,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("matched baseline target mismatch")
    prediction = np.asarray(baseline["prediction"], dtype=np.float64)[positions]
    return float(np.mean((target_values - prediction) ** 2)), len(positions)


def aggregate(artifacts: Path) -> None:
    nuisance_tables: list[pa.Table] = []
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for stage in ("selection", "refit"):
                paths = nuisance_paths(artifacts, symbol, fold, stage)
                if not paths["metric"].is_file() or not paths["complete"].is_file():
                    raise RuntimeError(
                        f"missing nuisance unit: {symbol} {fold} {stage}"
                    )
                nuisance_tables.append(pq.read_table(paths["metric"]))
    fold_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    expected = 0
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for horizon in HORIZONS:
                for target, specs in (
                    ("signed", SIGNED_SPECS),
                    ("absolute", ROLE_SPECS),
                    ("move", ROLE_SPECS),
                ):
                    for spec in specs:
                        expected += 1
                        paths = model_paths(
                            artifacts, target, symbol, fold, horizon, spec
                        )
                        if not paths["complete"].is_file():
                            raise RuntimeError(
                                f"missing model completion: {target} {symbol} {fold} h{horizon} {spec}"
                            )
                        complete = json.loads(paths["complete"].read_text())
                        if not _valid_model(paths, complete.get("config", {})):
                            raise RuntimeError(
                                f"invalid model unit: {target} {symbol} {fold} h{horizon} {spec}"
                            )
                        metric = pq.read_table(paths["metric"]).to_pylist()[0]
                        prediction = pq.read_table(paths["prediction"])
                        primary_name = "brier" if target == "move" else "mse"
                        primary_loss = float(metric[primary_name])
                        baseline_loss, matched_rows = _matched_baseline(
                            artifacts, symbol, fold, horizon, target, prediction
                        )
                        relative = (
                            (baseline_loss - primary_loss) / baseline_loss
                            if baseline_loss > 0
                            else None
                        )
                        row = {
                            "run_id": RUN_ID,
                            "symbol": symbol,
                            "fold": fold,
                            "test_date": metric["test_date"],
                            "horizon": horizon,
                            "target": target,
                            "spec": spec,
                            "feature_count": metric["feature_count"],
                            "rows": prediction.num_rows,
                            "best_iteration": metric["best_iteration"],
                            "primary_metric": primary_name,
                            "primary_loss": primary_loss,
                            "matched_baseline_loss": baseline_loss,
                            "relative_loss_reduction_pct": 100.0 * relative
                            if relative is not None
                            else None,
                            "mse": metric.get("mse"),
                            "mae": metric.get("mae"),
                            "brier": metric.get("brier"),
                            "log_loss": metric.get("log_loss"),
                            "oos_r2_zero": metric.get("oos_r2_zero"),
                        }
                        fold_rows.append(row)
                        support_rows.append(
                            {
                                "run_id": RUN_ID,
                                "symbol": symbol,
                                "fold": fold,
                                "horizon": horizon,
                                "target": target,
                                "spec": spec,
                                "candidate_rows": prediction.num_rows,
                                "matched_baseline_rows": matched_rows,
                                "key_mismatches": 0,
                                "nonfinite_target": 0,
                                "nonfinite_prediction": 0,
                                "baseline_source": "phase11_frozen_R1"
                                if target == "signed"
                                else "phase12_fitted_baseline",
                                "comparison_support": "candidate_test_intersection",
                            }
                        )
    if expected != 864 or len(fold_rows) != 864:
        raise RuntimeError(f"model aggregate cardinality mismatch: {len(fold_rows)}")
    compact_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            for target, specs in (
                ("signed", SIGNED_SPECS),
                ("absolute", ROLE_SPECS),
                ("move", ROLE_SPECS),
            ):
                for spec in specs:
                    selected = [
                        row
                        for row in fold_rows
                        if row["symbol"] == symbol
                        and row["horizon"] == horizon
                        and row["target"] == target
                        and row["spec"] == spec
                    ]
                    if len(selected) != 3:
                        raise RuntimeError("compact fold cardinality mismatch")
                    loss = float(np.mean([row["primary_loss"] for row in selected]))
                    baseline = float(
                        np.mean([row["matched_baseline_loss"] for row in selected])
                    )
                    compact_rows.append(
                        {
                            "run_id": RUN_ID,
                            "symbol": symbol,
                            "horizon": horizon,
                            "target": target,
                            "spec": spec,
                            "test_dates": 3,
                            "primary_metric": selected[0]["primary_metric"],
                            "date_equal_primary_loss": loss,
                            "date_equal_matched_baseline_loss": baseline,
                            "relative_loss_reduction_pct": 100.0
                            * (baseline - loss)
                            / baseline
                            if baseline > 0
                            else None,
                            "improving_dates": sum(
                                row["primary_loss"] < row["matched_baseline_loss"]
                                for row in selected
                            ),
                        }
                    )
    # The ordered residual estimand is nested: D2 adds O_perp after D1 has
    # already admitted P_perp.  Keep this comparison explicit instead of
    # inferring it from two separate R1-relative rows.
    ordered_increment_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            daily_rows: list[dict[str, Any]] = []
            for fold in FOLDS:
                d1 = [
                    row
                    for row in fold_rows
                    if row["symbol"] == symbol
                    and row["fold"] == fold
                    and row["horizon"] == horizon
                    and row["target"] == "signed"
                    and row["spec"] == "D1_p_perp"
                ]
                d2 = [
                    row
                    for row in fold_rows
                    if row["symbol"] == symbol
                    and row["fold"] == fold
                    and row["horizon"] == horizon
                    and row["target"] == "signed"
                    and row["spec"] == "D2_p_o_perp"
                ]
                if len(d1) != 1 or len(d2) != 1 or d1[0]["rows"] != d2[0]["rows"]:
                    raise RuntimeError("D2-D1 matched-support cardinality mismatch")
                reference, candidate = (
                    float(d1[0]["primary_loss"]),
                    float(d2[0]["primary_loss"]),
                )
                daily = {
                    "run_id": RUN_ID,
                    "aggregation": "test_date",
                    "symbol": symbol,
                    "fold": fold,
                    "test_date": d2[0]["test_date"],
                    "horizon": horizon,
                    "target": "signed",
                    "spec": "D2_vs_D1",
                    "primary_metric": "mse",
                    "candidate_loss": candidate,
                    "reference_loss": reference,
                    "relative_loss_reduction_pct": 100.0
                    * (reference - candidate)
                    / reference,
                    "improved": candidate < reference,
                    "improving_dates": None,
                }
                daily_rows.append(daily)
                ordered_increment_rows.append(daily)
            reference = float(np.mean([row["reference_loss"] for row in daily_rows]))
            candidate = float(np.mean([row["candidate_loss"] for row in daily_rows]))
            ordered_increment_rows.append(
                {
                    "run_id": RUN_ID,
                    "aggregation": "date_equal",
                    "symbol": symbol,
                    "fold": None,
                    "test_date": None,
                    "horizon": horizon,
                    "target": "signed",
                    "spec": "D2_vs_D1",
                    "primary_metric": "mse",
                    "candidate_loss": candidate,
                    "reference_loss": reference,
                    "relative_loss_reduction_pct": 100.0
                    * (reference - candidate)
                    / reference,
                    "improved": candidate < reference,
                    "improving_dates": sum(row["improved"] for row in daily_rows),
                }
            )
    promotion_rows: list[dict[str, Any]] = []
    target_specs = (
        ("signed", SIGNED_SPECS),
        ("absolute", ROLE_SPECS),
        ("move", ROLE_SPECS),
    )
    for target, specs in target_specs:
        for spec in specs:
            symbol_results: list[dict[str, Any]] = []
            for symbol in SYMBOLS:
                selected = [
                    row
                    for row in compact_rows
                    if row["target"] == target
                    and row["spec"] == spec
                    and row["symbol"] == symbol
                ]
                if len(selected) != len(HORIZONS):
                    raise RuntimeError("promotion horizon cardinality mismatch")
                passing_horizons = sum(
                    row["relative_loss_reduction_pct"] is not None
                    and row["relative_loss_reduction_pct"] > 0.0
                    and row["improving_dates"] == 3
                    for row in selected
                )
                symbol_results.append(
                    {
                        "run_id": RUN_ID,
                        "target": target,
                        "spec": spec,
                        "symbol": symbol,
                        "passing_horizons": passing_horizons,
                        "required_horizons": 4,
                        "symbol_gate_pass": passing_horizons >= 4,
                    }
                )
            passing_symbols = sum(row["symbol_gate_pass"] for row in symbol_results)
            for row in symbol_results:
                promotion_rows.append(
                    {
                        **row,
                        "passing_symbols": passing_symbols,
                        "required_symbols": 3,
                        "cross_symbol_gate_pass": passing_symbols >= 3,
                    }
                )
    ordered_aggregate = [
        row for row in ordered_increment_rows if row["aggregation"] == "date_equal"
    ]
    ordered_symbols: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        selected = [row for row in ordered_aggregate if row["symbol"] == symbol]
        if len(selected) != len(HORIZONS):
            raise RuntimeError("D2-D1 promotion horizon cardinality mismatch")
        passing_horizons = sum(
            row["relative_loss_reduction_pct"] > 0.0 and row["improving_dates"] == 3
            for row in selected
        )
        ordered_symbols.append(
            {
                "run_id": RUN_ID,
                "target": "signed",
                "spec": "D2_vs_D1",
                "symbol": symbol,
                "passing_horizons": passing_horizons,
                "required_horizons": 4,
                "symbol_gate_pass": passing_horizons >= 4,
            }
        )
    passing_symbols = sum(row["symbol_gate_pass"] for row in ordered_symbols)
    promotion_rows.extend(
        {
            **row,
            "passing_symbols": passing_symbols,
            "required_symbols": 3,
            "cross_symbol_gate_pass": passing_symbols >= 3,
        }
        for row in ordered_symbols
    )
    tables = artifacts / "tables"
    atomic_parquet(
        tables / "nuisance_metrics.parquet",
        pa.concat_tables(nuisance_tables, promote_options="default"),
    )
    atomic_parquet(tables / "fold_metrics.parquet", pa.Table.from_pylist(fold_rows))
    atomic_parquet(
        tables / "compact_summary.parquet", pa.Table.from_pylist(compact_rows)
    )
    atomic_parquet(
        tables / "ordered_residual_increment.parquet",
        pa.Table.from_pylist(ordered_increment_rows),
    )
    atomic_parquet(
        tables / "promotion_gate.parquet", pa.Table.from_pylist(promotion_rows)
    )
    atomic_parquet(
        tables / "support_audits.parquet", pa.Table.from_pylist(support_rows)
    )
    emit(
        "modeling_aggregate_complete",
        model_units=len(fold_rows),
        compact_rows=len(compact_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    nuisance = sub.add_parser("build-nuisance")
    nuisance.add_argument("--symbol", required=True, choices=SYMBOLS)
    nuisance.add_argument("--fold", required=True, choices=tuple(FOLDS))
    nuisance.add_argument("--stage", required=True, choices=("selection", "refit"))
    nuisance.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    nuisance.add_argument("--threads", type=int, default=8)
    nuisance.add_argument("--replace", action="store_true")
    fit = sub.add_parser("fit-unit")
    fit.add_argument("--symbol", required=True, choices=SYMBOLS)
    fit.add_argument("--fold", required=True, choices=tuple(FOLDS))
    fit.add_argument("--horizon", required=True, type=int, choices=HORIZONS)
    fit.add_argument("--target", required=True, choices=("signed", "absolute", "move"))
    fit.add_argument("--spec", required=True)
    fit.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--replace", action="store_true")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = Path(args.artifacts)
    if args.command == "build-nuisance":
        build_nuisance(
            args.symbol, args.fold, args.stage, artifacts, args.threads, args.replace
        )
    elif args.command == "fit-unit":
        fit_unit(
            args.symbol,
            args.fold,
            args.horizon,
            args.target,
            args.spec,
            artifacts,
            args.threads,
            args.replace,
        )
    else:
        aggregate(artifacts)


if __name__ == "__main__":
    main()
