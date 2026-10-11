#!/usr/bin/env python3
"""Conditional role of O_perp under a strong frozen R1 forecast.

All thresholds are estimated from each outer fold's refit dates with equal total
weight per date.  The corresponding test date is only assigned to those frozen
thresholds; it never affects the baseline model, nuisance residual, or bins.
"""

from __future__ import annotations

import argparse
import importlib.util
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
P11_DIR = HERE.parent / "11_lgbm_multihorizon_return"
DEFAULT_ARTIFACTS = HERE / "artifacts"
RUN_ID = "phase12_residual_role_v1"
KEYS = ("symbol", "source_date", "segment_id", "path_end_block_index")
QUINTILES = (0.2, 0.4, 0.6, 0.8)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P11 = _load_module("phase11_for_residual_role", P11_DIR / "codebook.py")
SYMBOLS = tuple(P11.SYMBOLS)
FOLDS = P11.FOLDS
HORIZONS = tuple(P11.HORIZONS)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    if pq.ParquetFile(temporary).metadata.num_rows != table.num_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row validation failed: {path}")
    os.replace(temporary, path)


def unit_root(artifacts: Path, symbol: str, fold: str, horizon: int) -> Path:
    return (
        artifacts
        / "residual_role_units"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"horizon={horizon}"
    )


def derived_path(artifacts: Path, symbol: str, fold: str) -> Path:
    return (
        artifacts
        / "data/derived_features"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / "stage=refit.parquet"
    )


def weighted_thresholds(
    values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]
) -> np.ndarray:
    if (
        len(values) == 0
        or values.shape != weights.shape
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid weighted quantile input")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative /= cumulative[-1]
    return np.asarray(
        [
            ordered[
                min(
                    int(np.searchsorted(cumulative, probability, side="left")),
                    len(ordered) - 1,
                )
            ]
            for probability in probabilities
        ],
        dtype=np.float64,
    )


def active_edges(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # Duplicate thresholds and a threshold at the observed maximum do not create
    # artificial bins. Equal observations are never split.
    return np.asarray(
        [edge for edge in np.unique(thresholds) if edge < np.max(values)],
        dtype=np.float64,
    )


def _date_equal_weights(parts: list[dict[str, np.ndarray]]) -> np.ndarray:
    if not parts or any(len(part["y"]) == 0 for part in parts):
        raise RuntimeError("empty refit date")
    return np.concatenate(
        [
            np.full(
                len(part["y"]), 1.0 / (len(parts) * len(part["y"])), dtype=np.float64
            )
            for part in parts
        ]
    )


def _join_o_perp(derived: pa.Table, rows: dict[str, np.ndarray]) -> np.ndarray:
    source_date = str(rows["source_date"][0])
    selected = derived.filter(pc.equal(derived["source_date"], source_date))
    source_index = np.asarray(selected["path_end_block_index"], dtype=np.int64)
    wanted_index = np.asarray(rows["path_end_block_index"], dtype=np.int64)
    positions = np.searchsorted(source_index, wanted_index)
    if np.any(positions >= len(source_index)) or not np.array_equal(
        source_index[positions], wanted_index
    ):
        raise RuntimeError(f"derived keys do not cover model support: {source_date}")
    for name in KEYS[:-1]:
        if not np.array_equal(
            np.asarray(selected[name])[positions], np.asarray(rows[name])
        ):
            raise RuntimeError(f"derived key mismatch: {name} {source_date}")
    result = np.asarray(selected["o_perp_xp"], dtype=np.float64)[positions]
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite o_perp_xp")
    return result


def _saved_test_prediction(
    symbol: str, fold: str, horizon: int, test: dict[str, np.ndarray]
) -> np.ndarray:
    path = P11.unit_paths(P11.ARTIFACTS, symbol, fold, horizon, "R1_baseline")[
        "prediction"
    ]
    saved = pq.read_table(
        path, columns=list(KEYS) + ["target_return_bps", "prediction"]
    )
    if saved.num_rows != len(test["y"]):
        raise RuntimeError("saved test prediction row mismatch")
    for name in KEYS:
        if not np.array_equal(np.asarray(saved[name]), np.asarray(test[name])):
            raise RuntimeError(f"saved test prediction key mismatch: {name}")
    if not np.allclose(
        np.asarray(saved["target_return_bps"], dtype=np.float64),
        test["y"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("saved test target mismatch")
    prediction = np.asarray(saved["prediction"], dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite saved prediction")
    return prediction


def _config(symbol: str, fold: str, horizon: int) -> dict[str, Any]:
    spec = FOLDS[fold]
    return {
        "run_id": RUN_ID,
        "symbol": symbol,
        "fold": fold,
        "horizon": horizon,
        "refit_dates": list(spec["refit"]),
        "test_date": str(spec["test"]),
        "strong_alpha_probability": 0.8,
        "o_perp_probabilities": list(QUINTILES),
    }


def _valid_unit(root: Path, config: dict[str, Any]) -> bool:
    try:
        complete = json.loads((root / "complete.json").read_text())
        return (
            complete.get("status") == "complete"
            and complete.get("config") == config
            and pq.ParquetFile(root / "cells.parquet").metadata.num_rows
            == complete["cell_rows"]
            and pq.ParquetFile(root / "thresholds.parquet").metadata.num_rows == 5
        )
    except Exception:
        return False


def run_unit(
    symbol: str,
    fold: str,
    horizon: int,
    artifacts: Path,
    threads: int,
    replace: bool = False,
) -> None:
    if symbol not in SYMBOLS or fold not in FOLDS or horizon not in HORIZONS:
        raise ValueError("unit outside contract")
    root = unit_root(artifacts, symbol, fold, horizon)
    config = _config(symbol, fold, horizon)
    if not replace and _valid_unit(root, config):
        print(
            json.dumps({"event": "residual_role_skip", **config}, sort_keys=True),
            flush=True,
        )
        return

    complete_path = P11.unit_paths(P11.ARTIFACTS, symbol, fold, horizon, "R1_baseline")[
        "complete"
    ]
    model_path = P11.unit_paths(P11.ARTIFACTS, symbol, fold, horizon, "R1_baseline")[
        "model"
    ]
    model_complete = json.loads(complete_path.read_text())
    features = model_complete.get("features")
    if not isinstance(features, list) or len(features) != 29:
        raise RuntimeError("invalid frozen R1 feature contract")
    spec = FOLDS[fold]
    refit_dates = tuple(spec["refit"])
    test_date = str(spec["test"])
    derived = pq.read_table(
        derived_path(artifacts, symbol, fold), columns=list(KEYS) + ["o_perp_xp"]
    )
    refit = [
        P11.load_date(P11.ARTIFACTS, symbol, day, horizon, features)
        for day in refit_dates
    ]
    test = P11.load_date(P11.ARTIFACTS, symbol, test_date, horizon, features)
    # LightGBM ignores constructor overrides for a serialized Booster on some
    # versions; its frozen model parameters remain deterministic for inference.
    booster = lgb.Booster(model_file=str(model_path), params={"num_threads": threads})
    refit_prediction_parts = [
        np.asarray(booster.predict(part["x"]), dtype=np.float64) for part in refit
    ]
    if any(not np.isfinite(values).all() for values in refit_prediction_parts):
        raise RuntimeError("nonfinite refit prediction")
    refit_prediction = np.concatenate(refit_prediction_parts)
    refit_o = np.concatenate([_join_o_perp(derived, part) for part in refit])
    weights = _date_equal_weights(refit)
    alpha_q80 = float(weighted_thresholds(np.abs(refit_prediction), weights, (0.8,))[0])
    o_thresholds = weighted_thresholds(refit_o, weights, QUINTILES)
    o_edges = active_edges(refit_o, o_thresholds)

    test_prediction = _saved_test_prediction(symbol, fold, horizon, test)
    test_o = _join_o_perp(derived, test)
    test_bins = np.searchsorted(o_edges, test_o, side="left").astype(np.int16) + 1
    strong = np.abs(test_prediction) >= alpha_q80
    cell_rows: list[dict[str, Any]] = []
    for bin_index in range(1, len(o_edges) + 2):
        mask = strong & (test_bins == bin_index)
        n = int(np.sum(mask))
        residual = test["y"] - test_prediction
        cell_rows.append(
            {
                "run_id": RUN_ID,
                "aggregation": "test_date",
                "symbol": symbol,
                "fold": fold,
                "test_date": test_date,
                "horizon": horizon,
                "residual_feature": "o_perp_xp",
                "residual_bin": bin_index,
                "intended_bins": 5,
                "effective_bins": len(o_edges) + 1,
                "n": n,
                "dates_present": 1 if n else 0,
                "date_coverage": 1.0 if n else 0.0,
                "test_rows": len(test["y"]),
                "strong_alpha_n": int(np.sum(strong)),
                "strong_alpha_share": float(np.mean(strong)),
                "cell_share_of_test": float(n / len(test["y"])),
                "mean_o_perp_xp": float(np.mean(test_o[mask])) if n else None,
                "mean_aligned_residual": float(
                    np.mean(np.sign(test_prediction[mask]) * residual[mask])
                )
                if n
                else None,
                "mean_aligned_return": float(
                    np.mean(np.sign(test_prediction[mask]) * test["y"][mask])
                )
                if n
                else None,
                "mean_absolute_return": float(np.mean(np.abs(test["y"][mask])))
                if n
                else None,
                "mean_signed_residual": float(np.mean(residual[mask])) if n else None,
            }
        )
    threshold_rows = [
        {
            "run_id": RUN_ID,
            "symbol": symbol,
            "fold": fold,
            "test_date": test_date,
            "horizon": horizon,
            "variable": "abs_r1_prediction",
            "probability": 0.8,
            "threshold": alpha_q80,
            "is_active_edge": True,
            "refit_rows": len(refit_prediction),
            "method": "date_equal_weighted_empirical_refit_only",
        }
    ]
    threshold_rows.extend(
        {
            "run_id": RUN_ID,
            "symbol": symbol,
            "fold": fold,
            "test_date": test_date,
            "horizon": horizon,
            "variable": "o_perp_xp",
            "probability": probability,
            "threshold": float(threshold),
            "is_active_edge": bool(np.any(o_edges == threshold)),
            "refit_rows": len(refit_prediction),
            "method": "date_equal_weighted_empirical_refit_only",
        }
        for probability, threshold in zip(QUINTILES, o_thresholds, strict=True)
    )
    atomic_parquet(root / "cells.parquet", pa.Table.from_pylist(cell_rows))
    atomic_parquet(root / "thresholds.parquet", pa.Table.from_pylist(threshold_rows))
    atomic_json(
        root / "complete.json",
        {
            "status": "complete",
            "config": config,
            "cell_rows": len(cell_rows),
            "completed_utc": utc_now(),
        },
    )
    print(
        json.dumps(
            {"event": "residual_role_complete", "rows": len(cell_rows), **config},
            sort_keys=True,
        ),
        flush=True,
    )


def aggregate(artifacts: Path) -> None:
    cells: list[dict[str, Any]] = []
    thresholds: list[pa.Table] = []
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for horizon in HORIZONS:
                root = unit_root(artifacts, symbol, fold, horizon)
                config = _config(symbol, fold, horizon)
                if not _valid_unit(root, config):
                    raise RuntimeError(
                        f"missing or invalid residual-role unit: {symbol} {fold} h{horizon}"
                    )
                cells.extend(pq.read_table(root / "cells.parquet").to_pylist())
                thresholds.append(pq.read_table(root / "thresholds.parquet"))
    date_rows = list(cells)
    for row in date_rows:
        row["dates_present"] = 1 if row["n"] > 0 else 0
        row["date_coverage"] = 1.0 if row["n"] > 0 else 0.0
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            for bin_index in range(1, 6):
                selected = [
                    row
                    for row in date_rows
                    if row["symbol"] == symbol
                    and row["horizon"] == horizon
                    and row["residual_bin"] == bin_index
                ]
                if not selected:
                    continue
                available = [row for row in selected if row["n"] > 0]
                cells.append(
                    {
                        "run_id": RUN_ID,
                        "aggregation": "date_equal",
                        "symbol": symbol,
                        "fold": None,
                        "test_date": None,
                        "horizon": horizon,
                        "residual_feature": "o_perp_xp",
                        "residual_bin": bin_index,
                        "intended_bins": 5,
                        "effective_bins": None,
                        "n": sum(row["n"] for row in selected),
                        "test_rows": sum(row["test_rows"] for row in selected),
                        "strong_alpha_n": sum(
                            row["strong_alpha_n"] for row in selected
                        ),
                        "strong_alpha_share": float(
                            np.mean([row["strong_alpha_share"] for row in selected])
                        ),
                        "cell_share_of_test": float(
                            np.mean([row["cell_share_of_test"] for row in selected])
                        ),
                        "mean_o_perp_xp": float(
                            np.mean([row["mean_o_perp_xp"] for row in available])
                        )
                        if available
                        else None,
                        "mean_aligned_residual": float(
                            np.mean([row["mean_aligned_residual"] for row in available])
                        )
                        if available
                        else None,
                        "mean_aligned_return": float(
                            np.mean([row["mean_aligned_return"] for row in available])
                        )
                        if available
                        else None,
                        "mean_absolute_return": float(
                            np.mean([row["mean_absolute_return"] for row in available])
                        )
                        if available
                        else None,
                        "mean_signed_residual": float(
                            np.mean([row["mean_signed_residual"] for row in available])
                        )
                        if available
                        else None,
                        "dates_present": len(available),
                        "date_coverage": len(available) / len(FOLDS),
                    }
                )
    tables = artifacts / "tables"
    atomic_parquet(
        tables / "residual_conditional_role.parquet", pa.Table.from_pylist(cells)
    )
    atomic_parquet(
        tables / "residual_role_thresholds.parquet",
        pa.concat_tables(thresholds, promote_options="default"),
    )
    print(
        json.dumps(
            {
                "event": "residual_role_aggregate_complete",
                "unit_count": 72,
                "rows": len(cells),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    unit = subparsers.add_parser("run-unit")
    unit.add_argument("--symbol", required=True, choices=SYMBOLS)
    unit.add_argument("--fold", required=True, choices=tuple(FOLDS))
    unit.add_argument("--horizon", required=True, type=int, choices=HORIZONS)
    unit.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    unit.add_argument("--threads", type=int, default=4)
    unit.add_argument("--replace", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = Path(args.artifacts).resolve()
    if args.command == "run-unit":
        run_unit(
            args.symbol, args.fold, args.horizon, artifacts, args.threads, args.replace
        )
    else:
        aggregate(artifacts)


if __name__ == "__main__":
    main()
