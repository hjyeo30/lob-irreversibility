"""Cross-symbol descriptive statistics on published Daytime B=10 blocks.

Read-only. Streams the column subset needed for the manuscript's data section and
writes one per-symbol-date table plus one per-symbol aggregate table. No model is
fit and no score is recomputed here.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

BLOCK_ROOT = pathlib.Path(
    "/data/lob_irreversibility_interpretation_and_execution"
    "/derived/continuous_blocks_v1"
).expanduser()

# NVDA Daytime blocks were published by the upstream project before the
# multi-regime replay, so they carry no regime partition. That store is already
# restricted to 10:00-15:30, which the block timestamps confirm.
NVDA_DAYTIME_ROOT = pathlib.Path(
    "/data/lob_liquidity_time_irreversibility"
    "/derived/continuous_blocks_micro_v1/symbol=NVDA"
).expanduser()

OUTPUT_ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "phases/15_cross_symbol_eda/artifacts/tables"
)
PHASE07_TABLE_ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "phases/07_cross_symbol/artifacts/tables"
)

SYMBOLS = (
    "NVDA",
    "AAPL",
    "COST",
    "TSLA",
    "SAIA",
    "PEP",
    "CROX",
    "IWM",
    "PODD",
    "SPY",
    "QQQ",
    "SOFI",
)

COLUMNS = [
    "source_date",
    "realized_source_event_count",
    "completion_overshoot",
    "elapsed_ns",
    "start_spread_ticks",
    "start_bid_depth_l1",
    "start_ask_depth_l1",
    "start_bid_depth_l5",
    "start_ask_depth_l5",
    "start_midprice_x2_nano",
    "end_midprice_x2_nano",
    "mid_log_return",
    "depletion_plus_appearance_count",
    "cause_add_count",
    "cause_cancel_count",
    "cause_displayed_execution_count",
]

CORRELATION_COVARIATES = (
    "realized_volatility_bps",
    "depth_l1_median",
)
CANONICAL_PHASE07_SEEDS = (20260730, 20260731, 20260732)
CANONICAL_PHASE07_TEST_DATES = ("2026-07-21", "2026-07-22", "2026-07-23")
CANONICAL_PHASE07_PAIRED_SCHEMA = (
    ("symbol", "string", True),
    ("seed", "int64", True),
    ("source_date", "string", True),
    ("score_contract", "string", True),
    ("contrast", "string", True),
    ("nats_path", "double", True),
    ("nats_block", "double", True),
    ("nats_event", "double", True),
    ("nats_second", "double", True),
    ("positive_nats_event", "bool", True),
)
CORRELATION_DETAIL_PARQUET = "cross_symbol_rank_correlation_symbols_v3.parquet"
CORRELATION_DETAIL_CSV = "cross_symbol_rank_correlation_symbols_v3.csv"
CORRELATION_SUMMARY_PARQUET = "cross_symbol_rank_correlations_v3.parquet"
CORRELATION_SUMMARY_CSV = "cross_symbol_rank_correlations_v3.csv"
CORRELATION_TABLE_FRAMES = (
    CORRELATION_DETAIL_PARQUET,
    CORRELATION_DETAIL_CSV,
    CORRELATION_SUMMARY_PARQUET,
    CORRELATION_SUMMARY_CSV,
)
CORRELATION_COMPLETE = "cross_symbol_rank_correlations_v3.complete.json"


def _atomic_publish_text(text: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_publish_frame(frame: pd.DataFrame, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        if destination.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif destination.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"unsupported table suffix: {destination.suffix}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_schema(path: pathlib.Path) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in pq.read_schema(path)
    ]


def _parquet_inventory(path: pathlib.Path) -> dict[str, object]:
    metadata = pq.read_metadata(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "schema": _parquet_schema(path),
    }


def _json_inventory(path: pathlib.Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "status": payload.get("status"),
        "rows": payload.get("rows"),
    }


def _output_inventory(path: pathlib.Path) -> dict[str, object]:
    if path.suffix == ".parquet":
        inventory = _parquet_inventory(path)
        inventory["path"] = path.name
        inventory["format"] = "parquet"
        return inventory
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        return {
            "path": path.name,
            "format": "csv",
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "columns": list(frame.columns),
            "schema": [
                {"name": name, "type": str(dtype)}
                for name, dtype in frame.dtypes.items()
            ],
        }
    raise ValueError(f"unsupported output inventory suffix: {path.suffix}")


def _validated_correlation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    phase07_complete_path = PHASE07_TABLE_ROOT / "complete.json"
    phase07_contrast_path = PHASE07_TABLE_ROOT / "paired_contrasts.parquet"
    descriptives_path = OUTPUT_ROOT / "symbol_descriptives.parquet"
    for path in (phase07_complete_path, phase07_contrast_path, descriptives_path):
        if not path.is_file():
            raise FileNotFoundError(f"required published input missing: {path}")

    phase07_complete = json.loads(phase07_complete_path.read_text(encoding="utf-8"))
    if phase07_complete.get("status") != "complete":
        raise RuntimeError("Phase 07 compact publication is not complete")
    expected_phase07_rows = phase07_complete.get("rows", {}).get(
        "paired_contrasts.parquet"
    )
    if expected_phase07_rows != 216:
        raise RuntimeError(
            f"unexpected Phase 07 paired contrast completion rows: {expected_phase07_rows}"
        )

    actual_phase07_schema = tuple(
        (field.name, str(field.type), field.nullable)
        for field in pq.read_schema(phase07_contrast_path)
    )
    if actual_phase07_schema != CANONICAL_PHASE07_PAIRED_SCHEMA:
        raise RuntimeError(
            "Phase 07 paired contrast Arrow schema is not exact canonical schema; "
            f"expected={CANONICAL_PHASE07_PAIRED_SCHEMA}, actual={actual_phase07_schema}"
        )

    contrasts = pd.read_parquet(
        phase07_contrast_path,
        columns=[
            "symbol",
            "seed",
            "source_date",
            "score_contract",
            "contrast",
            "nats_event",
        ],
    )
    selected = contrasts.loc[
        (contrasts["score_contract"] == "bounded_log99")
        & (contrasts["contrast"] == "ordered_tcn_minus_pooled_mlp")
    ].copy()
    expected_symbols = set(SYMBOLS)
    if set(selected["symbol"]) != expected_symbols:
        raise RuntimeError("Phase 07 bounded contrast symbol universe is not exact")
    if (
        len(selected) != 108
        or selected.duplicated(["symbol", "seed", "source_date"]).any()
    ):
        raise RuntimeError("Phase 07 bounded contrast keys are not exact 12x3x3")
    expected_support = {
        (symbol, seed, source_date)
        for symbol in SYMBOLS
        for seed in CANONICAL_PHASE07_SEEDS
        for source_date in CANONICAL_PHASE07_TEST_DATES
    }
    actual_support = {
        (row.symbol, row.seed, row.source_date)
        for row in selected[["symbol", "seed", "source_date"]].itertuples(index=False)
    }
    if actual_support != expected_support:
        missing = sorted(expected_support - actual_support)
        unexpected = sorted(actual_support - expected_support)
        raise RuntimeError(
            "Phase 07 bounded contrast is not the exact canonical symbol x seed x date "
            f"Cartesian support; missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    counts = selected.groupby("symbol").agg(
        seeds=("seed", "nunique"),
        dates=("source_date", "nunique"),
        cells=("nats_event", "size"),
    )
    if not (
        (counts["seeds"] == 3) & (counts["dates"] == 3) & (counts["cells"] == 9)
    ).all():
        raise RuntimeError(
            "Phase 07 bounded contrast support is not 3 seeds x 3 dates per symbol"
        )
    if not np.isfinite(selected["nats_event"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Phase 07 bounded nats/event contains nonfinite values")

    descriptives = pd.read_parquet(
        descriptives_path,
        columns=["symbol", "days", *CORRELATION_COVARIATES],
    )
    if len(descriptives) != 12 or descriptives["symbol"].duplicated().any():
        raise RuntimeError("Phase 15 descriptives do not have one row per exact symbol")
    if set(descriptives["symbol"]) != expected_symbols:
        raise RuntimeError("Phase 15 descriptive symbol universe is not exact")
    if not (descriptives["days"] == 15).all():
        raise RuntimeError("Phase 15 descriptives do not have exact 15-day support")
    if not np.isfinite(
        descriptives.loc[:, CORRELATION_COVARIATES].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Phase 15 correlation covariates contain nonfinite values")
    return selected, descriptives


def calculate_cross_symbol_correlations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the owned n=12 descriptive Spearman tables.

    The temporal gap is averaged equally over the three held-out dates within
    seed, then equally over the three seeds. Ties use average ranks. Missing or
    nonfinite values fail closed; complete-case deletion is not allowed because
    the approved observational universe is exactly 12 symbols.
    """

    contrasts, descriptives = _validated_correlation_inputs()
    seed_means = (
        contrasts.groupby(["symbol", "seed"], as_index=False, sort=True)["nats_event"]
        .mean()
        .rename(columns={"nats_event": "temporal_gap_seed_date_equal_nats_event"})
    )
    symbol_gaps = (
        seed_means.groupby("symbol", as_index=False, sort=True)[
            "temporal_gap_seed_date_equal_nats_event"
        ]
        .mean()
        .rename(
            columns={
                "temporal_gap_seed_date_equal_nats_event": (
                    "temporal_gap_date_equal_seed_mean_bounded_nats_event"
                )
            }
        )
    )
    joined = symbol_gaps.merge(
        descriptives,
        on="symbol",
        how="inner",
        validate="one_to_one",
    ).sort_values("symbol", kind="stable")
    if len(joined) != 12:
        raise RuntimeError("exact symbol join did not preserve all 12 symbols")

    gap_column = "temporal_gap_date_equal_seed_mean_bounded_nats_event"
    gap_rank = joined[gap_column].rank(method="average", ascending=True)
    symbol_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for covariate in CORRELATION_COVARIATES:
        covariate_rank = joined[covariate].rank(method="average", ascending=True)
        rho = float(np.corrcoef(gap_rank.to_numpy(), covariate_rank.to_numpy())[0, 1])
        if not np.isfinite(rho):
            raise RuntimeError(f"nonfinite Spearman rho for {covariate}")
        rows = pd.DataFrame(
            {
                "symbol": joined["symbol"].astype("string"),
                "covariate": pd.Series([covariate] * len(joined), dtype="string"),
                "temporal_gap_bounded_nats_event": joined[gap_column].astype("float64"),
                "covariate_value": joined[covariate].astype("float64"),
                "temporal_gap_rank_average": gap_rank.astype("float64"),
                "covariate_rank_average": covariate_rank.astype("float64"),
                "descriptive_days": joined["days"].astype("int16"),
            }
        )
        symbol_rows.append(rows)
        summary_rows.append(
            {
                "covariate": covariate,
                "spearman_rho": rho,
                "n_symbols": 12,
                "sample_selection": "non_random_selected_cross_section",
                "temporal_gap_contract": "bounded_log99_nats_event_date_equal_then_3_seed_mean",
                "rank_tie_rule": "average",
                "missing_rule": "fail_closed_exact_12_no_complete_case_deletion",
                "inference_scope": "descriptive_only_no_p_value_regression_or_ci",
            }
        )

    details = pd.concat(symbol_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    if len(details) != 24 or len(summary) != 2:
        raise RuntimeError("unexpected correlation output row count")

    return details, summary


def _correlation_manifest(
    output_directory: pathlib.Path,
    *,
    published_at_utc: str,
) -> dict[str, object]:
    phase07_complete_path = PHASE07_TABLE_ROOT / "complete.json"
    phase07_contrast_path = PHASE07_TABLE_ROOT / "paired_contrasts.parquet"
    descriptives_path = OUTPUT_ROOT / "symbol_descriptives.parquet"
    return {
        "status": "complete",
        "published_at_utc": published_at_utc,
        "calculation_contract": "phase15_cross_symbol_correlation_v3",
        "observational_unit": "symbol",
        "sample": {
            "symbols": sorted(SYMBOLS),
            "n_symbols": 12,
            "selection": "non_random_selected_cross_section",
            "descriptive_days_per_symbol": 15,
            "phase07_test_dates_per_symbol": 3,
            "phase07_model_seeds_per_symbol": 3,
            "phase07_test_dates": list(CANONICAL_PHASE07_TEST_DATES),
            "phase07_model_seeds": list(CANONICAL_PHASE07_SEEDS),
        },
        "inputs": {
            "phase07_paired_contrasts": {
                **_parquet_inventory(phase07_contrast_path),
                "rows_selected": 108,
                "score_contract": "bounded_log99",
                "contrast": "ordered_tcn_minus_pooled_mlp",
            },
            "phase07_completion": _json_inventory(phase07_complete_path),
            "phase15_symbol_descriptives": _parquet_inventory(descriptives_path),
        },
        "method": {
            "temporal_gap_aggregation": "equal mean over 3 test dates within seed, then equal mean over 3 seeds",
            "correlation": "Spearman rank correlation as Pearson correlation of ranks",
            "tie_rule": "average ranks",
            "missing_rule": "fail closed unless all exact 12 symbols have finite joined values",
            "prohibited_outputs": ["p_value", "regression", "confidence_interval"],
            "covariates": list(CORRELATION_COVARIATES),
            "temporal_gap_support": {
                "symbols": 12,
                "test_dates_per_symbol_seed": 3,
                "model_seeds_per_symbol": 3,
                "selected_cells": 108,
                "exact_test_dates": list(CANONICAL_PHASE07_TEST_DATES),
                "exact_model_seeds": list(CANONICAL_PHASE07_SEEDS),
                "support_shape": "exact Cartesian product for every symbol",
                "identity_comparison": "raw typed values without coercion",
                "paired_table_arrow_schema": [
                    {"name": name, "type": type_name, "nullable": nullable}
                    for name, type_name, nullable in CANONICAL_PHASE07_PAIRED_SCHEMA
                ],
            },
        },
        "outputs": {
            name: _output_inventory(output_directory / name)
            for name in CORRELATION_TABLE_FRAMES
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "code": str(pathlib.Path(__file__).resolve()),
            "code_version": "phase15_cross_symbol_correlation_v3",
        },
        "validations": [
            "phase07_completion_exact_216_rows",
            "bounded_contrast_exact_12_symbols_x_3_seeds_x_3_dates",
            "canonical_seed_identities_exact_20260730_20260731_20260732",
            "canonical_test_date_identities_exact_2026_07_21_22_23",
            "identical_exact_cartesian_support_for_every_symbol",
            "phase07_paired_arrow_schema_exact_before_identity_comparison",
            "raw_seed_and_source_date_identity_no_lossy_coercion",
            "phase15_descriptives_exact_12_symbols_x_15_days",
            "exact_one_to_one_symbol_join",
            "all_values_finite",
            "output_rows_exact_24_detail_and_2_summary",
        ],
    }


def _assert_correlation_output_semantics(
    output_directory: pathlib.Path,
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    parquet_detail = pd.read_parquet(output_directory / CORRELATION_DETAIL_PARQUET)
    parquet_summary = pd.read_parquet(output_directory / CORRELATION_SUMMARY_PARQUET)
    csv_detail = pd.read_csv(output_directory / CORRELATION_DETAIL_CSV)
    csv_summary = pd.read_csv(output_directory / CORRELATION_SUMMARY_CSV)
    pd.testing.assert_frame_equal(
        details.reset_index(drop=True),
        parquet_detail.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        summary.reset_index(drop=True),
        parquet_summary.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        details.reset_index(drop=True),
        csv_detail.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-14,
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(
        summary.reset_index(drop=True),
        csv_summary.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-14,
        atol=1e-14,
    )


def _manifest_semantic_core(manifest: dict[str, object]) -> dict[str, object]:
    runtime = manifest.get("runtime")
    return {
        "status": manifest.get("status"),
        "calculation_contract": manifest.get("calculation_contract"),
        "observational_unit": manifest.get("observational_unit"),
        "sample": manifest.get("sample"),
        "inputs": manifest.get("inputs"),
        "method": manifest.get("method"),
        "outputs": manifest.get("outputs"),
        "code_version": runtime.get("code_version")
        if isinstance(runtime, dict)
        else None,
        "validations": manifest.get("validations"),
    }


def _verify_existing_correlation_publication(
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    try:
        _assert_correlation_output_semantics(OUTPUT_ROOT, details, summary)
    except (AssertionError, ValueError) as exc:
        raise RuntimeError(
            "existing correlation tables do not match current exact recalculation"
        ) from exc
    complete_path = OUTPUT_ROOT / CORRELATION_COMPLETE
    existing = json.loads(complete_path.read_text(encoding="utf-8"))
    expected = _correlation_manifest(
        OUTPUT_ROOT,
        published_at_utc=str(existing.get("published_at_utc")),
    )
    if _manifest_semantic_core(existing) != _manifest_semantic_core(expected):
        raise RuntimeError(
            "existing correlation manifest does not match current upstream and output inventory"
        )


def _first_publish_correlation_tables(
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_root = pathlib.Path(
        tempfile.mkdtemp(dir=OUTPUT_ROOT, prefix=".correlation-publication-")
    )
    try:
        details.to_parquet(
            temporary_root / CORRELATION_DETAIL_PARQUET,
            index=False,
        )
        details.to_csv(
            temporary_root / CORRELATION_DETAIL_CSV,
            index=False,
        )
        summary.to_parquet(
            temporary_root / CORRELATION_SUMMARY_PARQUET,
            index=False,
        )
        summary.to_csv(
            temporary_root / CORRELATION_SUMMARY_CSV,
            index=False,
        )
        _assert_correlation_output_semantics(temporary_root, details, summary)
        manifest = _correlation_manifest(
            temporary_root,
            published_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        (temporary_root / CORRELATION_COMPLETE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in (*CORRELATION_TABLE_FRAMES, CORRELATION_COMPLETE):
            descriptor = os.open(temporary_root / name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        destinations = [
            *(OUTPUT_ROOT / name for name in CORRELATION_TABLE_FRAMES),
            OUTPUT_ROOT / CORRELATION_COMPLETE,
        ]
        if any(path.exists() for path in destinations):
            raise RuntimeError(
                "correlation publication appeared during first-publish staging"
            )
        for name in CORRELATION_TABLE_FRAMES:
            os.link(temporary_root / name, OUTPUT_ROOT / name)
        os.link(
            temporary_root / CORRELATION_COMPLETE,
            OUTPUT_ROOT / CORRELATION_COMPLETE,
        )
        directory_descriptor = os.open(OUTPUT_ROOT, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def publish_cross_symbol_correlations() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """First-publish, or verify and skip, without overwriting any prior artifact."""

    details, summary = calculate_cross_symbol_correlations()
    destinations = [
        *(OUTPUT_ROOT / name for name in CORRELATION_TABLE_FRAMES),
        OUTPUT_ROOT / CORRELATION_COMPLETE,
    ]
    existing_count = sum(path.exists() for path in destinations)
    if existing_count == len(destinations):
        _verify_existing_correlation_publication(details, summary)
        return details, summary, "verified_skip"
    if existing_count:
        raise RuntimeError(
            f"partial correlation publication exists ({existing_count}/{len(destinations)}); refusing overwrite"
        )
    _first_publish_correlation_tables(details, summary)
    _verify_existing_correlation_publication(details, summary)
    return details, summary, "published"


def day_row(symbol: str, day_directory: pathlib.Path) -> dict[str, float] | None:
    pattern = (
        "target_b=10/*.parquet"
        if symbol == "NVDA"
        else "regime=daytime/segment=*/target_b=10/*.parquet"
    )
    parts = sorted(day_directory.glob(pattern))
    if not parts:
        return None
    frame = pd.concat(
        [pq.read_table(part, columns=COLUMNS).to_pandas() for part in parts],
        ignore_index=True,
    )
    if frame.empty:
        return None

    events = frame["realized_source_event_count"].to_numpy(dtype=np.float64)
    elapsed_seconds = frame["elapsed_ns"].to_numpy(dtype=np.float64) / 1e9
    returns = frame["mid_log_return"].to_numpy(dtype=np.float64)
    depth_l1 = frame["start_bid_depth_l1"].to_numpy(dtype=np.float64) + frame[
        "start_ask_depth_l1"
    ].to_numpy(dtype=np.float64)
    depth_l5 = frame["start_bid_depth_l5"].to_numpy(dtype=np.float64) + frame[
        "start_ask_depth_l5"
    ].to_numpy(dtype=np.float64)
    session_seconds = elapsed_seconds.sum()
    open_mid = frame["start_midprice_x2_nano"].iloc[0] / 2e9
    close_mid = frame["end_midprice_x2_nano"].iloc[-1] / 2e9

    return {
        "symbol": symbol,
        "source_date": str(frame["source_date"].iloc[0]),
        "blocks": float(len(frame)),
        "source_events": float(events.sum()),
        "events_per_second": float(events.sum() / session_seconds),
        "block_seconds_median": float(np.median(elapsed_seconds)),
        "overshoot_mean": float(frame["completion_overshoot"].mean()),
        "spread_ticks_median": float(np.median(frame["start_spread_ticks"])),
        "depth_l1_median": float(np.median(depth_l1)),
        "depth_l5_median": float(np.median(depth_l5)),
        "realized_volatility_bps": float(np.sqrt(np.sum(returns**2)) * 1e4),
        "open_close_return_bps": float(np.log(close_mid / open_mid) * 1e4),
        "block_return_rms_bps": float(np.sqrt(np.mean(returns**2)) * 1e4),
        "queue_events_per_block": float(
            frame["depletion_plus_appearance_count"].mean()
        ),
        "execution_share": float(
            frame["cause_displayed_execution_count"].sum()
            / max(
                frame[
                    [
                        "cause_add_count",
                        "cause_cancel_count",
                        "cause_displayed_execution_count",
                    ]
                ]
                .to_numpy()
                .sum(),
                1.0,
            )
        ),
        "cancel_to_add_ratio": float(
            frame["cause_cancel_count"].sum() / max(frame["cause_add_count"].sum(), 1.0)
        ),
        "mid_price": float(open_mid),
    }


def publish_descriptives() -> None:
    rows: list[dict[str, float]] = []
    for symbol in SYMBOLS:
        symbol_root = (
            NVDA_DAYTIME_ROOT if symbol == "NVDA" else BLOCK_ROOT / f"symbol={symbol}"
        )
        for day_directory in sorted(symbol_root.glob("source_date=*")):
            row = day_row(symbol, day_directory)
            if row is not None:
                rows.append(row)
        print(f"{symbol}: {sum(r['symbol'] == symbol for r in rows)} days", flush=True)

    daily = (
        pd.DataFrame(rows).sort_values(["symbol", "source_date"]).reset_index(drop=True)
    )

    aggregate = (
        daily.drop(columns=["source_date"])
        .groupby("symbol", as_index=False)
        .mean(numeric_only=True)
    )
    aggregate["daily_return_dispersion_bps"] = (
        daily.groupby("symbol")["open_close_return_bps"]
        .std()
        .reindex(aggregate["symbol"])
        .to_numpy()
    )
    aggregate["days"] = (
        daily.groupby("symbol").size().reindex(aggregate["symbol"]).to_numpy()
    )

    if len(daily) != 180 or len(aggregate) != 12:
        raise RuntimeError("descriptive support is not exact 12 symbols x 15 days")
    for name, frame in (
        ("symbol_date_descriptives.parquet", daily),
        ("symbol_date_descriptives.csv", daily),
        ("symbol_descriptives.parquet", aggregate),
        ("symbol_descriptives.csv", aggregate),
    ):
        _atomic_publish_frame(frame, OUTPUT_ROOT / name)

    pd.set_option("display.width", 200)
    print(aggregate.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correlations-only",
        action="store_true",
        help="Reuse published Phase 07/15 compact tables without rescanning raw blocks.",
    )
    arguments = parser.parse_args()
    if not arguments.correlations_only:
        publish_descriptives()
    _, summary, publication_status = publish_cross_symbol_correlations()
    print(f"correlation_publication={publication_status}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
