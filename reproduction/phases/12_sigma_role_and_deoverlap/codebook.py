#!/usr/bin/env python3
"""Phase 12 restartable CPU orchestration.

Scientific calculations live in ``reflection.py``, ``diagnostics.py`` and
``modeling.py``.  This file owns only input admission, deterministic queues,
static symbol ownership, subprocess lifecycle, and final publication checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PHASE = Path(__file__).resolve().parent
PROJECT = PHASE.parents[1]
PHASE10 = PROJECT / "phases/10_predictive_bridge/artifacts"
PHASE11 = PROJECT / "phases/11_lgbm_multihorizon_return/artifacts"
ARTIFACTS = PHASE / "artifacts"
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
FOLDS = ("F1", "F2", "F3")
HORIZONS = (2, 4, 8, 16, 32, 64)
STAGES = ("selection", "refit")
SIGNED_FEATURES = ("D1_p_perp", "D2_p_o_perp", "odd", "even", "lag2", "ewma")
ROLE_FEATURES = ("baseline", "even", "odd")
TARGET_FEATURES = {
    "signed": SIGNED_FEATURES,
    "absolute": ROLE_FEATURES,
    "move": ROLE_FEATURES,
}
DEFAULT_OWNERS = {"NVDA": "m3", "QQQ": "local", "SOFI": "m4", "SAIA": "m4"}
KEYS = ("symbol", "source_date", "segment_id", "path_end_block_index")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"timestamp_utc": utc_now(), "event": event, **fields}, sort_keys=True
        ),
        flush=True,
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_owner_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_OWNERS)
    raw = json.loads(path.read_text())
    if "symbol_owner" in raw:
        raw = raw["symbol_owner"]
    if set(raw) == set(SYMBOLS) and all(isinstance(raw[s], str) for s in SYMBOLS):
        owners = {s: raw[s] for s in SYMBOLS}
    else:
        owners = {}
        for host, symbols in raw.items():
            if not isinstance(symbols, list):
                raise ValueError("owner JSON must be symbol->host or host->[symbols]")
            for symbol in symbols:
                if symbol in owners:
                    raise ValueError(f"duplicate symbol owner: {symbol}")
                owners[symbol] = host
        if set(owners) != set(SYMBOLS):
            raise ValueError(f"owner JSON must assign exactly {SYMBOLS}")
    if any(not value for value in owners.values()):
        raise ValueError("empty owner name")
    return owners


def return_row(symbol: str, source_date: str) -> Path:
    return (
        PHASE11
        / "data/return_rows"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def score_row(symbol: str, source_date: str) -> Path:
    return (
        PHASE10
        / "data/scores"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def reflected_row(root: Path, symbol: str, source_date: str) -> Path:
    return (
        root
        / "data/reflected_scores"
        / f"symbol={symbol}"
        / f"source_date={source_date}.parquet"
    )


def derived_row(root: Path, symbol: str, fold: str, stage: str) -> Path:
    return (
        root
        / "data/derived_features"
        / f"symbol={symbol}"
        / f"fold={fold}"
        / f"stage={stage}.parquet"
    )


def unit_root(root: Path, stage: str, unit_id: str) -> Path:
    return root / "units" / stage / unit_id


def diagnostic_root(root: Path, unit: dict[str, Any]) -> Path:
    return (
        root
        / "units/diagnostics"
        / f"symbol={unit['symbol']}"
        / f"fold={unit['fold']}"
        / f"horizon={unit['horizon']}"
    )


def unit_output(root: Path, unit: dict[str, Any]) -> Path:
    stage = unit["stage"]
    if stage == "reflection":
        return reflected_row(root, unit["symbol"], unit["source_date"])
    if stage == "features":
        return derived_row(root, unit["symbol"], unit["fold"], unit["feature_stage"])
    if stage == "diagnostics":
        return diagnostic_root(root, unit) / "baseline_calibration.parquet"
    if stage == "models":
        return (
            root
            / "predictions"
            / f"target={unit['target']}"
            / f"symbol={unit['symbol']}"
            / f"fold={unit['fold']}"
            / f"horizon={unit['horizon']}"
            / f"spec={unit['feature_set']}.parquet"
        )
    raise ValueError(stage)


def unit_completion(root: Path, unit: dict[str, Any]) -> Path:
    output = unit_output(root, unit)
    if unit["stage"] == "diagnostics":
        return diagnostic_root(root, unit) / "complete.json"
    if unit["stage"] == "features":
        return (
            root
            / "nuisance"
            / f"symbol={unit['symbol']}"
            / f"fold={unit['fold']}"
            / f"stage={unit['feature_stage']}"
            / "complete.json"
        )
    if unit["stage"] == "models":
        return (
            root
            / "models"
            / f"target={unit['target']}"
            / f"symbol={unit['symbol']}"
            / f"fold={unit['fold']}"
            / f"horizon={unit['horizon']}"
            / f"spec={unit['feature_set']}"
            / "complete.json"
        )
    return output.with_suffix(".complete.json")


def make_units(owner_map: dict[str, str]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for source_date in DATES:
            units.append(
                {
                    "stage": "reflection",
                    "unit_id": f"reflection__{symbol}__{source_date}",
                    "symbol": symbol,
                    "source_date": source_date,
                    "owner": owner_map[symbol],
                }
            )
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for horizon in HORIZONS:
                units.append(
                    {
                        "stage": "diagnostics",
                        "unit_id": f"diagnostics__{symbol}__{fold}__h{horizon}",
                        "symbol": symbol,
                        "fold": fold,
                        "horizon": horizon,
                        "owner": owner_map[symbol],
                    }
                )
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for feature_stage in STAGES:
                units.append(
                    {
                        "stage": "features",
                        "unit_id": f"features__{symbol}__{fold}__{feature_stage}",
                        "symbol": symbol,
                        "fold": fold,
                        "feature_stage": feature_stage,
                        "owner": owner_map[symbol],
                    }
                )
    for symbol in SYMBOLS:
        for fold in FOLDS:
            for horizon in HORIZONS:
                for target, feature_sets in TARGET_FEATURES.items():
                    for feature_set in feature_sets:
                        units.append(
                            {
                                "stage": "models",
                                "unit_id": f"models__{target}__{feature_set}__{symbol}__{fold}__h{horizon}",
                                "symbol": symbol,
                                "fold": fold,
                                "horizon": horizon,
                                "target": target,
                                "feature_set": feature_set,
                                "owner": owner_map[symbol],
                            }
                        )
    ids = [unit["unit_id"] for unit in units]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate generated unit id")
    counts = Counter(unit["stage"] for unit in units)
    if counts != Counter(
        {"reflection": 24, "diagnostics": 72, "features": 24, "models": 864}
    ):
        raise RuntimeError(f"queue cardinality mismatch: {dict(counts)}")
    return units


def write_queue(args: argparse.Namespace) -> None:
    owner_map = read_owner_map(Path(args.owner_json) if args.owner_json else None)
    units = make_units(owner_map)
    if args.stage != "all":
        units = [unit for unit in units if unit["stage"] == args.stage]
    if args.host:
        units = [unit for unit in units if unit["owner"] == args.host]
    payload = {
        "run_id": RUN_ID,
        "created_utc": utc_now(),
        "stage": args.stage,
        "host": args.host,
        "symbol_owner": owner_map,
        "unit_count": len(units),
        "units": units,
    }
    atomic_json(Path(args.output), payload)
    emit(
        "queue_written",
        output=args.output,
        units=len(units),
        stage=args.stage,
        host=args.host,
    )


def finite_table(path: Path, required: Iterable[str] = ()) -> pa.Table:
    table = pq.read_table(path)
    if table.num_rows <= 0:
        raise RuntimeError(f"empty parquet: {path}")
    missing = set(required) - set(table.column_names)
    if missing:
        raise RuntimeError(f"missing columns {sorted(missing)}: {path}")
    for field in table.schema:
        if pa.types.is_floating(field.type):
            column = table[field.name]
            nonfinite = pc.fill_null(pc.invert(pc.is_finite(column)), False)
            if pc.any(nonfinite).as_py():
                raise RuntimeError(f"nonfinite column {field.name}: {path}")
    return table


def assert_unique_keys(table: pa.Table, keys: Iterable[str], path: Path) -> None:
    keys = tuple(keys)
    if not keys or not set(keys).issubset(table.column_names):
        return
    # Model/reflection units contain one symbol-date-segment.  Avoid an expensive
    # four-column group-by for their hundreds of large prediction tables.
    prefix_constant = all(
        pc.count_distinct(table[name]).as_py() == 1 for name in keys[:-1]
    )
    if prefix_constant:
        observed = int(pc.count_distinct(table[keys[-1]]).as_py())
    else:
        observed = table.select(keys).group_by(list(keys)).aggregate([]).num_rows
    if observed != table.num_rows:
        raise RuntimeError(f"duplicate keys: {path}")


def valid_completion(root: Path, unit: dict[str, Any]) -> bool:
    output, completion = unit_output(root, unit), unit_completion(root, unit)
    if not output.is_file() or not completion.is_file():
        return False
    try:
        record = json.loads(completion.read_text())
        if record.get("status") != "complete" or record.get("run_id", RUN_ID) != RUN_ID:
            return False
        if record.get("unit_id", unit["unit_id"]) != unit["unit_id"]:
            return False
        config = record.get("config", {})
        if "run_id" in config and config.get("run_id") != RUN_ID:
            return False
        expected_config = {
            name: unit[name]
            for name in ("symbol", "fold", "horizon", "target")
            if name in unit
        }
        if "feature_set" in unit:
            expected_config["spec"] = unit["feature_set"]
        if "feature_stage" in unit:
            expected_config["stage"] = unit["feature_stage"]
        if "source_date" in unit:
            expected_config["source_date"] = unit["source_date"]
        if config and any(
            config.get(name) != value for name, value in expected_config.items()
        ):
            return False
        table = finite_table(output)
        if unit["stage"] == "diagnostics":
            names = (
                "bin_thresholds",
                "baseline_calibration",
                "sigma_1d",
                "conditional_role",
                "tail_gate",
            )
            expected = record.get("table_rows", {})
            for name in names:
                part = diagnostic_root(root, unit) / f"{name}.parquet"
                observed = finite_table(part).num_rows
                if name in expected and int(expected[name]) != observed:
                    return False
        if unit["stage"] in ("reflection", "features"):
            assert_unique_keys(table, KEYS, output)
        if unit["stage"] == "features":
            metric = completion.parent / "metric.parquet"
            finite_table(metric)
            required = {
                "p_perp_x",
                "o_perp_xp",
                "ordered_odd",
                "pooled_odd",
                "gap_odd",
                "ordered_even",
                "pooled_even",
                "gap_even",
                "gap_innovation_lag2",
                "gap_innovation_ewma_hl4",
            }
            if not required.issubset(table.column_names):
                return False
        if unit["stage"] == "models":
            metric = completion.parent / "metric.parquet"
            model = completion.parent / "model.txt"
            finite_table(metric)
            if not model.is_file() or model.stat().st_size <= 0:
                return False
            assert_unique_keys(table, KEYS, output)
            if not {"target", "prediction"}.issubset(table.column_names):
                return False
        expected_rows = record.get("rows")
        return expected_rows is None or int(expected_rows) == table.num_rows
    except Exception:
        return False


def preflight(args: argparse.Namespace) -> None:
    failures: list[str] = []
    admitted_rows = 0
    for symbol in SYMBOLS:
        for source_date in DATES:
            returns_path, scores_path = (
                return_row(symbol, source_date),
                score_row(symbol, source_date),
            )
            for path in (returns_path, scores_path):
                if not path.is_file():
                    failures.append(f"missing input {path}")
            if not returns_path.is_file() or not scores_path.is_file():
                continue
            try:
                returns = pq.read_table(returns_path, columns=list(KEYS))
                scores = pq.read_table(scores_path, columns=list(KEYS))
                assert_unique_keys(returns, KEYS, returns_path)
                assert_unique_keys(scores, KEYS, scores_path)
                if scores.num_rows != returns.num_rows + 3:
                    raise RuntimeError("score/return coverage is not +3 rows")
                # Phase 11 drops the first score row for delta features and retains
                # the Phase 10 forecast prefix, leaving two unused tail scores.
                aligned = scores.slice(1, returns.num_rows)
                for key in KEYS:
                    left = np.asarray(returns[key])
                    right = np.asarray(aligned[key])
                    if not np.array_equal(left, right):
                        raise RuntimeError(f"score/return key mismatch: {key}")
                admitted_rows += returns.num_rows
            except Exception as exc:
                failures.append(f"invalid input {symbol} {source_date}: {exc}")
    for script in ("reflection.py", "diagnostics.py", "modeling.py"):
        if not (PHASE / script).is_file():
            failures.append(f"missing module {script}")
    free = shutil.disk_usage(ARTIFACTS.parent).free
    if free < args.min_free_gib * 1024**3:
        failures.append(f"free disk below {args.min_free_gib} GiB")
    if failures:
        raise RuntimeError("preflight failed:\n" + "\n".join(failures))
    audit = subprocess.run(
        [sys.executable, str(PHASE / "reflection.py"), "audit"], cwd=PHASE, check=False
    )
    if audit.returncode != 0:
        raise RuntimeError("reflection algebra audit failed")
    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemAvailable", "SwapTotal", "SwapFree"):
                memory[name] = int(value.strip().split()[0]) * 1024
    stat = os.statvfs(ARTIFACTS.parent)
    emit(
        "preflight_complete",
        input_units=24,
        input_rows=admitted_rows,
        free_disk_bytes=free,
        free_inodes=stat.f_favail,
        available_ram_bytes=memory.get("MemAvailable"),
        swap_used_bytes=memory.get("SwapTotal", 0) - memory.get("SwapFree", 0),
        load_average_1m=os.getloadavg()[0],
        cpu_count=os.cpu_count(),
        modules=3,
    )


def module_command(unit: dict[str, Any], root: Path, threads: int) -> list[str]:
    stage = unit["stage"]
    output = unit_output(root, unit)
    common = ["--artifacts", str(root), "--symbol", unit["symbol"]]
    if stage == "reflection":
        return [
            sys.executable,
            str(PHASE / "reflection.py"),
            "score-unit",
            *common,
            "--source-date",
            unit["source_date"],
            "--output",
            str(output),
            "--threads",
            str(threads),
        ]
    if stage == "diagnostics":
        return [
            sys.executable,
            str(PHASE / "diagnostics.py"),
            "run-unit",
            "--symbol",
            unit["symbol"],
            "--fold",
            unit["fold"],
            "--horizon",
            str(unit["horizon"]),
            "--output-root",
            str(root / "units/diagnostics"),
            "--threads",
            str(threads),
        ]
    if stage == "features":
        return [
            sys.executable,
            str(PHASE / "modeling.py"),
            "build-nuisance",
            "--artifacts",
            str(root),
            "--symbol",
            unit["symbol"],
            "--fold",
            unit["fold"],
            "--stage",
            unit["feature_stage"],
            "--threads",
            str(threads),
        ]
    if stage == "models":
        return [
            sys.executable,
            str(PHASE / "modeling.py"),
            "fit-unit",
            "--artifacts",
            str(root),
            "--symbol",
            unit["symbol"],
            "--fold",
            unit["fold"],
            "--horizon",
            str(unit["horizon"]),
            "--target",
            unit["target"],
            "--spec",
            unit["feature_set"],
            "--threads",
            str(threads),
        ]
    raise ValueError(stage)


def load_queue(
    path: Path, expected_stage: str | None, host: str | None
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("run_id") != RUN_ID or int(payload.get("unit_count", -1)) != len(
        payload.get("units", [])
    ):
        raise RuntimeError("invalid queue envelope")
    units = payload["units"]
    ids = [unit.get("unit_id") for unit in units]
    if None in ids or len(ids) != len(set(ids)):
        raise RuntimeError("missing or duplicate queue unit")
    if expected_stage and any(unit.get("stage") != expected_stage for unit in units):
        raise RuntimeError("mixed stage queue")
    if host and any(unit.get("owner") != host for unit in units):
        raise RuntimeError("queue contains unit owned by another host")
    return units


def run_queue(args: argparse.Namespace) -> None:
    # Child processes run from the phase directory, so publish against one
    # absolute root even when the controller received a workspace-relative path.
    root = Path(args.artifacts).resolve()
    units = load_queue(Path(args.queue), args.stage, args.host)
    completed = failed = skipped = 0
    started = time.monotonic()
    pending: list[dict[str, Any]] = []
    for unit in units:
        if valid_completion(root, unit):
            skipped += 1
            emit("unit_skip", total=len(units), unit_id=unit["unit_id"])
        else:
            pending.append(unit)

    def execute(unit: dict[str, Any]) -> tuple[dict[str, Any], int, bool]:
        command = module_command(unit, root, args.threads)
        emit(
            "unit_start", total=len(units), unit_id=unit["unit_id"], stage=unit["stage"]
        )
        result = subprocess.run(command, cwd=PHASE, check=False)
        return (
            unit,
            result.returncode,
            result.returncode == 0 and valid_completion(root, unit),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, unit) for unit in pending]
        for future in as_completed(futures):
            unit, returncode, valid = future.result()
            if valid:
                completed += 1
                elapsed = time.monotonic() - started
                done = completed + failed
                emit(
                    "unit_complete",
                    total=len(units),
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    unit_id=unit["unit_id"],
                    elapsed_seconds=elapsed,
                    eta_seconds=(elapsed / max(done, 1) * (len(pending) - done)),
                )
            else:
                failed += 1
                emit(
                    "unit_failure",
                    total=len(units),
                    unit_id=unit["unit_id"],
                    returncode=returncode,
                )
    if failed:
        raise RuntimeError(f"queue completed with {failed} failures")
    emit(
        "queue_complete",
        units=len(units),
        completed=completed,
        skipped=skipped,
        elapsed_seconds=time.monotonic() - started,
    )


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.artifacts).resolve()
    completion_path = root / "tables/complete.json"
    # A failed rebuild must not leave an older marker advertising a partially
    # refreshed table bundle as complete.
    completion_path.unlink(missing_ok=True)
    owners = read_owner_map(Path(args.owner_json) if args.owner_json else None)
    units = make_units(owners)
    invalid = [unit["unit_id"] for unit in units if not valid_completion(root, unit)]
    if invalid:
        raise RuntimeError(f"missing/invalid units {len(invalid)}: {invalid[:10]}")
    stage_counts = Counter(unit["stage"] for unit in units)

    # Each module owns its scientific aggregation; the orchestrator validates the
    # requested compact products and publishes only after every step succeeds.
    aggregate_commands = (
        [
            sys.executable,
            str(PHASE / "reflection.py"),
            "support-audit",
            "--output",
            str(root / "tables/reflection_support.parquet"),
        ],
        [
            sys.executable,
            str(PHASE / "diagnostics.py"),
            "aggregate",
            "--unit-root",
            str(root / "units/diagnostics"),
            "--tables-dir",
            str(root / "tables"),
        ],
        [
            sys.executable,
            str(PHASE / "residual_role.py"),
            "aggregate",
            "--artifacts",
            str(root),
        ],
        [
            sys.executable,
            str(PHASE / "modeling.py"),
            "aggregate",
            "--artifacts",
            str(root),
        ],
    )
    for command in aggregate_commands:
        result = subprocess.run(command, cwd=PHASE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"aggregate failed: {Path(command[1]).name}")

    required = (
        "bin_thresholds.parquet",
        "baseline_calibration.parquet",
        "sigma_1d.parquet",
        "conditional_role.parquet",
        "tail_gate.parquet",
        "reflection_support.parquet",
        "residual_conditional_role.parquet",
        "residual_role_thresholds.parquet",
        "nuisance_metrics.parquet",
        "fold_metrics.parquet",
        "compact_summary.parquet",
        "ordered_residual_increment.parquet",
        "promotion_gate.parquet",
        "support_audits.parquet",
    )
    table_rows: dict[str, int] = {}
    for name in required:
        path = root / "tables" / name
        table = finite_table(path)
        table_rows[name] = table.num_rows
    residual_units = 72
    stage_counts["residual_role"] = residual_units
    total_units = len(units) + residual_units
    completion = {
        "status": "complete",
        "run_id": RUN_ID,
        "completed_utc": utc_now(),
        "units": total_units,
        "stage_counts": dict(stage_counts),
        "invalid_units": 0,
        "duplicate_units": 0,
        "tables": table_rows,
    }
    atomic_json(completion_path, completion)
    emit("aggregate_complete", units=total_units, tables=table_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--min-free-gib", type=float, default=20.0)
    check.set_defaults(function=preflight)

    queue = commands.add_parser("make-queue")
    queue.add_argument(
        "--stage",
        choices=("all", "reflection", "diagnostics", "features", "models"),
        default="all",
    )
    queue.add_argument("--host")
    queue.add_argument("--owner-json")
    queue.add_argument("--output", required=True)
    queue.set_defaults(function=write_queue)

    run = commands.add_parser("run-queue")
    run.add_argument(
        "--stage",
        choices=("reflection", "diagnostics", "features", "models"),
        required=True,
    )
    run.add_argument("--queue", required=True)
    run.add_argument("--host")
    run.add_argument("--artifacts", default=str(ARTIFACTS))
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--workers", type=int, default=1)
    run.set_defaults(function=run_queue)

    final = commands.add_parser("aggregate")
    final.add_argument("--artifacts", default=str(ARTIFACTS))
    final.add_argument("--owner-json")
    final.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
