"""Phase 03 selected-scale scientific unit runtime."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.cuda_runtime import (
    UPSTREAM,
    _cuda_collect_scores_by_day,
    _cuda_test_predictions_and_metrics,
    _cuda_train_stage,
    configure_determinism,
    device_or_raise,
    fit_positive_day_equal_calibration_with_heartbeat,
    phase07_checkpoint_root_relocation,
)

DATES = (
    "2026-07-02",
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-13",
    "2026-07-14",
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
)
NEW_CELLS = ((10, 1), (10, 2), (10, 12), (5, 12), (30, 2))
EXPECTED_PATHS = {
    (10, 1): {"development": 1_751_699, "calibration": 671_864, "test": 582_772},
    (10, 2): {"development": 1_751_696, "calibration": 671_863, "test": 582_772},
    (10, 6): {"development": 1_751_687, "calibration": 671_859, "test": 582_767},
    (10, 12): {"development": 1_751_669, "calibration": 671_853, "test": 582_761},
    (5, 12): {"development": 3_441_643, "calibration": 1_319_800, "test": 1_144_244},
    (30, 2): {"development": 591_020, "calibration": 226_627, "test": 196_688},
}
MODELS = ("pooled_mlp", "ordered_tcn")
SEEDS = (20260730, 20260731, 20260732)
PHASE14_V5_ADMISSION = "phase14_v5_shuffled_order_scale_admission_v1"
PHASE14_V5_EXPECTED_PATHS = {
    "development": 1_751_687,
    "calibration": 671_859,
    "test": 582_767,
}
CANONICAL_BLOCK_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("symbol", pa.string()),
        ("source_date", pa.date32()),
        ("target_b", pa.uint16()),
        ("block_index", pa.uint32()),
        ("first_source_event_index", pa.uint64()),
        ("last_source_event_index", pa.uint64()),
        ("first_raw_ordinal", pa.uint64()),
        ("last_raw_ordinal", pa.uint64()),
        ("realized_source_event_count", pa.uint16()),
        ("completion_overshoot", pa.uint16()),
        ("block_start_ns", pa.int64()),
        ("block_end_ns", pa.int64()),
        ("elapsed_ns", pa.int64()),
        ("flags_or", pa.uint8()),
        *[
            (f"{prefix}_{name}", arrow_type)
            for prefix in ("start", "end")
            for name, arrow_type in (
                ("best_bid_nano", pa.int64()),
                ("best_ask_nano", pa.int64()),
                ("midprice_x2_nano", pa.int64()),
                ("spread_ticks", pa.float32()),
                ("bid_depth_l1", pa.uint64()),
                ("ask_depth_l1", pa.uint64()),
                ("bid_depth_l5", pa.uint64()),
                ("ask_depth_l5", pa.uint64()),
                ("imbalance_l1", pa.float32()),
                ("imbalance_l5", pa.float32()),
            )
        ],
        ("bid_depth_l1_net_change", pa.int64()),
        ("ask_depth_l1_net_change", pa.int64()),
        ("bid_depth_l5_net_change", pa.int64()),
        ("ask_depth_l5_net_change", pa.int64()),
        ("bid_depth_l1_gross_variation", pa.uint64()),
        ("ask_depth_l1_gross_variation", pa.uint64()),
        ("bid_depth_l5_gross_variation", pa.uint64()),
        ("ask_depth_l5_gross_variation", pa.uint64()),
        ("bid_best_depletion_count", pa.uint32()),
        ("ask_best_depletion_count", pa.uint32()),
        ("bid_best_appearance_count", pa.uint32()),
        ("ask_best_appearance_count", pa.uint32()),
        ("depletion_plus_appearance_count", pa.uint32()),
        ("appearance_minus_depletion_count", pa.int32()),
        ("mid_log_return", pa.float64()),
        ("mid_change_half_ticks", pa.int32()),
        ("bid_quote_change_ticks", pa.int32()),
        ("ask_quote_change_ticks", pa.int32()),
        ("mid_abs_variation_half_ticks", pa.uint64()),
        ("spread_abs_variation_ticks", pa.uint64()),
        ("same_price_refill_count", pa.uint32()),
        ("cause_add_count", pa.uint32()),
        ("cause_cancel_count", pa.uint32()),
        ("cause_replace_count", pa.uint32()),
        ("cause_displayed_execution_count", pa.uint32()),
        ("cause_other_count", pa.uint32()),
    ],
    metadata={
        b"schema_version": b"continuous_blocks_v1",
        b"scientific_unit": b"completion-closed source-event-count block",
        b"price_unit": b"1e-9 currency units",
    },
)


@dataclass(frozen=True)
class ScaleUnitConfig:
    b: int
    k: int
    model: str
    seed: int
    device: str
    input_root: Path
    output_root: Path
    work_root: Path
    log_root: Path
    hourly_cost_usd: float = 0.69
    batch_size: int = 1_024
    scaler_paths_per_day: int = 10_000
    max_epochs: int = 30
    min_epochs: int = 6
    patience: int = 3
    checkpoint_batches: int = 50
    resume: bool = True
    logical_unit_id: str | None = None
    storage_relative_path: str | None = None
    expected_paths_override: dict[str, int] | None = None
    prediction_symbol: str | None = None
    prediction_segment: str | None = None

    @property
    def unit_id(self) -> str:
        if self.logical_unit_id is not None:
            return self.logical_unit_id
        return f"B={self.b}_K={self.k}/{self.model}/seed={self.seed}"

    @property
    def storage_path(self) -> Path:
        if self.storage_relative_path is not None:
            return Path(self.storage_relative_path)
        return Path(f"B={self.b}_K={self.k}") / self.model / f"seed={self.seed}"

    def scientific_config(self) -> dict[str, Any]:
        value = {
            key: item
            for key, item in asdict(self).items()
            if key
            not in {
                "device",
                "input_root",
                "output_root",
                "work_root",
                "log_root",
                "hourly_cost_usd",
                "resume",
                "logical_unit_id",
                "storage_relative_path",
                "expected_paths_override",
                "prediction_symbol",
                "prediction_segment",
            }
        }
        value["dates"] = list(DATES)
        value["split"] = {
            "development": list(DATES[:9]),
            "calibration": list(DATES[9:12]),
            "test": list(DATES[12:]),
        }
        value["path_stride"] = 3
        value["path_offset_rule"] = "date.toordinal()%3"
        value["feature_projection"] = "continuous_path_37_v1"
        value["model_implementation"] = "immutable_upstream_phase05_v1"
        value["adamw_candidates"] = list(UPSTREAM.ADAMW_CANDIDATES)
        value["reversible_null_seed"] = UPSTREAM.NULL_SEED
        value["fp32"] = True
        value["tf32"] = False
        value["amp"] = False
        if self.expected_paths_override is not None:
            value["expected_path_counts"] = dict(self.expected_paths_override)
        if self.prediction_symbol is not None:
            value["prediction_symbol"] = self.prediction_symbol
        if self.prediction_segment is not None:
            value["prediction_segment"] = self.prediction_segment
        if phase14_v5_admitted(self):
            value["phase14_v5_admission"] = PHASE14_V5_ADMISSION
        return value


def phase14_v5_admitted(config: ScaleUnitConfig) -> bool:
    return bool(
        (config.b, config.k, config.model) == (10, 6, "ordered_tcn")
        and config.seed in SEEDS
        and config.logical_unit_id == f"phase14/v5/shuffled_ordered/seed={config.seed}"
        and config.expected_paths_override == PHASE14_V5_EXPECTED_PATHS
        and config.prediction_symbol == "NVDA"
        and config.prediction_segment == "daytime"
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def phase07_ephemeral_root_relocation(
    observed: dict[str, Any], expected: dict[str, Any], config: ScaleUnitConfig
) -> bool:
    """Admit the fail-closed shared-cache repair without invalidating paid checkpoints."""
    if not (config.logical_unit_id or "").startswith("phase07/"):
        return False
    observed_copy = json.loads(json.dumps(observed, sort_keys=True))
    expected_copy = json.loads(json.dumps(expected, sort_keys=True))
    try:
        old_root = Path(observed_copy["input"].pop("root")).resolve()
        new_root = Path(expected_copy["input"].pop("root")).resolve()
        old_root.relative_to(Path("/workspace/cache/phase07"))
        new_root.relative_to(Path("/tmp"))
    except (KeyError, TypeError, ValueError):
        return False
    return observed_copy == expected_copy


def input_preflight(config: ScaleUnitConfig, *, deep: bool = False) -> dict[str, Any]:
    observed_dates = tuple(UPSTREAM.discover_dates(config.input_root, config.b))
    if observed_dates != DATES:
        raise RuntimeError(
            f"input date mismatch: observed={observed_dates} expected={DATES}"
        )
    split = UPSTREAM.chronological_day_split(DATES)
    rows_by_date: dict[str, int] = {}
    files = bytes_total = 0
    for source_date in DATES:
        paths = UPSTREAM.partition_paths(config.input_root, source_date, config.b)
        if not paths:
            raise FileNotFoundError(f"missing input date={source_date} B={config.b}")
        rows = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            names = set(schema.names)
            if len(schema.names) != 61:
                raise RuntimeError(
                    f"input is not the approved 61-column schema: {path}"
                )
            if not set(UPSTREAM.REQUIRED_COLUMNS).issubset(names):
                raise RuntimeError(f"input schema missing model columns: {path}")
            if not schema.equals(CANONICAL_BLOCK_SCHEMA, check_metadata=True):
                raise RuntimeError(
                    f"input schema differs from canonical upstream: {path}"
                )
            rows += parquet.metadata.num_rows
            files += 1
            bytes_total += path.stat().st_size
        rows_by_date[source_date] = rows
        if deep:
            day = UPSTREAM.read_day(config.input_root, source_date, config.b)
            if len(day.block_index) != rows:
                raise RuntimeError(f"decoded/footer row mismatch: {source_date}")
            if np.any(day.block_end_ns < day.block_start_ns):
                raise RuntimeError(f"negative block duration: {source_date}")
            realized = day.intervals[
                :, UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]
            ]
            overshoot = day.intervals[
                :, UPSTREAM.INTERVAL_INDEX["completion_overshoot"]
            ]
            if np.any(realized < config.b) or not np.array_equal(
                overshoot, realized - config.b
            ):
                raise RuntimeError(
                    f"source-event/overshoot accounting mismatch: {source_date}"
                )

    def path_count(days: tuple[str, ...]) -> int:
        total = 0
        for source_date in days:
            available = rows_by_date[source_date] - config.k + 1
            offset = date.fromisoformat(source_date).toordinal() % 3
            if available > offset:
                total += (available - 1 - offset) // 3 + 1
        return total

    counts = {
        "development": path_count(split.development),
        "calibration": path_count(split.calibration),
        "test": path_count(split.test),
    }
    expected_counts = (
        config.expected_paths_override
        if config.expected_paths_override is not None
        else EXPECTED_PATHS[(config.b, config.k)]
    )
    if counts != expected_counts:
        raise RuntimeError(
            f"path count mismatch observed={counts} "
            f"expected={EXPECTED_PATHS[(config.b, config.k)]}"
        )
    return {
        "root": str(config.input_root),
        "files": files,
        "bytes": bytes_total,
        "rows_by_date": rows_by_date,
        "path_counts": counts,
    }


def validate_complete(final: Path, config: ScaleUnitConfig) -> dict[str, Any]:
    path = final / "complete.json"
    if not path.is_file():
        raise RuntimeError(f"missing complete record: {final}")
    value = json.loads(path.read_text(encoding="utf-8"))
    current_input = input_preflight(config)
    published_input = value.get("input")
    portable_input_match = isinstance(published_input, dict) and {
        key: item for key, item in published_input.items() if key != "root"
    } == {key: item for key, item in current_input.items() if key != "root"}
    if (
        value.get("status") != "complete"
        or value.get("unit_id") != config.unit_id
        or value.get("scientific_config") != config.scientific_config()
        # Physical roots legitimately differ between execution environments and the
        # canonical local NVMe mirror.  All content-bearing preflight fields
        # remain exact; only the location string is portable.
        or not portable_input_match
        or value.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError(f"complete record mismatch: {final}")
    artifacts = value.get("artifacts", [])
    required = {
        "selection.json",
        "scaler.json",
        "calibration.json",
        "model.pt",
        "metrics.json",
        "test_predictions.parquet",
    }
    if config.prediction_symbol is not None:
        required.add("null_test_predictions.parquet")
    if {item.get("filename") for item in artifacts} != required:
        raise RuntimeError(f"artifact inventory mismatch: {final}")
    for item in artifacts:
        artifact = final / item["filename"]
        if not artifact.is_file() or artifact.stat().st_size != item["bytes"]:
            raise RuntimeError(f"artifact size mismatch: {artifact}")
        if (
            "rows" in item
            and pq.ParquetFile(artifact).metadata.num_rows != item["rows"]
        ):
            raise RuntimeError(f"artifact row mismatch: {artifact}")
    prediction = final / "test_predictions.parquet"
    expected_counts = (
        config.expected_paths_override
        if config.expected_paths_override is not None
        else EXPECTED_PATHS[(config.b, config.k)]
    )
    expected_rows = 2 * expected_counts["test"]
    parquet = pq.ParquetFile(prediction)
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(f"prediction count mismatch: {prediction}")
    if config.prediction_symbol is not None:
        null_prediction = final / "null_test_predictions.parquet"
        if pq.ParquetFile(null_prediction).metadata.num_rows != expected_rows:
            raise RuntimeError(f"null prediction count mismatch: {null_prediction}")
    if not {"pair_id", "orientation", "label"}.issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"prediction schema mismatch: {prediction}")
    expected_pair_ids = []
    for source_date in DATES[12:]:
        day = UPSTREAM.read_day(config.input_root, source_date, config.b)
        starts = UPSTREAM.day_path_starts(day, config.k)
        expected_pair_ids.extend(
            f"{source_date}:B{config.b}:K{config.k}:{int(day.block_index[index])}"
            for index in starts
        )
    identity_columns = (
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
        "seed",
    )
    # Prediction batches are written as a forward block followed by its reverse
    # block. Parquet may coalesce several batches into one row group, so validate
    # both the observed and reversible-null streams by semantic pair boundaries.
    prediction_paths = [prediction]
    if config.prediction_symbol is not None:
        prediction_paths.append(final / "null_test_predictions.parquet")
    for checked_prediction in prediction_paths:
        checked_parquet = pq.ParquetFile(checked_prediction)
        required_columns = set(identity_columns) | {
            "orientation",
            "label",
            "raw_antisymmetric_score",
            "calibrated_antisymmetric_score",
            "probability_forward",
        }
        if not required_columns.issubset(checked_parquet.schema_arrow.names):
            raise RuntimeError(f"prediction schema mismatch: {checked_prediction}")
        pending_pairs: dict[str, tuple[tuple[Any, ...], float, float, float]] = {}
        pending_order: deque[str] = deque()
        forward_count = reverse_count = 0
        for batch in checked_parquet.iter_batches(batch_size=config.batch_size):
            values = batch.to_pydict()
            for row_index, pair_id in enumerate(values["pair_id"]):
                orientation = values["orientation"][row_index]
                label = values["label"][row_index]
                identity = tuple(values[name][row_index] for name in identity_columns)
                raw = float(values["raw_antisymmetric_score"][row_index])
                calibrated = float(values["calibrated_antisymmetric_score"][row_index])
                probability = float(values["probability_forward"][row_index])
                if not all(math.isfinite(x) for x in (raw, calibrated, probability)):
                    raise RuntimeError(
                        f"prediction nonfinite score: {checked_prediction}"
                    )
                if orientation == "forward" and label == 1:
                    if pair_id in pending_pairs:
                        raise RuntimeError(
                            f"duplicate prediction forward pair: {pair_id}"
                        )
                    if (
                        forward_count >= len(expected_pair_ids)
                        or pair_id != expected_pair_ids[forward_count]
                    ):
                        raise RuntimeError(
                            f"prediction global path coverage mismatch: {checked_prediction}"
                        )
                    pending_pairs[pair_id] = (identity, raw, calibrated, probability)
                    pending_order.append(pair_id)
                    forward_count += 1
                    if len(pending_pairs) > config.batch_size:
                        raise RuntimeError(
                            f"prediction forward run exceeds batch bound: {checked_prediction}"
                        )
                elif orientation == "reverse" and label == 0:
                    if not pending_order or pending_order[0] != pair_id:
                        raise RuntimeError(
                            f"prediction reverse pair order mismatch: {pair_id}"
                        )
                    pending_order.popleft()
                    forward = pending_pairs.pop(pair_id, None)
                    if forward is None:
                        raise RuntimeError(
                            f"prediction reverse without forward: {pair_id}"
                        )
                    (
                        forward_identity,
                        forward_raw,
                        forward_calibrated,
                        forward_probability,
                    ) = forward
                    if identity != forward_identity:
                        raise RuntimeError(
                            f"prediction pair metadata mismatch: {pair_id}"
                        )
                    if (
                        raw != -forward_raw
                        or calibrated != -forward_calibrated
                        or not math.isclose(
                            probability + forward_probability,
                            1.0,
                            rel_tol=0.0,
                            abs_tol=1e-7,
                        )
                    ):
                        raise RuntimeError(f"prediction pair score mismatch: {pair_id}")
                    reverse_count += 1
                else:
                    raise RuntimeError(
                        f"prediction orientation/label mismatch: {pair_id}"
                    )
        if pending_pairs or pending_order:
            raise RuntimeError(
                f"prediction pairs missing reverse rows: {checked_prediction}"
            )
        if forward_count != len(expected_pair_ids) or reverse_count != forward_count:
            raise RuntimeError(
                f"prediction global path coverage mismatch: {checked_prediction}"
            )
    return value


def augment_prediction_contract(path: Path, config: ScaleUnitConfig) -> None:
    """Add Phase 07 identity/bounded columns with bounded row-group memory."""
    if config.prediction_symbol is None or config.prediction_segment is None:
        return
    source = pq.ParquetFile(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.augment.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for row_group in range(source.num_row_groups):
            table = source.read_row_group(row_group)
            count = len(table)
            calibrated = np.asarray(
                table["calibrated_antisymmetric_score"], dtype=np.float64
            )
            table = table.append_column(
                "symbol", pa.array([config.prediction_symbol] * count, type=pa.string())
            )
            table = table.append_column(
                "segment",
                pa.array([config.prediction_segment] * count, type=pa.string()),
            )
            table = table.append_column("direction", table["orientation"])
            table = table.append_column(
                "bounded_log99_score",
                pa.array(
                    np.clip(calibrated, -np.log(99.0), np.log(99.0)),
                    type=pa.float64(),
                ),
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise RuntimeError(f"prediction parquet has no row groups: {path}")
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def run_scale_unit(config: ScaleUnitConfig) -> dict[str, Any]:
    phase07_primary = (
        (config.b, config.k) == (10, 6)
        and config.logical_unit_id is not None
        and config.logical_unit_id.startswith("phase07/")
    )
    phase09_length = (
        config.b == 10
        and config.k in (3, 4, 8)
        and config.logical_unit_id is not None
        and config.logical_unit_id.startswith("phase09/length/")
        and config.expected_paths_override is not None
    )
    phase14_v5 = phase14_v5_admitted(config)
    if (
        (config.b, config.k) not in NEW_CELLS
        and not phase07_primary
        and not phase09_length
        and not phase14_v5
    ):
        raise ValueError(
            f"cell outside approved Phase 03 scope: {(config.b, config.k)}"
        )
    if config.model not in MODELS or config.seed not in SEEDS:
        raise ValueError("model or seed outside approved Phase 03 scope")
    configure_determinism()
    device = device_or_raise(config.device)
    if device.type != "cuda":
        raise ValueError("Phase 03 production units require qualified CUDA")
    preflight = input_preflight(config)
    split = UPSTREAM.chronological_day_split(DATES)
    final = config.output_root / config.storage_path
    work = config.work_root / config.storage_path
    if final.exists():
        if config.resume:
            return validate_complete(final, config)
        raise FileExistsError(final)
    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / ".unit.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    lock_open = True
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"unit already has a writer: {config.unit_id}") from error
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
            observed_config = json.loads(spec_path.read_text(encoding="utf-8"))
            if observed_config != exact_config:
                if phase07_ephemeral_root_relocation(
                    observed_config, exact_config, config
                ):
                    atomic_json(spec_path, exact_config)
                else:
                    raise RuntimeError(f"work config mismatch: {work}")
            if not config.resume:
                raise FileExistsError(f"work exists; use resume: {work}")
        else:
            atomic_json(spec_path, exact_config)
        if config.storage_relative_path is None:
            staging = work / "staging"
        else:
            attempt_path = work / "publication_attempt.json"
            if attempt_path.is_file():
                attempt_id = str(
                    json.loads(attempt_path.read_text(encoding="utf-8"))["attempt_id"]
                )
            else:
                attempt_id = f"{os.getpid()}-{time.time_ns()}"
                atomic_json(attempt_path, {"attempt_id": attempt_id})
            staging = final.parent / f".{final.name}.{attempt_id}.tmp"
        if (staging / "complete.json").is_file():
            validate_complete(staging, config)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            validated = validate_complete(final, config)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            lock_open = False
            shutil.rmtree(work)
            return validated
        staging.mkdir(parents=True, exist_ok=True)
        log_path = (
            config.log_root
            / f"B{config.b}_K{config.k}_{config.model}_seed{config.seed}.jsonl"
        )
        UPSTREAM.emit(
            log_path,
            "unit_start",
            unit_id=config.unit_id,
            device=str(device),
            input_files=preflight["files"],
            input_bytes=preflight["bytes"],
        )
        scaler_path = work / "development_scaler.json"
        if scaler_path.is_file():
            scaler = UPSTREAM.ChannelScaler.from_dict(
                json.loads(scaler_path.read_text(encoding="utf-8"))
            )
        else:
            scaler = UPSTREAM.fit_development_scaler(
                config.input_root,
                split.development,
                config.b,
                config.k,
                config.scaler_paths_per_day,
            )
            atomic_json(scaler_path, scaler.to_dict())

        fold_rows: list[dict[str, Any]] = []
        folds_root = work / "folds"
        folds_root.mkdir(parents=True, exist_ok=True)
        for fold in UPSTREAM.expanding_inner_folds(split.development):
            fold_scaler_path = folds_root / f"fold={fold.fold}_scaler.json"
            if fold_scaler_path.is_file():
                fold_scaler = UPSTREAM.ChannelScaler.from_dict(
                    json.loads(fold_scaler_path.read_text(encoding="utf-8"))
                )
            else:
                fold_scaler = UPSTREAM.fit_development_scaler(
                    config.input_root,
                    fold.train,
                    config.b,
                    config.k,
                    config.scaler_paths_per_day,
                )
                atomic_json(fold_scaler_path, fold_scaler.to_dict())
            for candidate_index, candidate in enumerate(UPSTREAM.ADAMW_CANDIDATES):
                result_path = folds_root / (
                    f"fold={fold.fold}_candidate={candidate_index}.json"
                )
                stage_config = {
                    "unit": exact_config,
                    "stage": "selection",
                    "fold": asdict(fold),
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                }
                if result_path.is_file():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                    # JSON publication turns tuples (for example fold.train) into
                    # lists.  Compare the same JSON-normalized representation so
                    # an interrupted run can reuse a durably published fold while
                    # the torch checkpoint retains its exact in-memory config.
                    published_stage_config = json.loads(
                        json.dumps(stage_config, sort_keys=True)
                    )
                    if (
                        row.get("checkpoint_config") != published_stage_config
                        and phase07_checkpoint_root_relocation(
                            row.get("checkpoint_config"), published_stage_config
                        )
                        is None
                    ):
                        raise RuntimeError(f"stale fold result: {result_path}")
                    fold_rows.append(row)
                    continue
                checkpoint_path = result_path.with_suffix(".checkpoint.pt")
                _, history, best_epoch, best_loss = _cuda_train_stage(
                    device,
                    model_kind=config.model,
                    seed=config.seed,
                    input_root=config.input_root,
                    train_dates=fold.train,
                    evaluation_dates=(fold.evaluation,),
                    target_b=config.b,
                    k=config.k,
                    scaler=fold_scaler,
                    candidate=candidate,
                    max_epochs=config.max_epochs,
                    min_epochs=config.min_epochs,
                    patience=config.patience,
                    fixed_epochs=None,
                    batch_size=config.batch_size,
                    checkpoint_batches=config.checkpoint_batches,
                    checkpoint_path=checkpoint_path,
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
                checkpoint_path.unlink(missing_ok=True)
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
            target_b=config.b,
            k=config.k,
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
            config.b,
            config.k,
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
        test_metrics, audit = _cuda_test_predictions_and_metrics(
            model=final_model,
            input_root=config.input_root,
            dates=split.test,
            target_b=config.b,
            k=config.k,
            scaler=scaler,
            alpha=alpha,
            batch_size=config.batch_size,
            prediction_path=staging / "test_predictions.parquet",
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
        torch_payload = {
            "model_state": {
                name: value.detach().cpu()
                for name, value in final_model.state_dict().items()
            },
            "model": config.model,
            "seed": config.seed,
            "B": config.b,
            "K": config.k,
        }
        UPSTREAM.atomic_torch_save(staging / "model.pt", torch_payload)
        atomic_json(staging / "scaler.json", scaler.to_dict())
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
            target_b=config.b,
            k=config.k,
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
            config.b,
            config.k,
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
        null_prediction_path = (
            staging / "null_test_predictions.parquet"
            if config.prediction_symbol is not None
            else None
        )
        null_metrics, null_audit = _cuda_test_predictions_and_metrics(
            model=null_model,
            input_root=config.input_root,
            dates=split.test,
            target_b=config.b,
            k=config.k,
            scaler=scaler,
            alpha=null_alpha,
            batch_size=config.batch_size,
            prediction_path=null_prediction_path,
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
        null_threshold = (
            UPSTREAM.NULL_ABSOLUTE_CEILING
            if config.prediction_symbol is not None
            else max(UPSTREAM.NULL_ABSOLUTE_CEILING, 0.10 * abs(real_d))
        )
        null_bias_gate = {
            "observed_abs": abs(null_d),
            "threshold": null_threshold,
            "passed": abs(null_d) <= null_threshold,
        }
        passed = bool(null_audit["passed"] and null_bias_gate["passed"])
        metrics = {
            "B": config.b,
            "K": config.k,
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
                "bias_gate": null_bias_gate,
                "training_history": null_history,
            },
        }
        atomic_json(staging / "metrics.json", metrics)
        if not passed:
            raise RuntimeError(
                f"reversible-null gate failed audit={null_audit} bias={null_bias_gate}"
            )
        augment_prediction_contract(staging / "test_predictions.parquet", config)
        if null_prediction_path is not None:
            augment_prediction_contract(null_prediction_path, config)
        artifacts = []
        for path in sorted(staging.iterdir()):
            if not path.is_file():
                continue
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
        expected_counts = (
            config.expected_paths_override
            if config.expected_paths_override is not None
            else EXPECTED_PATHS[(config.b, config.k)]
        )
        expected_prediction_rows = 2 * expected_counts["test"]
        prediction_rows = pq.ParquetFile(
            staging / "test_predictions.parquet"
        ).metadata.num_rows
        if prediction_rows != expected_prediction_rows:
            raise RuntimeError(
                f"prediction row mismatch observed={prediction_rows} "
                f"expected={expected_prediction_rows}"
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
                "null_bias": null_bias_gate,
                "resume": {
                    "checkpoint_config_exact": True,
                    "cursor_history_parameters_enforced": True,
                    "phase01_forced_resume_probe_required_by_controller": True,
                },
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
        validate_complete(staging, config)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        validated = validate_complete(final, config)
        succeeded = True
        UPSTREAM.emit(
            log_path,
            "unit_complete",
            unit_id=config.unit_id,
            elapsed_seconds=elapsed,
            cost_usd=validated["runtime"]["measured_cost_usd"],
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        lock_open = False
        shutil.rmtree(work)
        return validated
    finally:
        if not succeeded and work.exists():
            elapsed = time.monotonic() - started
            atomic_json(
                runtime_path,
                {
                    "attempts": int(runtime["attempts"]) + 1,
                    "elapsed_seconds": float(runtime["elapsed_seconds"]) + elapsed,
                    "measured_cost_usd": (
                        float(runtime["measured_cost_usd"])
                        + elapsed * config.hourly_cost_usd / 3600.0
                    ),
                },
            )
        if lock_open:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
