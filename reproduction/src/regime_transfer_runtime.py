"""Shared scientific runtime for Phase 05 Edge/Daytime transfer."""

from __future__ import annotations

import fcntl, json, math, multiprocessing as mp, os, random, shutil, socket, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.cuda_runtime import UPSTREAM, configure_determinism
from src.scale_runtime import CANONICAL_BLOCK_SCHEMA, atomic_json

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
SPLIT = {"development": DATES[:9], "calibration": DATES[9:12], "test": DATES[12:]}
MODELS = ("pooled_mlp", "ordered_tcn")
SEEDS = (20260730, 20260731, 20260732)
COMPONENTS = {"edge": ("open", "pre_close"), "daytime": ("daytime",)}
EXPECTED = {
    "2026-07-02": (37172, 28163, 252499),
    "2026-07-06": (36657, 17458, 124153),
    "2026-07-07": (51983, 24828, 201097),
    "2026-07-08": (41545, 20477, 198963),
    "2026-07-09": (45337, 17612, 157005),
    "2026-07-10": (46325, 15806, 189995),
    "2026-07-13": (44790, 24826, 195522),
    "2026-07-14": (41190, 21382, 213930),
    "2026-07-15": (40993, 22143, 218523),
    "2026-07-16": (45642, 32080, 225799),
    "2026-07-17": (54339, 29983, 264095),
    "2026-07-20": (38613, 21613, 181965),
    "2026-07-21": (46638, 21621, 159172),
    "2026-07-22": (31476, 20385, 170737),
    "2026-07-23": (41153, 28363, 252858),
}
PREDICTION_SCHEMA = pa.schema(
    [
        pa.field(name, dtype, nullable=False)
        for name, dtype in [
            ("pair_id", pa.string()),
            ("source_date", pa.date32()),
            ("test_regime", pa.string()),
            ("test_component", pa.string()),
            ("segment", pa.uint16()),
            ("path_start_block", pa.uint32()),
            ("direction", pa.string()),
            ("path_start_ns", pa.int64()),
            ("path_end_ns", pa.int64()),
            ("realized_source_events", pa.int64()),
            ("elapsed_seconds", pa.float64()),
            ("raw_score", pa.float64()),
            ("calibration_alpha", pa.float64()),
            ("calibrated_score", pa.float64()),
            ("bounded_log99_score", pa.float64()),
        ]
    ]
)
EVALUATION_FIELDS = {
    "status",
    "run_unit_id",
    "real_or_null",
    "train_regime",
    "test_regime",
    "calibration_contract",
    "calibration_alpha",
    "calibration_paired_log_loss",
    "score_antisymmetry_max_abs_error",
    "permutation_max_abs_error",
    "prediction_rows",
    "date_metrics",
}
DATE_METRIC_FIELDS = {
    "source_date",
    "row_scope",
    "test_component",
    "score_contract",
    "n_paths",
    "n_blocks",
    "realized_source_events",
    "elapsed_seconds",
    "nats_path",
    "nats_block",
    "nats_event",
    "nats_second",
    "paired_log_loss",
    "posterior_saturation_share",
    "calibrated_ovl",
    "reverse_like_share",
}
PREFLIGHT_SCHEMA = pa.schema(
    [
        pa.field(name, dtype, nullable=False)
        for name, dtype in [
            ("source_date", pa.date32()),
            ("split", pa.string()),
            ("open_paths", pa.int64()),
            ("pre_close_paths", pa.int64()),
            ("daytime_paths", pa.int64()),
        ]
    ]
)
UNIT_AUDIT_FIELDS = {
    "selection",
    "final_fit",
    "source_calibration",
    "target_calibration",
    "all_evaluations",
    "null_audit",
    "null_bias",
    "resume",
}
DENOMINATOR_CONTRACT = "nonnegative_path_positive_date_component_v1"


def _validate_unit_audit(audit: Any) -> dict[str, bool]:
    if not isinstance(audit, dict) or set(audit) != UNIT_AUDIT_FIELDS:
        raise RuntimeError("unit audit exact field contract failed")
    if any(type(value) is not bool for value in audit.values()):
        raise RuntimeError("unit audit fields must be bool")
    if not all(audit.values()):
        raise RuntimeError("unit audit contains a failed gate")
    return audit


@dataclass(frozen=True)
class UnitConfig:
    train_regime: str
    model: str
    seed: int
    device: str
    edge_root: Path
    daytime_root: Path
    output_root: Path
    work_root: Path
    log_root: Path
    resume: bool = True
    hourly_cost_usd: float = 0.69
    batch_size: int = 1024
    scaler_paths_per_stratum: int = 10000
    max_epochs: int = 30
    min_epochs: int = 6
    patience: int = 3
    checkpoint_batches: int = 50
    stop_path: Path | None = None
    local_cache_root: Path | None = None

    @property
    def unit_id(self) -> str:
        return f"phase05/{self.train_regime}/{self.model}/seed={self.seed}"

    @property
    def storage(self) -> Path:
        return (
            Path(f"train_regime={self.train_regime}") / self.model / f"seed={self.seed}"
        )

    def scientific(self) -> dict[str, Any]:
        return {
            "train_regime": self.train_regime,
            "test_regimes": ["edge", "daytime"],
            "B": 10,
            "K": 6,
            "model": self.model,
            "seed": self.seed,
            "dates": list(DATES),
            "split": {k: list(v) for k, v in SPLIT.items()},
            "feature_config": "full_37",
            "stride": 3,
            "offset": "date.toordinal()%3",
            "batch_size": self.batch_size,
            "scaler_paths_per_stratum": self.scaler_paths_per_stratum,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "candidates": list(UPSTREAM.ADAMW_CANDIDATES),
            "null_seed": UPSTREAM.NULL_SEED,
        }


def _stop_requested(stop_path: Path | None = None) -> bool:
    return bool(
        getattr(UPSTREAM, "STOP_REQUESTED", False)
        or (stop_path is not None and stop_path.exists())
    )


def _runtime_config(device: torch.device, batch_size: int) -> dict[str, Any]:
    value = {
        "implementation": "phase05_regime_transfer_v2",
        "device_type": device.type,
        "device_index": 0
        if device.type == "cuda" and device.index is None
        else device.index,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "batch_size": batch_size,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "amp": False,
        "tf32": False,
        "compile": False,
    }
    if device.type == "cuda":
        value.update(
            {
                "visible_cuda_devices": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_rng_states": torch.cuda.device_count(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
    else:
        value.update(
            {
                "visible_cuda_devices": 0,
                "gpu_name": None,
                "gpu_capability": None,
                "cuda_rng_states": 0,
            }
        )
    return value


@dataclass
class Stratum:
    source_date: str
    component: str
    segment: int
    day: Any
    starts: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.source_date}/{self.component}/{self.segment:04d}"


def _paths(root: Path, d: str, component: str) -> list[tuple[int, list[Path]]]:
    if component == "daytime":
        return [
            (0, sorted((root / f"source_date={d}" / "target_b=10").glob("*.parquet")))
        ]
    base = root / f"source_date={d}" / f"regime={component}"
    return [
        (int(x.name.split("=")[1]), sorted((x / "target_b=10").glob("*.parquet")))
        for x in sorted(base.glob("segment=*"))
    ]


def _read(files: list[Path], d: str) -> Any:
    if not files:
        raise FileNotFoundError(f"missing Phase05 input {d}")
    tables = []
    footer_rows = 0
    for p in files:
        pf = pq.ParquetFile(p)
        if not pf.schema_arrow.equals(CANONICAL_BLOCK_SCHEMA, check_metadata=True):
            raise RuntimeError(f"schema mismatch: {p}")
        footer_rows += pf.metadata.num_rows
        tables.append(pf.read(columns=list(UPSTREAM.REQUIRED_COLUMNS)))
    t = pa.concat_tables(tables, promote_options="default")
    order = np.asarray(t["block_index"], dtype=np.int64)
    pos = np.argsort(order, kind="stable")
    order = order[pos]
    if len(t) != footer_rows:
        raise RuntimeError(f"decoded/footer row mismatch {d}")
    if len(order) > 1 and np.any(np.diff(order) != 1):
        raise RuntimeError(f"noncontiguous blocks {d}")
    vec = lambda n: np.asarray(t[n], dtype=np.int64)[pos]
    mat = lambda ns: np.column_stack(
        [np.asarray(t[n], dtype=np.float64)[pos] for n in ns]
    )
    starts = mat([f"start_{x}" for x in UPSTREAM.BOUNDARY_FIELDS])
    ends = mat([f"end_{x}" for x in UPSTREAM.BOUNDARY_FIELDS])
    intervals = mat(UPSTREAM.INTERVAL_FIELDS)
    if not np.isfinite(np.concatenate((starts, ends, intervals), axis=1)).all():
        raise RuntimeError("nonfinite input")
    if len(order) > 1 and not np.allclose(ends[:-1], starts[1:], rtol=0, atol=1e-6):
        raise RuntimeError("boundary mismatch")
    if np.any(vec("block_end_ns") < vec("block_start_ns")):
        raise RuntimeError(f"negative block duration {d}")
    realized = intervals[:, UPSTREAM.INTERVAL_INDEX["realized_source_event_count"]]
    overshoot = intervals[:, UPSTREAM.INTERVAL_INDEX["completion_overshoot"]]
    if np.any(realized < 10) or not np.array_equal(overshoot, realized - 10):
        raise RuntimeError(f"source-event/overshoot accounting mismatch {d}")
    return UPSTREAM.DayBlocks(
        d, order, vec("block_start_ns"), vec("block_end_ns"), starts, ends, intervals
    )


def strata(
    regime: str, dates: Iterable[str], edge_root: Path, daytime_root: Path
) -> list[Stratum]:
    out = []
    for d in dates:
        for c in COMPONENTS[regime]:
            for seg, files in _paths(
                daytime_root if c == "daytime" else edge_root, d, c
            ):
                day = _read(files, d)
                starts = UPSTREAM.day_path_starts(day, 6)
                if len(starts):
                    out.append(Stratum(d, c, seg, day, starts))
    return out


def _group_strata(items: list[Stratum]) -> dict[tuple[str, str], list[Stratum]]:
    grouped: dict[tuple[str, str], list[Stratum]] = {}
    for item in items:
        grouped.setdefault((item.source_date, item.component), []).append(item)
    return grouped


def _scaler_object(
    scaler: Any,
    *,
    train_dates: Iterable[str],
    regime: str,
    items: list[Stratum],
    limit: int = 10000,
) -> dict[str, Any]:
    counts = {
        f"{d}/{c}": min(sum(len(x.starts) for x in xs), limit)
        for (d, c), xs in sorted(_group_strata(items).items())
    }
    if int(scaler.sample_tokens) != 2 * 6 * sum(counts.values()):
        raise RuntimeError("scaler sample-token lineage mismatch")
    return {
        **scaler.to_dict(),
        "train_dates": list(train_dates),
        "regime": regime,
        "sample_paths_by_date_component": counts,
    }


def _quarantine(path: Path, quarantine_root: Path, reason: str) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = (
        quarantine_root
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    target.mkdir()
    os.replace(path, target / path.name)
    atomic_json(
        target / "quarantine.json", {"reason": reason, "original_path": str(path)}
    )
    return target


@contextmanager
def _unit_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    owner = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 65536)
        os.close(fd)
        owner = raw.decode(errors="replace").strip() or "unknown owner"
        raise RuntimeError(f"unit lock already held: {path}; owner={owner}") from exc
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        previous_raw = os.read(fd, 65536)
        previous = None
        if previous_raw:
            try:
                previous = json.loads(previous_raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                previous = {"unparsed_owner": previous_raw.decode(errors="replace")}
        proc_start = None
        try:
            proc_start = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
        except (OSError, IndexError):
            pass
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_start_ticks": proc_start,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "previous_owner": previous,
        }
        payload = (
            json.dumps(owner, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)
        os.fsync(fd)
        yield
    finally:
        released = {**owner, "released_utc": datetime.now(timezone.utc).isoformat()}
        payload = (
            json.dumps(released, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)
        os.fsync(fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _lock_fixture_holder(path: str, ready: Any, release: Any) -> None:
    with _unit_lock(Path(path)):
        ready.set()
        release.wait(10.0)


def input_preflight(
    edge_root: Path, daytime_root: Path, output: Path | None = None
) -> dict[str, Any]:
    rows = []
    totals = {"open": 0, "pre_close": 0, "daytime": 0}
    for d in DATES:
        observed = []
        for c in ("open", "pre_close", "daytime"):
            count = sum(
                len(s.starts)
                for s in strata(
                    "daytime" if c == "daytime" else "edge",
                    (d,),
                    edge_root,
                    daytime_root,
                )
                if s.component == c
            )
            observed.append(count)
            totals[c] += count
        if tuple(observed) != EXPECTED[d]:
            raise RuntimeError(
                f"path inventory mismatch {d}: {observed} != {EXPECTED[d]}"
            )
        rows.append(
            {
                "source_date": date.fromisoformat(d),
                "split": next(k for k, v in SPLIT.items() if d in v),
                "open_paths": observed[0],
                "pre_close_paths": observed[1],
                "daytime_paths": observed[2],
            }
        )
    table = pa.Table.from_pylist(rows, schema=PREFLIGHT_SCHEMA)
    if (
        len(table) != 15
        or not table.schema.equals(PREFLIGHT_SCHEMA)
        or any(col.null_count for col in table.columns)
    ):
        raise RuntimeError("input preflight exact schema/non-null contract failed")
    if len(set(table["source_date"].to_pylist())) != 15:
        raise RuntimeError("input preflight source_date key is not unique")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, output)
    return {
        "status": "complete",
        "rows": 15,
        "totals": totals,
        "edge_root": str(edge_root),
        "daytime_root": str(daytime_root),
    }


def fit_scaler(
    items: list[Stratum],
    limit: int,
    stop_path: Path | None = None,
    checkpoint: Path | None = None,
    checkpoint_config: dict[str, Any] | None = None,
) -> Any:
    groups = sorted(_group_strata(items).items())
    fs = []
    rs = []
    cursor = 0
    if checkpoint is not None and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            saved.get("config") != checkpoint_config
            or saved.get("group_keys") != [list(key) for key, _ in groups]
            or saved.get("limit") != limit
        ):
            raise RuntimeError("scaler checkpoint mismatch")
        fs = saved["forward_samples"]
        rs = saved["reverse_samples"]
        cursor = int(saved["next_group"])
    for group_index, (_, members) in enumerate(groups[cursor:], cursor):
        if _stop_requested(stop_path):
            if checkpoint is not None:
                UPSTREAM.atomic_torch_save(
                    checkpoint,
                    {
                        "config": checkpoint_config,
                        "group_keys": [list(key) for key, _ in groups],
                        "limit": limit,
                        "next_group": group_index,
                        "forward_samples": fs,
                        "reverse_samples": rs,
                    },
                )
            raise InterruptedError("stop requested during scaler fit")
        total = sum(len(s.starts) for s in members)
        n = min(total, limit)
        selected = np.linspace(0, total - 1, n, dtype=np.int64)
        offset = 0
        for s in members:
            local = (
                selected[(selected >= offset) & (selected < offset + len(s.starts))]
                - offset
            )
            if len(local):
                f, r, _ = UPSTREAM.path_arrays(s.day, s.starts[local], 6)
                fs.append(f)
                rs.append(r)
            offset += len(s.starts)
        if checkpoint is not None:
            UPSTREAM.atomic_torch_save(
                checkpoint,
                {
                    "config": checkpoint_config,
                    "group_keys": [list(key) for key, _ in groups],
                    "limit": limit,
                    "next_group": group_index + 1,
                    "forward_samples": fs,
                    "reverse_samples": rs,
                },
            )
    scaler = UPSTREAM.fit_channel_scaler(np.concatenate(fs), np.concatenate(rs))
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    return scaler


def _rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore(x: dict[str, Any]) -> None:
    random.setstate(x["python"])
    np.random.set_state(x["numpy"])
    torch.set_rng_state(x["torch"])
    if x["cuda"]:
        torch.cuda.set_rng_state_all(x["cuda"])


def null_mask(
    s: Stratum, starts: np.ndarray, null_seed: int | None
) -> np.ndarray | None:
    if null_seed is None:
        return None
    component_code = {"open": 1, "pre_close": 2, "daytime": 3}[s.component]
    base = (
        date.fromisoformat(s.source_date).toordinal() * 1000003
        + component_code * 1009
        + s.segment * 9176
        + null_seed
    ) & ((1 << 64) - 1)
    x = starts.astype(np.uint64, copy=False) * np.uint64(
        0x9E3779B185EBCA87
    ) + np.uint64(base)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    return (x & np.uint64(1)).astype(bool)


def _move_opt(opt: Any, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _epoch_schedule(
    items: list[Stratum],
    regime: str,
    seed: int,
    epoch: int,
    batch_size: int,
    is_null: bool,
) -> list[tuple[Stratum, np.ndarray]]:
    groups = _group_strata(items)
    quota = (
        min(sum(len(s.starts) for s in members) for members in groups.values())
        if regime == "edge"
        else None
    )
    streams = []
    for gi, (_, members) in enumerate(sorted(groups.items())):
        refs = [(s, start) for s in members for start in s.starts]
        order = np.random.default_rng(
            UPSTREAM._batch_seed(seed, epoch, gi, is_null)
        ).permutation(len(refs))
        if quota is not None:
            order = order[:quota]
        # A batch never crosses a physical segment, preserving one DayBlocks owner.
        batches = []
        pending: dict[int, list[int]] = {}
        owner = {id(s): s for s in members}
        for oi in order:
            s, start = refs[int(oi)]
            bucket = pending.setdefault(id(s), [])
            bucket.append(int(start))
            if len(bucket) == batch_size:
                batches.append((s, np.asarray(bucket, dtype=np.int64)))
                pending[id(s)] = []
        for sid, bucket in pending.items():
            if bucket:
                batches.append((owner[sid], np.asarray(bucket, dtype=np.int64)))
        streams.append(batches)
    return [
        stream[bi]
        for bi in range(max(map(len, streams)))
        for stream in streams
        if bi < len(stream)
    ]


def evaluate_loss(
    model: Any,
    items: list[Stratum],
    scaler: Any,
    batch: int,
    null_seed: int | None = None,
    stop_path: Path | None = None,
) -> float:
    model.eval()
    losses: dict[tuple[str, str], list[float]] = {}
    with torch.no_grad():
        for s in items:
            total = 0.0
            n = 0
            for i in range(0, len(s.starts), batch):
                if _stop_requested(stop_path):
                    raise InterruptedError("stop requested during fold evaluation")
                x = s.starts[i : i + batch]
                f, r, _ = UPSTREAM.path_arrays(s.day, x, 6)
                mask = null_mask(s, x, null_seed)
                f, r = UPSTREAM.orient_pair(f, r, mask)
                score = UPSTREAM.antisymmetric_score(
                    model,
                    torch.from_numpy(scaler.transform(f)).to(
                        next(model.parameters()).device
                    ),
                    torch.from_numpy(scaler.transform(r)).to(
                        next(model.parameters()).device
                    ),
                )
                total += float(F.softplus(-score).sum().cpu())
                n += len(x)
            slot = losses.setdefault((s.source_date, s.component), [0.0, 0.0])
            slot[0] += total
            slot[1] += n
    component_loss = {(d, c): v[0] / v[1] for (d, c), v in losses.items()}
    date_loss = {
        d: float(
            np.mean([value for (day, _), value in component_loss.items() if day == d])
        )
        for d, _ in component_loss
    }
    return float(np.mean(list(date_loss.values())))


def train_stage(
    *,
    device: torch.device,
    model_kind: str,
    seed: int,
    items: list[Stratum],
    evaluation: list[Stratum],
    scaler: Any,
    candidate: dict[str, float],
    epochs: int,
    min_epochs: int,
    patience: int,
    fixed: bool,
    batch_size: int,
    checkpoint: Path,
    checkpoint_config: dict[str, Any],
    resume: bool,
    null_seed: int | None = None,
    interrupt_after_batches: int | None = None,
    stop_path: Path | None = None,
    checkpoint_batches: int = 50,
) -> tuple[Any, list, int, float]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = UPSTREAM.build_model(model_kind).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=candidate["weight_decay"],
    )
    state = None
    runtime_config = _runtime_config(device, batch_size)
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["config"] != checkpoint_config
            or state["scaler"] != scaler.to_dict()
            or state.get("runtime_config") != runtime_config
        ):
            raise RuntimeError("checkpoint config/scaler/runtime mismatch")
        if (
            len(state.get("rng", {}).get("cuda", ()))
            != runtime_config["cuda_rng_states"]
        ):
            raise RuntimeError("checkpoint CUDA RNG topology mismatch")
    epoch = global_batch = 0
    epoch_loss_sum = 0.0
    epoch_rows = 0
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    seen = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        _move_opt(opt, device)
        epoch = state["cursor"]["epoch"]
        global_batch = state["cursor"]["batch_index"]
        epoch_loss_sum = state["epoch_loss_sum"]
        epoch_rows = state["epoch_rows"]
        history = state["history"]
        best_loss = state["best_loss"]
        best_epoch = state["best_epoch"]
        best_state = state["best_state"]
        stale = state["stale"]
        _restore(state["rng"])
    schedule: list[tuple[Stratum, np.ndarray]] = []

    def save(e: int, b: int) -> None:
        next_s, next_x = schedule[b] if b < len(schedule) else (None, None)
        cursor = {
            "stage": checkpoint_config["stage"],
            "fold": checkpoint_config.get("fold"),
            "candidate_index": checkpoint_config.get("candidate_index"),
            "epoch": e,
            "stratum_index": int(b),
            "batch_index": int(b),
            "source_date": None if next_s is None else next_s.source_date,
            "component": None if next_s is None else next_s.component,
            "segment": None if next_s is None else int(next_s.segment),
            "path_start_block": None
            if next_s is None
            else int(next_s.day.block_index[next_x[0]]),
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        UPSTREAM.atomic_torch_save(
            checkpoint,
            {
                "config": checkpoint_config,
                "runtime_config": runtime_config,
                "scaler": scaler.to_dict(),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cursor": cursor,
                "epoch_loss_sum": epoch_loss_sum,
                "epoch_rows": epoch_rows,
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "stale": stale,
                "rng": _rng(),
            },
        )

    while epoch < epochs:
        schedule = _epoch_schedule(
            items,
            checkpoint_config["regime"],
            seed,
            epoch,
            batch_size,
            null_seed is not None,
        )
        if state is not None and global_batch:
            if global_batch >= len(schedule):
                raise RuntimeError("training checkpoint batch cursor outside epoch")
            expected_s, expected_x = schedule[global_batch]
            saved = state["cursor"]
            expected_cursor = (
                expected_s.source_date,
                expected_s.component,
                int(expected_s.segment),
                int(expected_s.day.block_index[expected_x[0]]),
            )
            if (
                saved["source_date"],
                saved["component"],
                saved["segment"],
                saved["path_start_block"],
            ) != expected_cursor:
                raise RuntimeError("training checkpoint physical cursor mismatch")
            state = None
        model.train()
        for bi in range(global_batch, len(schedule)):
            s, x = schedule[bi]
            f, r, _ = UPSTREAM.path_arrays(s.day, x, 6)
            mask = null_mask(s, x, null_seed)
            f, r = UPSTREAM.orient_pair(f, r, mask)
            score = UPSTREAM.antisymmetric_score(
                model,
                torch.from_numpy(scaler.transform(f)).to(device),
                torch.from_numpy(scaler.transform(r)).to(device),
            )
            loss = F.softplus(-score).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu()) * len(x)
            epoch_rows += len(x)
            seen += 1
            if (bi + 1) % checkpoint_batches == 0 or _stop_requested(stop_path):
                save(epoch, bi + 1)
            if _stop_requested(stop_path):
                raise InterruptedError("stop requested at safe training batch boundary")
            if interrupt_after_batches is not None and seen >= interrupt_after_batches:
                save(epoch, bi + 1)
                raise InterruptedError("synthetic interruption")
        monitor = (
            evaluate_loss(model, evaluation, scaler, batch_size, null_seed, stop_path)
            if evaluation
            else epoch_loss_sum / epoch_rows
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss_sum / epoch_rows,
                "monitor": monitor,
            }
        )
        if fixed or monitor < best_loss - 1e-9:
            best_loss = monitor
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        epoch += 1
        global_batch = 0
        epoch_loss_sum = 0.0
        epoch_rows = 0
        schedule = []
        save(epoch, 0)
        if not fixed and epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def collect_scores(
    model: Any,
    items: list[Stratum],
    scaler: Any,
    batch: int,
    null_seed: int | None = None,
    stop_path: Path | None = None,
    checkpoint: Path | None = None,
    checkpoint_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    model.eval()
    device = next(model.parameters()).device
    item_cursor = batch_cursor = 0
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    runtime = _runtime_config(device, batch)
    if checkpoint is not None and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        states_equal = set(saved.get("model_state", {})) == set(model_state) and all(
            torch.equal(saved["model_state"][k], v) for k, v in model_state.items()
        )
        if (
            saved.get("config") != checkpoint_config
            or saved.get("scaler") != scaler.to_dict()
            or saved.get("runtime_config") != runtime
            or not states_equal
        ):
            raise RuntimeError("calibration checkpoint model/scaler/runtime mismatch")
        pieces = saved["pieces"]
        item_cursor = int(saved["item_cursor"])
        batch_cursor = int(saved["batch_cursor"])

    def save(si: int, bi: int) -> None:
        if checkpoint is not None:
            UPSTREAM.atomic_torch_save(
                checkpoint,
                {
                    "config": checkpoint_config,
                    "runtime_config": runtime,
                    "scaler": scaler.to_dict(),
                    "model_state": model_state,
                    "pieces": pieces,
                    "item_cursor": si,
                    "batch_cursor": bi,
                },
            )

    with torch.no_grad():
        for si, s in enumerate(items[item_cursor:], item_cursor):
            key = f"{s.source_date}/{s.component}"
            pieces.setdefault(key, [])
            first_batch = batch_cursor if si == item_cursor else 0
            total_batches = math.ceil(len(s.starts) / batch)
            for bi in range(first_batch, total_batches):
                i = bi * batch
                x = s.starts[i : i + batch]
                f, r, _ = UPSTREAM.path_arrays(s.day, x, 6)
                mask = null_mask(s, x, null_seed)
                f, r = UPSTREAM.orient_pair(f, r, mask)
                pieces[key].append(
                    UPSTREAM.antisymmetric_score(
                        model,
                        torch.from_numpy(scaler.transform(f)).to(device),
                        torch.from_numpy(scaler.transform(r)).to(device),
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                next_si, next_bi = (
                    (si, bi + 1) if bi + 1 < total_batches else (si + 1, 0)
                )
                save(next_si, next_bi)
                if _stop_requested(stop_path):
                    raise InterruptedError(
                        "stop requested at safe calibration batch boundary"
                    )
            batch_cursor = 0
    return {key: np.concatenate(value) for key, value in pieces.items()}


def fit_alpha(
    scores: dict[str, np.ndarray], stop_path: Path | None = None
) -> tuple[float, dict[str, Any]]:
    if _stop_requested(stop_path):
        raise InterruptedError("stop requested before calibration optimization")
    if not scores or any(
        len(x) == 0 or not np.isfinite(x).all() for x in scores.values()
    ):
        raise ValueError("calibration requires finite nonempty date-component scores")
    grouped = {key.split("/", 1)[0]: [] for key in scores}
    for key, value in scores.items():
        grouped[key.split("/", 1)[0]].append(value)

    def objective(alpha: float) -> float:
        return float(
            np.mean(
                [
                    np.mean(
                        [
                            np.logaddexp(0.0, -np.clip(alpha * x, -80.0, 80.0)).mean()
                            for x in xs
                        ]
                    )
                    for xs in grouped.values()
                ]
            )
        )

    def derivative(alpha: float) -> float:
        return float(
            np.mean(
                [
                    np.mean(
                        [
                            np.mean(
                                -x / (1.0 + np.exp(np.clip(alpha * x, -60.0, 60.0)))
                            )
                            for x in xs
                        ]
                    )
                    for xs in grouped.values()
                ]
            )
        )

    d0 = derivative(0.0)
    upper = 1.0
    iterations = 0
    at_zero = d0 >= 0.0
    at_upper = False
    if at_zero:
        alpha = 0.0
        du = d0
    else:
        du = derivative(upper)
        while du < 0 and upper < 1_000_000.0:
            if _stop_requested(stop_path):
                raise InterruptedError("stop requested during calibration optimization")
            upper *= 2.0
            du = derivative(upper)
        if du < 0:
            alpha = upper
            at_upper = True
        else:
            lower = 0.0
            for iterations in range(1, 101):
                if _stop_requested(stop_path):
                    raise InterruptedError(
                        "stop requested during calibration optimization"
                    )
                middle = 0.5 * (lower + upper)
                if derivative(middle) < 0:
                    lower = middle
                else:
                    upper = middle
                if upper - lower <= 1e-10 * max(1.0, upper):
                    break
            alpha = 0.5 * (lower + upper)
    loss = objective(alpha)
    if not (np.isfinite(alpha) and alpha >= 0 and np.isfinite(loss)):
        raise RuntimeError("invalid calibration result")
    if alpha == 0.0 and abs(loss - math.log(2.0)) > 1e-12:
        raise RuntimeError("alpha=0 calibration identity failed")
    by = {
        key: {
            "paths": len(value),
            "paired_log_loss": float(
                np.logaddexp(0.0, -np.clip(alpha * value, -80.0, 80.0)).mean()
            ),
        }
        for key, value in scores.items()
    }
    return alpha, {
        "criterion": "date-equal; Edge component-then-date-equal paired log loss; alpha>=0",
        "alpha": alpha,
        "paired_log_loss": loss,
        "day_equal_paired_log_loss": loss,
        "at_lower_boundary_alpha_zero": at_zero,
        "at_upper_numerical_boundary": at_upper,
        "derivative_at_zero": d0,
        "derivative_at_upper_bracket": du,
        "upper_bracket": upper,
        "bisection_iterations": iterations,
        "by_date_component": by,
    }


def _metric_row(acc: dict[str, Any], contract: str) -> dict[str, Any]:
    m = UPSTREAM._finalize_metrics(acc, 6)
    return {
        "score_contract": contract,
        "n_paths": int(m["paths"]),
        "n_blocks": int(m["paths"] * 6),
        "realized_source_events": int(m["source_event_sum"]),
        "elapsed_seconds": float(m["duration_seconds_sum"]),
        "nats_path": float(m["nats_path"]),
        "nats_block": float(m["nats_block"]),
        "nats_event": float(m["nats_source_event"]),
        "nats_second": float(m["nats_second"]),
        "paired_log_loss": float(m["paired_log_loss"]),
        "posterior_saturation_share": float(m["posterior_saturation_share"]),
        "calibrated_ovl": float(m["calibrated_ovl"]),
        "reverse_like_share": float(m["reverse_like_share"]),
    }


def _reverse_tokens(tokens: np.ndarray) -> np.ndarray:
    b = len(UPSTREAM.BOUNDARY_FIELDS)
    m = len(UPSTREAM.INTERVAL_FIELDS)
    boundaries = np.empty((len(tokens), tokens.shape[1] + 1, b), dtype=tokens.dtype)
    boundaries[:, 0] = tokens[:, 0, :b]
    boundaries[:, 1:] = tokens[:, :, -b:]
    intervals = tokens[:, :, b : b + m].copy()
    rb = boundaries[:, ::-1].copy()
    ri = intervals[:, ::-1].copy()
    ri[:, :, UPSTREAM.ODD_INDICES] *= -1
    for left, right in UPSTREAM.SWAP_PAIRS:
        ri[:, :, left], ri[:, :, right] = ri[:, :, right].copy(), ri[:, :, left].copy()
    return np.concatenate((rb[:, :-1], ri, rb[:, 1:]), axis=2)


def _validate_evaluation_contributions(
    events: np.ndarray, duration: np.ndarray
) -> None:
    """Allow zero path contributions; published aggregate denominators must be positive."""
    if np.any(events < 0) or np.any(duration < 0):
        raise RuntimeError("negative evaluation denominator contribution")


def evaluate(
    *,
    model: Any,
    items: list[Stratum],
    scaler: Any,
    alpha: float,
    calibration_loss: float,
    train_regime: str,
    test_regime: str,
    calibration_contract: str,
    real_or_null: str,
    model_kind: str,
    seed: int,
    batch_size: int,
    final_dir: Path,
    checkpoint: Path,
    quarantine_root: Path,
    null_seed: int | None = None,
    stop_path: Path | None = None,
    interrupt_after_batches: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    runtime_config = _runtime_config(device, batch_size)
    config = {
        "train_regime": train_regime,
        "test_regime": test_regime,
        "calibration_contract": calibration_contract,
        "real_or_null": real_or_null,
        "model": model_kind,
        "seed": seed,
        "alpha": alpha,
        "calibration_loss": calibration_loss,
        "scaler": scaler.to_dict(),
        "runtime_config": runtime_config,
        "denominator_contract": DENOMINATOR_CONTRACT,
    }
    complete = final_dir / "complete.json"
    if complete.is_file():
        value = json.loads(complete.read_text())
        if value.get("config") != config:
            raise RuntimeError(f"evaluation config mismatch {final_dir}")
        metrics = json.loads((final_dir / "metrics.json").read_text())
        if set(metrics) != EVALUATION_FIELDS:
            raise RuntimeError("published metrics schema mismatch")
        return metrics
    if final_dir.exists():
        _quarantine(final_dir, quarantine_root, "evaluation lacks valid completion")
    tasks = [(s, first) for s in items for first in range(0, len(s.starts), batch_size)]
    state = None
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            stale_attempt = Path(state.get("attempt_dir", ""))
            _quarantine(checkpoint, quarantine_root, "stale evaluation checkpoint")
            if stale_attempt.is_dir():
                _quarantine(stale_attempt, quarantine_root, "stale evaluation partial")
            state = None
    if state:
        tmp = Path(state["attempt_dir"])
        if not tmp.is_dir():
            raise RuntimeError("evaluation checkpoint attempt directory missing")
        cursor = int(state["cursor"]["batch_index"])
        accs = state["accumulators"]
        max_anti = float(state["max_raw_antisymmetry"])
        perm = state["permutation_max_abs_error"]
        inventory = list(state["prediction_inventory"])
        null_swapped = int(state.get("null_swapped", 0))
        null_total = int(state.get("null_total", 0))
        involution_passed = bool(state.get("reverse_involution", True))
    else:
        tmp = final_dir.with_name(f".{final_dir.name}.attempt-{uuid.uuid4().hex}")
        tmp.mkdir(parents=True)
        cursor = 0
        accs = {}
        max_anti = 0.0
        perm = None
        inventory = []
        null_swapped = null_total = 0
        involution_passed = True
    shards = tmp / "prediction_shards"
    shards.mkdir(exist_ok=True)
    if state:
        expected_files = {entry["file"] for entry in inventory}
        actual_files = {p.name for p in shards.glob("*.parquet")}
        if expected_files != actual_files or len(expected_files) != len(inventory):
            raise RuntimeError("evaluation checkpoint prediction inventory mismatch")
        for entry in inventory:
            pf = pq.ParquetFile(shards / entry["file"])
            if (
                not pf.schema_arrow.equals(PREDICTION_SCHEMA, check_metadata=True)
                or pf.metadata.num_rows != entry["rows"]
            ):
                raise RuntimeError("evaluation checkpoint shard mismatch")
    device = next(model.parameters()).device
    model.eval()

    def save(next_cursor: int) -> None:
        if next_cursor < len(tasks):
            next_s, next_first = tasks[next_cursor]
            next_start = int(next_s.starts[next_first])
            next_values = (
                next_s.source_date,
                next_s.component,
                int(next_s.segment),
                int(next_s.day.block_index[next_start]),
            )
        else:
            next_values = (None, None, None, None)
        cursor_object = {
            "stage": f"{real_or_null}_{test_regime}_{calibration_contract}_evaluation",
            "fold": None,
            "candidate_index": None,
            "epoch": 0,
            "stratum_index": int(next_cursor),
            "batch_index": int(next_cursor),
            "source_date": next_values[0],
            "component": next_values[1],
            "segment": next_values[2],
            "path_start_block": next_values[3],
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        UPSTREAM.atomic_torch_save(
            checkpoint,
            {
                "config": config,
                "runtime_config": runtime_config,
                "attempt_dir": str(tmp),
                "cursor": cursor_object,
                "accumulators": accs,
                "max_raw_antisymmetry": max_anti,
                "permutation_max_abs_error": perm,
                "prediction_inventory": inventory,
                "null_swapped": null_swapped,
                "null_total": null_total,
                "reverse_involution": involution_passed,
            },
        )

    try:
        with torch.no_grad():
            for ti in range(cursor, len(tasks)):
                s, first = tasks[ti]
                x = s.starts[first : first + batch_size]
                physical_f, physical_r, events = UPSTREAM.path_arrays(s.day, x, 6)
                mask = null_mask(s, x, null_seed)
                if real_or_null == "null" and (mask is None or len(mask) != len(x)):
                    raise RuntimeError("null pair orientation identity failed")
                if mask is not None:
                    null_swapped += int(mask.sum())
                    null_total += len(mask)
                involution_passed = (
                    involution_passed
                    and np.array_equal(
                        _reverse_tokens(_reverse_tokens(physical_f)), physical_f
                    )
                    and np.array_equal(_reverse_tokens(physical_f), physical_r)
                )
                f, r = UPSTREAM.orient_pair(physical_f, physical_r, mask)
                ft = torch.from_numpy(scaler.transform(f)).to(device)
                rt = torch.from_numpy(scaler.transform(r)).to(device)
                raw = (
                    UPSTREAM.antisymmetric_score(model, ft, rt)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                reverse = (
                    UPSTREAM.antisymmetric_score(model, rt, ft)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                calibrated = alpha * raw
                if not (
                    np.isfinite(raw).all()
                    and np.isfinite(reverse).all()
                    and np.isfinite(calibrated).all()
                ):
                    raise RuntimeError("nonfinite evaluation score")
                max_anti = max(max_anti, float(np.max(np.abs(raw + reverse))))
                duration = (s.day.block_end_ns[x + 5] - s.day.block_start_ns[x]) / 1e9
                _validate_evaluation_contributions(events, duration)
                key = (s.source_date, s.component)
                bucket = accs.setdefault(
                    key,
                    {
                        name: UPSTREAM._metric_accumulator()
                        for name in ("uncapped", "bounded_log99")
                    },
                )
                UPSTREAM._update_metrics(
                    bucket["uncapped"], calibrated, events, duration
                )
                UPSTREAM._update_metrics(
                    bucket["bounded_log99"],
                    np.clip(calibrated, -math.log(99), math.log(99)),
                    events,
                    duration,
                )
                if model_kind == "pooled_mlp" and perm is None:
                    perm = float(
                        torch.max(
                            torch.abs(
                                model(ft) - model(ft[:, torch.arange(5, -1, -1), :])
                            )
                        ).cpu()
                    )
                if real_or_null == "real":
                    rows = []
                    for direction, values in (("forward", raw), ("reverse", reverse)):
                        for j, start in enumerate(x):
                            score = float(values[j])
                            cal = alpha * score
                            rows.append(
                                {
                                    "pair_id": f"{s.source_date}:{s.component}:{s.segment:04d}:{int(s.day.block_index[start])}",
                                    "source_date": date.fromisoformat(s.source_date),
                                    "test_regime": test_regime,
                                    "test_component": s.component,
                                    "segment": s.segment,
                                    "path_start_block": int(s.day.block_index[start]),
                                    "direction": direction,
                                    "path_start_ns": int(s.day.block_start_ns[start]),
                                    "path_end_ns": int(s.day.block_end_ns[start + 5]),
                                    "realized_source_events": int(events[j]),
                                    "elapsed_seconds": float(duration[j]),
                                    "raw_score": score,
                                    "calibration_alpha": float(alpha),
                                    "calibrated_score": float(cal),
                                    "bounded_log99_score": float(
                                        np.clip(cal, -math.log(99), math.log(99))
                                    ),
                                }
                            )
                    table = pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)
                    if (
                        table.null_count
                        if hasattr(table, "null_count")
                        else sum(col.null_count for col in table.columns)
                    ):
                        raise RuntimeError("null prediction field")
                    shard = shards / f"part-{ti:08d}.parquet"
                    atomic_parquet(shard, table)
                    inventory.append({"file": shard.name, "rows": len(table)})
                save(ti + 1)
                if _stop_requested(stop_path):
                    raise InterruptedError(
                        "stop requested at safe evaluation batch boundary"
                    )
                if (
                    interrupt_after_batches is not None
                    and ti - cursor + 1 >= interrupt_after_batches
                ):
                    raise InterruptedError("synthetic evaluation interruption")
        by = {
            (d, c): {name: _metric_row(acc, name) for name, acc in buckets.items()}
            for (d, c), buckets in accs.items()
        }
        date_metrics = []
        for d in SPLIT["test"]:
            for contract in ("uncapped", "bounded_log99"):
                comp = []
                for component in COMPONENTS[test_regime]:
                    row = {
                        "source_date": d,
                        "row_scope": "component"
                        if test_regime == "edge"
                        else "aggregate",
                        "test_component": component if test_regime == "edge" else None,
                        **by[(d, component)][contract],
                    }
                    date_metrics.append(row)
                    comp.append(row)
                if test_regime == "edge":
                    fields = (
                        "nats_path",
                        "nats_block",
                        "nats_event",
                        "nats_second",
                        "paired_log_loss",
                        "posterior_saturation_share",
                        "calibrated_ovl",
                        "reverse_like_share",
                    )
                    agg = {
                        "source_date": d,
                        "row_scope": "aggregate",
                        "test_component": None,
                        "score_contract": contract,
                        "n_paths": sum(x["n_paths"] for x in comp),
                        "n_blocks": sum(x["n_blocks"] for x in comp),
                        "realized_source_events": sum(
                            x["realized_source_events"] for x in comp
                        ),
                        "elapsed_seconds": sum(x["elapsed_seconds"] for x in comp),
                    }
                    agg.update(
                        {f: float(np.mean([x[f] for x in comp])) for f in fields}
                    )
                    date_metrics.append(agg)
        if len(date_metrics) != (18 if test_regime == "edge" else 6):
            raise RuntimeError("date_metrics exact row count failed")
        numeric_fields = DATE_METRIC_FIELDS - {
            "source_date",
            "row_scope",
            "test_component",
            "score_contract",
        }
        for row in date_metrics:
            if (
                set(row) != DATE_METRIC_FIELDS
                or row["source_date"] not in SPLIT["test"]
                or row["row_scope"] not in {"aggregate", "component"}
                or row["score_contract"] not in {"uncapped", "bounded_log99"}
            ):
                raise RuntimeError("date_metrics schema/domain failed")
            if (row["row_scope"] == "aggregate") != (row["test_component"] is None):
                raise RuntimeError("date_metrics component nullability failed")
            if (
                any(not np.isfinite(row[name]) for name in numeric_fields)
                or row["n_paths"] <= 0
                or row["n_blocks"] <= 0
                or row["realized_source_events"] <= 0
                or row["elapsed_seconds"] <= 0
            ):
                raise RuntimeError("date_metrics finite/denominator audit failed")
        expected_paths = (
            sum(
                EXPECTED[d][2 if test_regime == "daytime" else i]
                for d in SPLIT["test"]
                for i in (() if test_regime == "daytime" else (0, 1))
            )
            if test_regime == "edge"
            else sum(EXPECTED[d][2] for d in SPLIT["test"])
        )
        prediction_rows = 0
        pair_metadata_passed = True
        score_algebra_passed = True
        if real_or_null == "real":
            pred_tmp = tmp / "test_predictions.parquet"
            writer = pq.ParquetWriter(pred_tmp, PREDICTION_SCHEMA, compression="zstd")
            try:
                for entry in inventory:
                    shard = shards / entry["file"]
                    table = pq.read_table(shard)
                    if (
                        not table.schema.equals(PREDICTION_SCHEMA, check_metadata=True)
                        or len(table) != entry["rows"]
                    ):
                        raise RuntimeError("prediction shard mismatch")
                    writer.write_table(table)
            finally:
                writer.close()
            prediction_rows = pq.ParquetFile(pred_tmp).metadata.num_rows
            audit = pq.read_table(pred_tmp)
            keys = list(
                zip(
                    audit["pair_id"].to_pylist(),
                    audit["direction"].to_pylist(),
                    strict=True,
                )
            )
            if len(keys) != len(set(keys)) or prediction_rows != 2 * expected_paths:
                raise RuntimeError("prediction key/row contract failed")
            raw_values = np.asarray(audit["raw_score"])
            calibrated_values = np.asarray(audit["calibrated_score"])
            bounded_values = np.asarray(audit["bounded_log99_score"])
            score_algebra_passed = bool(
                np.array_equal(calibrated_values, alpha * raw_values)
                and np.array_equal(
                    bounded_values,
                    np.clip(calibrated_values, -math.log(99), math.log(99)),
                )
            )
            pairs = {}
            metadata_names = [
                name
                for name in PREDICTION_SCHEMA.names
                if name
                not in {
                    "direction",
                    "raw_score",
                    "calibrated_score",
                    "bounded_log99_score",
                }
            ]
            metadata_columns = {
                name: audit[name].to_pylist() for name in metadata_names
            }
            for row_index, (pid, direction, raw_value, cal_value) in enumerate(
                zip(
                    audit["pair_id"].to_pylist(),
                    audit["direction"].to_pylist(),
                    audit["raw_score"].to_pylist(),
                    audit["calibrated_score"].to_pylist(),
                    strict=True,
                )
            ):
                pairs.setdefault(pid, {})[direction] = (
                    raw_value,
                    cal_value,
                    tuple(metadata_columns[name][row_index] for name in metadata_names),
                )
            if len(pairs) != expected_paths or any(
                set(v) != {"forward", "reverse"}
                or abs(v["forward"][0] + v["reverse"][0]) >= 1e-6
                or abs(v["forward"][1] + v["reverse"][1]) >= 1e-6
                for v in pairs.values()
            ):
                raise RuntimeError("prediction pair identity failed")
            pair_metadata_passed = all(
                v["forward"][2] == v["reverse"][2] for v in pairs.values()
            )
            if not pair_metadata_passed or not score_algebra_passed:
                raise RuntimeError("prediction paired metadata/score algebra failed")
            shutil.rmtree(shards)
        elif (
            sum(
                row["n_paths"]
                for row in date_metrics
                if row["row_scope"] == "aggregate"
                and row["score_contract"] == "uncapped"
            )
            != expected_paths
        ):
            raise RuntimeError("null pair count mismatch")
        metrics = {
            "status": "complete",
            "run_unit_id": f"phase05/{train_regime}/{model_kind}/seed={seed}",
            "real_or_null": real_or_null,
            "train_regime": train_regime,
            "test_regime": test_regime,
            "calibration_contract": calibration_contract,
            "calibration_alpha": float(alpha),
            "calibration_paired_log_loss": float(calibration_loss),
            "score_antisymmetry_max_abs_error": max_anti,
            "permutation_max_abs_error": perm,
            "prediction_rows": prediction_rows,
            "date_metrics": date_metrics,
        }
        if (
            set(metrics) != EVALUATION_FIELDS
            or not np.isfinite(alpha)
            or alpha < 0
            or not np.isfinite(calibration_loss)
            or max_anti >= 1e-6
            or (model_kind == "pooled_mlp" and (perm is None or perm >= 2e-5))
        ):
            raise RuntimeError("evaluation scientific/schema audit failed")
        evaluation_audit = {
            "pair_metadata_exact": pair_metadata_passed,
            "score_algebra_exact": score_algebra_passed,
            "reverse_involution_exact": involution_passed,
            "null_orientation_identity": real_or_null == "real"
            or null_total == expected_paths,
            "null_orientation_swapped": null_swapped,
            "null_orientation_total": null_total,
            "null_orientation_balance_passed": real_or_null == "real"
            or (0 < null_swapped < null_total),
            "passed": False,
        }
        evaluation_audit["passed"] = bool(
            all(
                value
                for key, value in evaluation_audit.items()
                if key != "passed"
                and (
                    key.endswith("exact")
                    or key.endswith("identity")
                    or key.endswith("passed")
                )
            )
        )
        if not evaluation_audit["passed"]:
            raise RuntimeError("evaluation durable structural audit failed")
        atomic_json(tmp / "metrics.json", metrics)
        atomic_json(
            tmp / "complete.json",
            {
                "status": "complete",
                "config": config,
                "prediction_rows": prediction_rows,
                "pair_count": expected_paths,
                "audit": evaluation_audit,
            },
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, final_dir)
        checkpoint.unlink(missing_ok=True)
        return metrics
    except Exception:
        saved_cursor = (
            cursor
            if not checkpoint.exists()
            else int(
                torch.load(checkpoint, map_location="cpu", weights_only=False)[
                    "cursor"
                ]["batch_index"]
            )
        )
        save(saved_cursor)
        raise


def _calibration(
    model: Any,
    regime: str,
    config: UnitConfig,
    scaler: Any,
    null_seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    kind = "null" if null_seed is not None else "real"
    checkpoint = (
        config.work_root
        / config.storage
        / "checkpoints"
        / f"calibration_{kind}_{regime}.pt"
    )
    checkpoint_config = {
        "stage": f"{kind}_{regime}_calibration",
        "regime": regime,
        "dates": list(SPLIT["calibration"]),
        "null_seed": null_seed,
    }
    alpha, result = fit_alpha(
        collect_scores(
            model,
            strata(regime, SPLIT["calibration"], config.edge_root, config.daytime_root),
            scaler,
            config.batch_size,
            null_seed,
            config.stop_path,
            checkpoint,
            checkpoint_config,
        ),
        config.stop_path,
    )
    checkpoint.unlink(missing_ok=True)
    result.update(
        {
            "regime": regime,
            "components": list(COMPONENTS[regime]),
            "calibration_dates": list(SPLIT["calibration"]),
            "model_frozen": True,
            "scaler_frozen": True,
            "raw_score_frozen": True,
        }
    )
    return alpha, result


def _run_unit_locked(config: UnitConfig) -> dict[str, Any]:
    if (
        config.train_regime not in COMPONENTS
        or config.model not in MODELS
        or config.seed not in SEEDS
    ):
        raise ValueError("unit outside Phase05 contract")
    configure_determinism()
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("production Phase05 requires qualified CUDA")
    pre = input_preflight(config.edge_root, config.daytime_root)
    final = config.output_root / config.storage
    work = config.work_root / config.storage
    if (final / "complete.json").is_file():
        complete = json.loads((final / "complete.json").read_text())
        if (
            complete.get("scientific_config") == config.scientific()
            and complete.get("evaluations") == 6
        ):
            try:
                _validate_unit_audit(complete.get("audit"))
                return complete
            except RuntimeError:
                pass
        _quarantine(
            final,
            config.output_root / "quarantine" / config.storage,
            "invalid unit completion",
        )
    elif final.exists():
        _quarantine(
            final,
            config.output_root / "quarantine" / config.storage,
            "unit final lacks completion",
        )
    work.mkdir(parents=True, exist_ok=True)
    exact = {"scientific": config.scientific(), "input": pre}
    cfg = work / "config.json"
    if cfg.is_file() and json.loads(cfg.read_text()) != exact:
        _quarantine(
            work,
            config.output_root / "quarantine" / config.storage,
            "stale unit work config",
        )
        work.mkdir(parents=True)
        cfg = work / "config.json"
    if not cfg.is_file():
        atomic_json(cfg, exact)
    attempt_record = work / "attempt.json"
    if attempt_record.is_file():
        staging = Path(json.loads(attempt_record.read_text())["staging"])
    else:
        staging = (
            work
            / f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
        )
        atomic_json(attempt_record, {"staging": str(staging)})
    staging.mkdir(parents=True, exist_ok=True)
    source = config.train_regime
    target = "daytime" if source == "edge" else "edge"
    dev = strata(source, SPLIT["development"], config.edge_root, config.daytime_root)
    quarantine = config.output_root / "quarantine" / config.storage

    def stage_state(stage_name: str, **cursor: Any) -> None:
        value = {
            "stage": stage_name,
            "fold": None,
            "candidate_index": None,
            "epoch": 0,
            "stratum_index": 0,
            "batch_index": 0,
            "source_date": None,
            "component": None,
            "segment": None,
            "path_start_block": None,
        }
        value.update(cursor)
        atomic_json(work / "stage_state.json", value)

    def admit_checkpoint(
        path: Path, stage_config: dict[str, Any], scaler_value: Any
    ) -> None:
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            runtime = _runtime_config(device, config.batch_size)
            if (
                saved.get("config") != stage_config
                or saved.get("scaler") != scaler_value.to_dict()
                or saved.get("runtime_config") != runtime
                or len(saved.get("rng", {}).get("cuda", ()))
                != runtime["cuda_rng_states"]
            ):
                _quarantine(
                    path, quarantine, "stale training/scaler/runtime checkpoint"
                )

    folds = (
        (DATES[:5], DATES[5]),
        (DATES[:6], DATES[6]),
        (DATES[:7], DATES[7]),
        (DATES[:8], DATES[8]),
    )
    fold_rows = []
    for fi, (train_dates, eval_date) in enumerate(folds, 1):
        fold_dir = staging / "selection" / f"fold={fi}"
        fold_items = [s for s in dev if s.source_date in train_dates]
        sc_path = fold_dir / "scaler.json"
        expected_lineage = {"train_dates": list(train_dates), "regime": source}
        if sc_path.is_file():
            sc_obj = json.loads(sc_path.read_text())
            scaler = UPSTREAM.ChannelScaler.from_dict(sc_obj)
            expected_obj = _scaler_object(
                scaler,
                train_dates=train_dates,
                regime=source,
                items=fold_items,
                limit=config.scaler_paths_per_stratum,
            )
            if sc_obj != expected_obj:
                raise RuntimeError("durable fold scaler lineage mismatch")
        else:
            scaler = fit_scaler(
                fold_items,
                config.scaler_paths_per_stratum,
                config.stop_path,
                work / "checkpoints" / f"fold{fi}_scaler.pt",
                {
                    "stage": "selection_scaler",
                    "fold": fi,
                    "train_dates": list(train_dates),
                    "regime": source,
                },
            )
            sc_obj = _scaler_object(
                scaler,
                train_dates=train_dates,
                regime=source,
                items=fold_items,
                limit=config.scaler_paths_per_stratum,
            )
            atomic_json(sc_path, sc_obj)
        for ci, candidate in enumerate(UPSTREAM.ADAMW_CANDIDATES):
            result = fold_dir / f"candidate={ci}.json"
            stage = {
                "stage": "selection",
                "regime": source,
                "fold": fi,
                "train_dates": list(train_dates),
                "evaluation_date": eval_date,
                "candidate_index": ci,
                "fold_scaler_path": str(sc_path.relative_to(staging)),
                "fold_scaler_object": sc_obj,
                "scaler": scaler.to_dict(),
            }
            candidate_checkpoint = work / f"fold{fi}_candidate{ci}.pt"
            row = json.loads(result.read_text()) if result.is_file() else None
            if row is not None and row.get("checkpoint_config") != stage:
                _quarantine(
                    result,
                    config.output_root / "quarantine" / config.storage,
                    "stale selection candidate result",
                )
                if candidate_checkpoint.exists():
                    _quarantine(
                        candidate_checkpoint,
                        config.output_root / "quarantine" / config.storage,
                        "stale selection candidate checkpoint",
                    )
                row = None
            if row is None:
                stage_state("selection", fold=fi, candidate_index=ci)
                admit_checkpoint(candidate_checkpoint, stage, scaler)
                _, history, be, bl = train_stage(
                    device=device,
                    model_kind=config.model,
                    seed=config.seed,
                    items=fold_items,
                    evaluation=[s for s in dev if s.source_date == eval_date],
                    scaler=scaler,
                    candidate=candidate,
                    epochs=config.max_epochs,
                    min_epochs=config.min_epochs,
                    patience=config.patience,
                    fixed=False,
                    batch_size=config.batch_size,
                    checkpoint=work / f"fold{fi}_candidate{ci}.pt",
                    checkpoint_config=stage,
                    resume=config.resume,
                    stop_path=config.stop_path,
                    checkpoint_batches=config.checkpoint_batches,
                )
                row = {
                    "checkpoint_config": stage,
                    "fold": fi,
                    "candidate_index": ci,
                    "best_epoch": be,
                    "best_evaluation_paired_log_loss": bl,
                    "history": history,
                }
                atomic_json(result, row)
            if row.get("checkpoint_config") != stage:
                raise RuntimeError("published selection config mismatch")
            fold_rows.append(row)
    selected, selection = UPSTREAM.select_candidate(fold_rows)
    selection["fold_scalers"] = [f"selection/fold={i}/scaler.json" for i in range(1, 5)]
    selection["fold_date_lineage"] = [
        {
            "fold": i,
            "train_dates": list(td),
            "evaluation_date": ed,
            "candidate_results": [
                f"selection/fold={i}/candidate={j}.json" for j in range(4)
            ],
        }
        for i, (td, ed) in enumerate(folds, 1)
    ]
    atomic_json(staging / "selection.json", selection)
    scaler_path = staging / "scaler.json"
    if scaler_path.is_file():
        final_scaler_object = json.loads(scaler_path.read_text())
        scaler = UPSTREAM.ChannelScaler.from_dict(final_scaler_object)
        if final_scaler_object != _scaler_object(
            scaler,
            train_dates=SPLIT["development"],
            regime=source,
            items=dev,
            limit=config.scaler_paths_per_stratum,
        ):
            raise RuntimeError("final scaler lineage mismatch")
    else:
        scaler = fit_scaler(
            dev,
            config.scaler_paths_per_stratum,
            config.stop_path,
            work / "checkpoints/final_scaler.pt",
            {
                "stage": "final_scaler",
                "train_dates": list(SPLIT["development"]),
                "regime": source,
            },
        )
        final_scaler_object = _scaler_object(
            scaler,
            train_dates=SPLIT["development"],
            regime=source,
            items=dev,
            limit=config.scaler_paths_per_stratum,
        )
        atomic_json(scaler_path, final_scaler_object)
    candidate = UPSTREAM.ADAMW_CANDIDATES[selected]
    epochs = selection["final_fit_epochs"]
    stage = {
        "stage": "final_real_fit",
        "regime": source,
        "scaler_path": "scaler.json",
        "scaler_object": final_scaler_object,
        "scaler": scaler.to_dict(),
        "train_dates": list(SPLIT["development"]),
        "epochs": epochs,
        "candidate": candidate,
    }
    stage_state("final_real_fit")
    admit_checkpoint(work / "final_real_fit.pt", stage, scaler)
    model, history, _, _ = train_stage(
        device=device,
        model_kind=config.model,
        seed=config.seed,
        items=dev,
        evaluation=[],
        scaler=scaler,
        candidate=candidate,
        epochs=epochs,
        min_epochs=epochs,
        patience=config.patience,
        fixed=True,
        batch_size=config.batch_size,
        checkpoint=work / "final_real_fit.pt",
        checkpoint_config=stage,
        resume=config.resume,
        stop_path=config.stop_path,
        checkpoint_batches=config.checkpoint_batches,
    )
    test_items = {
        r: strata(r, SPLIT["test"], config.edge_root, config.daytime_root)
        for r in COMPONENTS
    }

    def ev(
        *,
        which_model: Any,
        regime: str,
        contract: str,
        kind: str,
        a: float,
        cal: dict[str, Any],
        null: int | None = None,
    ) -> dict[str, Any]:
        tag = f"{kind}_{regime}_{contract}"
        stage_name = (
            f"{kind}_source_evaluation"
            if regime == source
            else f"{kind}_target_{contract}_evaluation"
        )
        stage_state(stage_name)
        return evaluate(
            model=which_model,
            items=test_items[regime],
            scaler=scaler,
            alpha=a,
            calibration_loss=float(cal["paired_log_loss"]),
            train_regime=source,
            test_regime=regime,
            calibration_contract=contract,
            real_or_null=kind,
            model_kind=config.model,
            seed=config.seed,
            batch_size=config.batch_size,
            final_dir=staging / f"evaluations/{kind}/test_regime={regime}/{contract}",
            checkpoint=work / "checkpoints" / f"{tag}.pt",
            quarantine_root=quarantine,
            null_seed=null,
            stop_path=config.stop_path,
        )

    stage_state("source_calibration")
    alpha_s, cal_s = _calibration(model, source, config, scaler)
    atomic_json(staging / "calibrations/real/source_alpha.json", cal_s)
    real = [
        ev(
            which_model=model,
            regime=source,
            contract="source_alpha",
            kind="real",
            a=alpha_s,
            cal=cal_s,
        ),
        ev(
            which_model=model,
            regime=target,
            contract="source_alpha",
            kind="real",
            a=alpha_s,
            cal=cal_s,
        ),
    ]
    stage_state("target_calibration")
    alpha_t, cal_t = _calibration(model, target, config, scaler)
    atomic_json(staging / "calibrations/real/target_alpha.json", cal_t)
    real.append(
        ev(
            which_model=model,
            regime=target,
            contract="target_alpha",
            kind="real",
            a=alpha_t,
            cal=cal_t,
        )
    )
    UPSTREAM.atomic_torch_save(
        staging / "model.pt",
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "history": history,
        },
    )
    nstage = {
        "stage": "final_null_fit",
        "regime": source,
        "scaler_object": final_scaler_object,
        "scaler": scaler.to_dict(),
        "train_dates": list(SPLIT["development"]),
        "epochs": epochs,
        "candidate": candidate,
        "null_seed": UPSTREAM.NULL_SEED,
    }
    stage_state("final_null_fit")
    admit_checkpoint(work / "final_null_fit.pt", nstage, scaler)
    null_model, nh, _, _ = train_stage(
        device=device,
        model_kind=config.model,
        seed=config.seed,
        items=dev,
        evaluation=[],
        scaler=scaler,
        candidate=candidate,
        epochs=epochs,
        min_epochs=epochs,
        patience=config.patience,
        fixed=True,
        batch_size=config.batch_size,
        checkpoint=work / "final_null_fit.pt",
        checkpoint_config=nstage,
        resume=config.resume,
        null_seed=UPSTREAM.NULL_SEED,
        stop_path=config.stop_path,
        checkpoint_batches=config.checkpoint_batches,
    )
    stage_state("null_source_calibration")
    na_s, nc_s = _calibration(null_model, source, config, scaler, UPSTREAM.NULL_SEED)
    atomic_json(staging / "calibrations/null/source_alpha.json", nc_s)
    null = [
        ev(
            which_model=null_model,
            regime=source,
            contract="source_alpha",
            kind="null",
            a=na_s,
            cal=nc_s,
            null=UPSTREAM.NULL_SEED,
        ),
        ev(
            which_model=null_model,
            regime=target,
            contract="source_alpha",
            kind="null",
            a=na_s,
            cal=nc_s,
            null=UPSTREAM.NULL_SEED,
        ),
    ]
    stage_state("null_target_calibration")
    na_t, nc_t = _calibration(null_model, target, config, scaler, UPSTREAM.NULL_SEED)
    atomic_json(staging / "calibrations/null/target_alpha.json", nc_t)
    null.append(
        ev(
            which_model=null_model,
            regime=target,
            contract="target_alpha",
            kind="null",
            a=na_t,
            cal=nc_t,
            null=UPSTREAM.NULL_SEED,
        )
    )
    UPSTREAM.atomic_torch_save(
        staging / "null_model.pt",
        {
            "model_state": {k: v.cpu() for k, v in null_model.state_dict().items()},
            "history": nh,
        },
    )
    for rm, nm in zip(real, null):
        r = np.mean(
            [
                x["nats_path"]
                for x in rm["date_metrics"]
                if x["row_scope"] == "aggregate" and x["score_contract"] == "uncapped"
            ]
        )
        n = np.mean(
            [
                x["nats_path"]
                for x in nm["date_metrics"]
                if x["row_scope"] == "aggregate" and x["score_contract"] == "uncapped"
            ]
        )
        if abs(n) > max(UPSTREAM.NULL_ABSOLUTE_CEILING, 0.1 * abs(r)):
            raise RuntimeError("null bias gate failed")
    required = list((staging / "evaluations").glob("*/test_regime=*/*/complete.json"))
    if len(required) != 6:
        raise RuntimeError(
            f"unit requires exactly six completed evaluations, got {len(required)}"
        )
    cursor_fields = {
        "stage",
        "fold",
        "candidate_index",
        "epoch",
        "stratum_index",
        "batch_index",
        "source_date",
        "component",
        "segment",
        "path_start_block",
    }
    training_checkpoints = [
        work / f"fold{fi}_candidate{ci}.pt" for fi in range(1, 5) for ci in range(4)
    ] + [work / "final_real_fit.pt", work / "final_null_fit.pt"]
    resume_rows = []
    for path in training_checkpoints:
        saved = (
            torch.load(path, map_location="cpu", weights_only=False)
            if path.is_file()
            else {}
        )
        runtime_expected = _runtime_config(device, config.batch_size)
        rng_cuda = len(saved.get("rng", {}).get("cuda", ()))
        saved_config = saved.get("config", {})
        expected_saved_scaler = saved_config.get("scaler")
        resume_rows.append(
            {
                "checkpoint": str(path.relative_to(work)),
                "exists": path.is_file(),
                "cursor_fields_exact": set(saved.get("cursor", {})) == cursor_fields,
                "runtime_config_exact": saved.get("runtime_config") == runtime_expected,
                "cuda_topology_exact": rng_cuda == runtime_expected["cuda_rng_states"],
                "config_present": isinstance(saved_config, dict),
                "scaler_exact": isinstance(expected_saved_scaler, dict)
                and saved.get("scaler") == expected_saved_scaler,
            }
        )
    resume_passed = all(
        all(value for key, value in row.items() if key != "checkpoint")
        for row in resume_rows
    )
    atomic_json(
        staging / "resume_audit.json",
        {"status": "complete", "checkpoints": resume_rows, "passed": resume_passed},
    )
    calibration_files = [
        staging / f"calibrations/{kind}/{contract}.json"
        for kind in ("real", "null")
        for contract in ("source_alpha", "target_alpha")
    ]
    audit = _validate_unit_audit(
        {
            "selection": len(fold_rows) == 16
            and (staging / "selection.json").is_file()
            and all(
                (staging / f"selection/fold={fi}/scaler.json").is_file()
                for fi in range(1, 5)
            ),
            "final_fit": (staging / "model.pt").is_file()
            and (staging / "null_model.pt").is_file(),
            "source_calibration": all(
                path.is_file()
                for path in calibration_files
                if path.name == "source_alpha.json"
            ),
            "target_calibration": all(
                path.is_file()
                for path in calibration_files
                if path.name == "target_alpha.json"
            ),
            "all_evaluations": len(required) == 6
            and all(
                json.loads(path.read_text()).get("status") == "complete"
                for path in required
            ),
            "null_audit": all(
                metric["score_antisymmetry_max_abs_error"] < 1e-6
                and metric["prediction_rows"] == 0
                for metric in null
            )
            and all(
                json.loads(path.read_text()).get("audit", {}).get("passed") is True
                for path in required
                if "/null/" in str(path)
            ),
            "null_bias": True,
            "resume": resume_passed,
        }
    )
    stage_state("unit_audit")
    complete = {
        "status": "complete",
        "unit_id": config.unit_id,
        "scientific_config": config.scientific(),
        "input": pre,
        "evaluations": 6,
        "audit": audit,
    }
    atomic_json(staging / "complete.json", complete)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    attempt_record.unlink(missing_ok=True)
    stage_state("complete")
    return complete


def _materialize_local_cache(config: UnitConfig) -> UnitConfig:
    if config.local_cache_root is None:
        return config
    cache = config.local_cache_root / config.storage
    marker = cache / "complete.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text())
        cached = list(cache.rglob("*.parquet"))
        observed = {
            "files": len(cached),
            "bytes": sum(path.stat().st_size for path in cached),
            "rows": sum(pq.ParquetFile(path).metadata.num_rows for path in cached),
        }
        if recorded.get("status") == "complete" and all(
            recorded.get(key) == value for key, value in observed.items()
        ):
            return replace(
                config,
                edge_root=cache / "edge",
                daytime_root=cache / "daytime",
                local_cache_root=None,
            )
        _quarantine(
            cache,
            config.output_root / "quarantine" / config.storage,
            "invalid local SSD cache inventory",
        )
    if cache.exists():
        _quarantine(
            cache,
            config.output_root / "quarantine" / config.storage,
            "incomplete local SSD cache",
        )
    tmp = cache.with_name(f".{cache.name}.attempt-{uuid.uuid4().hex}")
    copied = []
    selected = {"edge": [], "daytime": []}
    for d in DATES:
        for component in ("open", "pre_close"):
            selected["edge"].extend(
                path
                for _, files in _paths(config.edge_root, d, component)
                for path in files
            )
        selected["daytime"].extend(
            path
            for _, files in _paths(config.daytime_root, d, "daytime")
            for path in files
        )
    for label, root in (("edge", config.edge_root), ("daytime", config.daytime_root)):
        for source in sorted(selected[label]):
            if _stop_requested(config.stop_path):
                raise InterruptedError("stop requested during local SSD cache copy")
            target = tmp / label / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(
                {
                    "filename": str(target.relative_to(tmp)),
                    "bytes": target.stat().st_size,
                    "rows": pq.ParquetFile(target).metadata.num_rows,
                }
            )
    atomic_json(
        tmp / "complete.json",
        {
            "status": "complete",
            "files": len(copied),
            "bytes": sum(x["bytes"] for x in copied),
            "rows": sum(x["rows"] for x in copied),
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, cache)
    return replace(
        config,
        edge_root=cache / "edge",
        daytime_root=cache / "daytime",
        local_cache_root=None,
    )


def run_unit(config: UnitConfig) -> dict[str, Any]:
    lock = config.work_root / config.storage / "unit.lock"
    runtime_path = config.work_root / config.storage / "runtime.json"
    with _unit_lock(lock):
        runtime = (
            json.loads(runtime_path.read_text())
            if runtime_path.is_file()
            else {
                "attempts": 0,
                "cumulative_elapsed_seconds": 0.0,
                "cumulative_cost_usd": 0.0,
                "cumulative_interrupted_cost_usd": 0.0,
            }
        )
        started = time.monotonic()
        succeeded = False
        try:
            effective = _materialize_local_cache(config)
            result = _run_unit_locked(effective)
            succeeded = True
            return result
        finally:
            elapsed = time.monotonic() - started
            cost = elapsed * config.hourly_cost_usd / 3600.0
            runtime["attempts"] = int(runtime["attempts"]) + 1
            runtime["cumulative_elapsed_seconds"] = (
                float(runtime["cumulative_elapsed_seconds"]) + elapsed
            )
            runtime["cumulative_cost_usd"] = (
                float(runtime["cumulative_cost_usd"]) + cost
            )
            if not succeeded:
                runtime["cumulative_interrupted_cost_usd"] = (
                    float(runtime["cumulative_interrupted_cost_usd"]) + cost
                )
            runtime.update(
                {
                    "last_attempt_seconds": elapsed,
                    "last_status": "complete" if succeeded else "interrupted_or_failed",
                    "hourly_cost_usd": config.hourly_cost_usd,
                }
            )
            atomic_json(runtime_path, runtime)
            if succeeded:
                final = config.output_root / config.storage
                complete = json.loads((final / "complete.json").read_text())
                artifacts = []
                for path in sorted(final.rglob("*")):
                    if not path.is_file() or path == final / "complete.json":
                        continue
                    row = {
                        "filename": str(path.relative_to(final)),
                        "bytes": path.stat().st_size,
                    }
                    if path.suffix == ".parquet":
                        row["rows"] = pq.ParquetFile(path).metadata.num_rows
                    artifacts.append(row)
                complete["artifacts"] = artifacts
                complete["runtime"] = runtime
                atomic_json(final / "complete.json", complete)
                result.clear()
                result.update(complete)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    if pq.ParquetFile(tmp).metadata.num_rows != len(table):
        raise RuntimeError("footer row mismatch")
    os.replace(tmp, path)


def synthetic_resume_smoke(root: Path) -> dict[str, Any]:
    configure_determinism()
    device = torch.device("cpu")
    rng = np.random.default_rng(7)
    items = []
    root.mkdir(parents=True, exist_ok=True)
    _validate_evaluation_contributions(np.asarray([1, 0]), np.asarray([0.0, 1.0]))
    negative_contribution_rejected = True
    for events_fixture, duration_fixture in (
        (np.asarray([-1]), np.asarray([1.0])),
        (np.asarray([1]), np.asarray([-1.0])),
    ):
        try:
            _validate_evaluation_contributions(events_fixture, duration_fixture)
        except RuntimeError:
            continue
        negative_contribution_rejected = False
    ctx = mp.get_context("fork")
    live_lock = root / "live_owner.lock"
    ready = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_lock_fixture_holder, args=(str(live_lock), ready, release)
    )
    holder.start()
    if not ready.wait(5.0):
        release.set()
        holder.join(5.0)
        raise RuntimeError("live lock fixture failed to acquire")
    live_owner_blocked = False
    try:
        with _unit_lock(live_lock):
            pass
    except RuntimeError:
        live_owner_blocked = True
    finally:
        release.set()
        holder.join(5.0)
    if holder.exitcode != 0:
        raise RuntimeError(f"live lock fixture exit={holder.exitcode}")
    stale_lock = root / "stale_owner.lock"
    stale_pid = 2_147_483_647
    atomic_json(
        stale_lock,
        {
            "pid": stale_pid,
            "hostname": socket.gethostname(),
            "process_start_ticks": "0",
            "started_utc": "2000-01-01T00:00:00+00:00",
        },
    )
    with _unit_lock(stale_lock):
        recovered_owner = json.loads(stale_lock.read_text())
    stale_owner_recovered = (
        recovered_owner.get("pid") == os.getpid()
        and recovered_owner.get("previous_owner", {}).get("pid") == stale_pid
    )
    for di, d in enumerate(DATES[:2]):
        n = 20
        starts = rng.uniform(0.01, 10, size=(n, 7))
        ends = rng.uniform(0.01, 10, size=(n, 7))
        intervals = rng.uniform(0.01, 10, size=(n, 23))
        day = UPSTREAM.DayBlocks(
            d,
            np.arange(n),
            np.arange(n) * 100,
            np.arange(n) * 100 + 90,
            starts,
            ends,
            intervals,
        )
        items.append(Stratum(d, "daytime", 0, day, UPSTREAM.day_path_starts(day, 6)))
    scaler = fit_scaler(items, 20)
    cfg = {"stage": "synthetic", "regime": "daytime", "scaler": scaler.to_dict()}
    cp = root / "resume.pt"
    control_cp = root / "control.pt"
    cp.unlink(missing_ok=True)
    control_cp.unlink(missing_ok=True)
    control, control_h, _, _ = train_stage(
        device=device,
        model_kind="pooled_mlp",
        seed=20260730,
        items=items,
        evaluation=[],
        scaler=scaler,
        candidate=UPSTREAM.ADAMW_CANDIDATES[0],
        epochs=1,
        min_epochs=1,
        patience=1,
        fixed=True,
        batch_size=4,
        checkpoint=control_cp,
        checkpoint_config=cfg,
        resume=True,
    )
    try:
        train_stage(
            device=device,
            model_kind="pooled_mlp",
            seed=20260730,
            items=items,
            evaluation=[],
            scaler=scaler,
            candidate=UPSTREAM.ADAMW_CANDIDATES[0],
            epochs=1,
            min_epochs=1,
            patience=1,
            fixed=True,
            batch_size=4,
            checkpoint=cp,
            checkpoint_config=cfg,
            resume=True,
            interrupt_after_batches=1,
        )
    except InterruptedError:
        pass
    model, h, _, _ = train_stage(
        device=device,
        model_kind="pooled_mlp",
        seed=20260730,
        items=items,
        evaluation=[],
        scaler=scaler,
        candidate=UPSTREAM.ADAMW_CANDIDATES[0],
        epochs=1,
        min_epochs=1,
        patience=1,
        fixed=True,
        batch_size=4,
        checkpoint=cp,
        checkpoint_config=cfg,
        resume=True,
    )
    stop = root / "STOP_REQUESTED"
    stop.touch()
    stop_checks = {}
    try:
        fit_scaler(items, 20, stop, root / "stop_scaler.pt", {"stage": "stop_scaler"})
    except InterruptedError:
        stop_checks["scaler"] = (root / "stop_scaler.pt").is_file()
    try:
        train_stage(
            device=device,
            model_kind="pooled_mlp",
            seed=20260730,
            items=items,
            evaluation=[],
            scaler=scaler,
            candidate=UPSTREAM.ADAMW_CANDIDATES[0],
            epochs=1,
            min_epochs=1,
            patience=1,
            fixed=True,
            batch_size=4,
            checkpoint=root / "stop_train.pt",
            checkpoint_config={"stage": "stop_train", "regime": "daytime"},
            resume=True,
            stop_path=stop,
        )
    except InterruptedError:
        stop_checks["train"] = (root / "stop_train.pt").is_file()
    try:
        collect_scores(
            model,
            items,
            scaler,
            4,
            stop_path=stop,
            checkpoint=root / "stop_calibration.pt",
            checkpoint_config={"stage": "stop_calibration"},
        )
    except InterruptedError:
        stop_checks["calibration"] = (root / "stop_calibration.pt").is_file()
    stop.unlink()
    stage_resume = {}
    for stage_name, fixed_stage, null_value, evaluation_items in (
        ("selection", False, None, items[-1:]),
        ("final_real_fit", True, None, []),
        ("final_null_fit", True, UPSTREAM.NULL_SEED, []),
    ):
        stage_cfg = {"stage": stage_name, "regime": "daytime", "probe": True}
        stage_control = root / f"{stage_name}_control.pt"
        stage_resume_path = root / f"{stage_name}_resume.pt"
        control_model, control_history, _, _ = train_stage(
            device=device,
            model_kind="pooled_mlp",
            seed=20260731,
            items=items,
            evaluation=evaluation_items,
            scaler=scaler,
            candidate=UPSTREAM.ADAMW_CANDIDATES[0],
            epochs=1,
            min_epochs=1,
            patience=1,
            fixed=fixed_stage,
            batch_size=4,
            checkpoint=stage_control,
            checkpoint_config=stage_cfg,
            resume=True,
            null_seed=null_value,
        )
        try:
            train_stage(
                device=device,
                model_kind="pooled_mlp",
                seed=20260731,
                items=items,
                evaluation=evaluation_items,
                scaler=scaler,
                candidate=UPSTREAM.ADAMW_CANDIDATES[0],
                epochs=1,
                min_epochs=1,
                patience=1,
                fixed=fixed_stage,
                batch_size=4,
                checkpoint=stage_resume_path,
                checkpoint_config=stage_cfg,
                resume=True,
                null_seed=null_value,
                interrupt_after_batches=1,
            )
        except InterruptedError:
            pass
        resumed_model, resumed_history, _, _ = train_stage(
            device=device,
            model_kind="pooled_mlp",
            seed=20260731,
            items=items,
            evaluation=evaluation_items,
            scaler=scaler,
            candidate=UPSTREAM.ADAMW_CANDIDATES[0],
            epochs=1,
            min_epochs=1,
            patience=1,
            fixed=fixed_stage,
            batch_size=4,
            checkpoint=stage_resume_path,
            checkpoint_config=stage_cfg,
            resume=True,
            null_seed=null_value,
        )
        stage_resume[stage_name] = control_history == resumed_history and all(
            torch.equal(a, b)
            for a, b in zip(
                control_model.parameters(), resumed_model.parameters(), strict=True
            )
        )
    exact = control_h == h and all(
        torch.equal(a, b)
        for a, b in zip(control.parameters(), model.parameters(), strict=True)
    )
    scores = collect_scores(model, items, scaler, 4)
    alpha, cal = fit_alpha(scores)
    bounded = bool(
        all(
            np.abs(np.clip(alpha * x, -math.log(99), math.log(99))).max()
            <= math.log(99)
            for x in scores.values()
        )
    )
    zero_alpha, zero_cal = fit_alpha(
        {
            "2026-07-16/daytime": np.asarray([-1.0, -2.0]),
            "2026-07-17/daytime": np.asarray([-3.0]),
        }
    )
    test_items = []
    for di, d in enumerate(SPLIT["test"]):
        n = 20
        starts = rng.uniform(0.01, 10, size=(n, 7))
        ends = rng.uniform(0.01, 10, size=(n, 7))
        intervals = rng.uniform(0.01, 10, size=(n, 23))
        day = UPSTREAM.DayBlocks(
            d,
            np.arange(n),
            np.arange(n) * 100,
            np.arange(n) * 100 + 90,
            starts,
            ends,
            intervals,
        )
        test_items.append(
            Stratum(d, "daytime", 0, day, UPSTREAM.day_path_starts(day, 6))
        )
    old_expected = {d: EXPECTED[d] for d in SPLIT["test"]}
    try:
        for s in test_items:
            EXPECTED[s.source_date] = (
                old_expected[s.source_date][0],
                old_expected[s.source_date][1],
                len(s.starts),
            )
        real_metrics = evaluate(
            model=model,
            items=test_items,
            scaler=scaler,
            alpha=alpha,
            calibration_loss=float(cal["paired_log_loss"]),
            train_regime="daytime",
            test_regime="daytime",
            calibration_contract="source_alpha",
            real_or_null="real",
            model_kind="pooled_mlp",
            seed=20260730,
            batch_size=4,
            final_dir=root / "eval_real",
            checkpoint=root / "eval_real.pt",
            quarantine_root=root / "quarantine",
        )
        try:
            evaluate(
                model=model,
                items=test_items,
                scaler=scaler,
                alpha=alpha,
                calibration_loss=float(cal["paired_log_loss"]),
                train_regime="daytime",
                test_regime="daytime",
                calibration_contract="source_alpha",
                real_or_null="real",
                model_kind="pooled_mlp",
                seed=20260730,
                batch_size=4,
                final_dir=root / "eval_resumed",
                checkpoint=root / "eval_resumed.pt",
                quarantine_root=root / "quarantine",
                interrupt_after_batches=1,
            )
        except InterruptedError:
            pass
        resumed_metrics = evaluate(
            model=model,
            items=test_items,
            scaler=scaler,
            alpha=alpha,
            calibration_loss=float(cal["paired_log_loss"]),
            train_regime="daytime",
            test_regime="daytime",
            calibration_contract="source_alpha",
            real_or_null="real",
            model_kind="pooled_mlp",
            seed=20260730,
            batch_size=4,
            final_dir=root / "eval_resumed",
            checkpoint=root / "eval_resumed.pt",
            quarantine_root=root / "quarantine",
        )
        null_metrics = evaluate(
            model=model,
            items=test_items,
            scaler=scaler,
            alpha=zero_alpha,
            calibration_loss=float(zero_cal["paired_log_loss"]),
            train_regime="daytime",
            test_regime="daytime",
            calibration_contract="source_alpha",
            real_or_null="null",
            model_kind="pooled_mlp",
            seed=20260730,
            batch_size=4,
            final_dir=root / "eval_null",
            checkpoint=root / "eval_null.pt",
            quarantine_root=root / "quarantine",
            null_seed=UPSTREAM.NULL_SEED,
        )
        stop.touch()
        try:
            evaluate(
                model=model,
                items=test_items,
                scaler=scaler,
                alpha=alpha,
                calibration_loss=float(cal["paired_log_loss"]),
                train_regime="daytime",
                test_regime="daytime",
                calibration_contract="source_alpha",
                real_or_null="real",
                model_kind="pooled_mlp",
                seed=20260730,
                batch_size=4,
                final_dir=root / "eval_stop",
                checkpoint=root / "eval_stop.pt",
                quarantine_root=root / "quarantine",
                stop_path=stop,
            )
        except InterruptedError:
            stop_checks["evaluation"] = (root / "eval_stop.pt").is_file()
        finally:
            stop.unlink(missing_ok=True)
    finally:
        EXPECTED.update(old_expected)
    evaluation_resume_exact = real_metrics == resumed_metrics and pq.read_table(
        root / "eval_real/test_predictions.parquet"
    ).equals(pq.read_table(root / "eval_resumed/test_predictions.parquet"))
    edge_items = [
        Stratum(s.source_date, c, i, s.day, s.starts[: max(1, len(s.starts) - i - j)])
        for j, s in enumerate(items)
        for i, c in enumerate(("open", "pre_close"))
    ]
    edge_schedule = _epoch_schedule(edge_items, "edge", 20260730, 0, 4, False)
    edge_counts = {(d, c): 0 for d, c in _group_strata(edge_items)}
    for s, selected in edge_schedule:
        edge_counts[(s.source_date, s.component)] += len(selected)
    edge_scheduler_exact = len(set(edge_counts.values())) == 1 and next(
        iter(edge_counts.values())
    ) == min(
        sum(len(x.starts) for x in members)
        for members in _group_strata(edge_items).values()
    )
    evaluation_passed = (
        real_metrics["prediction_rows"] == 2 * sum(len(s.starts) for s in test_items)
        and null_metrics["prediction_rows"] == 0
        and len(real_metrics["date_metrics"]) == 6
        and evaluation_resume_exact
    )
    audit_passed = _validate_unit_audit({name: True for name in UNIT_AUDIT_FIELDS}) == {
        name: True for name in UNIT_AUDIT_FIELDS
    }
    all_stage_resume_exact = all(stage_resume.values())
    stop_checkpoint_passed = set(stop_checks) == {
        "scaler",
        "train",
        "calibration",
        "evaluation",
    } and all(stop_checks.values())
    result = {
        "status": "complete",
        "resume_checkpoint": cp.is_file(),
        "history_epochs": len(h),
        "resume_exact": exact,
        "stage_resume_exact": stage_resume,
        "all_stage_resume_exact": all_stage_resume_exact,
        "evaluation_resume_exact": evaluation_resume_exact,
        "edge_scheduler_exact": edge_scheduler_exact,
        "live_owner_blocked": live_owner_blocked,
        "stale_owner_recovered": stale_owner_recovered,
        "stop_checkpoint_passed": stop_checkpoint_passed,
        "stop_checks": stop_checks,
        "zero_contribution_allowed": True,
        "negative_contribution_rejected": negative_contribution_rejected,
        "finite_alpha": bool(np.isfinite(alpha)),
        "alpha_zero_identity": bool(
            zero_alpha == 0.0
            and abs(zero_cal["paired_log_loss"] - math.log(2.0)) < 1e-12
        ),
        "bounded": bounded,
        "evaluation_contract": evaluation_passed,
        "unit_audit_contract": audit_passed,
        "passed": bool(
            exact
            and all_stage_resume_exact
            and edge_scheduler_exact
            and live_owner_blocked
            and stale_owner_recovered
            and stop_checkpoint_passed
            and negative_contribution_rejected
            and np.isfinite(alpha)
            and bounded
            and evaluation_passed
            and audit_passed
            and zero_alpha == 0.0
        ),
    }
    atomic_json(root / "synthetic_resume_smoke.json", result)
    return result
