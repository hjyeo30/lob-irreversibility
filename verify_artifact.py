#!/usr/bin/env python3
"""Run data-free integrity, packaging, and scientific contract checks."""

from __future__ import annotations

import py_compile
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".html", ".gitignore"}
PROHIBITED_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".csv",
    ".dbn",
    ".feather",
    ".jsonl",
    ".log",
    ".parquet",
    ".pt",
    ".pth",
    ".zst",
}
PROHIBITED_PATTERNS = (
    re.compile("/" + r"home/[^/\s\"']+", re.IGNORECASE),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)
REQUIRED = (
    "contracts/reversal_registry.yaml",
    "data/README.md",
    "docs/PAPER_TO_CODE.md",
    "figures/fig_overview.html",
    "figures/fig_cross_section.html",
    "figures/make_figures.py",
    "reproduction/src/cuda_runtime.py",
    "reproduction/src/transition_surrogate_runtime.py",
    "reproduction/src/qr_simulator_runtime.py",
    "reproduction/controls/short_term_controls.py",
    "reproduction/controls/shuffled_order_control.py",
    "reproduction/README.md",
    "reproduction/drivers/README.md",
    "reproduction/src/README.md",
    "reproduction/upstream/phases/05_antisymmetric_sequence_models/codebook.py",
)
ALLOWED_FONT_FILES = {
    "figures/fonts/IBMPlexSans-Bold.ttf",
    "figures/fonts/IBMPlexSans-Medium.ttf",
    "figures/fonts/IBMPlexSans-Regular.ttf",
    "figures/fonts/IBMPlexSans-SemiBold.ttf",
    "figures/fonts/LICENSE.txt",
}


def files() -> list[Path]:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    if any(tracked):
        return sorted(ROOT / name for name in tracked if name)
    ignored_directories = {".git", ".venv", "__pycache__", "build"}
    ignored_suffixes = {".pdf", ".png"}
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not ignored_directories.intersection(path.relative_to(ROOT).parts)
        and path.suffix.lower() not in ignored_suffixes
    )


def verify_inventory(paths: list[Path]) -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required files: {missing}")
    forbidden = [
        str(path.relative_to(ROOT))
        for path in paths
        if path.suffix.lower() in PROHIBITED_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"data or model artifacts are bundled: {forbidden}")
    font_files = {
        str(path.relative_to(ROOT))
        for path in paths
        if path.relative_to(ROOT).parts[:2] == ("figures", "fonts")
    }
    if font_files != ALLOWED_FONT_FILES:
        raise RuntimeError(
            "unexpected or missing vendored font files: "
            f"expected={sorted(ALLOWED_FONT_FILES)}, actual={sorted(font_files)}"
        )


def verify_text(paths: list[Path]) -> None:
    findings: list[str] = []
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for pattern in PROHIBITED_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    if findings:
        raise RuntimeError(
            "credential or local-path findings:\n" + "\n".join(findings)
        )


def compile_python(paths: list[Path]) -> None:
    with tempfile.TemporaryDirectory(prefix="review-artifact-bytecode-") as folder:
        for index, path in enumerate(path for path in paths if path.suffix == ".py"):
            py_compile.compile(
                str(path), cfile=str(Path(folder) / f"{index}.pyc"), doraise=True
            )


def run_smoke() -> None:
    commands = (
        [
            sys.executable,
            str(ROOT / "reproduction/src/qr_simulator_runtime.py"),
            "smoke",
        ],
        [sys.executable, str(ROOT / "reproduction/controls/short_term_controls.py")],
        [sys.executable, str(ROOT / "reproduction/controls/shuffled_order_control.py")],
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "reproduction")
    for command in commands:
        subprocess.run(command, cwd=ROOT / "reproduction", env=environment, check=True)


def main() -> None:
    paths = files()
    verify_inventory(paths)
    verify_text(paths)
    compile_python(paths)
    run_smoke()
    print(f"artifact verification passed: {len(paths)} files")


if __name__ == "__main__":
    main()
