# Reproduction source map

This directory contains only calculation code used by a result, control, or
input contract in the paper. It is split by provenance and purpose rather than
by the machines on which the original jobs ran.

| Directory | Role | Read this when |
|---|---|---|
| `upstream/` | Foundational MBO replay, block construction, linear baseline, and primary neural estimators | Auditing the data representation or core estimator |
| `src/` | Paper-specific scientific runtimes and validators | Auditing a reported comparison or simulator benchmark |
| `controls/` | Small, portable definitions for the clock-free, circular-padding, and shuffled-order controls | Auditing the identifying controls |
| `phases/` | Return-prediction bridge and leakage-safe follow-up calculations | Auditing the prediction paragraph |
| `drivers/` | Exact experiment-unit matrix and retained execution boundary | Checking completeness without reading implementations |

The files in `upstream/` and `phases/` preserve their phase names because those
names identify calculation lineage. The flat `src/` directory avoids artificial
wrappers around tightly coupled runtimes; its thematic grouping is documented
in [`src/README.md`](src/README.md). No provider SDK, cloud launcher, billing
ledger, credential helper, downloaded data, or generated result is included.

Most `src/` files are importable scientific libraries rather than standalone
commands. The executable codebooks expose `--help`; the exact original unit
universe is fixed separately in [`drivers/README.md`](drivers/README.md).
