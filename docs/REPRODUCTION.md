# Reproduction guide

## 1. Environment

```bash
uv sync --frozen
uv run python verify_artifact.py
```

The lockfile fixes the Python dependency graph. CUDA model fitting requires a
compatible PyTorch/CUDA environment; the default lock uses CPU PyTorch for
portable inspection and figure generation.

## 2. Data-free checks

`verify_artifact.py` rejects bundled observations and local-path or secret
patterns, compiles every Python file, imports the central estimator runtime, and
runs the QR simulator's bounded contract smoke test.

## 3. Figure reproduction

```bash
uv run python figures/make_figures.py
uv run playwright install chromium
uv run python figures/render_html.py figures/fig_overview.html
```

Figures 2--5 are written by Matplotlib. Figure 1 is an HTML/SVG layout printed
to vector PDF. Linux Libertine and Linux Biolinum are required for exact matching
to the ACM manuscript typography.

The plotting commands read only bundled aggregate values. Figure 2 validates
its 12-symbol, three-date panel and checks that averaging the daily increments
agrees with the difference between the mean ordered and pooled rates. This check
can be run independently with `uv run python figures/make_figures.py --check-data`
and is also included in `verify_artifact.py`. It performs no model fitting.
See [the figure guide](../figures/README.md) for units and aggregation.

## 4. Data-backed scientific order

With compatible inputs mounted under `/data`, the retained calculations follow
this order:

1. reconstruct and validate the book with the two codebooks under
   `reproduction/upstream/phases/02_*` and `03c_*`;
2. publish continuous blocks with `reproduction/src/block_producer.py`;
3. fit and evaluate estimators with `cuda_runtime.py` and `scale_runtime.py`;
4. run the first-order, surrogate, session, ablation, and cross-symbol runtimes;
5. run the three retained return-prediction phase scripts;
6. inspect and validate the portable control definitions under
   `reproduction/controls/`;
7. simulate and validate QR paths with the retained simulator runtimes.

The public snapshot intentionally does not reproduce the original provider and
machine orchestration. It also does not claim a portable end-to-end command for
the specialized clock-free, circular-padding, shuffled-order, or QR neural
refits; their retained boundary is listed in `reproduction/drivers/README.md`.
Runtime functions that remain executable take explicit roots/configurations.
Callers should preserve unit-specific temporary outputs and publish only after
the retained validators pass.

## 5. Expected primary contract

- symbol and window: NVDA, daytime 10:00--15:30 New York
- target events per block: `B=10`
- blocks per path: `K=6`
- stride: 3 blocks with date-ordinal offset modulo 3
- models: pooled MLP and ordered TCN
- seeds: 20260730, 20260731, 20260732
- reporting: three held-out dates weighted equally
- score cap: `[-log(99), log(99)]`

The expected path counts are 1,751,687 development, 671,859 calibration, and
582,767 test paths. A mismatch indicates a different input stream, session
boundary, reset segmentation, or completion-overshoot implementation.
