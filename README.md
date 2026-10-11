# Limit Order Book Path Irreversibility

Code and audit materials for the ICAIF 2026 paper
*Neural Estimation of Irreversibility in Real and Simulated Limit Order Book
Paths* by **Hyeonjun Yeo, Doohwi Cha, and Yoontae Hwang** (accepted for oral presentation).

This repository contains the scientific
implementations and audit contracts behind the reported analyses, plus the exact
sources for all five figures. It contains no market data, trained checkpoints,
intermediate outputs, provider-control code, or billing records.

## What is included

- MBO replay, five-level book reconstruction, event-count blocks, and the
  37-channel path representation
- physical path reversal and antisymmetric scoring
- invariant logistic, pooled MLP, and ordered TCN estimators
- path-length and resolution comparisons
- first-order and coarse-transition surrogate controls
- session-window, feature-family, and twelve-symbol analyses
- return-prediction comparisons
- clock-free, circular-padding, and shuffled-order controls
- the queue-reactive simulator and its temporal-order benchmark
- the five publication figures from the numerical values reported in the paper

The licensed Nasdaq MBO input is not redistributed. Re-executing the scientific
runtimes therefore requires compatible Databento `XNAS.ITCH` MBO files and
explicit local input/output roots. The figure sources are self-contained and do
not require the proprietary data.

## Repository map

```text
contracts/       feature, reversal, regime, and output contracts
data/            expected input layout and schemas; contains no observations
docs/            paper-to-code map, scope, and reproduction notes
figures/         exact sources for Figures 1--5
reproduction/    scientific source snapshot, grouped by analysis stage
verify_artifact.py
```

For a quick audit, start with this README, then read
[the paper-to-code map](docs/PAPER_TO_CODE.md) and
[the experiment matrix](reproduction/drivers/README.md). Review
[`reproduction/README.md`](reproduction/README.md) only when following the
calculation layers in detail. This route keeps the large scientific runtimes
out of the way until they are needed.

The files under `reproduction/upstream/` are the immutable replay and estimator
implementations on which the later experiments depend. Scientific runtime
helpers live under `reproduction/src/`, the clock-free, circular-padding and
shuffle definitions under `reproduction/controls/`, and the return-prediction
analyses under `reproduction/phases/`.

## Quick start

Python 3.12 and `uv` are required.

```bash
uv sync --frozen
uv run python verify_artifact.py
uv run python figures/make_figures.py
```

The last command writes Figures 3--5 as PDF and PNG. Figures 1--2 use HTML/SVG
sources and headless Chromium:

```bash
uv run playwright install chromium
uv run python figures/render_html.py figures/fig_overview.html figures/fig_cross_section.html
```

Linux Libertine/Biolinum must be installed for the exact paper typography. The
IBM Plex Sans faces used by the figures are included under their upstream font
license. See [the reproduction guide](docs/REPRODUCTION.md) for the data-backed
scope, execution order, and expected resources.

## Data and computational limits

The original sample has 15 trading days and twelve symbols. Full replay and
model fitting are large-data jobs and are not a quick unit test. Paths must stay
within a trading date, session window, and reset segment. Development,
calibration, and held-out dates must not be mixed. The compact verifier checks
source integrity and runs data-free contract tests only.

## Code provenance

The scientific sources are preserved from the reviewed research snapshot. File-level mapping
to the paper is recorded in [PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md). Production
controllers used to provision remote machines, transfer inputs, account for
cloud spend, and recover failed workers are intentionally absent. The retained
files define the estimands, transformations, model components, scientific unit
contracts, validations, aggregations, and figure encodings reported in the
paper. Machine-specific launch adapters omitted from this snapshot are listed
explicitly in the experiment matrix.

## Citation

Please cite the paper when using this code:

```bibtex
@inproceedings{yeo2026irreversibility,
  author = {Yeo, Hyeonjun and Cha, Doohwi and Hwang, Yoontae},
  title = {Neural Estimation of Irreversibility in Real and Simulated Limit Order Book Paths},
  booktitle = {7th ACM International Conference on AI in Finance},
  year = {2026},
  note = {Accepted for publication}
}
```

The DOI will be added when assigned. GitHub's citation metadata is also available
in [CITATION.cff](CITATION.cff).

## License

The original research code, figure source code, and accompanying documentation
are available under the [MIT License](LICENSE). The IBM Plex font files retain
their [SIL Open Font License](figures/fonts/LICENSE.txt). Third-party Python
packages retain their own licenses and are installed through the lockfile.
The code license does not grant rights to the licensed market data or to the
paper's published version, which is not bundled here.
