# Camera-ready figures

The scripts reproduce the five figures in the camera-ready manuscript using
bundled aggregate values. No fitting or licensed market input is needed.

```bash
uv sync --frozen
uv run python figures/make_figures.py
uv run playwright install chromium
uv run python figures/render_html.py figures/fig_overview.html
```

Figures 2--5 are vector PDFs under `figures/`, with 400-dpi PNG previews under
`build/figure_previews/`. Figure 1 uses Chromium. Linux Libertine/Biolinum must be
installed for the manuscript typography; the HTML renderer expects their OTF
files under the user's `.local/share/fonts` directory. IBM Plex Sans fonts and
their license are bundled in `fonts/`.

## Figure 2: cross-symbol repetition

The left panel compares pooled rates and ordered-minus-pooled increments for
12 selected symbols. The right panel shows whether the positive increment
recurs on each of three test dates, July 21--23, 2026. Each point is one
symbol/date mean over three network initializations. The mean bars average
both dates and initializations equally. Both panels use thousandths of a nat
per event; the JSON stores unscaled nats per event. Symbols are ordered by
mean increment. Date-specific markers and small vertical offsets distinguish
coincident points without changing their values.

`data/cross_symbol.json` contains 36 date-specific increments and 12 mean rows
for B=10, K=6, 37 features, daytime, and the clipped log(99) score contract.
The inputs are aggregate fitted-model estimates, not market observations.
The date points show observed variation, not confidence intervals. Symbols on
the same date are not treated as independent trading dates.

These values come from the completed cross-symbol result tables
`paired_contrasts.parquet` and `unit_metrics.parquet`. Their original
fingerprints and the completion-marker fingerprint are recorded in the JSON.
The paired contrasts were filtered to bounded_log99, then averaged over
initializations within symbol/date; the mean model rates average the same
three dates and three initializations. No estimate was refitted for this figure.

The loader checks complete and unique symbol/date coverage, finite values,
three initializations per point, and agreement between the mean of the date
increments and the difference of the mean model rates. It also checks agreement
with the previously reported rounded rates. Run just these checks with:

```bash
uv run python figures/make_figures.py --check-data
```

The former single-panel `fig_cross_section.html` is preserved in Git history;
it is not the source of the current Figure 2.

## Other panels

Figure 1 keeps the original overview. Its middle surrogate row compares the
surrogate rate with the observed ordered rate; it is not a pooled/ordered gap.
Figure 3 explicitly labels unclipped curves and uses the camera-ready panel
height. Figure 4 keeps the one-step and feature-group comparisons. Figure 5
uses the camera-ready spacing, short category labels and small-value labels.
All underlying numerical values in Figures 3--5 are unchanged.
