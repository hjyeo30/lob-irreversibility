"""Render the manuscript figures at ACM sigconf column widths.

Numbers are transcribed from the reviewed phase tables listed beside each block.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent

COLUMN_WIDTH = 3.33
TEXT_WIDTH = 7.0

# Labels are set in IBM Plex Sans, which journals ask for in figure lettering and
# which keeps the panels from reading as blocks of body text, while mathematical
# notation stays in Linux Libertine so that a symbol in a figure has the same
# shape as the same symbol in the running text. The faces ship with the paper in
# figures/fonts, so nothing has to be installed system-wide.
for _face in sorted((OUT / "fonts").glob("IBMPlexSans-*.ttf")):
    font_manager.fontManager.addfont(str(_face))

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Linux Libertine O",
        "mathtext.it": "Linux Libertine O:italic",
        "mathtext.bf": "Linux Libertine O:bold",
        "mathtext.cal": "Linux Libertine O:italic",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 400,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

# One chromatic hue across the whole paper, matching figures/fig_overview.html.
# INK carries the ordered series, because it is the same idea as the forward
# path in Figure 1, CLAY carries the increment that the reversal exposes, and
# SLATE carries the pooled baseline. The three separate by lightness as well as
# by hue, so every panel survives a monochrome printout. MUTE is reserved for
# reference lines and annotation, so that no data series is drawn in a color
# that reads as less important. The neutrals lean to the same warm side of the
# wheel as CLAY does, so that the darkest ink reads as a companion to the red
# rather than as printer's black set beside it.
INK = "#3A3531"
CLAY = "#B23B2C"
SLATE = "#8B837A"
MUTE = "#5C554D"
RULE = "#D2CDC6"

# Matplotlib draws spines, ticks, tick labels and the grid in pure black and a
# cool grey unless told otherwise, which would leave these panels on a neutral
# chassis facing the warm Figure 1 across a spread. In Fig. 5 the default black
# was the most abundant ink on the page, outweighing every data color. The
# chassis is therefore put on the same palette as the marks it carries.
mpl.rcParams.update(
    {
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": RULE,
        "grid.alpha": 0.9,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    preview = OUT.parent / "build" / "figure_previews"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=400)
    plt.close(fig)
    print(f"wrote {name}.pdf")


def figure_length() -> None:
    # Phase 09 length_curve.parquet and Phase 13 S1 bounded_uncapped_curve /
    # bounded score, three test dates x three seeds.
    lengths = np.array([1, 2, 3, 4, 6, 8, 12])
    ordered_bounded = np.array([0.02182, 0.05268, 0.06130, 0.06037, 0.05373, 0.04614, 0.03445])
    ordered_uncapped = np.array([0.02185, 0.05577, 0.07155, 0.07781, 0.08522, 0.08663, 0.08473])
    gap_bounded = np.array([-0.00135, 0.00866, 0.01740, 0.01890, 0.01786, 0.01541, 0.01021])
    gap_uncapped = np.array([-0.00138, 0.00884, 0.02270, 0.02956, 0.04107, 0.04717, 0.05157])

    # Pooled unclipped rate, reconstructed as ordered minus increment.
    pooled_unclipped = ordered_uncapped - gap_uncapped

    # Phase 03 scale_and_length, bounded nats/event, fixed nominal horizon BK=60.
    resolutions = [r"$(5,12)$", r"$(10,6)$", r"$(30,2)$"]
    res_pooled = np.array([0.04468, 0.03587, 0.01555])
    res_ordered = np.array([0.06624, 0.05373, 0.01754])

    saturation_ordered = np.array([0.00041, 0.03771, 0.15032, 0.27875, 0.51272, 0.66883, 0.83793])
    saturation_pooled = np.array([0.00078, 0.03083, 0.07733, 0.12838, 0.21672, 0.28346, 0.39264])

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.40))

    ax = axes[0]
    lines = [
        ax.plot(lengths, ordered_uncapped, "-s", color=INK, label="ordered, unclipped")[0],
        ax.plot(lengths, pooled_unclipped, "-o", color=SLATE, label="pooled, unclipped")[0],
        ax.plot(lengths, gap_uncapped, "-^", color=CLAY, label="increment, unclipped")[0],
        # Full strength rather than a tint: at 55% opacity this line printed at
        # 2.5:1 against white, and in greyscale it matched the pooled series it
        # is meant to contrast with. The dash and the open marker separate it.
        ax.plot(lengths, gap_bounded, "--v", color=CLAY, markerfacecolor="white",
                label="increment, clipped")[0],
    ]
    ax.axhline(0, color=MUTE, linewidth=0.6)
    ax.set_xlabel(r"path length $K$")
    ax.set_ylabel("nats per event")
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_ylim(-0.008, 0.098)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08])
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    ax.plot(lengths, saturation_ordered, "-s", color=INK)
    ax.plot(lengths, saturation_pooled, "-o", color=SLATE)
    ax.set_xlabel(r"path length $K$")
    ax.set_ylabel("fraction at the bound")
    ax.set_xticks([1, 2, 4, 6, 8, 12])
    ax.set_ylim(0, 1.0)
    ax.set_title("(b)", loc="left")
    ax.annotate("ordered", (12, saturation_ordered[-1]), xytext=(-3, 3), textcoords="offset points", ha="right", color=INK)
    ax.annotate("pooled", (12, saturation_pooled[-1]), xytext=(-3, 3), textcoords="offset points", ha="right", color=MUTE)

    ax = axes[2]
    positions = np.arange(3)
    ax.bar(positions - 0.19, res_pooled, 0.36, color=SLATE)
    ax.bar(positions + 0.19, res_ordered, 0.36, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel(r"resolution and length $(B,K)$")
    ax.set_ylabel("nats per event")
    ax.set_ylim(0, 0.084)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_title("(c)", loc="left")
    for x, lo, hi in zip(positions, res_pooled, res_ordered):
        ax.annotate(
            "", xy=(x + 0.19, hi), xytext=(x - 0.19, lo),
            arrowprops=dict(arrowstyle="-", color=CLAY, linewidth=0.8, linestyle=(0, (2, 1.4))),
        )
        ax.text(x, hi + 0.004, rf"$\Delta={hi - lo:.4f}$", fontsize=7.5, ha="center", color=CLAY)

    fig.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        handlelength=1.5,
        columnspacing=1.4,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.235, top=0.78, wspace=0.30)
    save(fig, "fig_length")


def figure_identification() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(COLUMN_WIDTH, 1.50),
        constrained_layout=True,
        width_ratios=[1.0, 1.3],
    )

    # Phase 04 history_contrasts / real_surrogate_contrasts, bounded nats/event.
    ax = axes[0]
    lengths = [r"$K{=}2$", r"$K{=}6$", r"$K{=}12$"]
    residual_one_step = [0.03052, 0.03243, 0.01572]
    residual_surrogate = [0.015346, 0.016087, 0.007779]
    positions = np.arange(3)
    ax.bar(positions - 0.19, residual_one_step, 0.36, color=INK, label="one-step")
    # The surrogate is a control, and CLAY is the color of the ordering
    # increment everywhere else in the paper. Drawing a construction that
    # destroys ordering in the color of ordering made panel (a) contradict
    # panel (b), so controls are hatched in MUTE, the reference color, and a
    # solid fill is reserved for an observed quantity.
    ax.bar(
        positions + 0.19,
        residual_surrogate,
        0.36,
        facecolor="white",
        edgecolor=MUTE,
        hatch="////",
        linewidth=0.7,
        label="surrogate",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(lengths)
    ax.set_ylabel("residual (nats per event)")
    ax.set_ylim(0, 0.050)
    ax.axhline(0, color=MUTE, linewidth=0.6)
    ax.legend(
        loc="upper right",
        handlelength=1.0,
        borderpad=0.1,
        labelspacing=0.2,
        fontsize=7.5,
    )
    ax.set_title("(a)", loc="left")

    # Phase 06 ablation_contrasts, bounded nats/event, three seeds x three dates.
    ax = axes[1]
    names = ["state", "flow", "liq.", "price", "no state", "all"]
    ordered = [0.00704, 0.01004, 0.00094, 0.00074, 0.05267, 0.05373]
    difference = [0.00294, 0.00530, 0.00061, 0.00046, 0.03032, 0.01786]
    positions = np.arange(len(names))
    ax.bar(positions - 0.19, ordered, 0.36, color=INK, label="ordered")
    ax.bar(positions + 0.19, difference, 0.36, color=CLAY, label="increment")
    ax.set_xticks(positions)
    # Six labels in half a column, so they are set on a steeper slant than the
    # three-category panels to keep "price" clear of "no state".
    ax.set_xticklabels(
        names, fontsize=7.2, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("nats per event")
    ax.set_ylim(0, 0.079)
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.legend(
        loc="upper left",
        handlelength=1.0,
        borderpad=0.1,
        labelspacing=0.2,
        fontsize=7.5,
    )
    ax.set_title("(b)", loc="left")
    save(fig, "fig_identification")


def figure_simulator() -> None:
    # Phase 14 primary compact result, bounded nats/event, three test dates x
    # three seeds. Pooled rates are ordered minus the reported increment.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.70), constrained_layout=True)

    sources = ["real", "QR-A", "QR-B"]
    ordered = np.array([0.0537297, 0.0516500, 0.0517085])
    increment = np.array([0.0178623, 0.000649947, 0.000753825])
    pooled = ordered - increment

    ax = axes[0]
    positions = np.arange(3)
    ax.bar(positions - 0.19, pooled, 0.36, color=SLATE, label="pooled")
    ax.bar(positions + 0.19, ordered, 0.36, color=INK, label="ordered")
    ax.set_xticks(positions)
    ax.set_xticklabels(sources)
    ax.set_ylabel("nats per event")
    ax.set_ylim(0, 0.088)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.legend(loc="upper center", ncol=2, handlelength=1.2, borderpad=0.1, columnspacing=1.0)
    ax.set_title("(a)", loc="left")
    for x, lo, hi, gap in zip(positions, pooled, ordered, increment):
        ax.annotate(
            "", xy=(x + 0.19, hi), xytext=(x - 0.19, lo),
            arrowprops=dict(arrowstyle="-", color=CLAY, linewidth=0.8, linestyle=(0, (2, 1.4))),
        )
        ax.text(
            x, max(lo, hi) + 0.0035, rf"$\Delta={gap:.4f}$",
            fontsize=7.5, ha="center", color=CLAY,
        )

    # Phase 14 V5 shuffled-order control shares the 34-feature representation.
    ax = axes[1]
    labels = [
        "real", "QR-A", "QR-B", "real", "QR-A/B", "shuffled",
    ]
    values = np.array(
        [0.0178623, 0.000649947, 0.000753825, 0.00705838, 0.000779023, 0.000548]
    )
    # Every bar here is an ordering increment, so all of them carry the
    # increment color. INK and SLATE are reserved for the ordered and pooled
    # rates of panel (a) and would read as those series if reused.
    colors = [CLAY] * 6
    hatches = ["", "", "", "", "", "///"]
    positions = np.array([0.0, 1.0, 2.0, 3.7, 4.7, 5.7])
    bars = ax.bar(positions, values, 0.62, color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            # The shuffled refit is a control, hatched on white like the
            # surrogate controls in Fig. 3(a) and panel (c). A white hatch over
            # a solid clay bar was invisible at this bar height anyway.
            bar.set_facecolor("white")
            bar.set_hatch(hatch)
            bar.set_edgecolor(MUTE)
            bar.set_linewidth(0.7)
    for x, value in zip(positions, values):
        if value < 0.001:
            ax.text(x, value + 0.0007, f"{value:.4f}", fontsize=6.5, ha="center", color=MUTE, rotation=60)
    ax.axhline(0, color=MUTE, linewidth=0.6)
    ax.axvline(2.85, color=MUTE, linewidth=0.6, linestyle=(0, (2, 1.6)))
    ax.set_xticks(positions)
    # Six categories in a third of the text width, so the labels are set on a
    # slant rather than shortened past the point of being readable.
    ax.set_xticklabels(
        labels, fontsize=7.2, rotation=30, ha="right", rotation_mode="anchor"
    )
    # Every bar in this panel is a difference, not a rate, so it does not carry
    # the same axis label as panels (a) and (c).
    ax.set_ylabel("increment (nats per event)")
    ax.set_ylim(0, 0.0215)
    ax.text(1.0, 0.0196, "37 features", fontsize=7.5, ha="center", color=MUTE)
    ax.text(4.7, 0.0196, "34 features", fontsize=7.5, ha="center", color=MUTE)
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    original = np.array([0.0537297, 0.0516500])
    surrogate = np.array([0.0376000, 0.0516052])
    positions = np.arange(2)
    ax.bar(positions - 0.19, original, 0.36, color=INK, label="observed")
    ax.bar(
        positions + 0.19, surrogate, 0.36,
        facecolor="white", edgecolor=MUTE, hatch="////", linewidth=0.7, label="surrogate",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["real", "QR-A"])
    ax.set_ylabel("nats per event")
    ax.set_ylim(0, 0.088)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.legend(loc="upper center", ncol=2, handlelength=1.2, borderpad=0.1, columnspacing=1.0)
    ax.set_title("(c)", loc="left")
    ax.annotate(r"$-30\%$", (0.0, 0.0585), fontsize=7.5, ha="center", color=CLAY)
    ax.annotate(r"$-0.09\%$", (1.0, 0.0565), fontsize=7.5, ha="center", color=CLAY)

    save(fig, "fig_simulator")


# Phase 07 cross-symbol, clipped rates averaged equally over three test dates and
# three initializations, as (pooled, ordered, gap low, gap high). The gap range is
# across initializations and reproduces the interval reported in the ledger.
GAPS = {
    "AAPL": (0.027274, 0.045663, 0.017778, 0.018964),
    "NVDA": (0.035867, 0.053730, 0.017278, 0.018237),
    "COST": (0.033078, 0.050875, 0.017431, 0.018082),
    "TSLA": (0.037276, 0.053606, 0.015904, 0.016831),
    "SAIA": (0.013224, 0.029340, 0.015569, 0.017058),
    "PEP": (0.037539, 0.053525, 0.015675, 0.016372),
    "CROX": (0.027527, 0.042807, 0.013758, 0.017381),
    "IWM": (0.031238, 0.045818, 0.013936, 0.015436),
    "PODD": (0.016244, 0.030110, 0.012906, 0.014400),
    "SPY": (0.038594, 0.050833, 0.011898, 0.012426),
    "QQQ": (0.039000, 0.050203, 0.010829, 0.011748),
    "SOFI": (0.049552, 0.053326, 0.002752, 0.004687),
}


# Retained for reference: the cross-section figure moved to
# figures/fig_cross_section.html so that it shares the stacked-bar grammar of
# Figure 1, and this table is the source the SVG bar widths were computed from.
# Daytime median best-quote depth in shares, phase 15 symbol_descriptives.
BEST_DEPTH = {
    "AAPL": 154,
    "NVDA": 248,
    "COST": 52,
    "TSLA": 120,
    "SAIA": 44,
    "PEP": 145,
    "CROX": 119,
    "IWM": 529,
    "PODD": 98,
    "SPY": 335,
    "QQQ": 320,
    "SOFI": 6851,
}


def load_cross_section() -> tuple[list[dict], list[dict], list[str]]:
    """Validate the complete symbol/date panel and both ways of computing its mean."""
    data = json.loads((OUT / 'data/cross_symbol.json').read_text())
    daily, means = data['daily'], data['means']
    dates = data['sample']['test_dates']
    if data['sample'] != {'B': 10, 'K': 6, 'features': 37, 'session': 'daytime',
                         'test_dates': ['2026-07-21', '2026-07-22', '2026-07-23'],
                         'initializations_per_date': 3, 'score_contract': 'bounded_log99'}:
        raise ValueError('Unexpected Figure 2 sample contract')
    symbols = {row['symbol'] for row in means}
    keys = {(row['symbol'], row['source_date']) for row in daily}
    if len(means) != 12 or symbols != set(GAPS) or len(daily) != 36 or keys != {
            (symbol, date) for symbol in symbols for date in dates}:
        raise ValueError('Missing or duplicate symbol/date rows')
    for row in daily:
        if row['n_initializations'] != 3 or not np.isfinite(row['ordering_increment']) or row['ordering_increment'] <= 0:
            raise ValueError('Invalid date-specific estimate')
    for row in means:
        values = [row[key] for key in ['pooled_mlp', 'ordered_tcn', 'ordering_increment']]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite mean estimate')
        gap = row['ordered_tcn'] - row['pooled_mlp']
        date_mean = np.mean([day['ordering_increment'] for day in daily if day['symbol'] == row['symbol']])
        if abs(gap - row['ordering_increment']) > 1e-12 or abs(gap - date_mean) > 1e-12:
            raise ValueError('Date means and paired model contrast disagree')
        for actual, rounded in zip(values[:2], GAPS[row['symbol']][:2]):
            if abs(actual - rounded) > 1.1e-6:
                raise ValueError('Mean differs from reported rounded rate')
    return daily, sorted(means, key=lambda row: row['ordering_increment'], reverse=True), dates


def figure_cross_section() -> None:
    daily, means, dates = load_cross_section()
    # Same ordering in both panels. Small deterministic vertical offsets reveal
    # coincident dates without changing any point's horizontal position.
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True,
                                 gridspec_kw={'width_ratios': [1.20, 1.0]})
    y = np.arange(12)
    pooled = np.array([row['pooled_mlp'] for row in means]) * 1000
    increment = np.array([row['ordering_increment'] for row in means]) * 1000
    ax.barh(y, pooled, height=.58, color=SLATE, label='pooled')
    ax.barh(y, increment, left=pooled, height=.58, color=CLAY, label='ordered − pooled')
    for yi, share in zip(y, increment / (pooled + increment)):
        ax.text(62.0, yi, f'{share:.0%}', va='center', ha='right', fontsize=7.4)
    ax.text(62.0, -1.02, r'$\Delta$/ordered', ha='right', va='center', fontsize=7.4)
    ax.set_yticks(y, [row['symbol'] for row in means], fontsize=8)
    ax.set_ylim(11.6, -1.7)
    ax.set_xlim(0, 64)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel(r'rate ($10^{-3}$ nats per event)')
    ax.set_title('(a) Mean rates', loc='left', pad=25)
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.01), ncol=2, fontsize=7.5,
              handlelength=1.2, columnspacing=1.0, borderaxespad=0)
    for date, marker, offset, color in zip(dates, ['o', '^', 's'], [-.18, 0, .18],
                                          [INK, CLAY, SLATE]):
        lookup = {row['symbol']: row['ordering_increment'] for row in daily if row['source_date'] == date}
        values = np.array([lookup[row['symbol']] for row in means]) * 1000
        axd.scatter(values, y + offset, s=21, marker=marker, color=color,
                    label='Jul ' + date[-2:], edgecolors='white', linewidths=.35, zorder=3)
    axd.axvline(0, color=MUTE, lw=.8)
    axd.set_xlim(-.5, max(22, float(max(row['ordering_increment'] for row in daily) * 1000 + 1.5)))
    axd.set_xticks([0, 5, 10, 15, 20])
    axd.set_xlabel(r'increment ($10^{-3}$ nats per event)')
    axd.set_title('(b) Individual test dates', loc='left', pad=25)
    axd.legend(loc='lower left', bbox_to_anchor=(-.03, 1.01), ncol=3, fontsize=7.5,
               handletextpad=.3, columnspacing=1.0, borderaxespad=0)
    axd.tick_params(axis='y', left=False)
    for a in (ax, axd):
        a.grid(False, axis='y')
    fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.83, wspace=.22)
    save(fig, 'fig_cross_section')


if __name__ == "__main__":
    if sys.argv[1:] == ['--check-data']:
        load_cross_section()
        print('Figure 2 aggregates verified: 12 symbols, 36 date points, paired means agree')
    elif sys.argv[1:]:
        raise SystemExit('Usage: make_figures.py [--check-data]')
    else:
        figure_cross_section()
        figure_length()
        figure_identification()
        figure_simulator()
