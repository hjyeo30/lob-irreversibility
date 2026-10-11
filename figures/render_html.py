"""Render an HTML figure source to a vector PDF with headless Chromium.

The page box is taken from the @page rule in the stylesheet, so the PDF comes
out at exactly the width the manuscript includes it at.

    uv run --with playwright --with otf2ttf python figures/render_html.py \
        figures/fig_overview.html

Chromium's PDF backend cannot embed CFF-flavoured OpenType, and falls back to
Type 3 fonts, which ACM production rejects.  The Libertine and Biolinum faces
that acmart uses are therefore converted to TrueType outlines once into
build/fonts, which the stylesheet loads, and they then embed as CIDFontType2.
A PNG preview is written beside the manuscript's other figure previews.
"""

from __future__ import annotations

import sys
from pathlib import Path

import otf2ttf
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
PREVIEW = BUILD / "figure_previews"
FONT_DIR = BUILD / "fonts"
FONT_SRC = Path.home() / ".local/share/fonts"
FACES = [
    "LinLibertine_R",
    "LinLibertine_RI",
    "LinLibertine_RB",
    "LinLibertine_RBI",
    "LinBiolinum_R",
    "LinBiolinum_RB",
]


def ensure_truetype_faces() -> None:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for face in FACES:
        src, dst = FONT_SRC / f"{face}.otf", FONT_DIR / f"{face}.ttf"
        if not src.exists():
            raise SystemExit(f"missing font {src}")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        otf2ttf.main(["-o", str(dst), str(src)])


def render(src: Path) -> None:
    out_pdf = src.with_suffix(".pdf")
    PREVIEW.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # Without this Chromium rounds TrueType advance widths to whole units
        # when it lays the page out, which opens ragged gaps after letters such
        # as s and m.  CFF outlines keep their fractional advances either way.
        browser = p.chromium.launch(args=["--font-render-hinting=none"])
        page = browser.new_page(device_scale_factor=4)
        page.goto(src.resolve().as_uri())
        page.wait_for_timeout(400)
        page.pdf(path=str(out_pdf), prefer_css_page_size=True, print_background=True)
        box = page.evaluate(
            "() => { const b = document.body.getBoundingClientRect();"
            " return {w: b.width, h: b.height}; }"
        )
        page.set_viewport_size({"width": round(box["w"]), "height": round(box["h"])})
        page.screenshot(path=str(PREVIEW / f"{src.stem}.png"))
        browser.close()
    print(f"wrote {out_pdf.name}")


if __name__ == "__main__":
    ensure_truetype_faces()
    for arg in sys.argv[1:] or [HERE / "fig_overview.html"]:
        render(Path(arg))
