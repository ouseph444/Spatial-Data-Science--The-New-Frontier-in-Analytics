#!/usr/bin/env python3
"""Render the course cells to a Jupyter-Notebook-looking PDF.

Pipeline:  cells -> HTML (markdown + pygments) -> WeasyPrint -> PDF

Requires:  pip install markdown pygments weasyprint
"""
from __future__ import annotations

import html
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import markdown                                     # noqa: E402
from pygments import highlight                      # noqa: E402
from pygments.lexers import PythonLexer, BashLexer  # noqa: E402
from pygments.formatters import HtmlFormatter       # noqa: E402

from part0_setup import CELLS as C0                 # noqa: E402
from part1_beginner import CELLS as C1              # noqa: E402
from part2_intermediate import CELLS as C2          # noqa: E402
from part3_advanced import CELLS as C3              # noqa: E402
from part4_capstone import CELLS as C4              # noqa: E402

ALL_CELLS = C0 + C1 + C2 + C3 + C4

TITLE = "Spatial Data Science: The New Frontier in Analytics"
SUBTITLE = "A practical Python GIS course — GeoPandas · Shapely · Rasterio"

MD_EXT = ["extra", "sane_lists", "admonition", "attr_list", "md_in_html"]

CSS = r"""
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-center {
    content: "Spatial Data Science: The New Frontier in Analytics  ·  page " counter(page);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 7.5pt; color: #8b93a7;
  }
}
@page :first { margin: 0; @bottom-center { content: ""; } }

* { box-sizing: border-box; }

body {
  font-family: "Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif;
  font-size: 9.3pt;
  line-height: 1.52;
  color: #1f2430;
}

/* ------------------------------------------------------------ cover ----- */
.cover {
  height: 297mm; width: 210mm; padding: 44mm 24mm 20mm 24mm;
  background: #0f1c33; color: #ffffff; page-break-after: always;
}
.cover .kicker { font-size: 10pt; letter-spacing: .28em; text-transform: uppercase;
                 color: #7fd1c1; margin-bottom: 14mm; }
.cover h1 { font-size: 30pt; line-height: 1.14; margin: 0 0 8mm 0; color: #fff;
            border: none; padding: 0; }
.cover .sub { font-size: 12.5pt; color: #b9c4da; margin-bottom: 22mm; line-height:1.5; }
.cover .rule { height: 3px; width: 46mm; background: #7fd1c1; margin-bottom: 12mm; }
.cover .meta { font-size: 9.5pt; color: #8f9dba; line-height: 1.9; }
.cover .meta b { color: #dfe7f5; font-weight: 600; }
.cover .box { margin-top: 16mm; padding: 7mm 8mm; border: 1px solid #2b3d5e;
              border-radius: 4px; background: #16243d; font-size: 9pt; color:#c3cfe4; }

/* ------------------------------------------------------------- toc ------ */
.toc { page-break-after: always; }
.toc h2 { border: none; margin-top: 0; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc li { margin: 0; padding: 1.6pt 0; border-bottom: 1px dotted #dde2ec; }
.toc li.l1 { font-weight: 700; font-size: 10pt; margin-top: 7pt; border-bottom: 1px solid #c8d0de; }
.toc li.l2 { padding-left: 7mm; font-size: 9pt; }
.toc li.l3 { padding-left: 14mm; font-size: 8.4pt; color: #4a5468; }
.toc a { color: inherit; text-decoration: none; }

/* ------------------------------------------------------- markdown cell -- */
.cell { margin: 0 0 3.2mm 0; }
.md { }
.md h1 {
  font-size: 17pt; margin: 9mm 0 4mm 0; padding-bottom: 2.2mm;
  border-bottom: 2.5px solid #0f1c33; color: #0f1c33;
  page-break-before: always; page-break-after: avoid;
  bookmark-level: 1; bookmark-state: closed;
}
.md h1:first-of-type { page-break-before: avoid; }
.md h2 {
  font-size: 13pt; margin: 7mm 0 3mm 0; padding-bottom: 1.4mm;
  border-bottom: 1px solid #c9d2e2; color: #16294a;
  page-break-after: avoid; bookmark-level: 2;
}
.md h3 { font-size: 11pt; margin: 5mm 0 2mm 0; color: #1d3357;
         page-break-after: avoid; bookmark-level: 3; }
.md h4 { font-size: 9.8pt; margin: 4mm 0 1.6mm 0; color: #33405c;
         page-break-after: avoid; }
.md p { margin: 0 0 2.4mm 0; }
.md ul, .md ol { margin: 0 0 2.6mm 0; padding-left: 6.5mm; }
.md li { margin-bottom: 1.1mm; }
.md hr { border: none; border-top: 1px solid #d8dee9; margin: 5mm 0; }
.md strong { color: #0e1830; }

.md code {
  font-family: "SFMono-Regular", "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace;
  font-size: 8.2pt; background: #f1f3f7; color: #b0326e;
  padding: 0.4pt 1.6pt; border-radius: 2px;
}
.md pre {
  background: #f6f8fa; border: 1px solid #e1e6ef; border-radius: 3px;
  padding: 2.4mm 3mm; margin: 0 0 3mm 0; overflow-wrap: break-word;
  page-break-inside: avoid;
}
.md pre code { background: none; color: #24292e; font-size: 8pt; padding: 0; }

.md blockquote {
  margin: 0 0 3mm 0; padding: 2.2mm 3.4mm; background: #fff8e6;
  border-left: 3.5px solid #e3a008; color: #533f03; font-size: 8.9pt;
}
.md blockquote p { margin: 0 0 1.4mm 0; }
.md blockquote p:last-child { margin-bottom: 0; }

.md table {
  border-collapse: collapse; width: 100%; margin: 0 0 3.4mm 0;
  font-size: 8.1pt; page-break-inside: avoid;
}
.md th {
  background: #eef1f7; border: 1px solid #ccd4e2; padding: 1.5mm 2mm;
  text-align: left; font-weight: 700; color: #16294a;
}
.md td { border: 1px solid #dbe1ec; padding: 1.4mm 2mm; vertical-align: top; }
.md tr:nth-child(even) td { background: #fafbfd; }

/* ---------------------------------------------------------- code cell --- */
/* NOT flexbox. WeasyPrint will not BEGIN a flex container on a partly-filled
   page, so a flex code cell that does not fit in the remaining space is pushed
   whole to the next page, stranding a half-empty one behind it. A positioned
   block fragments freely and fills the page. */
.codecell {
  position: relative; padding-left: 15mm; margin: 0 0 3.4mm 0;
}
.codecell .prompt {
  position: absolute; left: 0; top: 1.8mm; width: 13mm; text-align: right;
  font-family: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
  font-size: 7.6pt; color: #2d7ff9; font-weight: 600;
}
.codecell .input {
  background: #f7f7f8; border: 1px solid #d7dbe3;
  border-left: 3.5px solid #2d7ff9; border-radius: 2px; padding: 1.8mm 2.6mm;
}
.codecell pre {
  margin: 0; font-family: "SFMono-Regular", "SF Mono", Menlo, Consolas,
  "DejaVu Sans Mono", monospace; font-size: 7.7pt; line-height: 1.42;
  white-space: pre-wrap; word-break: break-word; color: #24292e;
}
.codecell:not(.long) { page-break-inside: avoid; }
.codecell.long { page-break-inside: auto; }

/* markdown cells that are "explanation" / "expected output" get a tint */
.md.note      { border-left: 3px solid #57a3a0; padding: 2.4mm 0 0.6mm 3.2mm;
                background: #f4faf9; }
.md.output    { border-left: 3px solid #9a6ec2; padding: 2.4mm 0 0.6mm 3.2mm;
                background: #faf7fd; }
.md.exercise  { border-left: 3px solid #d97706; padding: 2.4mm 0 0.6mm 3.2mm;
                background: #fffaf1; }

/* A markdown cell that is ONLY a heading and is followed by a code cell must
   NOT use page-break-after: avoid. WeasyPrint cannot start a fragmentable flex
   container on the same page as an avoided break, so the heading is pushed to
   a fresh page and the rest of that page is wasted. */
.md.hdr-before-code h1, .md.hdr-before-code h2,
.md.hdr-before-code h3, .md.hdr-before-code h4 { page-break-after: auto; }
"""


# --------------------------------------------------------------------------
def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s\-\.]", "", text).strip().lower()
    return re.sub(r"[\s\.]+", "-", s)[:70]


def md_to_html(src: str) -> str:
    return markdown.markdown(src, extensions=MD_EXT, output_format="html5")


def strip_fences(src: str) -> list[tuple[bool, str]]:
    """Split markdown into (in_code_fence, line) pairs so the TOC scanner can
    skip '# comment' lines inside ```bash / ```python blocks."""
    out, fence = [], False
    for line in src.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append((True, line))
        else:
            out.append((fence, line))
    return out


def is_heading_only(src: str) -> bool:
    """True if the markdown cell's content is a single heading and nothing else."""
    body = [ln for fenced, ln in strip_fences(src) if not fenced and ln.strip()]
    return len(body) == 1 and body[0].lstrip().startswith("#")


def classify(src: str) -> str:
    """Give explanation / expected-output / exercise cells their own tint."""
    head = src.lstrip()[:120].lower()
    if head.startswith("**explanation") or "\n**explanation.**" in src[:400].lower():
        return "note"
    if head.startswith("**expected output"):
        return "output"
    if head.startswith("## exercises") or head.startswith("# exercises") or \
       head.startswith("### exercise") or head.startswith("**exercise"):
        return "exercise"
    return ""


def build_html() -> str:
    formatter = HtmlFormatter(style="default", nowrap=False, cssclass="hl")
    pyg_css = formatter.get_style_defs(".hl")

    body_parts, toc_items = [], []
    counter = 0

    for idx, (kind, src) in enumerate(ALL_CELLS):
        next_is_code = (idx + 1 < len(ALL_CELLS) and ALL_CELLS[idx + 1][0] == "code")
        if kind == "markdown":
            # Collect headings for the table of contents and insert anchors -
            # but only OUTSIDE fenced code blocks, so a '# or' comment in a
            # bash snippet does not become a chapter.
            lines_out = []
            for fenced, line in strip_fences(src):
                m = None if fenced else re.match(r"^(#{1,3}) +(.+)$", line)
                if m:
                    level, text = len(m.group(1)), m.group(2).strip()
                    sid = f"h{len(toc_items)}-{slugify(text)}"
                    toc_items.append((level, text, sid))
                    lines_out.append(f'{m.group(1)} <a id="{sid}"></a>{text}')
                else:
                    lines_out.append(line)
            src_anchored = "\n".join(lines_out)
            extra = classify(src)
            if next_is_code and is_heading_only(src):
                extra += " hdr-before-code"
            body_parts.append(
                f'<div class="cell md {extra}">{md_to_html(src_anchored)}</div>')
        else:
            counter += 1
            lexer = BashLexer() if src.lstrip().startswith(("!pip", "!conda")) \
                else PythonLexer()
            hl = highlight(src, lexer, formatter)
            long_cls = " long" if src.count("\n") > 45 else ""
            body_parts.append(
                f'<div class="cell codecell{long_cls}">'
                f'<div class="prompt">In [{counter}]:</div>'
                f'<div class="input">{hl}</div></div>')

    toc_html = ['<div class="toc"><h2>Contents</h2><ul>']
    for level, text, sid in toc_items:
        toc_html.append(f'<li class="l{level}"><a href="#{sid}">{html.escape(text)}</a></li>')
    toc_html.append("</ul></div>")

    cover = f"""
    <div class="cover">
      <div class="kicker">Practical Python GIS</div>
      <div class="rule"></div>
      <h1>{html.escape(TITLE)}</h1>
      <div class="sub">{SUBTITLE}</div>
      <div class="meta">
        <b>Dataset</b> &nbsp; The Vallmara Basin, Republic of Kestria (entirely fictional)<br/>
        <b>Libraries</b> &nbsp; GeoPandas · Shapely · Rasterio · NumPy · pandas ·
        matplotlib · seaborn · SciPy · scikit-learn · statsmodels<br/>
        <b>Structure</b> &nbsp; 5 modules · 49 lessons · 4 exercise sets ·
        full solutions · end-to-end capstone<br/>
        <b>Code cells</b> &nbsp; {counter} runnable cells<br/>
        <b>Built</b> &nbsp; {date.today().isoformat()}
      </div>
      <div class="box">
        Run <code style="color:#7fd1c1">python generate_data.py</code> once to create the
        dataset, then work through <code style="color:#7fd1c1">Spatial_Data_Science_Course.ipynb</code>
        alongside this document. Every code cell in this PDF is exactly the code in the
        notebook — both are generated from the same source, so they cannot drift apart.
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title>
<style>{CSS}
{pyg_css}
.hl {{ background: transparent; }}
</style></head>
<body>
{cover}
{''.join(toc_html)}
{''.join(body_parts)}
</body></html>"""


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parent / "Spatial_Data_Science_Course.pdf"
    html_str = build_html()
    tmp_html = out.with_suffix(".build.html")
    tmp_html.write_text(html_str, encoding="utf-8")

    from weasyprint import HTML
    HTML(string=html_str, base_url=str(HERE)).write_pdf(str(out))
    tmp_html.unlink(missing_ok=True)
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
