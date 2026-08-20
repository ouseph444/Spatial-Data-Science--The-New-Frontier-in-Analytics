#!/usr/bin/env python3
"""Assemble the course cells into Spatial_Data_Science_Course.ipynb."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from part0_setup import CELLS as C0            # noqa: E402
from part1_beginner import CELLS as C1         # noqa: E402
from part2_intermediate import CELLS as C2     # noqa: E402
from part3_advanced import CELLS as C3         # noqa: E402
from part4_capstone import CELLS as C4         # noqa: E402

ALL_CELLS = C0 + C1 + C2 + C3 + C4


def build(out_path: Path) -> dict:
    cells = []
    for i, (kind, src) in enumerate(ALL_CELLS):
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        # nbformat 4.5+ requires a stable per-cell id; derive it from the
        # position so rebuilds produce identical notebooks.
        cid = f"cell-{i:04d}"
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": cid,
                          "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "id": cid, "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {
                "name": "python", "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "title": "Spatial Data Science: The New Frontier in Analytics",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return nb


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parent / "Spatial_Data_Science_Course.ipynb"
    nb = build(out)
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"Wrote {out}  ({len(nb['cells'])} cells: {n_md} markdown, {n_code} code)")
