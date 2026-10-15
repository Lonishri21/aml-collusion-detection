"""Shared figure style so every notebook's charts look and read the same."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Two-slot palette checked for colour-blind separation and contrast on a light background.
NORMAL = "#2a78d6"       # normal transactions / single-series bars
LAUNDERING = "#eb6834"   # laundering
INK = "#2b2b2b"          # text
MUTED = "#6b6b6b"        # secondary text, reference lines
GRID = "#e6e6e6"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
    })


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
