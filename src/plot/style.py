"""Shared matplotlib rcParams for publication figures.

Plot panels (``Plots``) and CD diagrams use separate dicts: merging them
would change font size / spine width on already published figures.
"""

from __future__ import annotations

from typing import Any

PLOT_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.3,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
}

CD_DIAGRAM_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "xtick.minor.width": 1.0,
    "text.usetex": False,  # Mude para True se tiver LaTeX instalado no sistema (opcional)
}
