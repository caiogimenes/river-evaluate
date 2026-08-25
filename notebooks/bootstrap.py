"""Ensure ``import src`` works when Jupyter's cwd is ``notebooks/``."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (directory that contains ``src/``)."""
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent.parent)
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "paths.py").is_file():
            return candidate
    raise ModuleNotFoundError(
        "Could not find the river-evaluate root (expected src/paths.py). "
        "Open the notebook from the repo or from notebooks/."
    )


def ensure_src_importable() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ensure_src_importable()
