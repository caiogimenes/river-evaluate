"""YAML config, CLI flags, and experiment RNG seeding."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.paths import REPO_ROOT, resolve_repo_path

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"

__all__ = [
    "DEFAULT_CONFIG",
    "apply_overrides",
    "load_config",
    "parse_args",
    "seed_everything",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Adaptive QO prequential evaluation suite.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (defaults match the original experiment).",
    )
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Pickle path (repo-relative or absolute).")
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args(argv)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.instances is not None:
        config["instances"] = args.instances
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs
    if args.output is not None:
        config.setdefault("output", {})["pickle"] = str(args.output)
    return config


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def resolve_config_path(path: Path) -> Path:
    """Resolve a config path relative to the repository root."""
    return path if path.is_absolute() else resolve_repo_path(path)
