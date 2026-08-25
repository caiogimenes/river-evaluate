"""Assemble the experiment dataset registry from YAML config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.data import (
    get_friedman_datasets,
    get_hyperplane_datasets,
    get_rbf_datasets,
    get_real_datasets,
)

__all__ = ["build_datasets"]


def build_datasets(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Callable[[], Any]]:
    instances = config["instances"]
    friedman_cfg = config["friedman"]
    datasets: dict[str, Callable[[], Any]] = {}
    datasets.update(
        get_friedman_datasets(
            friedman_cfg["drift_type"],
            n_datasets=friedman_cfg["n_datasets"],
            n_instances=instances,
            rng=rng,
        )
    )
    datasets.update(get_hyperplane_datasets(config["hyperplane"]["n_datasets"], rng=rng))
    datasets.update(get_rbf_datasets(config["rbf"]["n_datasets"], rng=rng))
    if config.get("real", {}).get("enabled", True):
        datasets.update(get_real_datasets())
    return datasets
