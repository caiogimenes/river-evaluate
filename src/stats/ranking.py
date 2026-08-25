"""Build the (dataset × model) matrix used by Friedman / CD / tradeoff plots."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from pandas import DataFrame

from src.evaluation.runner_log import RunnerLog

__all__ = ["rank_logs"]


def rank_logs(
    logs: Sequence[RunnerLog],
    att: str,
    models: Iterable[str],
    datasets: Iterable[str],
) -> DataFrame:
    models = list(models)
    datasets = list(datasets)
    model_set = set(models)
    dataset_set = set(datasets)
    for log in logs:
        if log.model not in model_set:
            raise ValueError(f"log.model {log.model!r} is not in models")
        if log.dataset not in dataset_set:
            raise ValueError(f"log.dataset {log.dataset!r} is not in datasets")

    friedman_matrix = np.zeros(shape=(len(datasets), len(models)))
    for log in logs:
        model_idx = models.index(log.model)
        dataset_idx = datasets.index(log.dataset)
        if att == "performance":
            friedman_matrix[dataset_idx, model_idx] = np.mean(log.performance)
        elif att == "memory":
            friedman_matrix[dataset_idx, model_idx] = np.mean(log.memory_usage)
        elif att == "time":
            friedman_matrix[dataset_idx, model_idx] = log.learn_time[-1]

    return DataFrame(friedman_matrix, columns=models, index=datasets)
