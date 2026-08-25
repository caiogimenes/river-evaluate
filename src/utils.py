from pandas import DataFrame
import numpy as np
from src.data import RunnerLog
from typing import List

from src.evaluation import (
    evaluate,
    run_prequential_eval,
    run_prequential_eval_parallel,
)

__all__ = [
    "rank_logs",
    "evaluate",
    "run_prequential_eval",
    "run_prequential_eval_parallel",
]


def rank_logs(logs: List[RunnerLog], att: str, models, datasets):
    models = list(models)
    datasets = list(datasets)
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
