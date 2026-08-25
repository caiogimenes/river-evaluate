"""Prequential evaluation: predict, update the metric, then learn."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from joblib import Parallel, delayed
from river.metrics import RMSE
from tqdm import tqdm

from src.evaluation.runner_log import RunnerLog

logger = logging.getLogger(__name__)

# Log a snapshot every 1% of the stream (at least once).
LOG_FRACTION = 100

__all__ = [
    "LOG_FRACTION",
    "evaluate",
    "evaluate_prequential",
    "evaluate_single_run",
    "run_prequential_eval",
    "run_prequential_eval_parallel",
]


def evaluate_prequential(
    dataset_name: str,
    stream: Iterable[tuple[dict[str, Any], Any]],
    model: Any,
    metric: Any,
    log_every: int,
    model_name: str,
) -> RunnerLog:
    """
    Avaliação pré-quencial: prediz, atualiza a métrica e então aprende.

    Amostras de complexidade/tempo/memória são registradas a cada ``log_every`` instâncias.
    """
    metric = metric.clone()
    log = RunnerLog(model_name=model_name, dataset_name=dataset_name)

    logger.info("Evaluating %s on %s", model_name, dataset_name)

    for i, (x, y) in enumerate(stream):
        start_pred = time.perf_counter()
        y_pred = model.predict_one(x)
        end_pred = time.perf_counter()

        metric.update(y, y_pred)

        start_learn = time.perf_counter()
        model.learn_one(x, y)
        end_learn = time.perf_counter()

        if (i + 1) % log_every == 0:
            log.update(
                steps=i + 1,
                performance=metric.get(),
                n_nodes=getattr(model, "n_nodes", 0),
                n_leaves=getattr(model, "n_leaves", 0),
                height=getattr(model, "height", 0),
                inference_time=(end_pred - start_pred) * 1e6,
                learn_time=(end_learn - start_learn) * 1e6,
                memory_usage=getattr(model, "_raw_memory_usage", 0),
            )

    logger.info("Final error: %.6f", metric.get())
    return log


def evaluate(dataset: dict, model: dict, metric: Any, print_every: int = 100) -> RunnerLog:
    """Compatibilidade com a API antiga baseada em dicts de um único item."""
    dataset_name, dataset_stream = next(iter(dataset.items()))
    model_name, eval_model = next(iter(model.items()))
    return evaluate_prequential(
        dataset_name=dataset_name,
        stream=dataset_stream,
        model=eval_model,
        metric=metric,
        log_every=print_every,
        model_name=model_name,
    )


def evaluate_single_run(
    dataset_tuple: tuple[str, Callable[[], Any]],
    model_tuple: tuple[str, Any],
    instances: int,
    print_every: int,
) -> RunnerLog:
    """Executa um par dataset/modelo isolado (processo joblib)."""
    dataset_name, dataset_factory = dataset_tuple
    model_name, model_proto = model_tuple

    stream = dataset_factory().take(instances)
    model = model_proto.clone()
    return evaluate_prequential(
        dataset_name=dataset_name,
        stream=stream,
        model=model,
        metric=RMSE(),
        log_every=print_every,
        model_name=model_name,
    )


def run_prequential_eval(
    models: Mapping[str, Any],
    datasets: Mapping[str, Callable[[], Any]],
    instances: int,
) -> list[RunnerLog]:
    logs: list[RunnerLog] = []
    log_every = max(instances // LOG_FRACTION, 1)
    for dataset_name, dataset_generator in datasets.items():
        for model_name, model in models.items():
            logs.append(
                evaluate(
                    {dataset_name: dataset_generator().take(instances)},
                    {model_name: model.clone()},
                    RMSE(),
                    print_every=log_every,
                )
            )
    return logs


def run_prequential_eval_parallel(
    models: Mapping[str, Any],
    datasets: Mapping[str, Callable[[], Any]],
    instances: int,
    n_jobs: int = -1,
) -> list[RunnerLog]:
    n_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    logger.info("Running on %s cores", n_cores)
    log_every = max(instances // LOG_FRACTION, 1)
    tasks = []
    for dataset_name, dataset_factory in datasets.items():
        for model_name, model_proto in models.items():
            tasks.append(
                delayed(evaluate_single_run)(
                    (dataset_name, dataset_factory),
                    (model_name, model_proto),
                    instances,
                    log_every,
                )
            )

    logs = Parallel(n_jobs=n_jobs, backend="loky", batch_size=2)(
        tqdm(tasks, total=len(tasks), desc="Progresso Geral")
    )
    return logs
