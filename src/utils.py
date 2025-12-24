import time
import numpy as np
from typing import List
from pandas import DataFrame
from src.data import RunnerLog
from river.metrics import RMSE

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

    return DataFrame(friedman_matrix, columns=models)


def friedman_statistics(avg_rank, N: int):
    """
    Calculates the Friedman statistic.
    :param avg_rank: list or array of average ranks for each algorithm
    :param N: number of datasets (rows)
    """
    avg_rank = np.array(avg_rank)
    k = len(avg_rank)

    # Calculate Friedman statistic
    chi_sq = (12 * N / (k * (k + 1))) * (np.sum(avg_rank ** 2) - (k * (k + 1) ** 2) / 4)
    return chi_sq


def iman_davenport(chi_sq, N: int, k: int):
    """
    Calculates the Iman-Davenport statistic.
    :param chi_sq: The result from the Friedman statistic
    :param N: number of datasets
    :param k: number of algorithms (columns)
    """
    # Prevent division by zero if predictions are identical (denominator becomes 0)
    denominator = (N * (k - 1) - chi_sq)
    if denominator == 0:
        return np.inf

    return (N - 1) * chi_sq / denominator

def evaluate(dataset, model, metric, print_every=10):
    """
    Executa a avaliação progressiva e captura métricas de performance e complexidade.
    Retorna dicionários com o histórico de cada métrica para plotagem.
    """
    dataset_name, dataset_stream = dataset.popitem()
    model_name, eval_model  = model.popitem()

    log = RunnerLog(
        model_name=model_name,
        dataset_name=dataset_name,
    )
    metric = metric.clone()
    print(f"Evaluating {model_name} on {dataset_name}")

    for i, (x, y) in enumerate(dataset_stream):
        # Medir tempo de inferência
        start_pred = time.perf_counter()
        y_pred = eval_model.predict_one(x)
        end_pred = time.perf_counter()

        metric.update(y, y_pred)

        # Medir tempo de aprendizagem
        start_learn = time.perf_counter()
        eval_model.learn_one(x, y)
        end_learn = time.perf_counter()

        if (i + 1) % print_every == 0:
            log.update(
                steps=i+1,
                performance=metric.get(),
                n_nodes=getattr(eval_model, 'n_nodes', 0),
                n_leaves=getattr(eval_model, "n_leaves", 0),
                height=getattr(eval_model, 'height', 0),
                inference_time=(end_pred - start_pred) * 1e6,
                learn_time=(end_learn - start_learn) * 1e6,
                memory_usage=getattr(eval_model, '_raw_memory_usage', 0)
            )
    print(f"Final {metric}: {metric.get():.6f}")
    print("-" * 50)
    return log

def run_prequential_eval(models, datasets, instances):
    logs = []
    for d_name, dataset_generator in datasets.items():
        data_gen = dataset_generator[0]()
        for model_name, model in models.items():
            eval_model = {
                model_name: model.clone()
            }
            eval_dataset_stream = {
                d_name: data_gen.take(instances)
            }
            logs.append(evaluate(
                eval_dataset_stream,
                eval_model,
                RMSE(),
                print_every= instances / 100
            ))
    return logs