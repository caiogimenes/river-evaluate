import time
from river.metrics import RMSE
from pandas import DataFrame
import numpy as np
from src.data import RunnerLog
from joblib import Parallel, delayed
from typing import List
from tqdm import tqdm
import os


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

def evaluate(dataset, model, metric, print_every=100):
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
    print(f"Final error: {metric.get():.6f}")
    print("-" * 50)
    return log

def run_prequential_eval(models, datasets, instances):
    logs = []
    for d_name, dataset_generator in datasets.items():
        for model_name, model in models.items():
            eval_model = {
                model_name: model.clone()
            }
            eval_dataset_stream = {
                d_name: dataset_generator().take(instances)
            }
            logs.append(evaluate(
                eval_dataset_stream,
                eval_model,
                RMSE(),
                print_every= instances // 100
            ))
    return logs

def evaluate_single_run(dataset_tuple, model_tuple, instances, print_every):
    """
    Executa um treino isolado e retorna o log.
    Imprime apenas o resultado final para não bagunçar o output paralelo.
    """
    d_name, dataset_factory = dataset_tuple
    m_name, model_proto = model_tuple

    # Instancia o stream e o modelo para este processo específico
    try:
        stream = dataset_factory()
    except TypeError:
        # Fallback caso a correção da vírgula em synth_data.py não tenha sido feita
        stream = dataset_factory[0]()

    model = model_proto.clone()
    metric = RMSE()
    log = RunnerLog(model_name=m_name, dataset_name=d_name)

    # Limita o tamanho do stream
    stream = stream.take(instances)

    for i, (x, y) in enumerate(stream):
        # Inferência
        start_pred = time.perf_counter()
        y_pred = model.predict_one(x)
        end_pred = time.perf_counter()

        metric.update(y, y_pred)

        # Treino
        start_learn = time.perf_counter()
        model.learn_one(x, y)
        end_learn = time.perf_counter()

        # Log periódico (apenas interno, sem print)
        if (i + 1) % print_every == 0:
            log.update(
                steps=i + 1,
                performance=metric.get(),
                n_nodes=getattr(model, 'n_nodes', 0),
                n_leaves=getattr(model, "n_leaves", 0),
                height=getattr(model, 'height', 0),
                inference_time=(end_pred - start_pred) * 1e6,
                learn_time=(end_learn - start_learn) * 1e6,
                memory_usage=getattr(model, '_raw_memory_usage', 0)
            )

    return log


def run_prequential_eval_parallel(models, datasets, instances, n_jobs=-1):
    """
    Executa avaliação em paralelo com barra de progresso.
    """
    # Prepara a lista de tarefas
    print(f"Running on {os.cpu_count() if n_jobs == -1 else n_jobs} cores")
    tasks = []
    for d_name, d_factory in datasets.items():
        for m_name, m_proto in models.items():
            tasks.append(
                delayed(evaluate_single_run)(
                    (d_name, d_factory),
                    (m_name, m_proto),
                    instances,
                    instances // 100
                )
            )

    # A mágica do tqdm com joblib:
    # Usamos o tqdm para criar uma barra baseada no número de tarefas
    logs = Parallel(n_jobs=n_jobs, backend='loky')(
        tqdm(tasks, total=len(tasks), desc="Progresso Geral")
    )

    return logs