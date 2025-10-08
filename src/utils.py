import time
from src.data import RunnerLog

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
                inference_time=(end_pred - start_pred) * 1_000_000,
                learn_time=(end_learn - start_learn) * 1_000_000,
                memory_usage=getattr(eval_model, '_raw_memory_usage', 0)
            )
    print(f"Final {metric}: {metric.get():.6f}")
    print("-" * 50)
    return log