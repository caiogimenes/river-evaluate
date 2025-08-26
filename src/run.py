import time
from river import tree
from river import metrics
from river import datasets
from src.data import RunnerLog

def run(model: tree.HoeffdingTreeRegressor, dataset, metric: metrics.RMSE, register_rate=100):
    log = RunnerLog(
        model=model.__class__.__name__,
        dataset=dataset,
    )
    metric = metric.clone()
    print(f"Evaluating {model.__class__.__name__}")


    for i, (x, y) in enumerate(dataset):
        start_pred = time.perf_counter()
        y_pred = model.predict_one(x)
        end_pred = time.perf_counter()

        metric.update(y, y_pred)

        start_learn = time.perf_counter()
        model.learn_one(x, y)
        end_learn = time.perf_counter()

        if (i + 1) % register_rate == 0:
            log.update(
                steps=i+1,
                performance=metric.get(),
                n_nodes=getattr(model, 'n_nodes', 0),
                n_leaves=getattr(model, "n_leaves", 0),
                height=getattr(model, 'height', 0),
                inference_time=(end_pred - start_pred) * 1_000_000,
                learn_time=(end_learn - start_learn) * 1_000_000,
                memory_usage=getattr(model, '_raw_memory_usage', 0)
            )

    print(f"Final {metric}: {metric.get():.6f}")
    return log

def main():
    log = run(
        model=tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(alpha=0.99)
        ),
        dataset=datasets.synth.ConceptDriftStream(
            stream=datasets.synth.FriedmanDrift(seed=42),
            drift_stream=datasets.synth.FriedmanDrift(seed=42, drift_type='gra', position=(10_000, 30_000)),
            seed=42, width=5000
        ),
        metric=metrics.RMSE()
    )