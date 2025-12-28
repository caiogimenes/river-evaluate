import pickle
from src.data import get_friedman_datasets, get_hyperplane_datasets, get_real_datasets, get_rbf_datasets
from src.models import get_models
from src.utils import run_prequential_eval, run_prequential_eval_parallel

INSTANCES = 1_000_000

if __name__ == "__main__":
    friedman_datasets = get_friedman_datasets("gsg", n_datasets=10, n_instances=INSTANCES)
    hyperplane_datasets = get_hyperplane_datasets(10)
    rbf_datasets = get_rbf_datasets(10)
    # real_datasets = get_real_datasets()

    test_datasets = rbf_datasets | friedman_datasets | hyperplane_datasets
    models = get_models()

    logs = run_prequential_eval_parallel(models, test_datasets, INSTANCES)

    with open("./logs/gradual.pkl", "wb") as file:
        pickle.dump(logs, file)