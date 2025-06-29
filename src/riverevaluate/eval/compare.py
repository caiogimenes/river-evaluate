from .base import Comparation
from typing import List, Dict
from time import time
from river.stream import iter_pandas
from river.metrics import MAE
from .base import TrainResult
from pandas import DataFrame, Series
from collections import defaultdict


class CompareModels(Comparation):
    def __init__(
        self,
        models: List,
        features: List[DataFrame | Series],
        targets: List[DataFrame | Series],
        metrics: List = None,
        models_map: Dict = None,
        dataset_map: Dict = None,
    ):
        super().__init__()
        self.models = models
        self.features = self._validate_features(features)
        self.targets = self._validade_targets(targets)
        self.metrics = metrics
        self.models_map = models_map
        self.dataset_map = dataset_map

    def _validate_features(self, features):
        valid_features = []
        for feats in features:
            if isinstance(feats, DataFrame):
                valid_features.append(feats.select_dtypes(exclude=["O", "category"]))
        return valid_features

    def _validade_targets(self, targets: List) -> List:
        valid_targets = []
        for target in targets:
            if isinstance(target, DataFrame):
                target_name = target.columns[0]
                valid_targets.append(target[target_name])
        return valid_targets

    def train_one(
        self, model, features, target, dataset_name, model_name
    ) -> TrainResult:
        result = TrainResult()
        start_time = time()
        mae = MAE()
        for x, y in iter_pandas(features, target):
            y_pred = model.predict_one(x)
            try:
                mae.update(y, y_pred)
            except Exception as e:
                print(x)
                print(y)
                raise Exception(e)
            model.learn_one(x, y)

        runtime = time() - start_time

        result.set_results(
            model=model,
            runtime=runtime,
            eval_metric=mae,
            dataset=dataset_name,
            model_name=model_name,
        )

        return result

    def run_trainings(self) -> Dict[str, Dict[str, TrainResult]]:
        results: Dict[Dict[TrainResult]] = defaultdict(defaultdict)

        for i, model in enumerate(self.models):
            dataset_index = 0
            for X, y in zip(self.features, self.targets):
                result: TrainResult = self.train_one(
                    model=model,
                    features=X,
                    target=y,
                    dataset_name=self.dataset_map[dataset_index],
                    model_name=self.models_map[i],
                )
                results[result.model_name][result.dataset] = result
                dataset_index += 1

        return results
