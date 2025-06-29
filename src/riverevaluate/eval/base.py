from abc import ABC, abstractmethod
from dataclasses import dataclass
from river.metrics.base import Metric


class Evaluation(ABC): ...


class Comparation(ABC):
    @abstractmethod
    def _validate_features(): ...

    @abstractmethod
    def _validade_targets(): ...


class Summary(ABC): ...


@dataclass
class TrainResult:
    model = None
    runtime: float = None
    eval_metric: Metric = None
    dataset: str = None
    model_name: str = None

    def set_results(self, model, runtime, eval_metric, dataset, model_name):
        self.model = model
        self.runtime = runtime
        self.eval_metric = eval_metric
        self.dataset = dataset
        self.model_name = model_name

    @property
    def memory_usage(self):
        return self.model._raw_memory_usage

    @property
    def params(self):
        return ["model", "runtime", "eval_metric", "dataset", "memory_usage"]


@dataclass
class Results: ...
