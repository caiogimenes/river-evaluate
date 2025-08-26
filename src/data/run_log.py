from dataclasses import dataclass
from typing import List
from pandas import DataFrame


@dataclass
class RunnerLog:
    model: str
    dataset: str
    steps: List[int]
    performance: List[float]
    n_nodes: List[int]
    n_leaves: List[int]
    height: List[int]
    inference_time: List[float]
    learn_time: List[float]
    memory_usage: List[float]

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.steps = []
        self.performance = []
        self.n_nodes = []
        self.n_leaves = []
        self.height = []
        self.inference_time = []
        self.learn_time = []
        self.memory_usage = []

    def update(self, **kwargs):
        self.steps.append(kwargs.get("steps"))
        self.performance.append(kwargs.get("performance"))
        self.n_nodes.append(kwargs.get("n_nodes"))
        self.n_leaves.append(kwargs.get("n_leaves"))
        self.height.append(kwargs.get("height"))
        self.inference_time.append(kwargs.get("inference_time"))
        self.learn_time.append(kwargs.get("learn_time"))
        self.memory_usage.append(kwargs.get("memory_usage"))

    def to_dataframe(self):
        return DataFrame(
            data=self.to_dict(),
            index=self.steps,
        )

    def to_dict(self):
        return {
            "steps": self.steps,
            "performance": self.performance,
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "height": self.height,
            "inference_time": self.inference_time,
            "learn_time": self.learn_time,
            "memory_usage": self.memory_usage
        }