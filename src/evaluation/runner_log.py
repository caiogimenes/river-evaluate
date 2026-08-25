"""Prequential run log (pickle-compatible via ``src.data.run_log``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["RunnerLog"]


@dataclass
class RunnerLog:
    """Per (model, dataset) timeseries recorded during prequential evaluation."""

    model: str
    dataset: str
    steps: list[int] = field(default_factory=list)
    performance: list[float] = field(default_factory=list)
    n_nodes: list[int] = field(default_factory=list)
    n_leaves: list[int] = field(default_factory=list)
    height: list[int] = field(default_factory=list)
    inference_time: list[float] = field(default_factory=list)
    learn_time: list[float] = field(default_factory=list)
    memory_usage: list[float] = field(default_factory=list)

    def __init__(self, model_name: str, dataset_name: str) -> None:
        self.model = model_name
        self.dataset = dataset_name
        self.steps = []
        self.performance = []
        self.n_nodes = []
        self.n_leaves = []
        self.height = []
        self.inference_time = []
        self.learn_time = []
        self.memory_usage = []

    def update(self, **kwargs: object) -> None:
        self.steps.append(kwargs.get("steps"))
        self.performance.append(kwargs.get("performance"))
        self.n_nodes.append(kwargs.get("n_nodes"))
        self.n_leaves.append(kwargs.get("n_leaves"))
        self.height.append(kwargs.get("height"))
        self.inference_time.append(kwargs.get("inference_time"))
        self.learn_time.append(kwargs.get("learn_time"))
        self.memory_usage.append(kwargs.get("memory_usage"))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.to_dict(),
            index=self.steps,
        )

    def to_dict(self) -> dict[str, list]:
        return {
            "steps": self.steps,
            "performance": self.performance,
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "height": self.height,
            "inference_time": self.inference_time,
            "learn_time": self.learn_time,
            "memory_usage": self.memory_usage,
        }
