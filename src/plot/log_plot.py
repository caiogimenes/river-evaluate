from src.data import RunnerLog
import matplotlib.pyplot as plt
from typing import List

class Plots:
    def __init__(self):
        self.figsize = (20,15)

    def plot_performance(self, logs: List[RunnerLog]):
        plt.figure(figsize=self.figsize)
        for log in logs:
            plt.title(label="Results for {dataset}".format(dataset=log.dataset.__class__.__name__))
            plt.plot(log.steps, log.performance, label=log.model)
        plt.legend()
        plt.show()
        return