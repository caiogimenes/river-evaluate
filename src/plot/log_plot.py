from src.data import RunnerLog
import matplotlib.pyplot as plt
from typing import List

class Plots:
    def __init__(self):
        self.figsize = (20,25)

    def plot_performance(self, logs: List[RunnerLog]):
        datasets = {log.dataset for log in logs}
        for dataset in datasets:
            plt.figure(figsize=self.figsize)
            for log in logs:
                if log.dataset == dataset:
                    plt.title(label="Results for {dataset}".format(dataset=dataset))
                    plt.plot(log.steps, log.performance, label=log.model)
            plt.legend()
            plt.show()
        return

    def plot_all(self, logs: List[RunnerLog]):
        datasets = {log.dataset for log in logs}
        for dataset in datasets:
            fig, axs = plt.subplots(3,2, figsize=self.figsize)
            fig.suptitle(f"Comparative analysis on {dataset}", fontsize=20)
            (ax1, ax2), (ax3, ax4), (ax5, ax6) = axs
            for log in logs:
                if log.dataset == dataset:
                    plt.title(label="Results for {dataset}".format(dataset=dataset))

                    ax1.set_title("Performance (RMSE)")
                    ax1.plot(log.steps, log.performance, label=log.model)

                    ax2.set_title("Complexity (Number of leaves)")
                    ax2.plot(log.steps, log.n_leaves, label=log.model)

                    ax3.set_title("Time - μs - (Inference)")
                    ax3.plot(log.steps, log.inference_time, label=log.model)

                    ax4.set_title("Time - μs - (Learn)")
                    ax4.plot(log.steps, log.learn_time, label=log.model)

                    ax5.set_title("Complexity (Height)")
                    ax5.plot(log.steps, log.height, label=log.model)

                    ax6.set_title("Memory usages")
                    ax6.plot(log.steps, log.memory_usage, label=log.model)

                    for ax in axs.flatten():
                        ax.set_xlabel("Instances")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.6)
            plt.show()
        return