from src.data import RunnerLog
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from itertools import cycle


class Plots:
    def __init__(self):
        # Configuração global de estilo para coincidir com LaTeX
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.linewidth': 1.0,
            'grid.alpha': 0.3,
            'legend.fontsize': 10,
            'lines.linewidth': 1.5
        })

        # Tamanho otimizado para uma página inteira de artigo (duas colunas ou página cheia)
        # Largura ~8-10 polegadas, Altura ~10-12 polegadas
        self.figsize = (10, 12)

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

    def _get_styles(self):
        """
        Gerador de estilos para diferenciar múltiplos algoritmos
        mesmo em preto e branco.
        """
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', 'x']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Cria um ciclo de combinações únicas
        combinations = []
        for i in range(len(colors)):
            combinations.append({
                'color': colors[i % len(colors)],
                'linestyle': line_styles[i % len(line_styles)],
                'marker': markers[i % len(markers)],
                'markevery': 0.1  # Não poluir a linha com marcadores em todos os pontos
            })
        return cycle(combinations)

    def export(self, logs: List[RunnerLog], save_path="./output/plots"):
        """
        Gera e salva painéis comparativos para cada dataset encontrado nos logs.

        Args:
            logs: Lista de objetos RunnerLog.
            save_path: Diretório onde salvar os arquivos PNG.
        """
        datasets = sorted(list({log.dataset for log in logs}))

        for i, dataset in enumerate(datasets):
            # Filtrar logs apenas deste dataset
            dataset_logs = [log for log in logs if log.dataset == dataset]

            # Configurar a grade 3x2
            fig, axs = plt.subplots(3, 2, figsize=self.figsize)

            # Título Geral da Figura (pode ser removido se for usar caption no LaTeX)
            # fig.suptitle(f"Comparative Analysis: {dataset}", fontsize=14, fontweight='bold', y=0.98)

            # Mapeamento dos eixos para facilitar
            (ax_perf, ax_leaves), (ax_inf_time, ax_learn_time), (ax_height, ax_mem) = axs

            # Gerador de estilos (reinicia para cada dataset para consistência de cores)
            style_cycle = self._get_styles()

            # Lista para coletar handles para uma legenda única
            handles, labels = [], []

            for log in dataset_logs:
                # Obter estilo único para este modelo
                st = next(style_cycle)
                lbl = log.model

                # Eixo X comum
                x = log.steps

                # Plot 1: Performance (RMSE)
                l1, = ax_perf.plot(x, log.performance, label=lbl, **st)

                # Plot 2: Complexity (Leaves)
                ax_leaves.plot(x, log.n_leaves, **st)

                # Plot 3: Cumulative Inference Time
                # Nota: Usei semilog se os valores variarem muito, mas mantive linear por padrão
                ax_inf_time.plot(x, np.cumsum(log.inference_time), **st)

                # Plot 4: Cumulative Learn Time
                ax_learn_time.plot(x, np.cumsum(log.learn_time), **st)

                # Plot 5: Complexity (Height)
                ax_height.plot(x, log.height, **st)

                # Plot 6: Memory Usage
                ax_mem.plot(x, log.memory_usage, **st)

                # Guardar handle para legenda
                if lbl not in labels:
                    handles.append(l1)
                    labels.append(lbl)

            # --- Configuração Específica dos Eixos ---

            # Dicionário para iterar configurações
            axes_settings = [
                (ax_perf, "Performance (RMSE)", "RMSE"),
                (ax_leaves, "Model Complexity", "Number of Leaves"),
                (ax_inf_time, "Cumulative Inference Time", "Time ($\mu$s)"),
                (ax_learn_time, "Cumulative Training Time", "Time ($\mu$s)"),
                (ax_height, "Tree Height", "Height"),
                (ax_mem, "Memory Usage", "Bytes")
            ]

            for ax, title, ylabel in axes_settings:
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_xlabel("Instances processed", fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)

                # Opcional: Notação científica no eixo X se forem muitos passos
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            # --- Legenda Única ---
            # Coloca a legenda abaixo de toda a figura
            fig.legend(handles, labels, loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=len(labels),
                       frameon=False, fontsize=11)

            # Ajuste de layout para não cortar nada
            plt.tight_layout()
            # Deixar espaço extra embaixo para a legenda
            plt.subplots_adjust(bottom=0.08, top=0.95)

            # Salvar
            filename = f"{save_path}/analysis_{i}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {filename}")
            plt.close(fig)  # Fecha para liberar memória


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
                    ax3.plot(log.steps, np.cumsum(log.inference_time), label=log.model)

                    ax4.set_title("Time - μs - (Learn)")
                    ax4.plot(log.steps, np.cumsum(log.learn_time), label=log.model)

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

    def plot_band_for_model(self, logs: List[RunnerLog], model):
        logs = [log for log in logs if log.model == model]
        performances = []
        for log in logs:
            performances.append(log.performance)
        array_performances = np.array(performances)
        array_performances.ravel()
        array_performances.reshape(len(performances), len(performances[0]))
        max_perf = np.max(array_performances, axis=0)
        min_perf = np.min(array_performances, axis=0)
        plt.fill_between(x=np.arange(len(performances[0])), y1=max_perf, y2=min_perf, alpha=0.5)
        plt.show()
        return

    def plot_performance_diff(self, logs: tuple):
        pass
