"""Performance vs efficiency trade-off scatter (IQR ellipses, 0.5–6.5 axes)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt

from src.evaluation.runner_log import RunnerLog
from src.paths import resolve_repo_path
from src.stats.ranking import rank_logs

__all__ = ["TradeoffPlot"]


class TradeoffPlot:
    """
    Gera um gráfico de dispersão com Quadrantes de Decisão e Elipses de Confiança,
    mostrando o trade-off entre Performance (Accuracy) e Eficiência (Memory).
    """

    def __init__(
        self,
        logs: Sequence[RunnerLog],
        models: Iterable[str],
        datasets: Iterable[str],
        figsize: tuple[float, float] = (10, 8),
        title: str = "Performance Trade-off (Confidence Zones)",
    ) -> None:
        self.logs = logs
        self.models = list(models)
        self.datasets = list(datasets)
        self.figsize = figsize
        self.title = title

        perf_matrix = rank_logs(logs, "performance", self.models, self.datasets)
        mem_matrix = rank_logs(logs, "memory", self.models, self.datasets)

        self.perf_ranks = perf_matrix.rank(axis=1, ascending=True, method="average")
        self.mem_ranks = mem_matrix.rank(axis=1, ascending=True, method="average")

        self.perf_stats = self._calculate_stats(self.perf_ranks)
        self.mem_stats = self._calculate_stats(self.mem_ranks)

    def _calculate_stats(self, ranks_df) -> dict[str, dict[str, float]]:
        stats = {}
        for model in self.models:
            col = ranks_df[model]
            median = col.median()
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            stats[model] = {'median': median, 'q1': q1, 'q3': q3}
        return stats

    def _is_proposed(self, model_name: str) -> bool:
        return "AQO" in model_name

    def plot(self, filename: str | Path | None = None, dpi: int = 300):
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        min_val, max_val = 0.5, 6.5
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # 1. Calcular o centro geométrico (mediana global) para os Quadrantes
        all_x_medians = [self.mem_stats[m]['median'] for m in self.models]
        all_y_medians = [self.perf_stats[m]['median'] for m in self.models]
        global_x_median = np.median(all_x_medians)
        global_y_median = np.median(all_y_medians)

        # 2. Desenhar o Quadrante Ideal (Canto Inferior Esquerdo: Baixo Rank = Melhor)
        optimal_zone = Rectangle(
            (min_val, min_val),
            global_x_median - min_val,
            global_y_median - min_val,
            linewidth=0, facecolor='#2ca02c', alpha=0.1, zorder=1
        )
        ax.add_patch(optimal_zone)

        # Linhas divisórias dos quadrantes
        ax.axhline(global_y_median, color='gray', linestyle='--', alpha=0.4, zorder=1)
        ax.axvline(global_x_median, color='gray', linestyle='--', alpha=0.4, zorder=1)

        # Rótulo intuitivo para a zona verde
        ax.text(min_val + 0.2, min_val + 0.2, 'Optimal Zone\n(Fast & Accurate)',
                color='#2ca02c', alpha=0.8, fontsize=11, fontweight='bold', va='bottom', ha='left', zorder=2)

        # Linha diagonal de referência
        ax.plot([min_val, max_val], [min_val, max_val], 'gray', linestyle='-', linewidth=1.5, alpha=0.3,
                label='Equilibrium')

        plot_configs = [
            ([m for m in self.models if not self._is_proposed(m)], 's', 'blue', 'Baselines'),
            ([m for m in self.models if self._is_proposed(m)], '^', 'red', 'Proposed AQO')
        ]

        # 3. Desenhar as Elipses (Nuvens de Confiança) e os Pontos (Medianas)
        for model_group, marker, color, _ in plot_configs:
            for model in model_group:
                x = self.mem_stats[model]['median']
                y = self.perf_stats[model]['median']

                # Largura (Memória Q3-Q1) e Altura (Performance Q3-Q1)
                width = max(self.mem_stats[model]['q3'] - self.mem_stats[model]['q1'], 0.1)
                height = max(self.perf_stats[model]['q3'] - self.perf_stats[model]['q1'], 0.1)

                # Elipse substituindo as antigas barras de erro ortogonais
                ellipse = Ellipse((x, y), width=width, height=height,
                                  facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.2, zorder=3)
                ax.add_patch(ellipse)

                # Marcador central (Mediana)
                ax.scatter(x, y, marker=marker, s=100, color=color, edgecolors='white', linewidth=1.2, zorder=4)

                # Nome do modelo limpo (Removendo redundância)
                clean_name = model.replace(" (baseline)", "")
                ax.annotate(clean_name, (x, y), xytext=(6, 6), textcoords='offset points',
                            fontsize=9, color=color, zorder=5)

        # Configurações finais dos eixos
        ax.set_xlabel('Efficiency Rank (Median & IQR Cloud) →', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Rank (Median & IQR Cloud) →', fontsize=12, fontweight='bold')
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=20)

        # Inverter os eixos se desejar que o melhor (1) fique no topo/direita (Opcional, mantido padrão)
        # ax.invert_xaxis()
        # ax.invert_yaxis()

        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Baselines'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Proposed AQO'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.95)

        plt.tight_layout()

        if filename:
            save_path = resolve_repo_path(filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")

        return fig

    def export_tikz(self, filename: str | Path | None = None) -> str:
        """Exporta a versão otimizada em quadrantes e elipses para código TikZ."""

        all_x_medians = [self.mem_stats[m]['median'] for m in self.models]
        all_y_medians = [self.perf_stats[m]['median'] for m in self.models]
        global_x = np.median(all_x_medians)
        global_y = np.median(all_y_medians)

        # Configuração base do eixo
        tikz_code = r"\begin{tikzpicture}" + "\n"
        tikz_code += r"  \begin{axis}[" + "\n"
        tikz_code += r"    width=12cm, height=10cm," + "\n"
        tikz_code += r"    xlabel={Efficiency Rank $\rightarrow$}," + "\n"
        tikz_code += r"    ylabel={Accuracy Rank $\rightarrow$}," + "\n"
        tikz_code += r"    title={Performance Trade-off (Confidence Zones)}," + "\n"
        tikz_code += r"    legend pos=outer north east," + "\n"
        tikz_code += r"    legend cell align={left}," + "\n"
        tikz_code += r"    xmin=0.5, xmax=6.5," + "\n"
        tikz_code += r"    ymin=0.5, ymax=6.5," + "\n"
        tikz_code += r"    axis background/.style={fill=white}," + "\n"
        tikz_code += r"    axis on top=false," + "\n"  # Garante que os marcadores fiquem acima das grades
        tikz_code += r"  ]" + "\n\n"

        # 1. LAYER DE FUNDO: Zona Verde (Quadrante Ideal)
        tikz_code += r"  % Optimal Zone Background" + "\n"
        tikz_code += f"  \\fill[green!10] (0.5, 0.5) rectangle ({global_x:.2f}, {global_y:.2f});\n"
        tikz_code += r"  \node[text=black!60!green, above right, font=\small\bfseries] at (0.5, 0.5) {Optimal Zone};\n\n"

        # 2. LAYER DE GUIAS: Linhas divisórias e diagonal
        tikz_code += r"  % Grid Lines and Equilibrium" + "\n"
        tikz_code += f"  \\draw[gray, dashed, opacity=0.5] (0.5, {global_y:.2f}) -- (6.5, {global_y:.2f});\n"
        tikz_code += f"  \\draw[gray, dashed, opacity=0.5] ({global_x:.2f}, 0.5) -- ({global_x:.2f}, 6.5);\n"
        tikz_code += r"  \addplot[gray, thick, opacity=0.3, mark=none, forget plot] coordinates {(0.5,0.5) (6.5,6.5)};" + "\n\n"

        plot_configs = [
            ([m for m in self.models if not self._is_proposed(m)], 'blue', 'square*'),
            ([m for m in self.models if self._is_proposed(m)], 'red', 'triangle*')
        ]

        # 3. LAYER DE INCERTEZA: Elipses (Nuvens de IQR)
        tikz_code += r"  % Confidence Ellipses (IQR)" + "\n"
        for model_group, color, mark in plot_configs:
            for model in model_group:
                x = self.mem_stats[model]['median']
                y = self.perf_stats[model]['median']
                width = max(self.mem_stats[model]['q3'] - self.mem_stats[model]['q1'], 0.1)
                height = max(self.perf_stats[model]['q3'] - self.perf_stats[model]['q1'], 0.1)

                x_rad = width / 2
                y_rad = height / 2

                tikz_code += f"  \\filldraw[fill={color}, draw={color}, fill opacity=0.15, draw opacity=0.6, thick] "
                tikz_code += f"({x:.2f}, {y:.2f}) ellipse [x radius={x_rad:.2f}, y radius={y_rad:.2f}];\n"
        tikz_code += "\n"

        # 4. LAYER DE DADOS E LEGENDAS: Marcadores e seus Rótulos
        tikz_code += r"  % Markers and Legends" + "\n"
        for model_group, color, mark in plot_configs:
            # Agrupar coordenadas do mesmo tipo para criar UMA única entrada de legenda precisa
            tikz_code += f"  \\addplot+[{color}, only marks, mark={mark}, mark size=3.5] coordinates {{\n"
            for model in model_group:
                x = self.mem_stats[model]['median']
                y = self.perf_stats[model]['median']
                tikz_code += f"    ({x:.2f}, {y:.2f})\n"
            tikz_code += "  };\n"

            label_entry = 'Proposed AQO' if color == 'red' else 'Baselines'
            tikz_code += f"  \\addlegendentry{{{label_entry}}}\n\n"

            # Rótulos nominais (distanciados do centro para evitar colisão visual)
            for model in model_group:
                x = self.mem_stats[model]['median']
                y = self.perf_stats[model]['median']
                clean_name = model.replace(" (baseline)", "")
                # inner sep=3pt cria uma margem de respiro ao redor do marcador
                tikz_code += f"  \\node[above right, inner sep=4pt, text={color}, font=\\scriptsize] at ({x:.2f}, {y:.2f}) {{{clean_name}}};\n"

        tikz_code += r"  \end{axis}" + "\n"
        tikz_code += r"\end{tikzpicture}" + "\n"

        if filename:
            save_path = resolve_repo_path(filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(tikz_code)
            print(f"Código TikZ salvo em: {save_path}")

        return tikz_code

    def export_pgfplots(self, filename: str | Path | None = None) -> str:
        """Exporta o documento LaTeX completo para compilação autônoma."""
        tikz_content = self.export_tikz()

        latex_code = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{pgfplots}
% Necessário para compatibilidade com os padrões mais recentes do pgfplots
\pgfplotsset{compat=1.18} 

\begin{document}
""" + tikz_content + r"""
\end{document}
"""

        if filename:
            save_path = resolve_repo_path(filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(latex_code)
            print(f"Arquivo LaTeX salvo em: {save_path}")
            print(f"Para compilar o PDF, execute: pdflatex {save_path}")

        return latex_code
