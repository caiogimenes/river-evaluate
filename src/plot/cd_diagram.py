import matplotlib.pyplot as plt
import pandas as pd

# Configuração global para estilo científico
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'text.usetex': False  # Mude para True se tiver LaTeX instalado no sistema (opcional)
})


def draw_nemenyi_diagram(df_ranks, cd, filename=None, width=10, height=4, title=None):
    """
    Gera um diagrama de Diferença Crítica (CD) de Nemenyi no padrão científico.

    Args:
        df_ranks (dict or pd.Series): Dicionário ou Series com {Algoritmo: Rank_Medio}
        cd (float): Valor da Diferença Crítica (CD).
        filename (str): Caminho para salvar o arquivo (ex: 'diagrama.png'). Se None, apenas mostra.
        width (int): Largura da figura.
        height (int): Altura da figura.
        title (str): Título do gráfico (opcional).
    """
    if isinstance(df_ranks, dict):
        df_ranks = pd.Series(df_ranks)

    # Ordenar rankings (menor é melhor)
    df_ranks = df_ranks.sort_values()
    names = df_ranks.index
    values = df_ranks.values

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(width, height))
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)

    # Limites do eixo
    min_rank = 1
    max_rank = len(values)

    # Margem lateral para os rótulos não cortarem
    padding = 0.5
    ax.set_xlim(min_rank - padding, max_rank + padding)
    ax.set_ylim(0, 2.5)  # Altura reduzida para focar na informação
    ax.axis('off')  # Remove bordas padrão

    # Posição da linha principal do eixo (escala de ranks)
    y_base = 0.8

    # Desenhar linha do eixo principal
    ax.plot([min_rank, max_rank], [y_base, y_base], color='black', linewidth=1.5, zorder=1)

    # Desenhar ticks e rótulos do eixo
    # Usamos passos inteiros ou meios passos dependendo da densidade
    for i in range(int(min_rank), int(max_rank) + 1):
        ax.plot([i, i], [y_base, y_base + 0.05], color='black', linewidth=1)
        ax.text(i, y_base + 0.1, str(i), ha='center', va='bottom', fontsize=10)

    # Legenda do eixo
    ax.text(min_rank, y_base + 0.25, 'Average Rank', ha='left', va='center', fontweight='bold', fontsize=11)

    # --- LÓGICA DE CLIQUES (Grupos sem diferença estatística) ---
    cliques = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[j] - values[i] < cd:
                if j == n - 1 or (values[j + 1] - values[i] >= cd):
                    cliques.append((i, j))
            else:
                break

    # Filtrar sub-cliques (manter apenas os maximais)
    final_cliques = []
    for c in cliques:
        is_sub = False
        for other in cliques:
            if c != other and other[0] <= c[0] and other[1] >= c[1]:
                is_sub = True
                break
        if not is_sub:
            final_cliques.append(c)

    # Desenhar as barras de conexão (cliques) ACIMA do eixo
    # Ajuste fino: movemos as barras para cima para não colidir com os pontos
    y_clique_start = y_base + 0.4
    y_step = 0.15  # Espaçamento entre barras sobrepostas

    for idx, (start_idx, end_idx) in enumerate(final_cliques):
        start_val = values[start_idx]
        end_val = values[end_idx]
        # Ciclo de posições para evitar empilhamento infinito se houver muitos cliques
        y_pos = y_clique_start + ((idx % 3) * y_step)

        # Linha grossa conectando os algoritmos
        ax.plot([start_val, end_val], [y_pos, y_pos], linewidth=3, color='black', alpha=0.8)

    # --- BARRA DE REFERÊNCIA DO CD ---
    # Colocada no canto superior esquerdo ou direito
    cd_x_start = min_rank
    cd_y = y_base + 1.2

    ax.plot([cd_x_start, cd_x_start + cd], [cd_y, cd_y], color='black', linewidth=2)
    # Ticks nas pontas da barra de CD
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.05, cd_y + 0.05], color='black', linewidth=1)
    ax.plot([cd_x_start + cd, cd_x_start + cd], [cd_y - 0.05, cd_y + 0.05], color='black', linewidth=1)

    ax.text(cd_x_start + cd / 2, cd_y + 0.1, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- DESENHAR ALGORITMOS ---
    # Alternar alturas para evitar sobreposição de texto
    for i, (name, r) in enumerate(zip(names, values)):
        # Ponto no eixo
        # Destaque condicional mantido
        color = 'red' if "AQO" in name else 'black'

        ax.scatter(r, y_base, color=color, s=40, zorder=3, edgecolors='white')

        # Linha tracejada descendo
        level = (i % 3)  # Níveis para alternar altura do texto
        # Ajuste para garantir que o texto não fique muito longe nem muito perto
        text_y_base = y_base - 0.1
        text_y = text_y_base - 0.1 - (level * 0.4)

        ax.plot([r, r], [y_base, text_y + 0.1], color=color, linestyle=':', linewidth=0.8, alpha=0.6)

        # Formatação do texto
        font_weight = 'bold' if "AQO" in name else 'normal'

        # Rótulo sem caixa (bbox) para visual mais limpo, ou caixa muito sutil
        ax.text(r, text_y, f'{name}\n{r:.2f}',
                ha='center', va='top', fontsize=10, color='black', fontweight=font_weight,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Salvar ou mostrar
    plt.tight_layout()

    if filename:
        # DPI 300 é o padrão para publicações
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
        print(f"Gráfico salvo em: {filename}")

    return fig

def draw_bonferroni_dunn_diagram(df_ranks, cd, control_label, width=15, height=10, title="Bonferroni-Dunn Diagram"):
    """
    Gera um diagrama Bonferroni-Dunn focado em um algoritmo de controle.
    """
    if isinstance(df_ranks, dict):
        df_ranks = pd.Series(df_ranks)

    # Ordenar rankings (menor é melhor)
    df_ranks = df_ranks.sort_values()
    names = df_ranks.index
    values = df_ranks.values

    # Verificar se o controle existe
    if control_label not in names:
        raise ValueError(f"O rótulo de controle '{control_label}' não foi encontrado nos rankings.")

    ctrl_rank = df_ranks[control_label]

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(title)

    min_rank = 1
    max_rank = len(values)

    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Posição da linha principal do eixo
    y_base = 1.0

    # Desenhar linha do eixo
    ax.plot([min_rank, max_rank], [y_base, y_base], color='black', linewidth=1.5)

    # Ticks e rótulos do eixo
    for i in range(int(min_rank), int(max_rank) + 1):
        ax.plot([i, i], [y_base, y_base + 0.1], color='black', linewidth=1)
        ax.text(i, y_base + 0.2, str(i), ha='center', va='bottom', fontsize=10)

    ax.text(min_rank - 0.2, y_base, 'Average Rank', ha='right', va='center', fontweight='bold')

    # --- LÓGICA BONFERRONI-DUNN ---
    # Desenhar a barra de "Threshold" (Limiar) a partir do controle
    # Qualquer algoritmo dentro de [ctrl_rank, ctrl_rank + CD] não é significativamente pior

    y_threshold = y_base + 1.0
    threshold_limit = ctrl_rank + cd

    # Linha horizontal indicando o intervalo do CD (Significância)
    ax.plot([ctrl_rank, threshold_limit], [y_threshold, y_threshold], color='red', linewidth=3, alpha=0.7)

    # Linhas verticais delimitando o intervalo
    ax.plot([ctrl_rank, ctrl_rank], [y_base, y_threshold], color='red', linestyle='--', alpha=0.5)
    ax.plot([threshold_limit, threshold_limit], [y_base, y_threshold], color='red', linestyle='--', alpha=0.5)

    # Texto indicando o CD
    ax.text((ctrl_rank + threshold_limit) / 2, y_threshold + 0.1, f'CD = {cd}',
            ha='center', va='bottom', color='red', fontweight='bold')

    ax.text(threshold_limit, y_threshold + 0.1, 'Threshold',
            ha='center', va='bottom', color='red', fontsize=8)

    # --- DESENHAR ALGORITMOS ---
    for i, (name, r) in enumerate(zip(names, values)):
        # Cor do ponto: Vermelho se for controle, Cinza se não for, Preto se for estatisticamente diferente
        if name == control_label:
            color = 'red'
            edge_color = 'red'
            font_weight = 'bold'
        elif r <= threshold_limit:
            # Dentro do intervalo do CD (Estatisticamente igual ao controle)
            color = 'gray'
            edge_color = 'gray'
            font_weight = 'normal'
        else:
            # Fora do intervalo (Estatisticamente pior que o controle)
            color = 'black'
            edge_color = 'black'
            font_weight = 'normal'

        # Ponto no eixo
        ax.scatter(r, y_base, color=color, s=50, zorder=3)

        # Alternar alturas para texto
        level = (i % 3)
        text_y = y_base - 0.3 - (level * 0.25)

        ax.plot([r, r], [y_base, text_y], color='gray', linestyle=':', linewidth=1)

        # Rótulo
        ax.text(r, text_y, f'{name}\n({r:.2f})',
                ha='center', va='top', fontsize=9, fontweight=font_weight,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge_color, alpha=0.8))

    return fig