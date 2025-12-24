import matplotlib.pyplot as plt
import pandas as pd

def draw_cd_diagram(df_ranks, cd, width=10, height=7, title="Diagram"):
    """
    Gera um diagrama de Diferença Crítica (CD) a partir de rankings e valor de CD.
    """
    if isinstance(df_ranks, dict):
        df_ranks = pd.Series(df_ranks)

    # Ordenar rankings (menor é melhor)
    df_ranks = df_ranks.sort_values()
    names = df_ranks.index
    values = df_ranks.values

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(width, height))
    fig.suptitle(title)
    # Definir limites do eixo
    min_rank = 1 # ou floor(values.min())
    max_rank = len(values) # ou ceil(values.max())

    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 3) # Altura abstrata para desenhar os elementos
    ax.axis('off') # Remove bordas padrão

    # Posição da linha principal do eixo
    y_base = 1.0

    # Desenhar linha do eixo
    ax.plot([min_rank, max_rank], [y_base, y_base], color='black', linewidth=1.5)

    # Desenhar ticks e rótulos do eixo
    for i in range(int(min_rank), int(max_rank) + 1):
        ax.plot([i, i], [y_base, y_base + 0.1], color='black', linewidth=1)
        ax.text(i, y_base + 0.2, str(i), ha='center', va='bottom', fontsize=10)

    ax.text(min_rank - 0.2, y_base, 'Average Rank', ha='right', va='center', fontweight='bold')

    # Lógica para as barras de "Clique" (grupos sem diferença significativa)
    # Um grupo está conectado se a diferença entre o melhor e o pior rank do grupo for < CD
    cliques = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[j] - values[i] < cd:
                # Potencial clique, vamos ver se é o mais longo a partir de i
                if j == n-1 or (values[j+1] - values[i] >= cd):
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

    # Desenhar as barras de conexão (cliques) acima do eixo
    y_clique_start = y_base + 0.5
    y_step = 0.15

    for idx, (start_idx, end_idx) in enumerate(final_cliques):
        start_val = values[start_idx]
        end_val = values[end_idx]
        y_pos = y_clique_start + (idx * y_step)

        # Linha grossa conectando os algoritmos
        ax.plot([start_val, end_val], [y_pos, y_pos], linewidth=4, color='black', alpha=0.7)

    # Desenhar barra de referência do CD (canto superior esquerdo)
    ax.plot([min_rank, min_rank + cd], [2.8, 2.8], color='red', linewidth=2)
    ax.text(min_rank + cd/2, 2.9, f'CD = {cd}', ha='center', va='bottom', color='red', fontweight='bold')

    # Desenhar os pontos e rótulos dos algoritmos
    # Alternar alturas para evitar sobreposição de texto
    for i, (name, r) in enumerate(zip(names, values)):
        # Ponto no eixo
        ax.scatter(r, y_base, color='black', s=50, zorder=3)

        # Linha tracejada descendo
        level = (i % 3) # Níveis para alternar altura do texto
        text_y = y_base - 0.3 - (level * 0.25)

        ax.plot([r, r], [y_base, text_y], color='gray', linestyle=':', linewidth=1)

        if "AQO" in name:
            text_color = "red"
        else:
            text_color = "grey"

        # Rótulo com caixa
        ax.text(r, text_y, f'{name}\n({r:.2f})',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=text_color, alpha=0.8))

    return fig