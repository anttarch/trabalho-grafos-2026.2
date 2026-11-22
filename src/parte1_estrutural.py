"""Parte I: Análise Estrutural Obrigatória do grafo Reddit Hyperlink Network.

Calcula todas as métricas estruturais exigidas, gera:
  - results/tables/parte1_metricas_resumo.csv e .json (com status Computada Sim/Não)
  - results/tables/parte1_top_hubs.csv
  - results/tables/parte1_componentes_undir.csv e parte1_componentes_scc.csv
  - results/tables/parte1_distribuicao_graus.csv
  - results/figures/parte1_distribuicao_graus.png / .pdf
  - results/figures/parte1_visualizacao_grafo.png / .pdf
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"


def load_igraph(csv_path: Path, directed: bool) -> ig.Graph:
    df = pd.read_csv(csv_path)
    edges = list(
        df[["source", "target", "weight_freq", "weight_dist", "weight_sent_mean", "neg_count"]]
        .itertuples(index=False, name=None)
    )
    return ig.Graph.TupleList(
        edges,
        directed=directed,
        edge_attrs=["weight_freq", "weight_dist", "weight_sent_mean", "neg_count"],
    )


def summarize_components(sizes: list[int]) -> pd.DataFrame:
    counts = Counter(sizes)
    rows = []
    for size, freq in sorted(counts.items(), key=lambda x: -x[0]):
        rows.append({"tamanho_componente": size, "quantidade_componentes": freq})
    return pd.DataFrame(rows)


def compute_exact_distance_metrics(g_gcc: ig.Graph) -> dict:
    """Calcula diâmetro, raio e comprimento médio dos caminhos exatos na GCC via igraph (C)."""
    t0 = time.perf_counter()
    print("  -> Calculando diâmetro exato na GCC...")
    diam = g_gcc.diameter(directed=False)
    t1 = time.perf_counter()
    print(f"     Diâmetro = {diam} ({t1 - t0:.2f}s)")

    print("  -> Calculando comprimento médio dos caminhos (L) exato na GCC...")
    avg_path = g_gcc.average_path_length(directed=False)
    t2 = time.perf_counter()
    print(f"     L = {avg_path:.6f} ({t2 - t1:.2f}s)")

    print("  -> Calculando raio exato na GCC...")
    rad = g_gcc.radius(mode="all")
    t3 = time.perf_counter()
    print(f"     Raio = {rad} ({t3 - t2:.2f}s)")

    return {
        "diameter": int(diam),
        "radius": int(rad),
        "avg_path_length": float(avg_path),
        "time_diameter_s": round(t1 - t0, 2),
        "time_avg_path_s": round(t2 - t1, 2),
        "time_radius_s": round(t3 - t2, 2),
    }


def plot_degree_distribution(deg_undir: np.ndarray, in_deg: np.ndarray, out_deg: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Painel 1: Distribuição de graus (PMF) em escala log-log
    for data, label, color, marker in [
        (deg_undir, "Não-dirigido (total)", "#1f77b4", "o"),
        (in_deg[in_deg > 0], "Dirigido (in-degree)", "#2ca02c", "^"),
        (out_deg[out_deg > 0], "Dirigido (out-degree)", "#d62728", "s"),
    ]:
        vals, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        axes[0].loglog(vals, probs, marker, markersize=3, alpha=0.65, label=label, color=color)

    axes[0].set_xlabel("Grau $k$ (escala log)")
    axes[0].set_ylabel("Probabilidade $P(k)$ (escala log)")
    axes[0].set_title("Distribuição de Graus (PMF)")
    axes[0].grid(True, which="both", ls="--", alpha=0.3)
    axes[0].legend(frameon=True, fontsize=8)

    # Painel 2: CCDF P(K >= k) em escala log-log
    for data, label, color in [
        (deg_undir, "Não-dirigido (total)", "#1f77b4"),
        (in_deg[in_deg > 0], "Dirigido (in-degree)", "#2ca02c"),
        (out_deg[out_deg > 0], "Dirigido (out-degree)", "#d62728"),
    ]:
        vals, counts = np.unique(data, return_counts=True)
        ccdf = np.cumsum(counts[::-1])[::-1] / counts.sum()
        axes[1].loglog(vals, ccdf, "-", linewidth=1.8, alpha=0.85, label=label, color=color)

    axes[1].set_xlabel("Grau $k$ (escala log)")
    axes[1].set_ylabel("$P(K \\geq k)$ (CCDF, escala log)")
    axes[1].set_title("Distribuição Acumulada Complementar (CCDF)")
    axes[1].grid(True, which="both", ls="--", alpha=0.3)
    axes[1].legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte1_distribuicao_graus.png", dpi=300)
    fig.savefig(FIGURES_DIR / "parte1_distribuicao_graus.pdf")
    plt.close(fig)


def plot_reduced_graph(g_gcc: ig.Graph, top_n: int = 50) -> None:
    """Visualização limpa dos top_n hubs da GCC com comunidades (Louvain)."""
    degrees = np.array(g_gcc.degree())
    top_indices = np.argsort(degrees)[::-1][:top_n].tolist()
    sub = g_gcc.subgraph(top_indices)

    # Detecção de comunidades para coloração
    communities = sub.community_multilevel(weights="weight_freq")
    membership = communities.membership

    # Converte para NetworkX para renderização de alta qualidade com Matplotlib
    G_nx = nx.Graph()
    for idx, v in enumerate(sub.vs):
        G_nx.add_node(v["name"], full_degree=int(degrees[top_indices[idx]]), comm=membership[idx])
    for e in sub.es:
        u = sub.vs[e.source]["name"]
        v = sub.vs[e.target]["name"]
        G_nx.add_edge(u, v, weight=float(e["weight_freq"]))

    # Mantém apenas as arestas mais fortes para não poluir visualmente (top 30% por peso)
    weights = [d["weight"] for _, _, d in G_nx.edges(data=True)]
    thresh = np.percentile(weights, 70) if weights else 0
    strong_edges = [(u, v) for u, v, d in G_nx.edges(data=True) if d["weight"] >= thresh]

    pos = nx.spring_layout(G_nx, k=0.55, iterations=120, seed=42, weight="weight")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    node_degrees = np.array([G_nx.nodes[n]["full_degree"] for n in G_nx.nodes()])
    node_sizes = 120 + 1800 * (node_degrees / node_degrees.max())
    node_colors = [G_nx.nodes[n]["comm"] for n in G_nx.nodes()]

    # Desenha arestas fracas muito sutis e fortes mais visíveis
    nx.draw_networkx_edges(G_nx, pos, alpha=0.07, width=0.5, edge_color="#888888", ax=ax)
    nx.draw_networkx_edges(
        G_nx, pos, edgelist=strong_edges, alpha=0.28, width=1.1, edge_color="#444444", ax=ax
    )

    nx.draw_networkx_nodes(
        G_nx,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        alpha=0.9,
        edgecolors="#222222",
        linewidths=0.7,
        ax=ax,
    )

    # Rótulos para os top 25 hubs (para manter legibilidade)
    top_label_nodes = sorted(
        G_nx.nodes(), key=lambda n: G_nx.nodes[n]["full_degree"], reverse=True
    )[:25]
    labels = {n: f"r/{n}" for n in top_label_nodes}
    nx.draw_networkx_labels(
        G_nx, pos, labels=labels, font_size=7.5, font_weight="bold", font_color="#111111", ax=ax
    )

    ax.set_title(
        f"Visualização Reduzida — Top {top_n} Subreddits (Hubs) na Maior Componente Conexa\n"
        "(Tamanho proporcional ao grau global; cores = comunidades Louvain)",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte1_visualizacao_grafo.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "parte1_visualizacao_grafo.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Carregando grafos tratados...")
    g_dir = load_igraph(DATA_PROCESSED_DIR / "edges_directed.csv.gz", directed=True)
    g_undir = load_igraph(DATA_PROCESSED_DIR / "edges_undirected.csv.gz", directed=False)
    g_gcc = load_igraph(DATA_PROCESSED_DIR / "edges_gcc.csv.gz", directed=False)

    print("[2/6] Calculando métricas básicas e graus...")
    n_dir, m_dir = g_dir.vcount(), g_dir.ecount()
    n_undir, m_undir = g_undir.vcount(), g_undir.ecount()
    n_gcc, m_gcc = g_gcc.vcount(), g_gcc.ecount()

    deg_undir = np.array(g_undir.degree())
    in_deg = np.array(g_dir.indegree())
    out_deg = np.array(g_dir.outdegree())
    tot_deg_dir = np.array(g_dir.degree())

    density_dir = g_dir.density()
    density_undir = g_undir.density()
    density_gcc = g_gcc.density()

    # Salva distribuição de graus não-dirigida
    vals, counts = np.unique(deg_undir, return_counts=True)
    pd.DataFrame({"grau": vals, "frequencia": counts, "probabilidade": counts / counts.sum()}).to_csv(
        TABLES_DIR / "parte1_distribuicao_graus.csv", index=False
    )

    # Top 15 hubs
    hub_rows = []
    dir_in_map = dict(zip(g_dir.vs["name"], in_deg))
    dir_out_map = dict(zip(g_dir.vs["name"], out_deg))
    for name, d in sorted(zip(g_undir.vs["name"], deg_undir), key=lambda x: -x[1])[:15]:
        hub_rows.append(
            {
                "subreddit": f"r/{name}",
                "grau_nao_dirigido": int(d),
                "in_degree": int(dir_in_map.get(name, 0)),
                "out_degree": int(dir_out_map.get(name, 0)),
            }
        )
    pd.DataFrame(hub_rows).to_csv(TABLES_DIR / "parte1_top_hubs.csv", index=False)

    print("[3/6] Calculando componentes conexas (WCC, SCC e não-dirigidas)...")
    wcc = g_dir.connected_components(mode="weak")
    scc = g_dir.connected_components(mode="strong")
    cc_undir = g_undir.connected_components()

    wcc_sizes = sorted(wcc.sizes(), reverse=True)
    scc_sizes = sorted(scc.sizes(), reverse=True)
    cc_undir_sizes = sorted(cc_undir.sizes(), reverse=True)

    df_comp_undir = summarize_components(cc_undir_sizes)
    df_comp_undir.to_csv(TABLES_DIR / "parte1_componentes_undir.csv", index=False)
    df_comp_scc = summarize_components(scc_sizes)
    df_comp_scc.to_csv(TABLES_DIR / "parte1_componentes_scc.csv", index=False)

    print("[4/6] Calculando clusterização e triângulos...")
    t0 = time.perf_counter()
    avg_clust_undir = g_undir.transitivity_avglocal_undirected(mode="zero")
    global_trans_undir = g_undir.transitivity_undirected()
    avg_clust_gcc = g_gcc.transitivity_avglocal_undirected(mode="zero")
    num_triangles = int(g_undir.motifs_randesu(size=3)[3])
    t_clust = time.perf_counter() - t0
    print(
        f"  -> C_local_medio = {avg_clust_undir:.6f}, Trans_global = {global_trans_undir:.6f}, "
        f"Triângulos = {num_triangles:,} ({t_clust:.2f}s)"
    )

    print("[5/6] Calculando distâncias (Diâmetro, Raio, Caminho Médio) na GCC...")
    dist_metrics = compute_exact_distance_metrics(g_gcc)

    print("[6/6] Gerando gráficos (distribuição de graus e visualização do subgrafo)...")
    plot_degree_distribution(deg_undir, in_deg, out_deg)
    plot_reduced_graph(g_gcc, top_n=50)

    # Monta tabela resumo obrigatória (indicando Computada Sim/Não)
    summary_rows = [
        ("Número de vértices (|V|) — Dirigido / Não-dirigido", f"{n_undir:,}", "Sim", "Após remoção de auto-loops"),
        ("Número de vértices (|V|) — Maior Componente (GCC)", f"{n_gcc:,} ({100*n_gcc/n_undir:.2f}%)", "Sim", "Extração da maior componente conexa"),
        ("Número de arestas (|E|) — Dirigido simples", f"{m_dir:,}", "Sim", "Agregado a partir de 858.488 links brutos"),
        ("Número de arestas (|E|) — Não-dirigido simples", f"{m_undir:,}", "Sim", "Fusão de pares (u,v) e (v,u)"),
        ("Número de arestas (|E|) — GCC (não-dirigido)", f"{m_gcc:,}", "Sim", "Arestas na maior componente conexa"),
        ("Grau mínimo / máximo / médio (Não-dirigido)", f"{deg_undir.min()} / {deg_undir.max():,} / {deg_undir.mean():.4f}", "Sim", f"Mediana = {np.median(deg_undir):.1f}"),
        ("In-degree mín / máx / médio (Dirigido)", f"{in_deg.min()} / {in_deg.max():,} / {in_deg.mean():.4f}", "Sim", f"Mediana = {np.median(in_deg):.1f}"),
        ("Out-degree mín / máx / médio (Dirigido)", f"{out_deg.min()} / {out_deg.max():,} / {out_deg.mean():.4f}", "Sim", f"Mediana = {np.median(out_deg):.1f}"),
        ("Distribuição de graus", "Ver Figura e CSV", "Sim", "PMF e CCDF em escala log-log"),
        ("Densidade (Dirigido)", f"{density_dir:.8f}", "Sim", "Grafo altamente esparso"),
        ("Densidade (Não-dirigido)", f"{density_undir:.8f}", "Sim", f"GCC = {density_gcc:.8f}"),
        ("Nº de componentes conexas (Não-dirigido / WCC)", f"{len(cc_undir_sizes):,}", "Sim", f"Maior = {cc_undir_sizes[0]:,}; 2ª maior = {cc_undir_sizes[1] if len(cc_undir_sizes)>1 else 0}"),
        ("Nº de componentes fortemente conexas (SCC)", f"{len(scc_sizes):,}", "Sim", f"Maior SCC = {scc_sizes[0]:,} ({100*scc_sizes[0]/n_dir:.2f}%)"),
        ("Tamanho de cada componente", "Ver CSVs de componentes", "Sim", "Distribuição completa salva em resultados"),
        ("Diâmetro (GCC)", f"{dist_metrics['diameter']}", "Sim", f"Exato na GCC ({dist_metrics['time_diameter_s']}s)"),
        ("Raio (GCC)", f"{dist_metrics['radius']}", "Sim", f"Exato na GCC ({dist_metrics['time_radius_s']}s)"),
        ("Comprimento médio dos caminhos L (GCC)", f"{dist_metrics['avg_path_length']:.4f}", "Sim", f"Exato na GCC ({dist_metrics['time_avg_path_s']}s)"),
        ("Coeficiente de clusterização médio (C)", f"{avg_clust_undir:.4f}", "Sim", f"GCC = {avg_clust_gcc:.4f}; Trans. global = {global_trans_undir:.4f}"),
        ("Número de triângulos", f"{num_triangles:,}", "Sim", "Contagem exata no grafo não-dirigido"),
        ("Visualização do grafo (versão reduzida)", "Ver Figura (Top 50 hubs)", "Sim", "Subgrafo induzido pelos 50 maiores hubs da GCC"),
    ]

    df_summary = pd.DataFrame(
        summary_rows, columns=["Medida", "Valor", "Computada", "Observacoes"]
    )
    df_summary.to_csv(TABLES_DIR / "parte1_metricas_resumo.csv", index=False)

    raw_json = {
        "n_dir": n_dir,
        "m_dir": m_dir,
        "n_undir": n_undir,
        "m_undir": m_undir,
        "n_gcc": n_gcc,
        "m_gcc": m_gcc,
        "deg_undir_min": int(deg_undir.min()),
        "deg_undir_max": int(deg_undir.max()),
        "deg_undir_mean": float(deg_undir.mean()),
        "deg_undir_median": float(np.median(deg_undir)),
        "in_deg_min": int(in_deg.min()),
        "in_deg_max": int(in_deg.max()),
        "in_deg_mean": float(in_deg.mean()),
        "out_deg_min": int(out_deg.min()),
        "out_deg_max": int(out_deg.max()),
        "out_deg_mean": float(out_deg.mean()),
        "tot_deg_dir_mean": float(tot_deg_dir.mean()),
        "density_dir": float(density_dir),
        "density_undir": float(density_undir),
        "density_gcc": float(density_gcc),
        "num_wcc": len(wcc_sizes),
        "largest_wcc": wcc_sizes[0],
        "num_scc": len(scc_sizes),
        "largest_scc": scc_sizes[0],
        "num_cc_undir": len(cc_undir_sizes),
        "largest_cc_undir": cc_undir_sizes[0],
        "avg_clustering_undir": float(avg_clust_undir),
        "avg_clustering_gcc": float(avg_clust_gcc),
        "global_transitivity_undir": float(global_trans_undir),
        "num_triangles": num_triangles,
        **dist_metrics,
    }
    with open(TABLES_DIR / "parte1_metricas_resumo.json", "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2, ensure_ascii=False)

    print("\n=== RESUMO PARTE I ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
