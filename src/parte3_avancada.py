"""Parte III: Análise Estrutural Avançada (Small-World, Power Law, Robustez, Descoberta e Extra).

1. Small-World (Apêndice A) — IMPLEMENTAÇÃO PRÓPRIA sem funções prontas de small-world:
   - Cálculo manual de C(G) e L(G) (BFS própria com amostragem) na GCC
   - Geração manual de grafo aleatório equivalente G_rand(n, m) + fórmula analítica ER
   - Cálculo de sigma = (C / C_rand) / (L / L_rand)
2. Lei de Potência (Apêndice B):
   - PMF com log-binning + CCDF em escala log-log
   - Ajuste linear log-log (R²) e estimativa MLE de gamma (Clauset et al.)
3. Robustez (Apêndice C):
   - Remoção aleatória de 5% (T=30 repetições) com boxplots (média ± DP)
   - Ataque direcionado aos 5% mais centrais (Degree e PageRank)
   - Métricas A (maior componente), B (nº componentes), C (distância média), D (fração isolados)
4. Descoberta mais interessante:
   - Assimetria e concentração de conflitos (LINK_SENTIMENT = -1) + evolução temporal mensal
5. (EXTRA) Comparação com modelos clássicos: Erdős-Rényi, Watts-Strogatz e Barabási-Albert.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"


# ===========================================================================
# 1. SMALL-WORLD (Implementação própria sem funções prontas de biblioteca)
# ===========================================================================

def build_set_adj(df: pd.DataFrame) -> list[set[int]]:
    nodes = sorted(set(df["source"]).union(set(df["target"])))
    idx = {name: i for i, name in enumerate(nodes)}
    adj: list[set[int]] = [set() for _ in range(len(nodes))]
    for u_s, v_s in df[["source", "target"]].itertuples(index=False, name=None):
        u, v = idx[u_s], idx[v_s]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def manual_avg_clustering(adj: list[set[int]]) -> float:
    """Calcula o coeficiente de clusterização médio C(G) manualmente."""
    n = len(adj)
    total_c = 0.0
    for u in range(n):
        nbrs = adj[u]
        k = len(nbrs)
        if k < 2:
            continue
        # Conta arestas entre vizinhos de u
        links_between = 0
        nbr_list = list(nbrs)
        for v in nbr_list:
            # Interseção entre vizinhos de u e vizinhos de v (cada aresta contada 2x)
            if len(adj[v]) < k:
                links_between += sum(1 for w in adj[v] if w in nbrs)
            else:
                links_between += sum(1 for w in nbrs if w in adj[v])
        # Como cada aresta entre vizinhos foi contada 2x, 2 * e_u = links_between
        total_c += links_between / (k * (k - 1))
    return total_c / n


def manual_gcc_nodes(adj: list[set[int]]) -> list[int]:
    """Extrai os vértices da maior componente conexa manualmente via BFS."""
    n = len(adj)
    visited = bytearray(n)
    best_comp: list[int] = []
    for start in range(n):
        if visited[start]:
            continue
        comp = [start]
        visited[start] = 1
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = 1
                    comp.append(v)
                    q.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp
    return best_comp


def manual_avg_path_length_sampled(
    adj: list[set[int]], nodes_subset: list[int], sample_size: int = 500, seed: int = 42
) -> tuple[float, float]:
    """Estima L(G) usando BFS própria a partir de sample_size vértices aleatórios."""
    rng = random.Random(seed)
    sources = rng.sample(nodes_subset, min(sample_size, len(nodes_subset)))
    n = len(adj)
    per_source_means = []

    for s in sources:
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        total_d = 0
        reached = 0
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    total_d += du + 1
                    reached += 1
                    q.append(v)
        if reached > 0:
            per_source_means.append(total_d / reached)

    arr = np.array(per_source_means, dtype=float)
    mean = float(arr.mean())
    ci95 = float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, ci95


def manual_generate_random_graph(n: int, m: int, seed: int = 42) -> list[set[int]]:
    """Gera grafo aleatório G(n, m) manualmente sem funções prontas de biblioteca."""
    rng = random.Random(seed)
    adj: list[set[int]] = [set() for _ in range(n)]
    added = 0
    while added < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            added += 1
    return adj


def run_small_world_analysis(df_gcc: pd.DataFrame) -> dict:
    print("[1/5] Executando análise Small-World (implementação própria)...")
    adj_gcc = build_set_adj(df_gcc)
    n = len(adj_gcc)
    m = sum(len(s) for s in adj_gcc) // 2
    avg_k = 2.0 * m / n

    c_real = manual_avg_clustering(adj_gcc)
    l_real_est, l_real_ci = manual_avg_path_length_sampled(
        adj_gcc, list(range(n)), sample_size=500, seed=42
    )

    # Grafo aleatório equivalente gerado manualmente
    adj_rand = manual_generate_random_graph(n, m, seed=42)
    c_rand_sim = manual_avg_clustering(adj_rand)
    rand_gcc_nodes = manual_gcc_nodes(adj_rand)
    l_rand_sim, l_rand_ci = manual_avg_path_length_sampled(
        adj_rand, rand_gcc_nodes, sample_size=500, seed=42
    )

    # Fórmulas analíticas Erdős-Rényi
    c_rand_theory = avg_k / n
    l_rand_theory = math.log(n) / math.log(avg_k)

    # Coeficiente small-world sigma (Humphries & Gurney, 2008)
    gamma_ratio = c_real / c_rand_sim if c_rand_sim > 0 else float("inf")
    lambda_ratio = l_real_est / l_rand_sim
    sigma = gamma_ratio / lambda_ratio

    res = {
        "n_gcc": n,
        "m_gcc": m,
        "avg_degree": round(avg_k, 4),
        "C_real_manual": round(c_real, 6),
        "L_real_manual_est": round(l_real_est, 6),
        "L_real_manual_ci95": round(l_real_ci, 6),
        "C_rand_sim_manual": round(c_rand_sim, 6),
        "L_rand_sim_manual": round(l_rand_sim, 6),
        "L_rand_sim_ci95": round(l_rand_ci, 6),
        "rand_gcc_size": len(rand_gcc_nodes),
        "C_rand_teorico_ER": round(c_rand_theory, 6),
        "L_rand_teorico_ER": round(l_rand_theory, 6),
        "razao_C_sobre_Crand": round(gamma_ratio, 2),
        "razao_L_sobre_Lrand": round(lambda_ratio, 4),
        "sigma_small_world": round(sigma, 2),
    }

    with open(TABLES_DIR / "parte3_small_world.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    pd.DataFrame(list(res.items()), columns=["metrica", "valor"]).to_csv(
        TABLES_DIR / "parte3_small_world.csv", index=False
    )

    print(
        f"     C(G)={c_real:.4f} vs C_rand={c_rand_sim:.6f} ({gamma_ratio:.1f}x maior) | "
        f"L(G)={l_real_est:.4f} vs L_rand={l_rand_sim:.4f} | sigma={sigma:.1f}"
    )
    return res


# ===========================================================================
# 2. LEI DE POTÊNCIA / POWER LAW (Apêndice B)
# ===========================================================================

def run_power_law_analysis(df_undir: pd.DataFrame) -> dict:
    print("[2/5] Executando análise de Lei de Potência (Power Law)...")
    g = ig.Graph.TupleList(
        df_undir[["source", "target"]].itertuples(index=False, name=None), directed=False
    )
    degrees = np.array(g.degree(), dtype=int)
    vals, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    ccdf = np.cumsum(counts[::-1])[::-1] / counts.sum()

    # Log-binning para ajuste limpo da PMF na cauda (k >= 2)
    bins = np.logspace(np.log10(1), np.log10(degrees.max() + 1), num=35)
    hist, edges = np.histogram(degrees, bins=bins)
    bin_widths = np.diff(edges)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    pdf_binned = hist / (len(degrees) * bin_widths)
    mask = (hist >= 5) & (bin_centers >= 2)

    slope_pmf, intercept_pmf, r_pmf, _, _ = stats.linregress(
        np.log10(bin_centers[mask]), np.log10(pdf_binned[mask])
    )
    gamma_logbin = -slope_pmf
    r2_logbin = r_pmf ** 2

    # Ajuste na CCDF (para P(K >= k) ~ k^{-(gamma-1)}, então gamma = 1 - slope)
    mask_ccdf = (vals >= 5) & (vals <= np.percentile(degrees, 99.9))
    slope_ccdf, intercept_ccdf, r_ccdf, _, _ = stats.linregress(
        np.log10(vals[mask_ccdf]), np.log10(ccdf[mask_ccdf])
    )
    gamma_ccdf = 1.0 - slope_ccdf
    r2_ccdf = r_ccdf ** 2

    # Estimador de Máxima Verossimilhança (Clauset-Shalizi-Newman) para k_min = 10
    k_min = 10
    tail = degrees[degrees >= k_min]
    gamma_mle = 1.0 + len(tail) / np.sum(np.log(tail / (k_min - 0.5)))

    # Gráfico log-log dedicado
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Painel 1: PMF bruta + log-binning + ajuste linear
    axes[0].loglog(vals, probs, "o", markersize=2.5, alpha=0.3, color="#1f77b4", label="Empírico $P(k)$")
    axes[0].loglog(
        bin_centers[mask],
        pdf_binned[mask],
        "s",
        markersize=5,
        color="#d62728",
        label="Log-binning",
    )
    fit_x = bin_centers[mask]
    fit_y = 10 ** (intercept_pmf + slope_pmf * np.log10(fit_x))
    axes[0].loglog(
        fit_x,
        fit_y,
        "--",
        linewidth=2,
        color="#111111",
        label=f"Ajuste: $\\gamma={gamma_logbin:.2f}$ ($R^2={r2_logbin:.3f}$)",
    )
    axes[0].set_xlabel("Grau $k$ (escala log)")
    axes[0].set_ylabel("$P(k)$ (escala log)")
    axes[0].set_title("Lei de Potência — PMF com Log-Binning")
    axes[0].grid(True, which="both", ls="--", alpha=0.3)
    axes[0].legend(frameon=True, fontsize=8.5)

    # Painel 2: CCDF + ajuste
    axes[1].loglog(vals, ccdf, "-", linewidth=2, color="#1f77b4", label="CCDF empírica $P(K \\geq k)$")
    fit_cx = vals[mask_ccdf]
    fit_cy = 10 ** (intercept_ccdf + slope_ccdf * np.log10(fit_cx))
    axes[1].loglog(
        fit_cx,
        fit_cy,
        "--",
        linewidth=2,
        color="#d62728",
        label=f"Ajuste CCDF: $\\gamma={gamma_ccdf:.2f}$ ($R^2={r2_ccdf:.3f}$)",
    )
    axes[1].set_xlabel("Grau $k$ (escala log)")
    axes[1].set_ylabel("$P(K \\geq k)$ (escala log)")
    axes[1].set_title(f"CCDF Log-Log (MLE $k_{{\\min}}={k_min}$: $\\hat{{\\gamma}}={gamma_mle:.2f}$)")
    axes[1].grid(True, which="both", ls="--", alpha=0.3)
    axes[1].legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte3_power_law.png", dpi=300)
    fig.savefig(FIGURES_DIR / "parte3_power_law.pdf")
    plt.close(fig)

    res = {
        "gamma_logbin": round(float(gamma_logbin), 4),
        "r2_logbin": round(float(r2_logbin), 4),
        "gamma_ccdf": round(float(gamma_ccdf), 4),
        "r2_ccdf": round(float(r2_ccdf), 4),
        "gamma_mle_kmin10": round(float(gamma_mle), 4),
        "n_tail_kmin10": int(len(tail)),
        "frac_deg_eq_1": round(float((degrees == 1).mean()), 4),
        "frac_deg_ge_100": round(float((degrees >= 100).mean()), 4),
    }
    with open(TABLES_DIR / "parte3_power_law.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(
        f"     gamma_logbin={gamma_logbin:.3f} (R²={r2_logbin:.3f}), "
        f"gamma_ccdf={gamma_ccdf:.3f} (R²={r2_ccdf:.3f}), gamma_MLE={gamma_mle:.3f}"
    )
    return res


# ===========================================================================
# 3. ROBUSTEZ (Apêndice C)
# ===========================================================================

def measure_robustness_metrics(
    g: ig.Graph, n_orig: int, sample_bfs: int = 200, seed: int = 42
) -> dict:
    n_rem = g.vcount()
    comps = g.connected_components()
    num_comps = len(comps)
    sizes = comps.sizes()
    max_comp_idx = int(np.argmax(sizes))
    gcc_size = sizes[max_comp_idx]
    s_rel_remaining = gcc_size / n_rem if n_rem > 0 else 0.0
    s_rel_orig = gcc_size / n_orig

    degrees = np.array(g.degree())
    isolated_frac = float((degrees == 0).sum() / n_rem) if n_rem > 0 else 0.0

    # Distância média na maior componente restante (amostragem de 200 origens via igraph C)
    sub_gcc = g.subgraph(comps[max_comp_idx])
    rng = random.Random(seed)
    k = min(sample_bfs, sub_gcc.vcount())
    if k > 1:
        sources = rng.sample(range(sub_gcc.vcount()), k)
        dist_matrix = sub_gcc.distances(source=sources)
        vals = [d for row in dist_matrix for d in row if 0 < d < float("inf")]
        avg_dist = float(np.mean(vals)) if vals else 0.0
    else:
        avg_dist = 0.0

    return {
        "gcc_size": int(gcc_size),
        "S_rel_remaining": float(s_rel_remaining),
        "S_rel_orig": float(s_rel_orig),
        "num_components": int(num_comps),
        "avg_distance": float(avg_dist),
        "isolated_frac": float(isolated_frac),
    }


def run_robustness_analysis(df_gcc: pd.DataFrame, T: int = 30) -> dict:
    print(f"[3/5] Executando análise de Robustez (T={T} repetições aleatórias + ataques direcionados)...")
    g_gcc = ig.Graph.TupleList(
        df_gcc[["source", "target"]].itertuples(index=False, name=None), directed=False
    )
    n = g_gcc.vcount()
    r = math.ceil(0.05 * n)

    baseline = measure_robustness_metrics(g_gcc, n_orig=n, sample_bfs=250, seed=42)

    # 3a. Remoção aleatória de 5% (T repetições)
    rng = random.Random(42)
    rand_records = []
    all_vertices = list(range(n))
    for rep in range(T):
        to_remove = set(rng.sample(all_vertices, r))
        keep = [v for v in all_vertices if v not in to_remove]
        g_sub = g_gcc.subgraph(keep)
        m = measure_robustness_metrics(g_sub, n_orig=n, sample_bfs=200, seed=100 + rep)
        m["repeticao"] = rep + 1
        rand_records.append(m)

    df_rand = pd.DataFrame(rand_records)
    df_rand.to_csv(TABLES_DIR / "parte3_robustez_aleatoria_bruto.csv", index=False)

    # 3b. Ataque direcionado — Top 5% por Grau (Degree)
    deg = np.array(g_gcc.degree())
    top_deg_remove = set(np.argsort(deg)[::-1][:r].tolist())
    keep_deg = [v for v in all_vertices if v not in top_deg_remove]
    attack_deg = measure_robustness_metrics(
        g_gcc.subgraph(keep_deg), n_orig=n, sample_bfs=250, seed=42
    )

    # Bônus: Ataque direcionado — Top 5% por PageRank
    pr = np.array(g_gcc.pagerank())
    top_pr_remove = set(np.argsort(pr)[::-1][:r].tolist())
    keep_pr = [v for v in all_vertices if v not in top_pr_remove]
    attack_pr = measure_robustness_metrics(
        g_gcc.subgraph(keep_pr), n_orig=n, sample_bfs=250, seed=42
    )

    # Tabela comparativa
    metrics_map = [
        ("A — Fração da Maior Componente (S)", "S_rel_remaining"),
        ("A' — Tamanho Absoluto da GCC", "gcc_size"),
        ("B — Número de Componentes (c)", "num_components"),
        ("C — Comprimento Médio dos Caminhos (L)", "avg_distance"),
        ("D — Fração de Nós Isolados", "isolated_frac"),
    ]
    comp_rows = []
    for label, col in metrics_map:
        mean_r = float(df_rand[col].mean())
        std_r = float(df_rand[col].std(ddof=1))
        comp_rows.append(
            {
                "metrica": label,
                "baseline_gcc": round(baseline[col], 5),
                "aleatoria_media": round(mean_r, 5),
                "aleatoria_dp": round(std_r, 5),
                "ataque_degree": round(attack_deg[col], 5),
                "ataque_pagerank": round(attack_pr[col], 5),
            }
        )
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(TABLES_DIR / "parte3_robustez_comparacao.csv", index=False)

    # Boxplots (média ± desvio padrão) da remoção aleatória + marcadores de ataque direcionado
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_specs = [
        ("S_rel_remaining", "Métrica A: Fração da Maior Componente ($S$)", axes[0, 0]),
        ("num_components", "Métrica B: Número de Componentes ($c$)", axes[0, 1]),
        ("avg_distance", "Métrica C: Distância Média ($L'$)", axes[1, 0]),
        ("isolated_frac", "Métrica D: Fração de Nós Isolados", axes[1, 1]),
    ]
    for col, title, ax in plot_specs:
        vals = df_rand[col].to_numpy()
        mean_v, std_v = vals.mean(), vals.std(ddof=1)

        bp = ax.boxplot(
            [vals],
            positions=[1],
            widths=0.4,
            patch_artist=True,
            boxprops={"facecolor": "#a6cee3", "edgecolor": "#1f78b4"},
            medianprops={"color": "#111111", "linewidth": 1.5},
        )
        # Média ± DP sobreposta no boxplot
        ax.errorbar(
            [1],
            [mean_v],
            yerr=[std_v],
            fmt="D",
            color="#08306b",
            ecolor="#08306b",
            capsize=5,
            markersize=6,
            label=f"Aleatória: {mean_v:.4f} ± {std_v:.4f}",
        )
        # Pontos do ataque direcionado
        ax.scatter(
            [2],
            [attack_deg[col]],
            color="#d62728",
            s=85,
            zorder=5,
            label=f"Ataque Degree: {attack_deg[col]:.4f}",
        )
        ax.scatter(
            [2.5],
            [attack_pr[col]],
            color="#ff7f0e",
            marker="^",
            s=85,
            zorder=5,
            label=f"Ataque PageRank: {attack_pr[col]:.4f}",
        )
        ax.axhline(
            baseline[col],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"Original: {baseline[col]:.4f}",
        )
        ax.set_xticks([1, 2, 2.5])
        ax.set_xticklabels(["Aleatória\n(T=30)", "Ataque\nDegree", "Ataque\nPageRank"], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        ax.legend(fontsize=7.5, frameon=True, loc="best")

    fig.suptitle(
        f"Parte III: Robustez da GCC (n={n:,}, remoção de 5% = {r:,} vértices)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte3_robustez_boxplots.png", dpi=300)
    fig.savefig(FIGURES_DIR / "parte3_robustez_boxplots.pdf")
    plt.close(fig)

    print(
        f"     S_rand={df_rand['S_rel_remaining'].mean():.4f} vs S_deg={attack_deg['S_rel_remaining']:.4f} | "
        f"c_rand={df_rand['num_components'].mean():.1f} vs c_deg={attack_deg['num_components']}"
    )
    return {
        "n_gcc": n,
        "r_removed": r,
        "T_repetitions": T,
        "baseline": baseline,
        "attack_degree": attack_deg,
        "attack_pagerank": attack_pr,
    }


# ===========================================================================
# 4. DESCOBERTA MAIS INTERESSANTE — Conflitos (LINK_SENTIMENT = -1) e Evolução
# ===========================================================================

def run_interesting_discovery(df_dir: pd.DataFrame) -> dict:
    print("[4/5] Executando análise da Descoberta Mais Interessante (conflitos e temporalidade)...")
    # Agregação por subreddit de saída (quem inicia conflitos) e entrada (quem recebe)
    out_stats = (
        df_dir.groupby("source", as_index=False)
        .agg(
            out_links=("weight_freq", "sum"),
            out_neg=("neg_count", "sum"),
            out_deg=("target", "size"),
        )
    )
    out_stats["neg_out_rate"] = out_stats["out_neg"] / out_stats["out_links"]

    in_stats = (
        df_dir.groupby("target", as_index=False)
        .agg(
            in_links=("weight_freq", "sum"),
            in_neg=("neg_count", "sum"),
            in_deg=("source", "size"),
        )
    )
    in_stats["neg_in_rate"] = in_stats["in_neg"] / in_stats["in_links"]

    # Concentração dos conflitos: quanto % dos links negativos saem dos top 1% subreddits emissores
    out_sorted = out_stats.sort_values("out_neg", ascending=False).reset_index(drop=True)
    total_neg = int(out_sorted["out_neg"].sum())
    top_1pct_k = max(1, int(math.ceil(0.01 * len(out_sorted))))
    top_1pct_neg_share = float(out_sorted.iloc[:top_1pct_k]["out_neg"].sum() / total_neg)
    top_10_neg_share = float(out_sorted.iloc[:10]["out_neg"].sum() / total_neg)

    top_aggressors = out_sorted.head(12).copy()
    top_aggressors["source"] = "r/" + top_aggressors["source"]
    top_aggressors.to_csv(TABLES_DIR / "parte3_descoberta_top_emissores_negativos.csv", index=False)

    top_targets = in_stats.sort_values("in_neg", ascending=False).head(12).copy()
    top_targets["target"] = "r/" + top_targets["target"]
    top_targets.to_csv(TABLES_DIR / "parte3_descoberta_top_alvos_negativos.csv", index=False)

    # Série temporal mensal a partir dos arquivos brutos (ou fallback para CSV já salvo)
    ts_frames = []
    for fname in ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"]:
        p = DATA_RAW_DIR / fname
        if p.exists():
            ts_frames.append(
                pd.read_csv(p, sep="\t", usecols=["TIMESTAMP", "LINK_SENTIMENT"])
            )

    monthly_csv = TABLES_DIR / "parte3_descoberta_serie_mensal.csv"
    if ts_frames:
        df_ts = pd.concat(ts_frames, ignore_index=True)
        df_ts["month"] = df_ts["TIMESTAMP"].str.slice(0, 7)
        monthly = (
            df_ts.groupby("month", as_index=False)
            .agg(
                total_links=("LINK_SENTIMENT", "size"),
                neg_links=("LINK_SENTIMENT", lambda s: int((s == -1).sum())),
            )
        )
        # Remove mês incompleto 2013-12 (apenas algumas horas de 31/12/2013)
        monthly = monthly[monthly["month"] >= "2014-01"].copy()
        monthly["neg_pct"] = 100.0 * monthly["neg_links"] / monthly["total_links"]
        monthly.to_csv(monthly_csv, index=False)
    elif monthly_csv.exists():
        print("     [AVISO] Arquivos brutos .tsv ausentes em data/raw/; reutilizando série mensal salva.")
        monthly = pd.read_csv(monthly_csv)
    else:
        raise FileNotFoundError(
            "Arquivos brutos não encontrados em data/raw/ nem série mensal salva em "
            f"{monthly_csv}. Execute src/download_data.py primeiro."
        )

    # Figura com 2 painéis: Top 10 emissores de conflito + Evolução mensal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    top10 = top_aggressors.head(10).iloc[::-1]
    axes[0].barh(top10["source"], top10["out_neg"], color="#d62728", alpha=0.85, edgecolor="#222222")
    axes[0].set_xlabel("Total de hiperlinks negativos enviados")
    axes[0].set_title(
        f"Top 10 Emissores de Links Negativos\n"
        f"(Top 10 = {100*top_10_neg_share:.1f}% | Top 1% = {100*top_1pct_neg_share:.1f}% dos conflitos)"
    )
    axes[0].grid(True, axis="x", ls="--", alpha=0.35)

    x_idx = np.arange(len(monthly))
    axes[1].plot(x_idx, monthly["total_links"] / 1000.0, "-o", markersize=3, color="#1f77b4", label="Total (milhares)")
    ax2 = axes[1].twinx()
    ax2.plot(x_idx, monthly["neg_pct"], "-s", markersize=3, color="#d62728", alpha=0.85, label="% Negativos")
    tick_pos = x_idx[::6]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(monthly["month"].iloc[tick_pos], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Hiperlinks mensais (milhares)", color="#1f77b4")
    ax2.set_ylabel("Fração de links negativos (%)", color="#d62728")
    axes[1].set_title("Evolução Temporal dos Hiperlinks e Taxa de Conflito (2014–2017)")
    axes[1].grid(True, ls="--", alpha=0.3)

    # Legenda combinada
    l1, lb1 = axes[1].get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    axes[1].legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte3_descoberta_conflitos.png", dpi=300)
    fig.savefig(FIGURES_DIR / "parte3_descoberta_conflitos.pdf")
    plt.close(fig)

    peak_row = monthly.loc[monthly["total_links"].idxmax()]
    peak_neg_row = monthly.loc[monthly["neg_pct"].idxmax()]
    res = {
        "total_neg_links": total_neg,
        "top_10_neg_share": round(top_10_neg_share, 4),
        "top_1pct_subreddits_count": top_1pct_k,
        "top_1pct_neg_share": round(top_1pct_neg_share, 4),
        "peak_month_total": str(peak_row["month"]),
        "peak_month_total_links": int(peak_row["total_links"]),
        "peak_month_neg_pct_month": str(peak_neg_row["month"]),
        "peak_month_neg_pct": round(float(peak_neg_row["neg_pct"]), 2),
    }
    with open(TABLES_DIR / "parte3_descoberta_resumo.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(
        f"     Top 10 subreddits concentram {100*top_10_neg_share:.1f}% dos links negativos; "
        f"Top 1% concentram {100*top_1pct_neg_share:.1f}%! Pico total em {peak_row['month']}."
    )
    return res


# ===========================================================================
# 5. (EXTRA) COMPARAÇÃO COM MODELOS CLÁSSICOS (ER, WS, BA)
# ===========================================================================

def run_classical_models_comparison(df_gcc: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Executando questão EXTRA — comparação com ER, Watts-Strogatz e Barabási-Albert...")
    g_real = ig.Graph.TupleList(
        df_gcc[["source", "target"]].itertuples(index=False, name=None), directed=False
    )
    n = g_real.vcount()
    m = g_real.ecount()
    avg_k = 2 * m / n
    m_ba = max(1, int(round(avg_k / 2)))  # ~5 arestas por novo nó
    nei_ws = max(1, int(round(avg_k / 2)))

    random.seed(42)
    np.random.seed(42)
    g_er = ig.Graph.Erdos_Renyi(n=n, m=m, directed=False)
    g_ws = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=nei_ws, p=0.05)
    g_ba = ig.Graph.Barabasi(n=n, m=m_ba, directed=False)

    def summarize(name: str, g: ig.Graph) -> dict:
        comps = g.connected_components()
        gcc = g.subgraph(comps[int(np.argmax(comps.sizes()))])
        deg = np.array(g.degree())
        c_local = g.transitivity_avglocal_undirected(mode="zero")
        rng = random.Random(42)
        sources = rng.sample(range(gcc.vcount()), min(300, gcc.vcount()))
        dmat = gcc.distances(source=sources)
        vals = [d for row in dmat for d in row if 0 < d < float("inf")]
        l_est = float(np.mean(vals))
        return {
            "modelo": name,
            "vertices": g.vcount(),
            "arestas": g.ecount(),
            "grau_medio": round(float(deg.mean()), 3),
            "grau_max": int(deg.max()),
            "clustering_medio_C": round(float(c_local), 6),
            "caminho_medio_L_est": round(l_est, 4),
        }

    rows = [
        summarize("Reddit Hyperlinks (GCC Real)", g_real),
        summarize("Erdős-Rényi G(n, m)", g_er),
        summarize(f"Watts-Strogatz (nei={nei_ws}, p=0.05)", g_ws),
        summarize(f"Barabási-Albert (m={m_ba})", g_ba),
    ]
    df_models = pd.DataFrame(rows)
    df_models.to_csv(TABLES_DIR / "parte3_modelos_classicos.csv", index=False)
    print("\n=== COMPARAÇÃO COM MODELOS CLÁSSICOS (EXTRA) ===")
    print(df_models.to_string(index=False))
    return df_models


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_gcc = pd.read_csv(DATA_PROCESSED_DIR / "edges_gcc.csv.gz")
    df_undir = pd.read_csv(DATA_PROCESSED_DIR / "edges_undirected.csv.gz")
    df_dir = pd.read_csv(DATA_PROCESSED_DIR / "edges_directed.csv.gz")

    run_small_world_analysis(df_gcc)
    run_power_law_analysis(df_undir)
    run_robustness_analysis(df_gcc, T=30)
    run_interesting_discovery(df_dir)
    run_classical_models_comparison(df_gcc)
    print("\n[CONCLUÍDO] Parte III finalizada com sucesso!")


if __name__ == "__main__":
    main()
