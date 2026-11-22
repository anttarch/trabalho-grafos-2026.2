"""Parte II: Algoritmos da Disciplina — Complexidade Teórica vs. Tempo Real Observado.

Implementa em Python puro (listas de adjacência, lista de arestas em Kruskal e
matriz de distâncias em Floyd-Warshall) e mede tempo real com média,
desvio padrão e Intervalo de Confiança de 95% (Normal Z para n >= 30; t-Student para n < 30):
  1. BFS (Busca em Largura)                      — GCC completa (n=30, Z)
  2. DFS (Busca em Profundidade, iterativa)      — GCC completa (n=30, Z)
  3. Verificação de Eulerianidade                — Grafo completo (n=30, Z)
  4. Dijkstra (pesos positivos weight_dist)      — GCC completa (n=30, Z)
  5. Bellman-Ford (pesos positivos e assinados)  — Subgrafo top-500 (n=30 Z / n=10 t)
  6. Floyd-Warshall                              — Subgrafo top-200 (n=10, t)
  7. Algoritmo de Tarjan (SCCs)                  — Grafo dirigido completo (n=30, Z)
  8. AGM: Kruskal e Prim (heap binário)          — GCC completa (n=30, Z)

Salva:
  - results/tables/parte2_algoritmos_resumo.csv e .json
  - results/tables/parte2_tempos_brutos.csv
  - results/figures/parte2_tempos_algoritmos.png e .pdf
"""

from __future__ import annotations

import heapq
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"


# ---------------------------------------------------------------------------
# Estruturas auxiliares de grafo em Python puro (listas de adjacência)
# ---------------------------------------------------------------------------

def build_adj_undirected(df: pd.DataFrame) -> tuple[list[list[tuple[int, float]]], dict[str, int]]:
    nodes = sorted(set(df["source"]).union(set(df["target"])))
    idx = {name: i for i, name in enumerate(nodes)}
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]
    for u_s, v_s, w in df[["source", "target", "weight_dist"]].itertuples(index=False, name=None):
        u, v = idx[u_s], idx[v_s]
        wf = float(w)
        adj[u].append((v, wf))
        adj[v].append((u, wf))
    return adj, idx


def build_adj_directed(
    df: pd.DataFrame, weight_col: str = "weight_dist"
) -> tuple[list[list[tuple[int, float]]], dict[str, int]]:
    nodes = sorted(set(df["source"]).union(set(df["target"])))
    idx = {name: i for i, name in enumerate(nodes)}
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]
    for u_s, v_s, w in df[["source", "target", weight_col]].itertuples(index=False, name=None):
        adj[idx[u_s]].append((idx[v_s], float(w)))
    return adj, idx


# ---------------------------------------------------------------------------
# 1. BFS — O(|V| + |E|)
# ---------------------------------------------------------------------------

def run_bfs(adj: list[list[tuple[int, float]]], start: int) -> int:
    visited = bytearray(len(adj))
    visited[start] = 1
    q = deque([start])
    count = 1
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = 1
                count += 1
                q.append(v)
    return count


# ---------------------------------------------------------------------------
# 2. DFS (iterativa) — O(|V| + |E|)
# ---------------------------------------------------------------------------

def run_dfs(adj: list[list[tuple[int, float]]], start: int) -> int:
    visited = bytearray(len(adj))
    stack = [start]
    count = 0
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = 1
        count += 1
        for v, _ in adj[u]:
            if not visited[v]:
                stack.append(v)
    return count


# ---------------------------------------------------------------------------
# 3. Verificação de Eulerianidade (não-dirigido) — O(|V| + |E|)
# ---------------------------------------------------------------------------

def check_eulerian_undirected(adj: list[list[tuple[int, float]]]) -> tuple[str, int]:
    n = len(adj)
    odd_count = 0
    start = -1
    non_isolated = 0
    for u in range(n):
        deg = len(adj[u])
        if deg > 0:
            non_isolated += 1
            if start == -1:
                start = u
            if deg % 2 == 1:
                odd_count += 1

    if non_isolated == 0:
        return "Euleriano (vazio)", 0

    # Verifica conectividade entre vértices de grau > 0
    reached = run_bfs(adj, start)
    if reached != non_isolated:
        return "Não-Euleriano (desconexo)", odd_count
    if odd_count == 0:
        return "Circuito Euleriano", 0
    if odd_count == 2:
        return "Caminho Euleriano (Semi-Euleriano)", 2
    return "Não-Euleriano", odd_count


# ---------------------------------------------------------------------------
# 4. Dijkstra (heap binário) — O((|V| + |E|) log |V|)
# ---------------------------------------------------------------------------

def run_dijkstra(adj: list[list[tuple[int, float]]], start: int) -> tuple[int, float]:
    n = len(adj)
    inf = float("inf")
    dist = [inf] * n
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    finite = [x for x in dist if x < inf]
    return len(finite), (sum(finite) / len(finite) if finite else inf)


# ---------------------------------------------------------------------------
# 5. Bellman-Ford — O(|V| * |E|)
# ---------------------------------------------------------------------------

def run_bellman_ford(
    n: int, edges: list[tuple[int, int, float]], start: int
) -> tuple[bool, int]:
    inf = float("inf")
    dist = [inf] * n
    dist[start] = 0.0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            du = dist[u]
            if du != inf and du + w < dist[v]:
                dist[v] = du + w
                updated = True
        if not updated:
            break

    # Verifica ciclo negativo alcançável
    has_neg_cycle = False
    for u, v, w in edges:
        du = dist[u]
        if du != inf and du + w < dist[v]:
            has_neg_cycle = True
            break

    reached = sum(1 for x in dist if x < inf)
    return has_neg_cycle, reached


# ---------------------------------------------------------------------------
# 6. Floyd-Warshall — O(|V|^3) tempo, O(|V|^2) espaço
# ---------------------------------------------------------------------------

def run_floyd_warshall(n: int, edges: list[tuple[int, int, float]]) -> float:
    inf = float("inf")
    dist = [[inf] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w

    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == inf:
                continue
            di = dist[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < inf:
                total += dist[i][j]
                count += 1
    return total / count if count else inf


# ---------------------------------------------------------------------------
# 7. Algoritmo de Tarjan (SCCs, versão iterativa) — O(|V| + |E|)
# ---------------------------------------------------------------------------

def run_tarjan_scc(adj: list[list[tuple[int, float]]]) -> tuple[int, int]:
    n = len(adj)
    index = [-1] * n
    lowlink = [0] * n
    on_stack = bytearray(n)
    scc_stack: list[int] = []
    cur_index = 0
    num_sccs = 0
    max_scc = 0

    for start in range(n):
        if index[start] != -1:
            continue
        # Pilha de chamada explícita: (vértice, posição_no_vizinho)
        call_stack: list[tuple[int, int]] = [(start, 0)]
        index[start] = lowlink[start] = cur_index
        cur_index += 1
        scc_stack.append(start)
        on_stack[start] = 1

        while call_stack:
            u, pos = call_stack[-1]
            neighbors = adj[u]
            if pos < len(neighbors):
                call_stack[-1] = (u, pos + 1)
                v = neighbors[pos][0]
                if index[v] == -1:
                    index[v] = lowlink[v] = cur_index
                    cur_index += 1
                    scc_stack.append(v)
                    on_stack[v] = 1
                    call_stack.append((v, 0))
                elif on_stack[v]:
                    if index[v] < lowlink[u]:
                        lowlink[u] = index[v]
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    if lowlink[u] < lowlink[parent]:
                        lowlink[parent] = lowlink[u]
                if lowlink[u] == index[u]:
                    size = 0
                    while True:
                        w = scc_stack.pop()
                        on_stack[w] = 0
                        size += 1
                        if w == u:
                            break
                    num_sccs += 1
                    if size > max_scc:
                        max_scc = size

    return num_sccs, max_scc


# ---------------------------------------------------------------------------
# 8a. AGM — Kruskal (Union-Find com path compression + rank) — O(|E| log |E|)
# ---------------------------------------------------------------------------

def run_kruskal(n: int, edges: list[tuple[float, int, int]]) -> tuple[int, float]:
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sorted_edges = sorted(edges, key=lambda e: e[0])
    mst_weight = 0.0
    mst_edges = 0

    for w, u, v in sorted_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            mst_weight += w
            mst_edges += 1
            if mst_edges == n - 1:
                break

    return mst_edges, mst_weight


# ---------------------------------------------------------------------------
# 8b. AGM — Prim (heap binário) — O((|V| + |E|) log |V|)
# ---------------------------------------------------------------------------

def run_prim(adj: list[list[tuple[int, float]]], start: int = 0) -> tuple[int, float]:
    n = len(adj)
    in_mst = bytearray(n)
    heap: list[tuple[float, int]] = [(0.0, start)]
    mst_weight = 0.0
    visited_count = 0

    while heap and visited_count < n:
        w, u = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u] = 1
        mst_weight += w
        visited_count += 1
        for v, vw in adj[u]:
            if not in_mst[v]:
                heapq.heappush(heap, (vw, v))

    return visited_count - 1, mst_weight


# ---------------------------------------------------------------------------
# Benchmark e estatística (Normal Z para n >= 30; t-Student para n < 30)
# ---------------------------------------------------------------------------

def benchmark(
    name: str,
    graph_label: str,
    complexity: str,
    fn: Callable[[], object],
    n_runs: int,
    warmup: int = 1,
) -> tuple[dict, list[float]]:
    print(f"  -> Executando {name} ({graph_label}, n={n_runs})...")
    result = None
    for _ in range(warmup):
        result = fn()

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n_runs > 1 else 0.0

    if n_runs >= 30:
        dist_name = "Normal (Z)"
        crit = float(stats.norm.ppf(0.975))  # ~1.95996
    else:
        dist_name = f"t-Student (gl={n_runs - 1})"
        crit = float(stats.t.ppf(0.975, df=n_runs - 1))

    margin = crit * (std / math.sqrt(n_runs))
    ci_low = max(0.0, mean - margin)
    ci_high = mean + margin

    print(
        f"     Média = {mean:.4f}s ± {margin:.4f}s (DP={std:.4f}s, {dist_name}) | Resultado: {result}"
    )
    row = {
        "algoritmo": name,
        "grafo_utilizado": graph_label,
        "complexidade_teorica": complexity,
        "n_execucoes": n_runs,
        "distribuicao_ic": dist_name,
        "media_s": round(mean, 6),
        "desvio_padrao_s": round(std, 6),
        "margem_erro_95_s": round(margin, 6),
        "ic95_inf_s": round(ci_low, 6),
        "ic95_sup_s": round(ci_high, 6),
        "resultado_observado": str(result),
    }
    return row, times


def plot_benchmark(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y_pos = np.arange(len(df))
    colors = [
        "#1f77b4" if "Completo" in g or "GCC" in g else "#ff7f0e"
        for g in df["grafo_utilizado"]
    ]

    ax.barh(
        y_pos,
        df["media_s"],
        xerr=df["margem_erro_95_s"],
        color=colors,
        alpha=0.85,
        capsize=4,
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax.set_yticks(y_pos)
    labels = [
        f"{row['algoritmo']}\n[{row['complexidade_teorica']}]"
        for _, row in df.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Tempo médio de execução em segundos (escala log) — barras de erro = IC 95%")
    ax.set_title("Parte II: Tempo Real Observado dos Algoritmos da Disciplina (com IC 95%)")
    ax.grid(True, axis="x", which="both", ls="--", alpha=0.35)

    for i, row in df.iterrows():
        ax.text(
            row["ic95_sup_s"] * 1.08,
            i,
            f"{row['media_s']:.4f}s (±{row['margem_erro_95_s']:.4f}s)",
            va="center",
            fontsize=8,
        )

    # Legenda manual para distinguir grafo completo vs subgrafo reduzido
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="#222222", label="Grafo Completo / GCC (~66k nós)"),
        Patch(facecolor="#ff7f0e", edgecolor="#222222", label="Subgrafo Reduzido (Top-500 / Top-200)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8.5)

    ax.set_xlim(right=df["ic95_sup_s"].max() * 4.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parte2_tempos_algoritmos.png", dpi=300)
    fig.savefig(FIGURES_DIR / "parte2_tempos_algoritmos.pdf")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Carregando grafos tratados para Parte II...")
    df_dir = pd.read_csv(DATA_PROCESSED_DIR / "edges_directed.csv.gz")
    df_undir = pd.read_csv(DATA_PROCESSED_DIR / "edges_undirected.csv.gz")
    df_gcc = pd.read_csv(DATA_PROCESSED_DIR / "edges_gcc.csv.gz")
    df_sub500 = pd.read_csv(DATA_PROCESSED_DIR / "edges_sub500_directed.csv.gz")

    adj_gcc, idx_gcc = build_adj_undirected(df_gcc)
    adj_undir, _ = build_adj_undirected(df_undir)
    adj_dir, _ = build_adj_directed(df_dir, "weight_dist")
    sub500_nodes = sorted(set(df_sub500["source"]).union(set(df_sub500["target"])))
    idx_sub500 = {name: i for i, name in enumerate(sub500_nodes)}

    # Vértice inicial padrão: r/askreddit (maior hub)
    start_gcc = idx_gcc["askreddit"]
    start_sub500 = idx_sub500["askreddit"]

    # Lista de arestas para Kruskal na GCC
    kruskal_edges = [
        (float(w), idx_gcc[u], idx_gcc[v])
        for u, v, w in df_gcc[["source", "target", "weight_dist"]].itertuples(index=False, name=None)
    ]

    # Listas de arestas para Bellman-Ford no subgrafo top-500
    bf_edges_pos = [
        (idx_sub500[u], idx_sub500[v], float(w))
        for u, v, w in df_sub500[["source", "target", "weight_dist"]].itertuples(index=False, name=None)
    ]
    bf_edges_signed = [
        (idx_sub500[u], idx_sub500[v], float(w))
        for u, v, w in df_sub500[["source", "target", "weight_sent_mean"]].itertuples(index=False, name=None)
    ]

    # Subgrafo top-200 (por grau na GCC) para Floyd-Warshall em Python puro
    top200_names = {
        name
        for name, _ in sorted(
            ((name, len(adj_gcc[i])) for name, i in idx_gcc.items()),
            key=lambda x: -x[1],
        )[:200]
    }
    df_sub200 = df_sub500[
        df_sub500["source"].isin(top200_names) & df_sub500["target"].isin(top200_names)
    ]
    sub200_nodes = sorted(set(df_sub200["source"]).union(set(df_sub200["target"])))
    idx_sub200 = {name: i for i, name in enumerate(sub200_nodes)}
    fw_edges = [
        (idx_sub200[u], idx_sub200[v], float(w))
        for u, v, w in df_sub200[["source", "target", "weight_dist"]].itertuples(index=False, name=None)
    ]

    print("[2/3] Executando benchmarks dos algoritmos...")
    rows = []
    raw_times = {}

    benchmarks_spec = [
        (
            "BFS",
            f"GCC ({len(adj_gcc):,} nós, {len(df_gcc):,} arestas)",
            "O(|V| + |E|)",
            lambda: f"{run_bfs(adj_gcc, start_gcc):,} nós alcançados",
            30,
        ),
        (
            "DFS (iterativa)",
            f"GCC ({len(adj_gcc):,} nós, {len(df_gcc):,} arestas)",
            "O(|V| + |E|)",
            lambda: f"{run_dfs(adj_gcc, start_gcc):,} nós alcançados",
            30,
        ),
        (
            "Eulerianidade",
            f"Completo não-dirigido ({len(adj_undir):,} nós)",
            "O(|V| + |E|)",
            lambda: (lambda r: f"{r[0]} ({r[1]:,} nós de grau ímpar)")(
                check_eulerian_undirected(adj_undir)
            ),
            30,
        ),
        (
            "Tarjan (SCCs)",
            f"Completo dirigido ({len(adj_dir):,} nós, {len(df_dir):,} arestas)",
            "O(|V| + |E|)",
            lambda: (lambda r: f"{r[0]:,} SCCs (maior={r[1]:,})")(run_tarjan_scc(adj_dir)),
            30,
        ),
        (
            "Dijkstra (weight_dist)",
            f"GCC ({len(adj_gcc):,} nós, {len(df_gcc):,} arestas)",
            "O((|V| + |E|) log |V|)",
            lambda: (lambda r: f"{r[0]:,} alcançados; dist média={r[1]:.4f}")(
                run_dijkstra(adj_gcc, start_gcc)
            ),
            30,
        ),
        (
            "AGM — Kruskal",
            f"GCC ({len(adj_gcc):,} nós, {len(df_gcc):,} arestas)",
            "O(|E| log |E|)",
            lambda: (lambda r: f"{r[0]:,} arestas; peso total={r[1]:.2f}")(
                run_kruskal(len(adj_gcc), kruskal_edges)
            ),
            30,
        ),
        (
            "AGM — Prim",
            f"GCC ({len(adj_gcc):,} nós, {len(df_gcc):,} arestas)",
            "O((|V| + |E|) log |V|)",
            lambda: (lambda r: f"{r[0]:,} arestas; peso total={r[1]:.2f}")(
                run_prim(adj_gcc, start_gcc)
            ),
            30,
        ),
        (
            "Bellman-Ford (positivos)",
            f"Subgrafo Top-500 ({len(idx_sub500)} nós, {len(bf_edges_pos):,} arestas)",
            "O(|V| * |E|)",
            lambda: (lambda r: f"ciclo_neg={r[0]}; {r[1]} nós alcançados")(
                run_bellman_ford(len(idx_sub500), bf_edges_pos, start_sub500)
            ),
            30,
        ),
        (
            "Bellman-Ford (assinados)",
            f"Subgrafo Top-500 ({len(idx_sub500)} nós, {len(bf_edges_signed):,} arestas)",
            "O(|V| * |E|)",
            lambda: (lambda r: f"ciclo_neg={r[0]}; {r[1]} nós alcançados")(
                run_bellman_ford(len(idx_sub500), bf_edges_signed, start_sub500)
            ),
            10,
        ),
        (
            "Floyd-Warshall",
            f"Subgrafo Top-200 ({len(idx_sub200)} nós, {len(fw_edges):,} arestas)",
            "O(|V|^3)",
            lambda: f"dist média ponderada={run_floyd_warshall(len(idx_sub200), fw_edges):.4f}",
            10,
        ),
    ]

    for name, graph_label, comp, fn, n_runs in benchmarks_spec:
        row, times = benchmark(name, graph_label, comp, fn, n_runs=n_runs, warmup=1)
        rows.append(row)
        raw_times[name] = times

    print("[3/3] Salvando tabelas e gráficos da Parte II...")
    df_res = pd.DataFrame(rows)
    df_res.to_csv(TABLES_DIR / "parte2_algoritmos_resumo.csv", index=False)
    with open(TABLES_DIR / "parte2_algoritmos_resumo.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Salva tempos individuais para reprodutibilidade
    max_len = max(len(v) for v in raw_times.values())
    df_raw_t = pd.DataFrame(
        {k: v + [np.nan] * (max_len - len(v)) for k, v in raw_times.items()}
    )
    df_raw_t.to_csv(TABLES_DIR / "parte2_tempos_brutos.csv", index=False)

    plot_benchmark(df_res)
    print("\n=== RESUMO PARTE II ===")
    print(
        df_res[
            [
                "algoritmo",
                "complexidade_teorica",
                "n_execucoes",
                "distribuicao_ic",
                "media_s",
                "desvio_padrao_s",
                "margem_erro_95_s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
