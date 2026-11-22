"""Etapa 0: Tratamento e preparação do grafo Reddit Hyperlink Network (SNAP).

Gera as seguintes versões tratadas em data/processed/:
  - edges_directed.csv.gz: grafo dirigido simples ponderado (sem auto-loops)
  - edges_undirected.csv.gz: grafo não-dirigido simples ponderado (sem auto-loops)
  - edges_gcc.csv.gz: maior componente conexa (GCC) do grafo não-dirigido
  - edges_sub500_directed.csv.gz: subgrafo dirigido induzido pelos top 500 hubs da GCC
  - edges_sub500_undirected.csv.gz: subgrafo não-dirigido induzido pelos top 500 hubs da GCC

Também salva um resumo completo do tratamento em results/tables/tratamento_resumo.json e .csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "results" / "tables"

RAW_FILES = [
    ("body", DATA_RAW_DIR / "soc-redditHyperlinks-body.tsv"),
    ("title", DATA_RAW_DIR / "soc-redditHyperlinks-title.tsv"),
]


def load_raw_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    file_stats = {}

    for label, path in RAW_FILES:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}. Execute src/download_data.py primeiro."
            )
        # Lê apenas as colunas necessárias (descarta PROPERTIES para economizar memória)
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"],
            dtype={
                "SOURCE_SUBREDDIT": "string",
                "TARGET_SUBREDDIT": "string",
                "TIMESTAMP": "string",
                "LINK_SENTIMENT": "int8",
            },
        )
        df["SOURCE_FILE"] = label
        file_stats[f"raw_edges_{label}"] = int(len(df))
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    return full_df, file_stats


def build_directed_simple(df: pd.DataFrame) -> pd.DataFrame:
    df_agg = (
        df.groupby(["source", "target"], as_index=False)
        .agg(
            weight_freq=("LINK_SENTIMENT", "size"),
            weight_sent_sum=("LINK_SENTIMENT", "sum"),
            weight_sent_mean=("LINK_SENTIMENT", "mean"),
            neg_count=("is_neg", "sum"),
        )
    )
    df_agg["weight_dist"] = 1.0 / df_agg["weight_freq"]
    return df_agg


def build_undirected_simple(df_dir: pd.DataFrame) -> pd.DataFrame:
    u = np.minimum(df_dir["source"].to_numpy(), df_dir["target"].to_numpy())
    v = np.maximum(df_dir["source"].to_numpy(), df_dir["target"].to_numpy())
    tmp = pd.DataFrame(
        {
            "source": u,
            "target": v,
            "weight_freq": df_dir["weight_freq"].to_numpy(),
            "weight_sent_sum": df_dir["weight_sent_sum"].to_numpy(),
            "neg_count": df_dir["neg_count"].to_numpy(),
        }
    )
    df_undir = (
        tmp.groupby(["source", "target"], as_index=False)
        .agg(
            weight_freq=("weight_freq", "sum"),
            weight_sent_sum=("weight_sent_sum", "sum"),
            neg_count=("neg_count", "sum"),
        )
    )
    df_undir["weight_sent_mean"] = df_undir["weight_sent_sum"] / df_undir["weight_freq"]
    df_undir["weight_dist"] = 1.0 / df_undir["weight_freq"]
    return df_undir


def extract_gcc_and_subgraphs(
    df_dir: pd.DataFrame, df_undir: pd.DataFrame, top_k: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    g_undir = ig.Graph.TupleList(
        df_undir[["source", "target"]].itertuples(index=False, name=None),
        directed=False,
    )
    components = g_undir.connected_components()
    gcc_idx = max(range(len(components)), key=lambda i: len(components[i]))
    gcc_nodes = {g_undir.vs[i]["name"] for i in components[gcc_idx]}

    df_gcc = df_undir[
        df_undir["source"].isin(gcc_nodes) & df_undir["target"].isin(gcc_nodes)
    ].copy()

    # Subgrafo induzido pelos top_k vértices de maior grau na GCC
    g_gcc = g_undir.subgraph(components[gcc_idx])
    degrees = g_gcc.degree()
    names = g_gcc.vs["name"]
    top_nodes = {
        name for name, _ in sorted(zip(names, degrees), key=lambda x: x[1], reverse=True)[:top_k]
    }

    df_sub_dir = df_dir[
        df_dir["source"].isin(top_nodes) & df_dir["target"].isin(top_nodes)
    ].copy()
    df_sub_undir = df_gcc[
        df_gcc["source"].isin(top_nodes) & df_gcc["target"].isin(top_nodes)
    ].copy()

    stats = {
        "gcc_nodes": len(gcc_nodes),
        "gcc_edges_undirected": int(len(df_gcc)),
        "gcc_node_fraction": round(len(gcc_nodes) / g_undir.vcount(), 6),
        "sub500_nodes": len(top_nodes),
        "sub500_edges_directed": int(len(df_sub_dir)),
        "sub500_edges_undirected": int(len(df_sub_undir)),
    }
    return df_gcc, df_sub_dir, df_sub_undir, stats


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Carregando arquivos brutos do SNAP...")
    df_raw, summary = load_raw_data()

    # Normalização de nomes de subreddits
    df_raw["source"] = df_raw["SOURCE_SUBREDDIT"].str.strip().str.lower()
    df_raw["target"] = df_raw["TARGET_SUBREDDIT"].str.strip().str.lower()
    df_raw["is_neg"] = (df_raw["LINK_SENTIMENT"] == -1).astype("int32")

    raw_nodes = set(df_raw["source"]).union(set(df_raw["target"]))
    summary["raw_edges_total"] = int(len(df_raw))
    summary["raw_nodes_total"] = len(raw_nodes)
    summary["raw_pos_edges"] = int((df_raw["LINK_SENTIMENT"] == 1).sum())
    summary["raw_neg_edges"] = int((df_raw["LINK_SENTIMENT"] == -1).sum())
    summary["raw_neg_fraction"] = round(summary["raw_neg_edges"] / summary["raw_edges_total"], 6)
    summary["timestamp_min"] = str(df_raw["TIMESTAMP"].min())
    summary["timestamp_max"] = str(df_raw["TIMESTAMP"].max())

    # Remoção de valores ausentes (se houver) e auto-loops
    missing_mask = df_raw["source"].isna() | df_raw["target"].isna()
    summary["removed_missing"] = int(missing_mask.sum())
    df_clean = df_raw[~missing_mask].copy()

    self_loop_mask = df_clean["source"] == df_clean["target"]
    summary["removed_self_loops"] = int(self_loop_mask.sum())
    df_clean = df_clean[~self_loop_mask].copy()

    clean_nodes = set(df_clean["source"]).union(set(df_clean["target"]))
    summary["clean_multigraph_edges"] = int(len(df_clean))
    summary["clean_nodes"] = len(clean_nodes)

    print("[2/5] Agregando multigrafo -> grafo dirigido simples ponderado...")
    df_dir = build_directed_simple(df_clean)
    summary["directed_simple_edges"] = int(len(df_dir))
    summary["removed_parallel_directed_edges"] = (
        summary["clean_multigraph_edges"] - summary["directed_simple_edges"]
    )

    print("[3/5] Construindo grafo não-dirigido simples ponderado...")
    df_undir = build_undirected_simple(df_dir)
    summary["undirected_simple_edges"] = int(len(df_undir))

    print("[4/5] Extraindo Maior Componente Conexa (GCC) e subgrafo reduzido (top 500)...")
    df_gcc, df_sub_dir, df_sub_undir, gcc_stats = extract_gcc_and_subgraphs(
        df_dir, df_undir, top_k=500
    )
    summary.update(gcc_stats)

    print("[5/5] Salvando arquivos tratados em data/processed/...")
    df_dir.to_csv(DATA_PROCESSED_DIR / "edges_directed.csv.gz", index=False, compression="gzip")
    df_undir.to_csv(DATA_PROCESSED_DIR / "edges_undirected.csv.gz", index=False, compression="gzip")
    df_gcc.to_csv(DATA_PROCESSED_DIR / "edges_gcc.csv.gz", index=False, compression="gzip")
    df_sub_dir.to_csv(
        DATA_PROCESSED_DIR / "edges_sub500_directed.csv.gz", index=False, compression="gzip"
    )
    df_sub_undir.to_csv(
        DATA_PROCESSED_DIR / "edges_sub500_undirected.csv.gz", index=False, compression="gzip"
    )

    with open(TABLES_DIR / "tratamento_resumo.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(list(summary.items()), columns=["metrica", "valor"]).to_csv(
        TABLES_DIR / "tratamento_resumo.csv", index=False
    )

    print("\n=== RESUMO DO TRATAMENTO ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
