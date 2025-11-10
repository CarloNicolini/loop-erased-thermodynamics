#!/usr/bin/env python3
"""Run Wilson sampling experiments and compare to dense effective-resistance marginals.

Saves CSVs and scatter plots to artifacts/ for inclusion in the manuscript.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# make local src importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from wilson.wilson import Wilson


try:
    from scipy import linalg as spla
    from scipy import stats
except Exception:
    spla = None
    stats = None


def effective_resistance_marginals(G: nx.Graph):
    """Compute effective-resistance marginals p_e = w_e * R_uv using pseudoinverse.

    Returns dict mapping canonical edges to p_e.
    """
    idx = {v: i for i, v in enumerate(G.nodes())}
    # build Laplacian
    A = nx.to_numpy_array(G, nodelist=list(G.nodes()), weight="weight")
    degs = np.sum(A, axis=1)
    L = np.diag(degs) - A
    # pseudoinverse (Moore-Penrose)
    Lp = np.linalg.pinv(L)
    pe = {}
    for u, v, data in G.edges(data=True):
        i, j = idx[u], idx[v]
        R = Lp[i, i] + Lp[j, j] - 2 * Lp[i, j]
        w = data.get("weight", 1.0)
        pe[(min(u, v), max(u, v))] = float(w * R)
    return pe


def empirical_marginals_by_wilson(G: nx.Graph, trials=10000, seed=0):
    W = Wilson(random_state=seed)
    W.fit(G)
    counts = {(min(u, v), max(u, v)): 0 for u, v in G.edges()}
    for t in range(trials):
        sample, roots = W.sample(seed=seed + t)
        # sample may be list of directed (u,v) pairs or DiGraph
        if isinstance(sample, nx.DiGraph):
            edges = list(sample.edges())
        else:
            edges = list(sample)
        for u, v in edges:
            counts[(min(u, v), max(u, v))] += 1
    freqs = {e: counts[e] / float(trials) for e in counts}
    return freqs


def run_one_graph(kind: str, G: nx.Graph, trials=10000, outdir: Path = None):
    name = f"{kind}_n{G.number_of_nodes()}"
    print(
        f"Running experiment {name}: n={G.number_of_nodes()}, m={G.number_of_edges()}"
    )
    pe = effective_resistance_marginals(G)
    emp = empirical_marginals_by_wilson(G, trials=trials, seed=42)

    # assemble arrays for plotting
    edges = sorted(pe.keys())
    p_theory = np.array([pe[e] for e in edges])
    p_emp = np.array([emp[e] for e in edges])
    abs_err = np.abs(p_theory - p_emp)

    if outdir is None:
        outdir = ROOT / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    # save CSV
    import csv

    csv_path = outdir / f"{name}_marginals.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "p_theory", "p_emp", "abs_err"])
        for (u, v), pt, pe_ in zip(edges, p_theory, p_emp):
            w.writerow([u, v, pt, pe_, abs(pt - pe_)])

    # scatter plot with seaborn styling
    sns.set_theme(style="whitegrid", context="paper", palette="deep")
    fig, ax = plt.subplots(figsize=(5, 5))
    # scatter with small jitter and nicer palette
    ax.scatter(p_theory, p_emp, s=18, alpha=0.75, edgecolors="w", linewidths=0.25)
    mn = min(p_theory.min(), p_emp.min())
    mx = max(p_theory.max(), p_emp.max())
    ax.plot([mn, mx], [mn, mx], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("theory $p_e$", fontsize=10)
    ax.set_ylabel("empirical $\hat p_e$", fontsize=10)
    ax.set_title(name, fontsize=11)
    fig.tight_layout()
    fig_path_png = outdir / f"{name}_emp_vs_theory.png"
    fig_path_pdf = outdir / f"{name}_emp_vs_theory.pdf"
    fig.savefig(fig_path_png, dpi=200)
    # save PDF high-quality for LaTeX inclusion
    fig.savefig(fig_path_pdf, dpi=300)
    plt.close(fig)

    # simple metrics
    try:
        if stats is not None:
            pearson = float(stats.pearsonr(p_theory, p_emp)[0])
        else:
            pearson = float(np.corrcoef(p_theory, p_emp)[0, 1])
    except Exception:
        pearson = float(np.nan)
    rmse = float(np.sqrt(np.mean((p_theory - p_emp) ** 2)))
    maxabs = float(np.max(abs_err))

    metrics = dict(
        name=name,
        n=G.number_of_nodes(),
        m=G.number_of_edges(),
        trials=trials,
        pearson=pearson,
        rmse=rmse,
        maxabs=maxabs,
        csv=str(csv_path),
        fig_png=str(fig_path_png),
        fig_pdf=str(fig_path_pdf),
    )
    with open(outdir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Saved {csv_path}, {fig_path_png} and {fig_path_pdf}. Metrics: pearson={pearson:.4g}, rmse={rmse:.4g}"
    )
    return metrics


def main():
    outdir = Path("artifacts")
    # small instances for quick runs; you can scale n up for paper runs
    experiments = []
    experiments.append(("ER", nx.erdos_renyi_graph(50, 0.05, seed=1)))
    experiments.append(("BA", nx.barabasi_albert_graph(50, 3, seed=2)))
    # grid
    Ggrid = nx.grid_2d_graph(8, 6)
    Ggrid = nx.convert_node_labels_to_integers(Ggrid)
    experiments.append(("GRID", Ggrid))

    all_metrics = []
    for name, G in experiments:
        # ensure weights
        for u, v in G.edges():
            G[u][v]["weight"] = G[u][v].get("weight", 1.0)
        m = run_one_graph(name, G, trials=1000, outdir=outdir)
        all_metrics.append(m)

    with open(outdir / "summary_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("All experiments done. Outputs in", outdir)


if __name__ == "__main__":
    main()
