#!/usr/bin/env python3
"""
Sample s(q) via Wilson's q-forest sampler across a grid of q values and
store results as CSV plus a validation plot.

Usage example (activate env first):

  source .venv/bin/activate
  python scripts/sample_s_vs_q.py --graph ER --n 100 --p 0.05 \
      --q-min 1e-3 --q-max 1e3 --num-q 30 --samples-per-q 256 --seed 42

Outputs go to artifacts/tomography/<graph_id>/
"""

import argparse
import math
import os
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wilson.wilson import Wilson  # noqa: E402


def relabel_consecutive(G: nx.Graph) -> nx.Graph:
    """Relabel nodes to 0..n-1 to match Wilson implementation assumptions."""
    mapping = {u: i for i, u in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def build_graph(kind: str, args: argparse.Namespace) -> nx.Graph:
    kind = kind.upper()
    if kind == "ER":
        G = nx.erdos_renyi_graph(args.n, args.p, seed=args.seed)
    elif kind == "BA":
        G = nx.barabasi_albert_graph(args.n, args.m, seed=args.seed)
    elif kind == "REG":
        G = nx.random_regular_graph(args.d, args.n, seed=args.seed)
    elif kind == "GRID":
        G = nx.grid_2d_graph(args.rows, args.cols)
    else:
        raise ValueError(f"Unsupported graph kind: {kind}")
    return relabel_consecutive(G)


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    L = nx.laplacian_matrix(G).toarray()
    return np.linalg.eigvalsh(L)


def compute_s_true(q: float, lambdas: np.ndarray) -> float:
    return float(np.sum(q / (q + lambdas)))


def sample_s_mc(G: nx.Graph, q: float, samples: int, base_seed: int | None) -> tuple[float, float]:
    """Monte Carlo estimate of s(q) by averaging number of roots in q-forests.

    Returns (mean, standard_error).
    """
    wil = Wilson(q=q, random_state=base_seed)
    wil.fit(G)
    num_roots = np.empty(samples, dtype=float)
    for i in range(samples):
        # deterministic per-sample seed for reproducibility while preserving independence
        s = None if base_seed is None else base_seed + i
        _, roots = wil.sample(seed=s)
        num_roots[i] = len(roots)
    mean = float(num_roots.mean())
    se = float(num_roots.std(ddof=1) / math.sqrt(samples)) if samples > 1 else 0.0
    return mean, se


def graph_id(kind: str, args: argparse.Namespace) -> str:
    kind = kind.upper()
    if kind == "ER":
        return f"ER_n{args.n}_p{args.p:.3f}"
    if kind == "BA":
        return f"BA_n{args.n}_m{args.m}"
    if kind == "REG":
        return f"REG_n{args.n}_d{args.d}"
    if kind == "GRID":
        return f"GRID_{args.rows}x{args.cols}"
    return kind


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample s(q) across a q-grid using Wilson sampling.")
    parser.add_argument("--graph", required=True, choices=["ER", "BA", "REG", "GRID"], help="Graph family")
    parser.add_argument("--n", type=int, default=100, help="Number of nodes (ER/BA/REG)")
    parser.add_argument("--p", type=float, default=0.05, help="ER edge prob")
    parser.add_argument("--m", type=int, default=3, help="BA 'm' parameter")
    parser.add_argument("--d", type=int, default=4, help="Regular degree (REG)")
    parser.add_argument("--rows", type=int, default=10, help="GRID rows")
    parser.add_argument("--cols", type=int, default=10, help="GRID cols")
    parser.add_argument("--q-min", type=float, default=1e-3)
    parser.add_argument("--q-max", type=float, default=1e3)
    parser.add_argument("--num-q", type=int, default=30)
    parser.add_argument("--samples-per-q", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=str(ROOT / "artifacts" / "tomography"), help="Output directory base")
    parser.add_argument("--skip-theory", action="store_true", help="Skip eigenvalue-based ground truth")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_context("talk")
    sns.set_style("whitegrid")

    G = build_graph(args.graph, args)
    gid = graph_id(args.graph, args)
    out_dir = Path(args.outdir) / gid
    os.makedirs(out_dir, exist_ok=True)

    lambdas = None
    if not args.skip_theory:
        try:
            lambdas = laplacian_eigenvalues(G)
        except Exception as e:
            print(f"Warning: could not compute eigenvalues ({e}); proceeding without theory.")
            lambdas = None

    q_grid = np.logspace(math.log10(args.q_min), math.log10(args.q_max), args.num_q)

    records: list[dict[str, float]] = []
    for q in q_grid:
        s_mean, s_se = sample_s_mc(G, float(q), args.samples_per_q, args.seed)
        rec: dict[str, float] = {"q": float(q), "s_mc": s_mean, "s_mc_se": s_se, "g_mc": s_mean / float(q)}
        if lambdas is not None:
            s_true = compute_s_true(float(q), lambdas)
            rec["s_true"] = s_true
            rec["g_true"] = s_true / float(q)
        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("q").reset_index(drop=True)
    csv_path = out_dir / "s_vs_q.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot s(q)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df["q"], df["s_mc"], yerr=df["s_mc_se"], fmt="o", ms=4, lw=1, label="Wilson MC")
    if "s_true" in df.columns:
        ax.plot(df["q"], df["s_true"], label="theory", color="C1", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("s(q)")
    ax.set_title(f"s(q) via Wilson sampling — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "s_vs_q.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Also plot g(q)=s/q for inversion visibility
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["q"], df["g_mc"], "o", ms=4, label="g_mc = s/q")
    if "g_true" in df.columns:
        ax.plot(df["q"], df["g_true"], label="g_true", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("g(q) = s(q)/q")
    ax.set_title(f"g(q) for Stieltjes inversion — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "g_vs_q.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()


