#!/usr/bin/env python3
"""Validate forest-heat-kernel identities via Wilson sampling.

⚠️  DEPRECATION NOTICE ⚠️
This script is deprecated. Please use the unified CLI instead:

  python -m wilson.cli validate

See: docs/CLI_USAGE.md for full documentation.

This script:
- Builds several graph ensembles (ER, BA, Grid, Regular)
- Computes spectral heat trace Z(β) and spectral s(q) = Σ q/(q+λ_i)
- Estimates s(q) via Wilson sampling (expected number of roots)
- Validates the Laplace identity s(q) = ∫_0^∞ e^{-t} Z(β t) dt with β=1/q
- Produces publication-ready plots saved under notebooks/artifacts/

Usage: python scripts/validate_wilson_heattrace.py
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import the local package when running from repo root
from wilson.wilson import Wilson
from numpy.polynomial.laguerre import laggauss


@dataclass
class ExperimentConfig:
    er_n: int = 100
    er_p: float = 0.05
    ba_n: int = 100
    ba_m: int = 3
    grid_shape: tuple[int, int] = (10, 10)
    reg_n: int = 100
    reg_d: int = 4
    q_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 2, 20))
    beta_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 2, 20))
    n_wilson_samples: int = 512
    random_state: int = 42
    gl_nodes: int = 64  # Gauss-Laguerre nodes for ∫ e^{-t} f(t) dt
    outdir: Path = field(default_factory=lambda: Path("notebooks/artifacts"))


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_simple_integer_labels(G: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def build_graphs(cfg: ExperimentConfig) -> dict[str, nx.Graph]:
    rng = np.random.default_rng(cfg.random_state)
    graphs: dict[str, nx.Graph] = {}

    G_er = nx.erdos_renyi_graph(
        cfg.er_n, cfg.er_p, seed=int(rng.integers(1, 2**31 - 1))
    )
    graphs[f"ER_n{cfg.er_n}_p{cfg.er_p:.3f}"] = G_er

    G_ba = nx.barabasi_albert_graph(
        cfg.ba_n, cfg.ba_m, seed=int(rng.integers(1, 2**31 - 1))
    )
    graphs[f"BA_n{cfg.ba_n}_m{cfg.ba_m}"] = G_ba

    gx, gy = cfg.grid_shape
    G_grid = nx.grid_2d_graph(gx, gy)
    graphs[f"GRID_{gx}x{gy}"] = to_simple_integer_labels(G_grid)

    G_reg = nx.random_regular_graph(
        cfg.reg_d, cfg.reg_n, seed=int(rng.integers(1, 2**31 - 1))
    )
    graphs[f"REG_n{cfg.reg_n}_d{cfg.reg_d}"] = G_reg

    return graphs


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    L = nx.laplacian_matrix(G).astype(float).toarray()
    return np.linalg.eigvalsh(L)


def heat_trace_from_spectrum(beta: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    # Z(β) = Σ_i exp(-β λ_i)
    beta = np.asarray(beta, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    return np.exp(-np.outer(beta, lambdas)).sum(axis=1)


def s_from_spectrum(q: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    return (q[:, None] / (q[:, None] + lambdas[None, :])).sum(axis=1)


def s_from_Z_via_gauss_laguerre(
    q: np.ndarray, lambdas: np.ndarray, n_nodes: int
) -> np.ndarray:
    # s(q) = ∫_0^∞ e^{-t} Z(β t) dt, β = 1/q
    # Use Gauss-Laguerre quadrature: ∫ e^{-t} f(t) dt ≈ Σ w_j f(t_j)
    t_nodes, w_nodes = laggauss(n_nodes)  # nodes/weights for e^{-t}
    s_vals = []
    for qi in q:
        beta = 1.0 / float(qi)
        # Z(β t_j) for all nodes t_j
        Z_vals = np.exp(-np.outer(beta * t_nodes, lambdas)).sum(axis=1)
        s_vals.append(float(w_nodes @ Z_vals))
    return np.asarray(s_vals)


def estimate_s_via_wilson(
    G: nx.Graph, q: float, n_samples: int, random_state: int
) -> tuple[float, float]:
    est = Wilson(q=q, random_state=random_state).fit(G)
    samples = est.transform(n_samples=n_samples, seed=random_state)
    root_counts = np.array([len(roots) for (_, roots) in samples], dtype=float)
    mean = float(root_counts.mean())
    # standard error of the mean
    sem = float(root_counts.std(ddof=1) / math.sqrt(max(1, n_samples)))
    return mean, sem


def plot_s_curves(
    name: str,
    q: np.ndarray,
    s_spec: np.ndarray,
    s_quad: np.ndarray,
    s_mc_mean: np.ndarray,
    s_mc_sem: np.ndarray,
    outdir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.semilogx(q, s_spec, label="spectral s(q)", color="#1f77b4", lw=1)
    plt.semilogx(
        q, s_quad, label="from Z via Gauss-Laguerre", color="#2ca02c", lw=1, ls="--"
    )
    plt.errorbar(
        q,
        s_mc_mean,
        yerr=s_mc_sem,
        fmt="o",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=1.0,
        capsize=3,
        label="Wilson MC",
    )
    plt.xlabel("q")
    plt.ylabel("s(q) = E[#roots]")
    plt.title(f"s(q) validation — {name}")
    plt.legend(frameon=True)
    plt.tight_layout()
    ensure_outdir(outdir)
    fname = outdir / f"validation_s_vs_q_{name}.pdf"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def plot_Z_curves(
    name: str,
    beta: np.ndarray,
    Z_spec: np.ndarray,
    outdir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.semilogx(beta, Z_spec, label="spectral Z(β)", color="#1f77b4", lw=1)
    plt.xlabel("β")
    plt.ylabel("Z(β) = Tr e^{-βL}")
    plt.title(f"Heat trace — {name}")
    plt.legend(frameon=True)
    plt.tight_layout()
    ensure_outdir(outdir)
    fname = outdir / f"validation_Z_vs_beta_{name}.pdf"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def run_for_graph(name: str, G: nx.Graph, cfg: ExperimentConfig) -> None:
    print(f"\n=== {name} ===")
    lambdas = laplacian_eigenvalues(G)
    print(
        f"n={G.number_of_nodes()}, m={G.number_of_edges()}, max λ={lambdas.max():.3f}"
    )

    # Spectral ground truth
    Z_spec = heat_trace_from_spectrum(cfg.beta_values, lambdas)
    s_spec = s_from_spectrum(cfg.q_values, lambdas)

    # Laplace forward check via Gauss-Laguerre quadrature
    s_quad = s_from_Z_via_gauss_laguerre(cfg.q_values, lambdas, n_nodes=cfg.gl_nodes)

    # Wilson Monte Carlo estimates
    s_mc_mean = np.zeros_like(cfg.q_values)
    s_mc_sem = np.zeros_like(cfg.q_values)
    for i, q in enumerate(cfg.q_values):
        mean, sem = estimate_s_via_wilson(
            G, float(q), cfg.n_wilson_samples, cfg.random_state + i
        )
        s_mc_mean[i] = mean
        s_mc_sem[i] = sem

    # Report relative errors
    rel_err_quad = np.linalg.norm(s_quad - s_spec) / max(1e-12, np.linalg.norm(s_spec))
    rel_err_mc = np.linalg.norm(s_mc_mean - s_spec) / max(1e-12, np.linalg.norm(s_spec))
    print(f"rel error s_from_Z (GL {cfg.gl_nodes}): {rel_err_quad:.3e}")
    print(f"rel error s Wilson MC (n={cfg.n_wilson_samples}): {rel_err_mc:.3e}")

    # Plots
    plot_s_curves(name, cfg.q_values, s_spec, s_quad, s_mc_mean, s_mc_sem, cfg.outdir)
    plot_Z_curves(name, cfg.beta_values, Z_spec, cfg.outdir)


def main() -> None:
    cfg = ExperimentConfig()
    ensure_outdir(cfg.outdir)
    graphs = build_graphs(cfg)
    for name, G in graphs.items():
        run_for_graph(name, G, cfg)
    print(f"\nFigures saved under: {cfg.outdir}")


if __name__ == "__main__":
    main()
