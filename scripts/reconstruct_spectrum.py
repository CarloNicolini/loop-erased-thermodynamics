#!/usr/bin/env python3
"""
Inverse spectral density reconstruction from s(q) measurements.

This script loads s_vs_q.csv (from sample_s_vs_q.py), performs a
nonnegative Tikhonov-regularized least-squares fit of the Stieltjes
transform to recover a discrete spectral measure on a lambda grid,
and produces comparison plots and CSV outputs.

Usage example:

  source .venv/bin/activate
  python scripts/sample_s_vs_q.py --graph ER --n 100 --p 0.05 --seed 42
  python scripts/reconstruct_spectrum.py --graph ER --n 100 --p 0.05 --seed 42 --compute_theory --use-theory-g

If s_vs_q.csv is not present, this script can compute s(q) directly as well
with --compute-s.
"""

import argparse
import math
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import lsq_linear
from scipy.stats import gaussian_kde

from wilson.wilson import Wilson  # noqa: E402


def relabel_consecutive(G: nx.Graph) -> nx.Graph:
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


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    L = nx.laplacian_matrix(G).toarray()
    return np.linalg.eigvalsh(L)


def compute_bin_widths(lam_grid: np.ndarray) -> np.ndarray:
    """Compute bin widths Δλ around grid points using mid-edge differences.

    The first and last widths are extrapolated by half-intervals.
    """
    lam_grid = np.asarray(lam_grid, dtype=float)
    assert lam_grid.ndim == 1 and lam_grid.size >= 2
    mids = 0.5 * (lam_grid[1:] + lam_grid[:-1])
    left_edge = max(0.0, lam_grid[0] - (mids[0] - lam_grid[0]))
    right_edge = lam_grid[-1] + (lam_grid[-1] - mids[-1])
    edges = np.concatenate([[left_edge], mids, [right_edge]])
    widths = np.diff(edges)
    widths[widths <= 0] = np.min(widths[widths > 0])
    return widths


def ensure_s_data(args: argparse.Namespace, out_dir: Path, G: nx.Graph) -> pd.DataFrame:
    csv_path = out_dir / "s_vs_q.csv"
    if csv_path.exists() and not args.compute_s:
        return pd.read_csv(csv_path)

    # Compute a default q-grid and sample s(q) quickly if requested
    q_grid = np.logspace(math.log10(args.q_min), math.log10(args.q_max), args.num_q)
    records: list[dict[str, float]] = []
    wil = None
    for q in q_grid:
        wil = Wilson(q=float(q), random_state=args.seed)
        wil.fit(G)
        s_vals = []
        for i in range(args.samples_per_q):
            s = None if args.seed is None else args.seed + i
            _, roots = wil.sample(seed=s)
            s_vals.append(len(roots))
        s_vals = np.asarray(s_vals, dtype=float)
        rec: dict[str, float] = {
            "q": float(q),
            "s_mc": float(s_vals.mean()),
            "s_mc_se": float(s_vals.std(ddof=1) / math.sqrt(len(s_vals)))
            if len(s_vals) > 1
            else 0.0,
        }
        records.append(rec)
    df = pd.DataFrame.from_records(records).sort_values("q").reset_index(drop=True)
    df["g_mc"] = df["s_mc"] / df["q"]
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df


def build_second_difference_matrix(m: int) -> np.ndarray:
    if m < 3:
        return np.zeros((0, m))
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def invert_stieltjes_density(
    q: np.ndarray,
    g: np.ndarray,
    g_se: np.ndarray | None,
    lam_grid: np.ndarray,
    delta_lam: np.ndarray,
    n_nodes: float,
    mass_target: float,
    gamma_mass: float,
    tau_smooth: float,
    tau_tv: float,
    tv_iters: int,
    tv_eps: float,
) -> tuple[np.ndarray, float]:
    """Recover probability density ρ(λ) (Σ Δλ ρ = mass_target) from g(q) = Σ_i 1/(q+λ_i).

    Discretization: g ≈ n * Aρ where A_{jℓ} = Δλ_ℓ / (q_j + λ_ℓ).
    Objective: weighted LS + mass penalty + optional L2 smooth + TV (IRLS), with ρ ≥ 0.
    Returns (rho_hat, rrmse).
    """
    J = q.shape[0]
    M = lam_grid.shape[0]
    # Data matrix in density mode
    A_data = (delta_lam.reshape(1, M)) / (q.reshape(J, 1) + lam_grid.reshape(1, M))
    A_data *= float(n_nodes)
    b_data = g.reshape(J)

    # Optional heteroscedastic weighting using standard errors of g
    if g_se is not None:
        w = 1.0 / (np.maximum(g_se.reshape(J), 1e-12) ** 2)
        wsqrt = np.sqrt(w)
        A_data = (wsqrt.reshape(J, 1)) * A_data
        b_data = wsqrt * b_data

    # Mass penalty row for Σ Δλ ρ = mass_target
    A_mass = math.sqrt(gamma_mass) * delta_lam.reshape(1, M)
    b_mass = np.array([math.sqrt(gamma_mass) * float(mass_target)], dtype=float)

    # L2 smoothness rows
    D2 = build_second_difference_matrix(M)
    A_smooth = math.sqrt(max(tau_smooth, 0.0)) * D2
    b_smooth = np.zeros(A_smooth.shape[0], dtype=float)

    # Initial solution without TV
    A_stack = np.vstack([A_data, A_mass, A_smooth])
    b_stack = np.concatenate([b_data, b_mass, b_smooth])
    res = lsq_linear(
        A_stack, b_stack, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto", verbose=0
    )
    rho = np.maximum(res.x, 0.0)

    # TV via IRLS if requested
    tau_tv = float(max(tau_tv, 0.0))
    if tau_tv > 0.0 and D2.shape[0] > 0:
        for _ in range(int(tv_iters)):
            diff = D2 @ rho
            w_tv = 1.0 / np.sqrt(diff * diff + tv_eps * tv_eps)
            A_tv = math.sqrt(tau_tv) * (w_tv.reshape(-1, 1) * D2)
            b_tv = np.zeros(A_tv.shape[0], dtype=float)
            A_stack = np.vstack([A_data, A_mass, A_smooth, A_tv])
            b_stack = np.concatenate([b_data, b_mass, b_smooth, b_tv])
            res = lsq_linear(
                A_stack,
                b_stack,
                bounds=(0.0, np.inf),
                method="trf",
                lsmr_tol="auto",
                verbose=0,
            )
            rho = np.maximum(res.x, 0.0)

    # Relative RMSE in unweighted space
    g_pred = (
        float(n_nodes)
        * (delta_lam.reshape(1, M) / (q.reshape(J, 1) + lam_grid.reshape(1, M)))
    ) @ rho
    rrmse = float(np.linalg.norm(g_pred - g) / (np.linalg.norm(g) + 1e-12))
    return rho, rrmse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverse spectral density reconstruction from s(q)"
    )
    parser.add_argument(
        "--graph",
        required=True,
        choices=["ER", "BA", "REG", "GRID"],
        help="Graph family",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of nodes (ER/BA/REG)"
    )
    parser.add_argument("--p", type=float, default=0.05, help="ER edge prob")
    parser.add_argument("--m", type=int, default=3, help="BA 'm' parameter")
    parser.add_argument("--d", type=int, default=4, help="Regular degree (REG)")
    parser.add_argument("--rows", type=int, default=10, help="GRID rows")
    parser.add_argument("--cols", type=int, default=10, help="GRID cols")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir", type=str, default=str(ROOT / "artifacts" / "tomography")
    )
    parser.add_argument(
        "--s-csv",
        type=str,
        default=None,
        help="Path to s_vs_q.csv; defaults to outdir/<gid>/s_vs_q.csv",
    )
    parser.add_argument(
        "--compute-s", action="store_true", help="Compute s(q) if not present"
    )
    parser.add_argument("--q-min", type=float, default=1e-3)
    parser.add_argument("--q-max", type=float, default=1e3)
    parser.add_argument("--num-q", type=int, default=30)
    parser.add_argument("--samples-per-q", type=int, default=256)
    parser.add_argument(
        "--lambda-grid", type=int, default=200, help="Number of lambda grid points"
    )
    parser.add_argument(
        "--log-lam-grid",
        action="store_true",
        help="Use log-spaced lambda grid (denser near 0)",
    )
    parser.add_argument(
        "--gamma-mass",
        type=float,
        default=1e4,
        help="Mass penalty strength (density mode)",
    )
    parser.add_argument(
        "--tau-smooth",
        type=float,
        default=1e-2,
        help="L2 smoothness on density (second diff)",
    )
    parser.add_argument(
        "--tau-tv", type=float, default=0.0, help="Total variation strength (IRLS)"
    )
    parser.add_argument(
        "--tv-iters", type=int, default=8, help="IRLS iterations for TV regularization"
    )
    parser.add_argument(
        "--tv-eps", type=float, default=1e-6, help="TV epsilon (Hube-like)"
    )
    parser.add_argument(
        "--compute-theory",
        action="store_true",
        help="Compute true eigenvalues for validation",
    )
    parser.add_argument(
        "--use-theory-g",
        action="store_true",
        help="Use exact g(q) from eigenvalues to isolate inversion errors",
    )
    parser.add_argument(
        "--fix-zero-mass",
        action="store_true",
        help="Subtract exact zero-eigen mass (components/n) from g and enforce reduced mass",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_context("talk")
    sns.set_style("whitegrid")

    G = build_graph(args.graph, args)
    gid = graph_id(args.graph, args)
    out_dir = Path(args.outdir) / gid
    os.makedirs(out_dir, exist_ok=True)

    if args.s_csv is None:
        s_df = ensure_s_data(args, out_dir, G)
    else:
        s_df = pd.read_csv(args.s_csv)
        if "g_mc" not in s_df.columns:
            s_df["g_mc"] = s_df["s_mc"] / s_df["q"]

    q = s_df["q"].to_numpy(float)
    if args.use_theory_g and args.compute_theory:
        # Build exact g(q) if possible
        try:
            lambdas_exact = laplacian_eigenvalues(G)
            g_exact = np.array(
                [np.sum(1.0 / (float(qq) + lambdas_exact)) for qq in q], dtype=float
            )
            g = g_exact
        except Exception as e:
            print(f"Warning: could not compute exact g(q) ({e}); falling back to g_mc.")
            g = s_df["g_mc"].to_numpy(float)
    else:
        g = s_df["g_mc"].to_numpy(float)
    # Standard errors in g-space if available
    g_se = None
    if "s_mc_se" in s_df.columns:
        g_se = s_df["s_mc_se"].to_numpy(float) / np.maximum(q, 1e-12)

    # Lambda grid upper bound via Laplacian bound: lambda_max <= 2 * d_max
    d_max = max(d for _, d in G.degree())
    lam_max = max(1.0, 2.0 * float(d_max))
    if args.log_lam_grid:
        lam_min_pos = lam_max / 1000.0
        lam_pos = np.logspace(
            np.log10(lam_min_pos), np.log10(lam_max), max(2, args.lambda_grid - 1)
        )
        lam_grid = np.concatenate(([0.0], lam_pos[:-1]))  # keep size ~ args.lambda_grid
    else:
        lam_grid = np.linspace(0.0, lam_max, args.lambda_grid)

    delta_lam = compute_bin_widths(lam_grid)

    n_nodes = float(G.number_of_nodes())

    # Optionally remove exact zero-eigen mass α = components/n from g and enforce reduced mass
    mass_target = 1.0
    alpha_zero = 0.0
    if args.fix_zero_mass:
        try:
            comps = nx.number_connected_components(G)
        except Exception:
            comps = 1
        alpha_zero = float(comps) / n_nodes
        g = g - (n_nodes * alpha_zero) / np.maximum(q, 1e-12)
        mass_target = max(0.0, 1.0 - alpha_zero)

    rho_hat, rrmse = invert_stieltjes_density(
        q,
        g,
        g_se,
        lam_grid,
        delta_lam,
        n_nodes,
        mass_target,
        args.gamma_mass,
        args.tau_smooth,
        args.tau_tv,
        args.tv_iters,
        args.tv_eps,
    )
    print(f"Fit relative RMSE on g(q): {rrmse:.4e}")

    # Compute fitted curves
    A_full = (
        delta_lam.reshape(1, -1) / (q.reshape(-1, 1) + lam_grid.reshape(1, -1))
    ) * n_nodes
    g_fit_remainder = A_full @ rho_hat
    # Add back zero-mass contribution if subtracted
    g_fit_total = g_fit_remainder + (n_nodes * alpha_zero) / np.maximum(q, 1e-12)
    s_fit = q * g_fit_total

    # Add fits to s_df and save
    s_df = s_df.copy()
    s_df["g_fit"] = g_fit_total
    s_df["s_fit"] = s_fit
    s_csv = out_dir / "s_vs_q_fit.csv"
    s_df.to_csv(s_csv, index=False)
    print(f"Saved: {s_csv}")

    # Optionally compute truth
    lambdas = None
    if args.compute_theory:
        try:
            lambdas = laplacian_eigenvalues(G)
        except Exception as e:
            print(f"Warning: could not compute eigenvalues ({e})")
            lambdas = None

    # Spectrum DataFrame
    spec_df = pd.DataFrame(
        {
            "lambda": lam_grid,
            "delta_lambda": delta_lam,
            "density_hat": rho_hat,
            "prob_mass_hat": rho_hat * delta_lam,
            "count_hat": n_nodes * rho_hat * delta_lam,
        }
    )
    if lambdas is not None:
        # Build histogram on edges consistent with our center grid
        mids = 0.5 * (lam_grid[1:] + lam_grid[:-1])
        left_edge = max(0.0, lam_grid[0] - (mids[0] - lam_grid[0]))
        right_edge = lam_grid[-1] + (lam_grid[-1] - mids[-1])
        edges = np.concatenate([[left_edge], mids, [right_edge]])
        counts, edges = np.histogram(lambdas, bins=edges)
        bin_widths = np.diff(edges)
        density_true = (counts.astype(float) / n_nodes) / np.maximum(bin_widths, 1e-12)
        # Expand to piecewise-constant values aligned with centers (first len = M)
        density_true_centers = density_true
        spec_df["density_true"] = density_true_centers
        # KDE of true eigenvalues (for pattern comparison)
        try:
            kde = gaussian_kde(lambdas)
            lam_kde = np.linspace(lam_grid[0], lam_grid[-1], 1000)
            density_kde = kde(lam_kde)
        except Exception:
            kde = None
            lam_kde = None
            density_kde = None

    spec_csv = out_dir / "spectrum.csv"
    spec_df.to_csv(spec_csv, index=False)
    print(f"Saved: {spec_csv}")

    # Plots
    # 1) s(q)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        s_df["q"],
        s_df["s_mc"],
        yerr=s_df.get("s_mc_se", None),
        fmt="o",
        ms=4,
        lw=1,
        label="Wilson MC",
    )
    if "s_fit" in s_df.columns:
        ax.plot(s_df["q"], s_df["s_fit"], label="fit", lw=2)
    if "s_true" in s_df.columns:
        ax.plot(s_df["q"], s_df["s_true"], label="theory", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("s(q)")
    # ax.set_title(f"s(q): MC vs fit — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "s_vs_q_fit.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # 2) Spectral density
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        spec_df["lambda"], spec_df["density_hat"], label="reconstructed density", lw=2
    )
    if "density_true" in spec_df.columns:
        # Use edges-derived step; plot against centers for visual consistency
        ax.step(
            spec_df["lambda"],
            spec_df["density_true"],
            where="mid",
            label="true histogram",
            alpha=0.6,
        )
        # Rug plot of true eigenvalues
        ax.scatter(
            lambdas,
            np.zeros_like(lambdas),
            marker="|",
            color="k",
            alpha=0.3,
            label="eigs",
        )
        if "density_kde" in locals() and density_kde is not None:
            ax.plot(lam_kde, density_kde, label="true KDE", color="C2", lw=2, alpha=0.9)
    ax.set_xlabel("lambda")
    ax.set_ylabel("spectral density (normalized)")
    # ax.set_title(f"Spectral density reconstruction — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "spectrum.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
