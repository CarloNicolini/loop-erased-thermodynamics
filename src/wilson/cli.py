#!/usr/bin/env python3
"""
Unified CLI for Wilson algorithm experiments and spectral analysis.

This module provides three main subcommands:
- sample-s: Sample s(q) via Wilson's q-forest sampler
- reconstruct: Inverse spectral density reconstruction from s(q) measurements
- validate: Validate forest-heat-kernel identities via Wilson sampling
"""

import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from numpy.polynomial.laguerre import laggauss
from rich import print as rprint
from scipy.optimize import lsq_linear
from typing_extensions import Annotated

from wilson.wilson import Wilson

app = typer.Typer(
    help="Wilson algorithm CLI for thermodynamic graph analysis",
    no_args_is_help=True,
)


# ============================================================================
# Utility functions
# ============================================================================


def relabel_consecutive(G: nx.Graph) -> nx.Graph:
    """
    Relabel graph nodes to consecutive integers 0..n-1.

    Parameters
    ----------
    G : nx.Graph
        Input graph with arbitrary node labels.

    Returns
    -------
    nx.Graph
        Graph with nodes relabeled to 0..n-1.
    """
    mapping = {u: i for i, u in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def build_graph(
    kind: str,
    n: int | None = None,
    p: float | None = None,
    m: int | None = None,
    d: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    seed: int | None = None,
) -> nx.Graph:
    """
    Build a graph from standard families.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', or 'GRID'.
    n : int, optional
        Number of nodes (for ER, BA, REG).
    p : float, optional
        Edge probability (for ER).
    m : int, optional
        Number of edges per new node (for BA).
    d : int, optional
        Degree (for REG).
    rows : int, optional
        Number of rows (for GRID).
    cols : int, optional
        Number of columns (for GRID).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Generated graph with consecutive integer node labels.

    Raises
    ------
    ValueError
        If required parameters are missing or graph kind is unsupported.
    """
    kind = kind.upper()
    if kind == "ER":
        if n is None or p is None:
            raise ValueError("ER graph requires --n and --p")
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif kind == "BA":
        if n is None or m is None:
            raise ValueError("BA graph requires --n and --m")
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "REG":
        if n is None or d is None:
            raise ValueError("REG graph requires --n and --d")
        G = nx.random_regular_graph(d, n, seed=seed)
    elif kind == "GRID":
        if rows is None or cols is None:
            raise ValueError("GRID graph requires --rows and --cols")
        G = nx.grid_2d_graph(rows, cols)
    else:
        raise ValueError(f"Unsupported graph kind: {kind}")
    return relabel_consecutive(G)


def graph_id(
    kind: str,
    n: int | None = None,
    p: float | None = None,
    m: int | None = None,
    d: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
) -> str:
    """
    Generate a human-readable graph identifier string.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', or 'GRID'.
    n : int, optional
        Number of nodes.
    p : float, optional
        Edge probability (ER).
    m : int, optional
        BA parameter.
    d : int, optional
        Degree (REG).
    rows : int, optional
        Grid rows.
    cols : int, optional
        Grid columns.

    Returns
    -------
    str
        Graph identifier string like 'ER_n100_p0.050' or 'GRID_10x10'.
    """
    kind = kind.upper()
    if kind == "ER":
        return f"ER_n{n}_p{p:.3f}"
    if kind == "BA":
        return f"BA_n{n}_m{m}"
    if kind == "REG":
        return f"REG_n{n}_d{d}"
    if kind == "GRID":
        return f"GRID_{rows}x{cols}"
    return kind


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """
    Compute Laplacian eigenvalues of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    np.ndarray
        Sorted array of Laplacian eigenvalues.
    """
    L = nx.laplacian_matrix(G).toarray()
    return np.linalg.eigvalsh(L)


def compute_s_true(q: float, lambdas: np.ndarray) -> float:
    """
    Compute exact s(q) from eigenvalues.

    Parameters
    ----------
    q : float
        Temperature parameter.
    lambdas : np.ndarray
        Laplacian eigenvalues.

    Returns
    -------
    float
        s(q) = sum(q / (q + lambda_i)).
    """
    return float(np.sum(q / (q + lambdas)))


def compute_bin_widths(lam_grid: np.ndarray) -> np.ndarray:
    """
    Compute bin widths around grid points using mid-edge differences.

    The first and last widths are extrapolated by half-intervals to
    ensure proper normalization.

    Parameters
    ----------
    lam_grid : np.ndarray
        1D array of lambda grid points.

    Returns
    -------
    np.ndarray
        Array of bin widths, same length as lam_grid.
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


def build_second_difference_matrix(m: int) -> np.ndarray:
    """
    Build second-difference matrix for smoothness regularization.

    Parameters
    ----------
    m : int
        Size of the density vector.

    Returns
    -------
    np.ndarray
        Second-difference matrix of shape (m-2, m).
    """
    if m < 3:
        return np.zeros((0, m))
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


# ============================================================================
# Subcommand 1: sample-s
# ============================================================================


@app.command(name="sample-s")
def sample_s(
    graph: Annotated[str, typer.Option(help="Graph family: ER, BA, REG, GRID")] = "ER",
    n: Annotated[int, typer.Option(help="Number of nodes (ER/BA/REG)")] = 100,
    p: Annotated[float, typer.Option(help="Edge probability (ER)")] = 0.05,
    m: Annotated[int, typer.Option(help="BA 'm' parameter")] = 3,
    d: Annotated[int, typer.Option(help="Regular degree (REG)")] = 4,
    rows: Annotated[int, typer.Option(help="GRID rows")] = 10,
    cols: Annotated[int, typer.Option(help="GRID columns")] = 10,
    q_min: Annotated[float, typer.Option(help="Minimum q value")] = 1e-3,
    q_max: Annotated[float, typer.Option(help="Maximum q value")] = 1e3,
    num_q: Annotated[int, typer.Option(help="Number of q points")] = 30,
    samples_per_q: Annotated[int, typer.Option(help="MC samples per q")] = 256,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    outdir: Annotated[
        str, typer.Option(help="Output directory base")
    ] = "artifacts/tomography",
    skip_theory: Annotated[
        bool, typer.Option(help="Skip eigenvalue ground truth")
    ] = False,
) -> None:
    """
    Sample s(q) via Wilson's q-forest sampler across a grid of q values.

    This command generates a graph from the specified family, computes s(q)
    via Monte Carlo sampling using Wilson's algorithm, and optionally compares
    against exact eigenvalue-based calculations. Results are saved as CSV
    files and validation plots.

    Parameters
    ----------
    graph : str
        Graph family (ER, BA, REG, GRID).
    n : int
        Number of nodes for ER/BA/REG graphs.
    p : float
        Edge probability for Erdős-Rényi graphs.
    m : int
        Number of edges per new node for Barabási-Albert graphs.
    d : int
        Degree for regular graphs.
    rows : int
        Number of rows for grid graphs.
    cols : int
        Number of columns for grid graphs.
    q_min : float
        Minimum q value for log-spaced grid.
    q_max : float
        Maximum q value for log-spaced grid.
    num_q : int
        Number of q points to sample.
    samples_per_q : int
        Number of Monte Carlo samples per q value.
    seed : int
        Random seed for reproducibility.
    outdir : str
        Base output directory (results saved in outdir/<graph_id>/).
    skip_theory : bool
        If True, skip computing exact eigenvalue-based ground truth.

    Examples
    --------
    Sample s(q) for an Erdős-Rényi graph:

        $ wilson sample-s --graph ER --n 100 --p 0.05 --num-q 30

    Sample for a Barabási-Albert graph:

        $ wilson sample-s --graph BA --n 100 --m 3 --samples-per-q 512

    Notes
    -----
    - Outputs are saved to `outdir/<graph_id>/s_vs_q.csv`
    - Two plots are generated: s(q) and g(q) = s(q)/q
    - Standard errors are computed from MC samples
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # Build graph
    G = build_graph(graph, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed)
    gid = graph_id(graph, n=n, p=p, m=m, d=d, rows=rows, cols=cols)
    out_dir = Path(outdir) / gid
    os.makedirs(out_dir, exist_ok=True)

    rprint(f"[bold cyan]Graph:[/bold cyan] {gid}")
    rprint(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Optionally compute eigenvalues
    lambdas = None
    if not skip_theory:
        try:
            rprint("[yellow]Computing eigenvalues...[/yellow]")
            lambdas = laplacian_eigenvalues(G)
        except Exception as e:
            rprint(f"[red]Warning:[/red] could not compute eigenvalues ({e})")
            lambdas = None

    # Sample s(q) across q-grid
    q_grid = np.logspace(math.log10(q_min), math.log10(q_max), num_q)
    rprint(f"[yellow]Sampling s(q) for {num_q} q-values...[/yellow]")

    records: list[dict[str, float]] = []
    for q in q_grid:
        wil = Wilson(q=float(q), random_state=seed)
        wil.fit(G)
        num_roots = np.empty(samples_per_q, dtype=float)
        for i in range(samples_per_q):
            s = None if seed is None else seed + i
            _, roots = wil.sample(seed=s)
            num_roots[i] = len(roots)
        s_mean = float(num_roots.mean())
        s_se = (
            float(num_roots.std(ddof=1) / math.sqrt(samples_per_q))
            if samples_per_q > 1
            else 0.0
        )
        rec: dict[str, float] = {
            "q": float(q),
            "s_mc": s_mean,
            "s_mc_se": s_se,
            "g_mc": s_mean / float(q),
        }
        if lambdas is not None:
            s_true = compute_s_true(float(q), lambdas)
            rec["s_true"] = s_true
            rec["g_true"] = s_true / float(q)
        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("q").reset_index(drop=True)
    csv_path = out_dir / "s_vs_q.csv"
    df.to_csv(csv_path, index=False)
    rprint(f"[green]✓[/green] Saved: {csv_path}")

    # Plot s(q)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        df["q"],
        df["s_mc"],
        yerr=df["s_mc_se"],
        fmt="o",
        ms=4,
        lw=1,
        label="Wilson MC",
    )
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
    rprint(f"[green]✓[/green] Saved: {fig_path}")

    # Plot g(q) = s/q
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
    rprint(f"[green]✓[/green] Saved: {fig_path}")


# ============================================================================
# Subcommand 2: reconstruct
# ============================================================================


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
    """
    Recover spectral density ρ(λ) from g(q) measurements via regularized inversion.

    This function solves a nonnegative Tikhonov-regularized least-squares problem
    to recover the spectral density from Stieltjes transform measurements.

    Parameters
    ----------
    q : np.ndarray
        Array of q values where g(q) was measured.
    g : np.ndarray
        Measured g(q) = s(q)/q values.
    g_se : np.ndarray or None
        Standard errors for g measurements (for heteroscedastic weighting).
    lam_grid : np.ndarray
        Lambda grid for discretized density.
    delta_lam : np.ndarray
        Bin widths for lambda grid.
    n_nodes : float
        Number of nodes in the graph.
    mass_target : float
        Target total probability mass (usually 1.0).
    gamma_mass : float
        Penalty strength for mass constraint.
    tau_smooth : float
        L2 smoothness regularization strength.
    tau_tv : float
        Total variation regularization strength (IRLS).
    tv_iters : int
        Number of IRLS iterations for TV regularization.
    tv_eps : float
        TV epsilon for Huber-like approximation.

    Returns
    -------
    rho_hat : np.ndarray
        Recovered spectral density on lam_grid.
    rrmse : float
        Relative RMSE between fitted and measured g(q).

    Notes
    -----
    The discretization assumes g ≈ n * A ρ where A_{jl} = Δλ_l / (q_j + λ_l).
    The objective includes:
    - Weighted least squares fit to g(q)
    - Mass penalty to enforce ∫ ρ dλ = mass_target
    - L2 smoothness on second differences
    - Optional TV regularization via IRLS
    """
    J = q.shape[0]
    M = lam_grid.shape[0]
    # Data matrix in density mode
    A_data = (delta_lam.reshape(1, M)) / (q.reshape(J, 1) + lam_grid.reshape(1, M))
    A_data *= float(n_nodes)
    b_data = g.reshape(J)

    # Optional heteroscedastic weighting
    if g_se is not None:
        w = 1.0 / (np.maximum(g_se.reshape(J), 1e-12) ** 2)
        wsqrt = np.sqrt(w)
        A_data = (wsqrt.reshape(J, 1)) * A_data
        b_data = wsqrt * b_data

    # Mass penalty row
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


@app.command()
def reconstruct(
    graph: Annotated[str, typer.Option(help="Graph family: ER, BA, REG, GRID")] = "ER",
    n: Annotated[int, typer.Option(help="Number of nodes (ER/BA/REG)")] = 100,
    p: Annotated[float, typer.Option(help="Edge probability (ER)")] = 0.05,
    m: Annotated[int, typer.Option(help="BA 'm' parameter")] = 3,
    d: Annotated[int, typer.Option(help="Regular degree (REG)")] = 4,
    rows: Annotated[int, typer.Option(help="GRID rows")] = 10,
    cols: Annotated[int, typer.Option(help="GRID columns")] = 10,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    outdir: Annotated[
        str, typer.Option(help="Output directory base")
    ] = "artifacts/tomography",
    s_csv: Annotated[
        str | None, typer.Option(help="Path to s_vs_q.csv (optional)")
    ] = None,
    compute_s: Annotated[
        bool, typer.Option(help="Compute s(q) if not present")
    ] = False,
    q_min: Annotated[float, typer.Option(help="Minimum q for s computation")] = 1e-3,
    q_max: Annotated[float, typer.Option(help="Maximum q for s computation")] = 1e3,
    num_q: Annotated[int, typer.Option(help="Number of q points")] = 30,
    samples_per_q: Annotated[int, typer.Option(help="MC samples per q")] = 256,
    lambda_grid: Annotated[
        int, typer.Option(help="Number of lambda grid points")
    ] = 200,
    log_lam_grid: Annotated[
        bool, typer.Option(help="Use log-spaced lambda grid")
    ] = False,
    gamma_mass: Annotated[float, typer.Option(help="Mass penalty strength")] = 1e4,
    tau_smooth: Annotated[float, typer.Option(help="L2 smoothness strength")] = 1e-2,
    tau_tv: Annotated[float, typer.Option(help="TV strength (IRLS)")] = 0.0,
    tv_iters: Annotated[int, typer.Option(help="IRLS iterations for TV")] = 8,
    tv_eps: Annotated[float, typer.Option(help="TV epsilon")] = 1e-6,
    compute_theory: Annotated[
        bool, typer.Option(help="Compute true eigenvalues")
    ] = False,
    use_theory_g: Annotated[
        bool, typer.Option(help="Use exact g(q) for inversion")
    ] = False,
    fix_zero_mass: Annotated[
        bool, typer.Option(help="Subtract zero-eigen mass")
    ] = False,
    kde_bw: Annotated[
        float, typer.Option(help="KDE bandwidth method or scale factor")
    ] = 1.0,
    kde_normalize: Annotated[
        bool, typer.Option(help="Normalize KDE to unit integral")
    ] = True,
) -> None:
    """
    Inverse spectral density reconstruction from s(q) measurements.

    This command performs Tikhonov-regularized least-squares inversion of
    the Stieltjes transform to recover a discrete spectral density from
    s(q) measurements. The recovered density can be compared against the
    true eigenvalue histogram and KDE.

    Parameters
    ----------
    graph : str
        Graph family (ER, BA, REG, GRID).
    n : int
        Number of nodes for ER/BA/REG graphs.
    p : float
        Edge probability for ER graphs.
    m : int
        BA parameter.
    d : int
        Degree for regular graphs.
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    seed : int
        Random seed.
    outdir : str
        Base output directory.
    s_csv : str or None
        Path to existing s_vs_q.csv file. If None, looks in outdir/<graph_id>/.
    compute_s : bool
        If True, compute s(q) from scratch if not present.
    q_min : float
        Minimum q for s(q) computation.
    q_max : float
        Maximum q for s(q) computation.
    num_q : int
        Number of q points.
    samples_per_q : int
        MC samples per q value.
    lambda_grid : int
        Number of lambda grid points for density reconstruction.
    log_lam_grid : bool
        Use log-spaced lambda grid (denser near zero).
    gamma_mass : float
        Mass constraint penalty strength.
    tau_smooth : float
        L2 smoothness regularization strength (second differences).
    tau_tv : float
        Total variation regularization strength (IRLS).
    tv_iters : int
        Number of IRLS iterations for TV.
    tv_eps : float
        TV epsilon for Huber-like approximation.
    compute_theory : bool
        Compute true eigenvalues for validation.
    use_theory_g : bool
        Use exact g(q) from eigenvalues (isolates inversion errors).
    fix_zero_mass : bool
        Subtract exact zero-eigenvalue mass from g and enforce reduced mass.
    kde_bw : float
        KDE bandwidth scale factor (multiplied by default bandwidth).
    kde_normalize : bool
        If True, normalize KDE to unit integral.

    Examples
    --------
    Reconstruct spectrum from existing s_vs_q.csv:

        $ wilson reconstruct --graph ER --n 100 --p 0.05

    Compute s(q) and reconstruct:

        $ wilson reconstruct --graph ER --n 100 --p 0.05 --compute-s --compute-theory

    Use exact g(q) to isolate inversion errors:

        $ wilson reconstruct --graph ER --n 100 --p 0.05 --compute-theory --use-theory-g

    Notes
    -----
    - Outputs: spectrum.csv, s_vs_q_fit.csv, and two plots
    - KDE uses seaborn for flexible normalization control
    - Regularization parameters control smoothness vs fidelity tradeoff
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # Build graph
    G = build_graph(graph, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed)
    gid = graph_id(graph, n=n, p=p, m=m, d=d, rows=rows, cols=cols)
    out_dir = Path(outdir) / gid
    os.makedirs(out_dir, exist_ok=True)

    rprint(f"[bold cyan]Graph:[/bold cyan] {gid}")
    rprint(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Load or compute s(q) data
    csv_path = out_dir / "s_vs_q.csv" if s_csv is None else Path(s_csv)
    if csv_path.exists() and not compute_s:
        rprint(f"[yellow]Loading s(q) data from {csv_path}...[/yellow]")
        s_df = pd.read_csv(csv_path)
    elif compute_s:
        rprint("[yellow]Computing s(q) from scratch...[/yellow]")
        q_grid = np.logspace(math.log10(q_min), math.log10(q_max), num_q)
        records: list[dict[str, float]] = []
        for q in q_grid:
            wil = Wilson(q=float(q), random_state=seed)
            wil.fit(G)
            s_vals = []
            for i in range(samples_per_q):
                s = None if seed is None else seed + i
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
        s_df = (
            pd.DataFrame.from_records(records).sort_values("q").reset_index(drop=True)
        )
        s_df["g_mc"] = s_df["s_mc"] / s_df["q"]
        s_df.to_csv(csv_path, index=False)
        rprint(f"[green]✓[/green] Saved: {csv_path}")
    else:
        raise FileNotFoundError(
            f"s_vs_q.csv not found at {csv_path}. Use --compute-s to generate it."
        )

    if "g_mc" not in s_df.columns:
        s_df["g_mc"] = s_df["s_mc"] / s_df["q"]

    # Extract q and g
    q = s_df["q"].to_numpy(float)
    if use_theory_g and compute_theory:
        try:
            rprint("[yellow]Computing exact g(q) from eigenvalues...[/yellow]")
            lambdas_exact = laplacian_eigenvalues(G)
            g_exact = np.array(
                [np.sum(1.0 / (float(qq) + lambdas_exact)) for qq in q], dtype=float
            )
            g = g_exact
        except Exception as e:
            rprint(f"[red]Warning:[/red] could not compute exact g(q) ({e})")
            g = s_df["g_mc"].to_numpy(float)
    else:
        g = s_df["g_mc"].to_numpy(float)

    g_se = None
    if "s_mc_se" in s_df.columns:
        g_se = s_df["s_mc_se"].to_numpy(float) / np.maximum(q, 1e-12)

    # Lambda grid
    d_max = max(d for _, d in G.degree())
    lam_max = max(1.0, 2.0 * float(d_max))
    if log_lam_grid:
        lam_min_pos = lam_max / 1000.0
        lam_pos = np.logspace(
            np.log10(lam_min_pos), np.log10(lam_max), max(2, lambda_grid - 1)
        )
        lam_grid = np.concatenate(([0.0], lam_pos[:-1]))
    else:
        lam_grid = np.linspace(0.0, lam_max, lambda_grid)

    delta_lam = compute_bin_widths(lam_grid)
    n_nodes = float(G.number_of_nodes())

    # Optionally remove zero-mass
    mass_target = 1.0
    alpha_zero = 0.0
    if fix_zero_mass:
        try:
            comps = nx.number_connected_components(G)
        except Exception:
            comps = 1
        alpha_zero = float(comps) / n_nodes
        g = g - (n_nodes * alpha_zero) / np.maximum(q, 1e-12)
        mass_target = max(0.0, 1.0 - alpha_zero)

    # Invert
    rprint("[yellow]Inverting Stieltjes transform...[/yellow]")
    rho_hat, rrmse = invert_stieltjes_density(
        q,
        g,
        g_se,
        lam_grid,
        delta_lam,
        n_nodes,
        mass_target,
        gamma_mass,
        tau_smooth,
        tau_tv,
        tv_iters,
        tv_eps,
    )
    rprint(f"[green]Fit relative RMSE on g(q):[/green] {rrmse:.4e}")

    # Compute fitted curves
    A_full = (
        delta_lam.reshape(1, -1) / (q.reshape(-1, 1) + lam_grid.reshape(1, -1))
    ) * n_nodes
    g_fit_remainder = A_full @ rho_hat
    g_fit_total = g_fit_remainder + (n_nodes * alpha_zero) / np.maximum(q, 1e-12)
    s_fit = q * g_fit_total

    # Save fitted s(q)
    s_df = s_df.copy()
    s_df["g_fit"] = g_fit_total
    s_df["s_fit"] = s_fit
    s_csv = out_dir / "s_vs_q_fit.csv"
    s_df.to_csv(s_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {s_csv}")

    # Optionally compute truth
    lambdas = None
    if compute_theory:
        try:
            rprint("[yellow]Computing eigenvalues for validation...[/yellow]")
            lambdas = laplacian_eigenvalues(G)
        except Exception as e:
            rprint(f"[red]Warning:[/red] could not compute eigenvalues ({e})")
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
        # Histogram on edges consistent with center grid
        mids = 0.5 * (lam_grid[1:] + lam_grid[:-1])
        left_edge = max(0.0, lam_grid[0] - (mids[0] - lam_grid[0]))
        right_edge = lam_grid[-1] + (lam_grid[-1] - mids[-1])
        edges = np.concatenate([[left_edge], mids, [right_edge]])
        counts, edges = np.histogram(lambdas, bins=edges)
        bin_widths = np.diff(edges)
        density_true = (counts.astype(float) / n_nodes) / np.maximum(bin_widths, 1e-12)
        spec_df["density_true"] = density_true

    spec_csv = out_dir / "spectrum.csv"
    spec_df.to_csv(spec_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {spec_csv}")

    # Plot s(q)
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
    ax.set_title(f"s(q): MC vs fit — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "s_vs_q_fit.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    rprint(f"[green]✓[/green] Saved: {fig_path}")

    # Plot spectral density
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        spec_df["lambda"], spec_df["density_hat"], label="reconstructed density", lw=2
    )
    if "density_true" in spec_df.columns:
        ax.step(
            spec_df["lambda"],
            spec_df["density_true"],
            where="mid",
            label="true histogram",
            alpha=0.6,
        )
        # Rug plot
        ax.scatter(
            lambdas,
            np.zeros_like(lambdas),
            marker="|",
            color="k",
            alpha=0.3,
            label="eigs",
        )
        # KDE using seaborn with normalize control
        try:
            rprint(f"[yellow]Computing KDE (normalize={kde_normalize})...[/yellow]")
            # Use seaborn's kdeplot with explicit normalization control
            lam_kde = np.linspace(lam_grid[0], lam_grid[-1], 1000)
            # Compute KDE density estimates
            kde_data = pd.DataFrame({"lambda": lambdas})
            sns.kdeplot(
                data=kde_data,
                x="lambda",
                bw_adjust=kde_bw,
                ax=ax,
                label="true KDE",
                color="C2",
                lw=2,
                alpha=0.9,
                # The 'common_norm' parameter controls normalization
                # When True (default), normalizes to integrate to 1
                common_norm=kde_normalize,
            )
        except Exception as e:
            rprint(f"[red]Warning:[/red] KDE failed ({e})")

    ax.set_xlabel("lambda")
    ax.set_ylabel("spectral density (normalized)")
    ax.set_title(f"Spectral density reconstruction — {gid}")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "spectrum.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    rprint(f"[green]✓[/green] Saved: {fig_path}")


# ============================================================================
# Subcommand 3: validate
# ============================================================================


def heat_trace_from_spectrum(beta: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    Compute heat trace Z(β) from eigenvalues.

    Parameters
    ----------
    beta : np.ndarray
        Array of inverse temperature values.
    lambdas : np.ndarray
        Laplacian eigenvalues.

    Returns
    -------
    np.ndarray
        Z(β) = sum_i exp(-β λ_i) for each β.
    """
    beta = np.asarray(beta, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    return np.exp(-np.outer(beta, lambdas)).sum(axis=1)


def s_from_spectrum(q: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    Compute s(q) from eigenvalues.

    Parameters
    ----------
    q : np.ndarray
        Array of q values.
    lambdas : np.ndarray
        Laplacian eigenvalues.

    Returns
    -------
    np.ndarray
        s(q) = sum_i q / (q + λ_i) for each q.
    """
    q = np.asarray(q, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    return (q[:, None] / (q[:, None] + lambdas[None, :])).sum(axis=1)


def s_from_Z_via_gauss_laguerre(
    q: np.ndarray, lambdas: np.ndarray, n_nodes: int
) -> np.ndarray:
    """
    Compute s(q) from heat trace Z(β) via Gauss-Laguerre quadrature.

    Uses the Laplace transform identity: s(q) = ∫_0^∞ e^{-t} Z(β t) dt with β = 1/q.

    Parameters
    ----------
    q : np.ndarray
        Array of q values.
    lambdas : np.ndarray
        Laplacian eigenvalues.
    n_nodes : int
        Number of quadrature nodes.

    Returns
    -------
    np.ndarray
        s(q) computed via numerical integration.
    """
    t_nodes, w_nodes = laggauss(n_nodes)
    s_vals = []
    for qi in q:
        beta = 1.0 / float(qi)
        Z_vals = np.exp(-np.outer(beta * t_nodes, lambdas)).sum(axis=1)
        s_vals.append(float(w_nodes @ Z_vals))
    return np.asarray(s_vals)


def estimate_s_via_wilson(
    G: nx.Graph, q: float, n_samples: int, random_state: int
) -> tuple[float, float]:
    """
    Estimate s(q) via Wilson Monte Carlo sampling.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    q : float
        Temperature parameter.
    n_samples : int
        Number of Monte Carlo samples.
    random_state : int
        Random seed.

    Returns
    -------
    mean : float
        Mean number of roots (estimate of s(q)).
    sem : float
        Standard error of the mean.
    """
    est = Wilson(q=q, random_state=random_state).fit(G)
    samples = est.transform(n_samples=n_samples, seed=random_state)
    root_counts = np.array([len(roots) for (_, roots) in samples], dtype=float)
    mean = float(root_counts.mean())
    sem = float(root_counts.std(ddof=1) / math.sqrt(max(1, n_samples)))
    return mean, sem


@app.command()
def validate(
    er_n: Annotated[int, typer.Option(help="ER nodes")] = 100,
    er_p: Annotated[float, typer.Option(help="ER edge probability")] = 0.05,
    ba_n: Annotated[int, typer.Option(help="BA nodes")] = 100,
    ba_m: Annotated[int, typer.Option(help="BA m parameter")] = 3,
    grid_rows: Annotated[int, typer.Option(help="Grid rows")] = 10,
    grid_cols: Annotated[int, typer.Option(help="Grid columns")] = 10,
    reg_n: Annotated[int, typer.Option(help="Regular graph nodes")] = 100,
    reg_d: Annotated[int, typer.Option(help="Regular graph degree")] = 4,
    q_min: Annotated[float, typer.Option(help="Minimum q value")] = 1e-2,
    q_max: Annotated[float, typer.Option(help="Maximum q value")] = 1e2,
    num_q: Annotated[int, typer.Option(help="Number of q points")] = 20,
    beta_min: Annotated[float, typer.Option(help="Minimum beta")] = 1e-2,
    beta_max: Annotated[float, typer.Option(help="Maximum beta")] = 1e2,
    num_beta: Annotated[int, typer.Option(help="Number of beta points")] = 20,
    n_wilson_samples: Annotated[int, typer.Option(help="Wilson MC samples")] = 512,
    gl_nodes: Annotated[int, typer.Option(help="Gauss-Laguerre nodes")] = 64,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    outdir: Annotated[
        str, typer.Option(help="Output directory")
    ] = "notebooks/artifacts",
) -> None:
    """
    Validate forest-heat-kernel identities via Wilson sampling.

    This command validates the Laplace transform identity relating the forest
    partition function s(q) to the heat trace Z(β). It computes:
    - Spectral heat trace Z(β) from eigenvalues
    - Spectral s(q) from eigenvalues
    - s(q) via Gauss-Laguerre quadrature from Z(β)
    - s(q) via Wilson Monte Carlo sampling
    and produces comparison plots for multiple graph families.

    Parameters
    ----------
    er_n : int
        Number of nodes for Erdős-Rényi graph.
    er_p : float
        Edge probability for ER graph.
    ba_n : int
        Number of nodes for Barabási-Albert graph.
    ba_m : int
        Number of edges per new node for BA graph.
    grid_rows : int
        Number of rows for grid graph.
    grid_cols : int
        Number of columns for grid graph.
    reg_n : int
        Number of nodes for regular graph.
    reg_d : int
        Degree for regular graph.
    q_min : float
        Minimum q value.
    q_max : float
        Maximum q value.
    num_q : int
        Number of q points.
    beta_min : float
        Minimum beta value.
    beta_max : float
        Maximum beta value.
    num_beta : int
        Number of beta points.
    n_wilson_samples : int
        Number of Wilson MC samples per q.
    gl_nodes : int
        Number of Gauss-Laguerre quadrature nodes.
    seed : int
        Random seed.
    outdir : str
        Output directory for plots.

    Examples
    --------
    Run validation with default parameters:

        $ wilson validate

    Run with more samples for higher accuracy:

        $ wilson validate --n-wilson-samples 2048 --gl-nodes 128

    Custom graph parameters:

        $ wilson validate --er-n 200 --er-p 0.1 --ba-n 200 --ba-m 5

    Notes
    -----
    - Generates validation plots for s(q) and Z(β) for each graph family
    - Reports relative errors for quadrature and Monte Carlo estimates
    - Validates the identity: s(q) = ∫_0^∞ e^{-t} Z(t/q) dt
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_context("paper")
    sns.set_style("whitegrid")

    out_path = Path(outdir)
    os.makedirs(out_path, exist_ok=True)

    rprint("[bold cyan]Forest-Heat Kernel Validation[/bold cyan]")

    # Build graphs
    rng = np.random.default_rng(seed)
    graphs: dict[str, nx.Graph] = {}

    G_er = nx.erdos_renyi_graph(er_n, er_p, seed=int(rng.integers(1, 2**31 - 1)))
    graphs[f"ER_n{er_n}_p{er_p:.3f}"] = G_er

    G_ba = nx.barabasi_albert_graph(ba_n, ba_m, seed=int(rng.integers(1, 2**31 - 1)))
    graphs[f"BA_n{ba_n}_m{ba_m}"] = G_ba

    G_grid = nx.grid_2d_graph(grid_rows, grid_cols)
    graphs[f"GRID_{grid_rows}x{grid_cols}"] = relabel_consecutive(G_grid)

    G_reg = nx.random_regular_graph(reg_d, reg_n, seed=int(rng.integers(1, 2**31 - 1)))
    graphs[f"REG_n{reg_n}_d{reg_d}"] = G_reg

    q_values = np.logspace(math.log10(q_min), math.log10(q_max), num_q)
    beta_values = np.logspace(math.log10(beta_min), math.log10(beta_max), num_beta)

    # Process each graph
    for name, G in graphs.items():
        rprint(f"\n[bold yellow]=== {name} ===[/bold yellow]")
        lambdas = laplacian_eigenvalues(G)
        rprint(
            f"  n={G.number_of_nodes()}, m={G.number_of_edges()}, λ_max={lambdas.max():.3f}"
        )

        # Spectral ground truth
        Z_spec = heat_trace_from_spectrum(beta_values, lambdas)
        s_spec = s_from_spectrum(q_values, lambdas)

        # Laplace forward check via Gauss-Laguerre
        rprint(
            f"[yellow]  Computing s from Z via Gauss-Laguerre ({gl_nodes} nodes)...[/yellow]"
        )
        s_quad = s_from_Z_via_gauss_laguerre(q_values, lambdas, n_nodes=gl_nodes)

        # Wilson Monte Carlo
        rprint(
            f"[yellow]  Estimating s via Wilson MC ({n_wilson_samples} samples)...[/yellow]"
        )
        s_mc_mean = np.zeros_like(q_values)
        s_mc_sem = np.zeros_like(q_values)
        for i, q in enumerate(q_values):
            mean, sem = estimate_s_via_wilson(G, float(q), n_wilson_samples, seed + i)
            s_mc_mean[i] = mean
            s_mc_sem[i] = sem

        # Report errors
        rel_err_quad = np.linalg.norm(s_quad - s_spec) / max(
            1e-12, np.linalg.norm(s_spec)
        )
        rel_err_mc = np.linalg.norm(s_mc_mean - s_spec) / max(
            1e-12, np.linalg.norm(s_spec)
        )
        rprint(f"  [green]rel error s_from_Z (GL):[/green] {rel_err_quad:.3e}")
        rprint(f"  [green]rel error s Wilson MC:[/green] {rel_err_mc:.3e}")

        # Plot s(q)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(q_values, s_spec, label="spectral s(q)", color="#1f77b4", lw=1)
        ax.semilogx(
            q_values,
            s_quad,
            label="from Z via Gauss-Laguerre",
            color="#2ca02c",
            lw=1,
            ls="--",
        )
        ax.errorbar(
            q_values,
            s_mc_mean,
            yerr=s_mc_sem,
            fmt="o",
            color="#d62728",
            ecolor="#ff9896",
            elinewidth=1.0,
            capsize=3,
            label="Wilson MC",
        )
        ax.set_xlabel("q")
        ax.set_ylabel("s(q) = E[#roots]")
        ax.set_title(f"s(q) validation — {name}")
        ax.legend(frameon=True)
        fig.tight_layout()
        fname = out_path / f"validation_s_vs_q_{name}.pdf"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        rprint(f"  [green]✓[/green] Saved: {fname}")

        # Also save PNG
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(q_values, s_spec, label="spectral s(q)", color="#1f77b4", lw=1)
        ax.semilogx(
            q_values,
            s_quad,
            label="from Z via Gauss-Laguerre",
            color="#2ca02c",
            lw=1,
            ls="--",
        )
        ax.errorbar(
            q_values,
            s_mc_mean,
            yerr=s_mc_sem,
            fmt="o",
            color="#d62728",
            ecolor="#ff9896",
            elinewidth=1.0,
            capsize=3,
            label="Wilson MC",
        )
        ax.set_xlabel("q")
        ax.set_ylabel("s(q) = E[#roots]")
        ax.set_title(f"s(q) validation — {name}")
        ax.legend(frameon=True)
        fig.tight_layout()
        fname_png = out_path / f"validation_s_vs_q_{name}.png"
        fig.savefig(fname_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        rprint(f"  [green]✓[/green] Saved: {fname_png}")

        # Plot Z(β)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(beta_values, Z_spec, label="spectral Z(β)", color="#1f77b4", lw=1)
        ax.set_xlabel("β")
        ax.set_ylabel("Z(β) = Tr e^{-βL}")
        ax.set_title(f"Heat trace — {name}")
        ax.legend(frameon=True)
        fig.tight_layout()
        fname = out_path / f"validation_Z_vs_beta_{name}.pdf"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        rprint(f"  [green]✓[/green] Saved: {fname}")

        # Also save PNG
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(beta_values, Z_spec, label="spectral Z(β)", color="#1f77b4", lw=1)
        ax.set_xlabel("β")
        ax.set_ylabel("Z(β) = Tr e^{-βL}")
        ax.set_title(f"Heat trace — {name}")
        ax.legend(frameon=True)
        fig.tight_layout()
        fname_png = out_path / f"validation_Z_vs_beta_{name}.png"
        fig.savefig(fname_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        rprint(f"  [green]✓[/green] Saved: {fname_png}")

    rprint(f"\n[bold green]✓ All figures saved to {out_path}[/bold green]")


if __name__ == "__main__":
    app()
