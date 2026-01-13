"""Common utility functions for CLI and library."""

import math
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from rich import print as rprint

from wilson.graphs import build_graph, graph_id


def setup_plotting_style(context: str = "paper", style: str = "whitegrid") -> None:
    """
    Configure plotting style for research-paper quality figures.

    Sets Helvetica font, enables tight layout, and configures seaborn
    for publication-quality output.

    Parameters
    ----------
    context : str
        Seaborn context ('paper', 'talk', 'poster', 'notebook').
        Default is 'paper' for publication quality.
    style : str
        Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks').

    Examples
    --------
    >>> setup_plotting_style("paper", "whitegrid")

    Notes
    -----
    This function configures:
    - Helvetica font family for all text
    - Automatic tight layout for better subplot handling
    - TeX rendering for mathematical notation
    - Seaborn context and style
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # Seaborn settings
    sns.set_context(context)
    sns.set_style(style)

    # Matplotlib settings for research-paper quality
    mpl.rcParams["figure.autolayout"] = True  # Enables tight layout
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Helvetica"

    # Enable TeX rendering for proper subscripts/superscripts
    plt.rcParams["text.usetex"] = False  # Use mathtext, not full LaTeX
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Helvetica"
    plt.rcParams["mathtext.it"] = "Helvetica:italic"
    plt.rcParams["mathtext.bf"] = "Helvetica:bold"


def format_label(text: str) -> str:
    """
    Format text for plot labels with proper TeX subscripts.

    Converts underscores to subscripts and wraps in math mode.
    Preserves Greek letters and mathematical notation.

    Parameters
    ----------
    text : str
        Raw label text (e.g., 's_mc', 'lambda_max').

    Returns
    -------
    str
        TeX-formatted label (e.g., '$s_{\mathrm{mc}}$', '$\lambda_{\mathrm{max}}$').

    Examples
    --------
    >>> format_label("s_mc")
    '$s_{\\mathrm{mc}}$'
    >>> format_label("lambda_max")
    '$\\lambda_{\\mathrm{max}}$'
    >>> format_label("Wilson MC")
    'Wilson MC'

    Notes
    -----
    - Underscores are converted to subscripts with roman font
    - Greek letter names are converted to symbols (lambda → λ)
    - Plain text without underscores is returned unchanged
    """
    # If no underscore, return as-is unless it's a Greek letter
    if "_" not in text:
        # Replace common Greek letter names
        greek_map = {
            "lambda": r"$\lambda$",
            "beta": r"$\beta$",
            "gamma": r"$\gamma$",
            "alpha": r"$\alpha$",
            "delta": r"$\delta$",
            "epsilon": r"$\epsilon$",
            "rho": r"$\rho$",
            "sigma": r"$\sigma$",
            "tau": r"$\tau$",
        }
        for greek, symbol in greek_map.items():
            if text.lower() == greek:
                return symbol
        return text

    # Split on underscore
    parts = text.split("_", 1)
    main = parts[0]
    subscript = parts[1] if len(parts) > 1 else ""

    # Convert Greek letter names to symbols
    greek_map = {
        "lambda": r"\lambda",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "alpha": r"\alpha",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "rho": r"\rho",
        "sigma": r"\sigma",
        "tau": r"\tau",
    }

    main_symbol = greek_map.get(main.lower(), main)

    # Format with roman subscript
    if subscript:
        return f"${main_symbol}_{{\mathrm{{{subscript}}}}}$"
    return f"${main_symbol}$"


def make_logspace_grid(x_min: float, x_max: float, num_points: int) -> np.ndarray:
    """
    Create log-spaced grid of points.

    Parameters
    ----------
    x_min : float
        Minimum value.
    x_max : float
        Maximum value.
    num_points : int
        Number of points.

    Returns
    -------
    np.ndarray
        Log-spaced array from x_min to x_max.

    Examples
    --------
    >>> grid = make_logspace_grid(1e-3, 1e3, 30)
    >>> len(grid)
    30
    """
    return np.logspace(math.log10(x_min), math.log10(x_max), num_points)


def setup_graph_and_output(
    graph_type: str,
    outdir: str,
    n: int | None = None,
    p: float | None = None,
    m: int | None = None,
    d: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    blocks: int | None = None,
    block_size: int | None = None,
    p_in: float | None = None,
    p_out: float | None = None,
    tau1: float | None = None,
    tau2: float | None = None,
    mu: float | None = None,
    average_degree: float | None = None,
    min_degree: int | None = None,
    max_degree: int | None = None,
    min_community: int | None = None,
    max_community: int | None = None,
    tol: float | None = 1e-7,
    max_iters: int | None = 500,
    seed: int | None = None,
) -> tuple[nx.Graph, str, Path]:
    """
    Build graph, generate ID, and create output directory.

    Parameters
    ----------
    graph_type : str
        Graph family (ER, BA, REG, GRID, SBM).
    outdir : str
        Base output directory.
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
    blocks : int, optional
        Number of communities (SBM).
    block_size : int, optional
        Community size (SBM).
    p_in : float, optional
        Intra-community edge probability (SBM).
    p_out : float, optional
        Inter-community edge probability (SBM).
    tau1 : float, optional
        Degree exponent (LFR).
    tau2 : float, optional
        Community-size exponent (LFR).
    mu : float, optional
        Mixing parameter (LFR).
    average_degree : float, optional
        Average degree (LFR).
    min_degree : int, optional
        Minimum degree (LFR).
    max_degree : int, optional
        Maximum degree (LFR).
    min_community : int, optional
        Minimum community size (LFR).
    max_community : int, optional
        Maximum community size (LFR).
    tol : float, optional
        Tolerance for LFR generation.
    max_iters : int, optional
        Maximum iterations for LFR generation.
    seed : int, optional
        Random seed.

    Returns
    -------
    G : nx.Graph
        Built graph.
    gid : str
        Graph identifier string.
    out_dir : Path
        Output directory path.

    Examples
    --------
    >>> G, gid, out_dir = setup_graph_and_output("ER", "artifacts", n=100, p=0.05, seed=42)
    >>> G.number_of_nodes()
    100
    >>> gid
    'ER_n100_p0.050'
    """
    G = build_graph(
        graph_type,
        n=n,
        p=p,
        m=m,
        d=d,
        rows=rows,
        cols=cols,
        blocks=blocks,
        block_size=block_size,
        p_in=p_in,
        p_out=p_out,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        tol=tol,
        max_iters=max_iters,
        seed=seed,
    )
    gid = graph_id(
        graph_type,
        n=n,
        p=p,
        m=m,
        d=d,
        rows=rows,
        cols=cols,
        blocks=blocks,
        block_size=block_size,
        p_in=p_in,
        p_out=p_out,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        tol=tol,
        max_iters=max_iters,
    )
    out_dir = Path(outdir) / gid
    os.makedirs(out_dir, exist_ok=True)

    rprint(f"[bold cyan]Graph:[/bold cyan] {gid}")
    rprint(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    return G, gid, out_dir


def compute_histogram_density(
    lambdas: np.ndarray,
    lam_grid: np.ndarray,
    n_nodes: float,
) -> np.ndarray:
    """
    Compute histogram density of eigenvalues on lambda grid.

    Uses consistent edge construction with bin_widths logic.

    Parameters
    ----------
    lambdas : np.ndarray
        Eigenvalues to histogram.
    lam_grid : np.ndarray
        Lambda grid centers.
    n_nodes : float
        Number of nodes for normalization.

    Returns
    -------
    np.ndarray
        Normalized density values at grid centers.

    Examples
    --------
    >>> lambdas = np.array([0.0, 1.0, 2.0, 3.0])
    >>> lam_grid = np.linspace(0, 4, 5)
    >>> density = compute_histogram_density(lambdas, lam_grid, 4.0)
    >>> len(density) == len(lam_grid)
    True
    """
    # Build edges consistent with grid centers
    mids = 0.5 * (lam_grid[1:] + lam_grid[:-1])
    left_edge = max(0.0, lam_grid[0] - (mids[0] - lam_grid[0]))
    right_edge = lam_grid[-1] + (lam_grid[-1] - mids[-1])
    edges = np.concatenate([[left_edge], mids, [right_edge]])

    # Compute histogram
    counts, _ = np.histogram(lambdas, bins=edges)
    bin_widths = np.diff(edges)
    density = (counts.astype(float) / n_nodes) / np.maximum(bin_widths, 1e-12)

    return density
