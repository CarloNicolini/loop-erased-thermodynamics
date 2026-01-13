"""Plotting utilities for Wilson algorithm visualizations."""

from pathlib import Path
import math
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from rich import print as rprint


def _component_colors(n: int) -> list[tuple[float, float, float]]:
    """Return up to n colours sampled from the tab20 palette."""

    if n <= 0:
        return []
    tab20 = plt.get_cmap("tab20")
    # Use even indices first to maximize contrast, then odds.
    order = list(range(0, tab20.N, 2)) + list(range(1, tab20.N, 2))
    colours: list[tuple[float, float, float]] = []
    for i in range(n):
        idx = order[i % len(order)]
        rgba = tab20(idx)
        colours.append((float(rgba[0]), float(rgba[1]), float(rgba[2])))
    return colours


def save_plot_both_formats(fig: plt.Figure, out_path: Path, base_name: str) -> None:
    """
    Save matplotlib figure in both PDF and PNG formats.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save.
    out_path : Path
        Output directory path.
    base_name : str
        Base filename without extension (e.g., 'validation_s_vs_q_ER_n100').

    Notes
    -----
    Saves with dpi=200 and bbox_inches='tight' for both formats.
    Automatically closes the figure after saving.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pathlib import Path
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_plot_both_formats(fig, Path("."), "test_plot")
    """
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fname = out_path / f"{base_name}.{ext}"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        rprint(f"  [green]✓[/green] Saved: {fname}")
    plt.close(fig)


def plot_s_vs_q(
    df: pd.DataFrame,
    out_path: Path,
    graph_id: str,
    save: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create s(q) plot with MC samples and optional theory/fit curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: q, s_mc, s_mc_se, and optionally s_true, s_fit.
    out_path : Path
        Output directory for saving.
    graph_id : str
        Graph identifier for title.
    save : bool
        If True, save the plot.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'q': [0.1, 1.0], 's_mc': [5.0, 10.0], 's_mc_se': [0.1, 0.2]})
    >>> fig, ax = plot_s_vs_q(df, Path("."), "ER_n100", save=False)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        df["q"],
        df["s_mc"],
        yerr=df.get("s_mc_se", None),
        fmt="o",
        ms=4,
        lw=1,
        label="Wilson MC",
    )
    if "s_fit" in df.columns:
        ax.plot(df["q"], df["s_fit"], label="Wilson MC", lw=1)
    if "s_spec" in df.columns:
        ax.plot(df["q"], df["s_spec"], label="Spectral", color="C1", lw=1)

    ax.set_xscale("log")
    ax.set_xlabel("$q$")
    ax.set_ylabel("$s(q)$")
    # ax.set_title(f"$s(q)$ via Wilson sampling — {graph_id}")
    ax.legend()

    if save:
        fig.tight_layout()
        fig_path = out_path / "s_vs_q.pdf"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        rprint(f"[green]✓[/green] Saved: {fig_path}")

    return fig, ax


def plot_g_vs_q(
    df: pd.DataFrame,
    out_path: Path,
    graph_id: str,
    save: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create g(q) = s(q)/q plot for Stieltjes inversion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: q, g_mc, and optionally g_true.
    out_path : Path
        Output directory for saving.
    graph_id : str
        Graph identifier for title.
    save : bool
        If True, save the plot.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        df["q"],
        df["g_mc"],
        yerr=df.get("g_mc_se", None),
        fmt="o",
        ms=4,
        lw=1,
        label="Wilson MC",
    )
    if "g_fit" in df.columns:
        ax.plot(df["q"], df["g_fit"], label="Wilson MC", lw=1)
    if "g_spec" in df.columns:
        ax.plot(df["q"], df["g_spec"], label="spectral", color="C1", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("$q$")
    ax.set_ylabel("$g(q) = s(q)/q$")
    # ax.set_title(f"$g(q)$ for Stieltjes inversion — {graph_id}")
    ax.legend()

    if save:
        fig.tight_layout()
        fig_path = out_path / "g_vs_q.pdf"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        rprint(f"[green]✓[/green] Saved: {fig_path}")

    return fig, ax


def plot_spectrum_reconstruction(
    spec_df: pd.DataFrame,
    out_path: Path,
    graph_id: str,
    lambdas: np.ndarray | None = None,
    kde_bw: float = 1.0,
    kde_normalize: bool = True,
    save: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create spectral density reconstruction plot.

    Parameters
    ----------
    spec_df : pd.DataFrame
        DataFrame with lambda, density_hat, and optionally density_true.
    out_path : Path
        Output directory.
    graph_id : str
        Graph identifier for title.
    lambdas : np.ndarray or None
        True eigenvalues for KDE and rug plot.
    kde_bw : float
        KDE bandwidth adjustment.
    kde_normalize : bool
        Normalize KDE to unit integral.
    save : bool
        If True, save the plot.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(spec_df["lambda"], spec_df["density_hat"], label="reconstructed", lw=1)

    if "density_true" in spec_df.columns and lambdas is not None:
        ax.step(
            spec_df["lambda"],
            spec_df["density_true"],
            where="mid",
            label="histogram",
            alpha=0.6,
        )
        # Rug plot
        ax.scatter(
            lambdas,
            np.zeros_like(lambdas),
            marker="|",
            color="k",
            alpha=0.3,
            label="eigenvalues",
        )
        # KDE
        try:
            rprint(f"[yellow]Computing KDE (normalize={kde_normalize})...[/yellow]")
            kde_data = pd.DataFrame({"lambda": lambdas})
            sns.kdeplot(
                data=kde_data,
                x="lambda",
                bw_adjust=kde_bw,
                ax=ax,
                label="KDE",
                color="C2",
                lw=1,
                alpha=0.9,
                common_norm=kde_normalize,
            )
        except Exception as e:
            rprint(f"[red]Warning:[/red] KDE failed ({e})")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"spectral density $\rho(\lambda)$")
    # ax.set_title(f"Spectral density reconstruction — {graph_id}")
    ax.legend()

    if save:
        fig.tight_layout()
        fig_path = out_path / "spectrum.pdf"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        rprint(f"[green]✓[/green] Saved: {fig_path}")

    return fig, ax


def plot_Z_reconstruction(
    beta: np.ndarray,
    Z_hat: np.ndarray,
    out_path: Path,
    graph_id: str,
    Z_true: np.ndarray | None = None,
    Z_ci_lower: np.ndarray | None = None,
    Z_ci_upper: np.ndarray | None = None,
    save: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create Z(β) reconstruction plot with optional CIs and ground truth.

    Parameters
    ----------
    beta : np.ndarray
        β-grid.
    Z_hat : np.ndarray
        Reconstructed partition function values.
    out_path : Path
        Output directory.
    graph_id : str
        Graph identifier for filenames.
    Z_true : np.ndarray, optional
        Spectral ground truth Z(β) values.
    Z_ci_lower, Z_ci_upper : np.ndarray, optional
        Lower and upper credible intervals for Z(β).
    save : bool, default True
        If True, save plot in PDF and PNG formats.
    """

    beta = np.asarray(beta, dtype=float)
    Z_hat = np.asarray(Z_hat, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    if Z_true is not None:
        Z_true = np.asarray(Z_true, dtype=float)
        ax.loglog(beta, Z_true, label="Z true", color="orange", lw=1.5)
    ax.loglog(beta, Z_hat, label="Z reconstructed", color="C0", lw=1.5)

    if Z_ci_lower is not None and Z_ci_upper is not None:
        ax.fill_between(
            beta,
            Z_ci_lower,
            Z_ci_upper,
            alpha=0.25,
            label="15–85% CI",
            color="C0",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$Z(\beta)$")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    if save:
        save_plot_both_formats(fig, out_path, f"Z_reconstruction_{graph_id}")

    return fig, ax


def plot_validation_s_vs_q(
    q_values: np.ndarray,
    s_spec: np.ndarray,
    s_quad: np.ndarray,
    s_mc_mean: np.ndarray,
    s_mc_sem: np.ndarray,
    name: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create s(q) validation plot comparing spectral, quadrature, and MC estimates.

    Parameters
    ----------
    q_values : np.ndarray
        Array of q values.
    s_spec : np.ndarray
        Spectral s(q) values.
    s_quad : np.ndarray
        s(q) from Gauss-Laguerre quadrature.
    s_mc_mean : np.ndarray
        Wilson MC mean estimates.
    s_mc_sem : np.ndarray
        Wilson MC standard errors.
    name : str
        Graph name for title.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(q_values, s_spec, label="spectral", color="#1f77b4", lw=1)
    ax.semilogx(
        q_values,
        s_quad,
        label="Gauss-Laguerre",
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
    ax.set_xlabel("$q$")
    ax.set_ylabel(r"$s(q) = \mathbb{E}[\mathrm{roots}]$")
    # ax.set_title(f"$s(q)$ validation — {name}")
    ax.legend(frameon=True)
    return fig, ax


def plot_validation_Z_vs_beta(
    beta_values: np.ndarray,
    Z_spec: np.ndarray,
    name: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create Z(β) heat trace validation plot.

    Parameters
    ----------
    beta_values : np.ndarray
        Array of beta values.
    Z_spec : np.ndarray
        Spectral heat trace values.
    name : str
        Graph name for title.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(beta_values, Z_spec, label="spectral", color="#1f77b4", lw=1)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$Z(\beta) = \mathrm{Tr}\, e^{-\beta L}$")
    # ax.set_title(f"Heat trace — {name}")
    ax.legend(frameon=True)
    return fig, ax
