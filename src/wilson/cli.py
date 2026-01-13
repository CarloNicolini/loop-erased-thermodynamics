#!/usr/bin/env python3
"""
Unified CLI for Wilson algorithm experiments and spectral analysis.

This module provides four main subcommands:
- sample-s: Sample s(q) via Wilson's q-forest sampler
- reconstruct: Inverse spectral density reconstruction from s(q) measurements
- validate: Validate forest-heat-kernel identities via Wilson sampling
- fractal: Estimate LERW fractal dimension on wired 2D grids
"""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from rich import print as rprint
from typing_extensions import Annotated

from wilson.wilson import sample_wilson
from wilson.inversion import reconstruct_spectrum_from_wilson_graph
from wilson.plotting import (
    plot_g_vs_q,
    plot_s_vs_q,
    plot_spectrum_reconstruction,
    plot_Z_reconstruction,
)
from wilson.utils import (
    make_logspace_grid,
    setup_graph_and_output,
    setup_plotting_style,
)

setup_plotting_style("paper", "whitegrid")

app = typer.Typer(
    help="Wilson algorithm CLI for thermodynamic graph analysis",
    no_args_is_help=True,
)


# Sample the s function


@app.command(name="sample")
def sample(
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
    samples_per_q: Annotated[int, typer.Option(help="MC samples per q")] = 32,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    outdir: Annotated[
        str, typer.Option(help="Output directory base")
    ] = "artifacts/sampling",
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

    Examples
    --------
    Sample s(q) for an Erdős-Rényi graph:

        $ wilson sample --graph ER --n 100 --p 0.05 --num-q 30

    Sample for a Barabási-Albert graph:

        $ wilson sample --graph BA --n 100 --m 3 --samples-per-q 512

    Notes
    -----
    - Outputs are saved to `outdir/<graph_id>/s_vs_q.csv`
    - Two plots are generated: s(q) and g(q) = s(q)/q
    - Standard errors are computed from MC samples
    """
    G, gid, out_dir = setup_graph_and_output(
        graph, outdir, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed
    )
    lambdas = nx.laplacian_spectrum(G)

    # Sample s(q) across q-grid in parallel
    q_grid = make_logspace_grid(q_min, q_max, num_q)
    rprint(f"[yellow]Sampling s(q) for {num_q} q-values (parallel)...[/yellow]")

    # Parallel execution with cached eigenvalues
    records = Parallel(n_jobs=-1)(
        delayed(sample_wilson)(
            q=q,
            G=G,
            n_samples=samples_per_q,
            seed=seed,
            lambdas=lambdas,
            compute_theory=(lambdas is not None),
        )
        for q in q_grid
    )

    df = pd.DataFrame.from_records(records).sort_values("q").reset_index(drop=True)
    csv_path = out_dir / "s_vs_q.csv"
    df.to_csv(csv_path, index=False)
    rprint(f"[green]✓[/green] Saved: {csv_path}")

    # Generate plots using library functions
    plot_s_vs_q(df, out_dir, gid, save=True)
    plot_g_vs_q(df, out_dir, gid, save=True)


@app.command(name="reconstruct")
def reconstruct(
    graph: Annotated[str, typer.Option(help="Graph family: ER, BA, REG, GRID")] = "ER",
    n: Annotated[int, typer.Option(help="Number of nodes (ER/BA/REG)")] = 100,
    p: Annotated[float, typer.Option(help="Edge probability (ER)")] = 0.05,
    m: Annotated[int, typer.Option(help="BA 'm' parameter")] = 3,
    d: Annotated[int, typer.Option(help="Regular degree (REG)")] = 4,
    rows: Annotated[int, typer.Option(help="GRID rows")] = 10,
    cols: Annotated[int, typer.Option(help="GRID columns")] = 10,
    q_min: Annotated[float, typer.Option(help="Minimum q value")] = 1e-3,
    q_max: Annotated[float, typer.Option(help="Maximum q value")] = 1e3,
    num_q: Annotated[int, typer.Option(help="Number of q points")] = 20,
    beta_min: Annotated[float, typer.Option(help="Minimum beta")] = 1e-2,
    beta_max: Annotated[float, typer.Option(help="Maximum beta")] = 1e2,
    num_beta: Annotated[int, typer.Option(help="Number of beta points")] = 20,
    samples_per_q: Annotated[int, typer.Option(help="Wilson MC samples per q")] = 256,
    n_bootstrap: Annotated[int, typer.Option(help="Bootstrap replicates")] = 5,
    mass_target: Annotated[float, typer.Option(help="Target density mass")] = 1.0,
    gamma_mass: Annotated[float, typer.Option(help="Mass penalty strength")] = 1e4,
    tau_smooth: Annotated[float, typer.Option(help="L2 smoothness penalty")] = 1e-2,
    tau_tv: Annotated[float, typer.Option(help="TV penalty strength")] = 0.0,
    tv_iters: Annotated[int, typer.Option(help="TV IRLS iterations")] = 4,
    tv_eps: Annotated[float, typer.Option(help="TV epsilon")] = 1e-6,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    outdir: Annotated[
        str, typer.Option(help="Output directory base")
    ] = "artifacts/thermo",
) -> None:
    """Reconstruct spectral density and Z(β) from Wilson s(q) measurements.

    This command:

    1. Generates a graph from the specified family.
    2. Samples s(q) via Wilson's q-forest sampler on a log-spaced q-grid.
    3. Inverts the Stieltjes transform g(q) = s(q)/q to recover a spectral
       density ρ(λ) via regularized least squares.
    4. Computes the partition function Z(β) on a β-grid by forward quadrature.
    5. Optionally runs a bootstrap by re-sampling Wilson forests to obtain
       uncertainty bands on Z(β).
    """

    G, gid, out_dir = setup_graph_and_output(
        graph, outdir, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed
    )

    # Grids in q and beta
    q_grid = make_logspace_grid(q_min, q_max, num_q)
    beta_grid = make_logspace_grid(beta_min, beta_max, num_beta)

    rprint(
        f"[yellow]Reconstructing spectrum from Wilson data on {gid} "
        f"({num_q} q-values, {num_beta} beta-values)...[/yellow]"
    )

    result = reconstruct_spectrum_from_wilson_graph(
        G=G,
        q=q_grid,
        beta=beta_grid,
        n_samples_per_q=samples_per_q,
        n_bootstrap=n_bootstrap,
        mass_target=mass_target,
        gamma_mass=gamma_mass,
        tau_smooth=tau_smooth,
        tau_tv=tau_tv,
        tv_iters=tv_iters,
        tv_eps=tv_eps,
        random_state=seed,
    )

    # Save reconstructed density
    lambdas_exact = nx.laplacian_spectrum(G)
    spec_df = pd.DataFrame(
        {
            "lambda": result.lam_grid,
            "density_hat": result.rho_hat,
        }
    )
    # Optional histogram-based true density for diagnostics
    if lambdas_exact.size > 0:
        counts, edges = np.histogram(
            lambdas_exact,
            bins=len(result.lam_grid),
            range=(result.lam_grid[0], result.lam_grid[-1]),
        )
        delta = np.diff(edges)
        density_true = counts / (counts.sum() * delta)
        # align to grid midpoints
        mids = 0.5 * (edges[1:] + edges[:-1])
        # simple interpolation onto lam_grid
        density_true_interp = np.interp(result.lam_grid, mids, density_true)
        spec_df["density_true"] = density_true_interp

    spec_csv = out_dir / "spectrum_reconstruction.csv"
    spec_df.to_csv(spec_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {spec_csv}")

    # Save Z(beta) reconstruction
    Z_df = pd.DataFrame({"beta": result.beta, "Z_hat": result.Z_hat})
    if result.Z_ci_lower is not None and result.Z_ci_upper is not None:
        Z_df["Z_ci_lower"] = result.Z_ci_lower
        Z_df["Z_ci_upper"] = result.Z_ci_upper
    if result.Z_true is not None:
        Z_df["Z_true"] = result.Z_true
    Z_csv = out_dir / "Z_reconstruction.csv"
    Z_df.to_csv(Z_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {Z_csv}")

    # Save g(q) data used for reconstruction
    g_df = pd.DataFrame(
        {
            "q": result.q,
            "g_mc": result.g_hat,
        }
    )
    if result.g_se is not None:
        g_df["g_mc_se"] = result.g_se
    g_csv = out_dir / "g_vs_q_reconstruct.csv"
    g_df.to_csv(g_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {g_csv}")

    # Plots
    plot_spectrum_reconstruction(
        spec_df,
        out_path=out_dir,
        graph_id=gid,
        lambdas=lambdas_exact,
        save=True,
    )
    plot_Z_reconstruction(
        beta=result.beta,
        Z_hat=result.Z_hat,
        out_path=out_dir,
        graph_id=gid,
        Z_true=result.Z_true,
        Z_ci_lower=result.Z_ci_lower,
        Z_ci_upper=result.Z_ci_upper,
        save=True,
    )

    rprint(f"[cyan]Reconstruction RMSE in g(q):[/cyan] {result.rrmse:.3e}")
    if result.Z_true is not None:
        rel_l2 = np.linalg.norm(result.Z_hat - result.Z_true) / np.linalg.norm(
            result.Z_true
        )
        rprint(f"[cyan]Relative L2 error on Z(beta):[/cyan] {rel_l2:.3e}")


if __name__ == "__main__":
    app()
