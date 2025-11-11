#!/usr/bin/env python3
"""
Unified CLI for Wilson algorithm experiments and spectral analysis.

This module provides four main subcommands:
- sample-s: Sample s(q) via Wilson's q-forest sampler
- reconstruct: Inverse spectral density reconstruction from s(q) measurements
- validate: Validate forest-heat-kernel identities via Wilson sampling
- fractal: Estimate LERW fractal dimension on wired 2D grids
"""

import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from rich import print as rprint
from typing_extensions import Annotated

from wilson.fractal import run_fractal_dimension_experiment
from wilson.inversion import compute_bin_widths, invert_stieltjes_density
from wilson.parallel import estimate_s_for_validate, sample_wilson_for_q
from wilson.plotting import (
    plot_g_vs_q,
    plot_s_vs_q,
    plot_spectrum_reconstruction,
    plot_validation_s_vs_q,
    plot_validation_Z_vs_beta,
    save_plot_both_formats,
)
from wilson.spectral import (
    heat_trace_from_spectrum,
    laplacian_eigenvalues,
    s_from_spectrum,
    s_from_Z_via_gauss_laguerre,
)
from wilson.utils import (
    compute_eigenvalues_safe,
    compute_histogram_density,
    make_logspace_grid,
    setup_graph_and_output,
    setup_plotting_style,
)
from wilson.validation import build_validation_graphs

app = typer.Typer(
    help="Wilson algorithm CLI for thermodynamic graph analysis",
    no_args_is_help=True,
)


# ============================================================================
# Subcommand 1: sample the s function
# ============================================================================


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
    setup_plotting_style("talk", "whitegrid")
    G, gid, out_dir = setup_graph_and_output(
        graph, outdir, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed
    )
    lambdas = compute_eigenvalues_safe(
        G, compute=not skip_theory, description="eigenvalues (once)"
    )

    # Sample s(q) across q-grid in parallel
    q_grid = make_logspace_grid(q_min, q_max, num_q)
    rprint(f"[yellow]Sampling s(q) for {num_q} q-values (parallel)...[/yellow]")

    # Parallel execution with cached eigenvalues
    records = Parallel(n_jobs=-1, backend="loky")(
        delayed(sample_wilson_for_q)(
            q=float(q),
            G=G,
            samples_per_q=samples_per_q,
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


# ============================================================================
# Subcommand 2: reconstruct
# ============================================================================


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
    setup_plotting_style("talk", "whitegrid")
    G, gid, out_dir = setup_graph_and_output(
        graph, outdir, n=n, p=p, m=m, d=d, rows=rows, cols=cols, seed=seed
    )

    # Load or compute s(q) data
    csv_path = out_dir / "s_vs_q.csv" if s_csv is None else Path(s_csv)
    if csv_path.exists() and not compute_s:
        rprint(f"[yellow]Loading s(q) data from {csv_path}...[/yellow]")
        s_df = pd.read_csv(csv_path)
    elif compute_s:
        rprint("[yellow]Computing s(q) from scratch (parallel)...[/yellow]")
        q_grid = make_logspace_grid(q_min, q_max, num_q)
        lambdas_for_s = compute_eigenvalues_safe(
            G, compute=True, description="eigenvalues for s(q)"
        )

        # Parallel execution
        records = Parallel(n_jobs=-1, backend="loky")(
            delayed(sample_wilson_for_q)(
                q=float(q),
                G=G,
                samples_per_q=samples_per_q,
                seed=seed,
                lambdas=lambdas_for_s,
                compute_theory=False,
            )
            for q in q_grid
        )
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
    lambdas = compute_eigenvalues_safe(
        G, compute=compute_theory, description="eigenvalues for validation"
    )

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
        # Compute histogram density using library function
        density_true = compute_histogram_density(lambdas, lam_grid, n_nodes)
        spec_df["density_true"] = density_true

    spec_csv = out_dir / "spectrum.csv"
    spec_df.to_csv(spec_csv, index=False)
    rprint(f"[green]✓[/green] Saved: {spec_csv}")

    # Generate plots using library functions
    plot_s_vs_q(s_df, out_dir, gid, save=True)
    plot_spectrum_reconstruction(
        spec_df, out_dir, gid, lambdas, kde_bw, kde_normalize, save=True
    )


# ============================================================================
# Subcommand 3: validate
# ============================================================================


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
    setup_plotting_style("paper", "whitegrid")

    out_path = Path(outdir)
    os.makedirs(out_path, exist_ok=True)

    rprint("[bold cyan]Forest-Heat Kernel Validation[/bold cyan]")

    # Build graphs using library function
    graphs = build_validation_graphs(
        er_n, er_p, ba_n, ba_m, grid_rows, grid_cols, reg_n, reg_d, seed
    )

    q_values = make_logspace_grid(q_min, q_max, num_q)
    beta_values = make_logspace_grid(beta_min, beta_max, num_beta)

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

        # Wilson Monte Carlo (parallel with cached eigenvalues)
        rprint(
            f"[yellow]  Estimating s via Wilson MC ({n_wilson_samples} samples, parallel)...[/yellow]"
        )
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(estimate_s_for_validate)(
                G, float(q), n_wilson_samples, seed + i, lambdas
            )
            for i, q in enumerate(q_values)
        )
        s_mc_mean = np.array([r[0] for r in results])
        s_mc_sem = np.array([r[1] for r in results])

        # Report errors
        rel_err_quad = np.linalg.norm(s_quad - s_spec) / max(
            1e-12, np.linalg.norm(s_spec)
        )
        rel_err_mc = np.linalg.norm(s_mc_mean - s_spec) / max(
            1e-12, np.linalg.norm(s_spec)
        )
        rprint(f"  [green]rel error s_from_Z (GL):[/green] {rel_err_quad:.3e}")
        rprint(f"  [green]rel error s Wilson MC:[/green] {rel_err_mc:.3e}")

        # Generate validation plots using library functions
        fig, ax = plot_validation_s_vs_q(
            q_values, s_spec, s_quad, s_mc_mean, s_mc_sem, name
        )
        save_plot_both_formats(fig, out_path, f"validation_s_vs_q_{name}")

        fig, ax = plot_validation_Z_vs_beta(beta_values, Z_spec, name)
        save_plot_both_formats(fig, out_path, f"validation_Z_vs_beta_{name}")

    rprint(f"\n[bold green]✓ All figures saved to {out_path}[/bold green]")


@app.command()
def fractal(
    rows: Annotated[int, typer.Option(help="Grid rows")] = 48,
    cols: Annotated[int, typer.Option(help="Grid columns")] = 48,
    samples: Annotated[int, typer.Option(help="Number of LERW samples")] = 20000,
    seed: Annotated[int, typer.Option(help="Random seed")] = 123,
    outdir: Annotated[
        str, typer.Option(help="Base output directory")
    ] = "notebooks/artifacts",
    log_bins: Annotated[int, typer.Option(help="Log-spaced bins for averaging")] = 18,
    max_scatter: Annotated[
        int,
        typer.Option(
            help="Max scatter points to plot (0 skips scatter, negative keeps all)"
        ),
    ] = 4000,
) -> None:
    """
    Estimate the LERW fractal dimension on a wired 2D grid.

    Samples loop-erased random walks, measures loop-erased length vs Euclidean
    span, fits the scaling exponent, and saves CSV/JSON artifacts alongside a
    diagnostic plot.
    """

    setup_plotting_style("paper", "whitegrid")
    summary = run_fractal_dimension_experiment(
        rows=rows,
        cols=cols,
        samples=samples,
        seed=seed,
        outdir=outdir,
        log_bins=log_bins,
        max_scatter=max_scatter,
    )
    rprint(
        "[bold cyan]Estimated fractal dimension:[/bold cyan] "
        f"d_f = {summary.df_estimate:.3f}"
        + (
            f" ± {1.96 * summary.df_stderr:.3f}"
            if np.isfinite(summary.df_stderr)
            else ""
        )
    )
    summary_path = (
        Path(outdir) / f"fractal_GRID_{rows}x{cols}" / "lerw_fractal_summary.json"
    )
    if summary_path.exists():
        rprint(f"[green]✓[/green] Summary saved to {summary_path}")


if __name__ == "__main__":
    app()
