"""Fractal-dimension experiment utilities for Wilson loop-erased walks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional

from wilson.plotting import save_plot_both_formats


@dataclass(slots=True)
class FractalSummary:
    """Container for LERW scaling statistics."""

    rows: int
    cols: int
    seed: int
    samples: int
    effective_samples: int
    radius_range: tuple[float, float]
    span_range: Optional[tuple[float, float]]
    df_estimate: float
    df_stderr: float
    df_ci95: tuple[float, float]
    intercept: float
    max_radius_range: tuple[float, float]
    log_bins: int
    bin_edges: list[float] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "grid": {"rows": self.rows, "cols": self.cols},
            "seed": self.seed,
            "samples": self.samples,
            "effective_samples": self.effective_samples,
            "radius_range": list(self.radius_range),
            "span_range": list(self.span_range)
            if self.span_range is not None
            else None,
            "df_estimate": self.df_estimate,
            "df_stderr": self.df_stderr,
            "df_ci95": list(self.df_ci95),
            "intercept": self.intercept,
            "max_radius_range": list(self.max_radius_range),
            "log_bins": self.log_bins,
            "bin_edges": self.bin_edges,
        }


def _loop_erased_random_walk(
    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    start: tuple[int, int],
    boundary: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Wilson-style loop-erased random walk until the wired boundary."""

    path: list[tuple[int, int]] = [start]
    visited: dict[tuple[int, int], int] = {start: 0}
    current = start

    while current not in boundary:
        neighbors = adjacency[current]
        nxt = neighbors[int(rng.integers(len(neighbors)))]
        if nxt in visited:
            loop_start = visited[nxt]
            for node in path[loop_start + 1 :]:
                visited.pop(node, None)
            path = path[: loop_start + 1]
        else:
            path.append(nxt)
            visited[nxt] = len(path) - 1
        current = path[-1]
    return path


def _fit_scaling(span: np.ndarray, length: np.ndarray) -> tuple[float, float, float]:
    """Fit log length vs log span; return slope, intercept, stderr."""

    x = np.log(span)
    y = np.log(length)
    X = np.column_stack((np.ones_like(x), x))
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(coeffs[0])
    slope = float(coeffs[1])

    residuals = y - X @ coeffs
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return slope, intercept, float("nan")

    sigma2 = float(residuals @ residuals / dof)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        slope_stderr = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        slope_stderr = float("nan")
    return slope, intercept, slope_stderr


def _bin_statistics(
    df: pd.DataFrame,
    x_col: str,
    log_bins: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    values = df[x_col].to_numpy()
    val_min = float(values.min())
    val_max = float(values.max())
    if val_max <= val_min:
        return pd.DataFrame(), None

    edges = np.logspace(np.log10(val_min), np.log10(val_max), log_bins + 1)
    binned = (
        df.assign(
            span_bin=pd.cut(
                df[x_col],
                edges,
                include_lowest=True,
                duplicates="drop",
            )
        )
        .groupby("span_bin", observed=True)
        .agg(
            span_mean=(x_col, "mean"),
            length_mean=("length_steps", "mean"),
            count=("length_steps", "size"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    return binned, edges


def run_fractal_dimension_experiment(
    rows: int,
    cols: int,
    samples: int,
    seed: int,
    outdir: str,
    log_bins: int,
    max_scatter: int,
) -> FractalSummary:
    """Execute LERW fractal-dimension estimation and save artifacts."""

    if rows < 3 or cols < 3:
        raise ValueError("Grid must be at least 3×3.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if log_bins < 1:
        raise ValueError("log_bins must be ≥ 1.")

    out_path = Path(outdir) / f"fractal_GRID_{rows}x{cols}"
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    grid = nx.grid_2d_graph(rows, cols)
    adjacency = {node: tuple(grid.neighbors(node)) for node in grid.nodes()}
    interior = [
        node
        for node in grid.nodes()
        if 0 < node[0] < rows - 1 and 0 < node[1] < cols - 1
    ]
    if not interior:
        raise RuntimeError("Grid has no interior nodes to start walks.")
    boundary = set(grid.nodes()) - set(interior)

    records: list[dict[str, float | int]] = []
    for idx in range(samples):
        start = interior[int(rng.integers(len(interior)))]
        path = _loop_erased_random_walk(adjacency, start, boundary, rng)
        coords = np.asarray(path, dtype=float)
        steps = len(coords) - 1
        span = float(np.linalg.norm(coords[-1] - coords[0]))
        radii = np.linalg.norm(coords - coords[0], axis=1)
        records.append(
            {
                "sample": idx,
                "length_steps": float(steps),
                "euclidean_span": span,
                "max_radius": float(radii.max()),
                "num_vertices": int(len(coords)),
                "start_row": int(coords[0][0]),
                "start_col": int(coords[0][1]),
                "end_row": int(coords[-1][0]),
                "end_col": int(coords[-1][1]),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path / "lerw_paths.csv", index=False)

    valid = df[(df["euclidean_span"] > 0.0) & (df["length_steps"] > 0.0)]
    if valid.shape[0] < 3:
        raise RuntimeError("Need at least three paths with positive span and length.")

    radius_valid = valid[valid["max_radius"] > 0.0]
    if radius_valid.shape[0] < 3:
        raise RuntimeError("Need at least three paths with positive radius.")

    radius_array = radius_valid["max_radius"].to_numpy()
    length_array_radius = radius_valid["length_steps"].to_numpy()
    slope, intercept, slope_stderr = _fit_scaling(radius_array, length_array_radius)

    if np.isfinite(slope_stderr):
        ci95 = (float(slope - 1.96 * slope_stderr), float(slope + 1.96 * slope_stderr))
    else:
        ci95 = (float("nan"), float("nan"))

    binned, edges = _bin_statistics(radius_valid, "max_radius", log_bins)
    if not binned.empty:
        binned.to_csv(out_path / "lerw_binned.csv", index=False)

    span_array = valid["euclidean_span"].to_numpy()
    summary = FractalSummary(
        rows=rows,
        cols=cols,
        seed=seed,
        samples=int(df.shape[0]),
        effective_samples=int(valid.shape[0]),
        radius_range=(float(radius_array.min()), float(radius_array.max())),
        span_range=(
            (float(span_array.min()), float(span_array.max()))
            if span_array.size > 0
            else None
        ),
        df_estimate=float(slope),
        df_stderr=float(slope_stderr),
        df_ci95=ci95,
        intercept=float(intercept),
        max_radius_range=(
            float(radius_valid["max_radius"].min()),
            float(radius_valid["max_radius"].max()),
        ),
        log_bins=log_bins,
        bin_edges=edges.tolist() if edges is not None else None,
    )

    with (out_path / "lerw_fractal_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    if max_scatter != 0:
        if max_scatter > 0 and valid.shape[0] > max_scatter:
            scatter_df = valid.sample(max_scatter, random_state=seed)
        else:
            scatter_df = valid
        ax.scatter(
            scatter_df["max_radius"],
            scatter_df["length_steps"],
            s=10,
            alpha=0.3,
            edgecolors="none",
            label=f"samples ({scatter_df.shape[0]:,})",
        )
    if not binned.empty:
        ax.plot(
            binned["span_mean"],
            binned["length_mean"],
            marker="o",
            color="C1",
            lw=2,
            label="bin mean",
        )

    radius_min, radius_max = summary.radius_range
    if radius_max > radius_min:
        span_fit = np.logspace(np.log10(radius_min), np.log10(radius_max), 256)
        ax.plot(
            span_fit,
            np.exp(intercept) * span_fit**slope,
            color="k",
            lw=2,
            ls="--",
            label=f"fit $d_f={slope:.3f}$",
        )
        ax.plot(
            span_fit,
            np.exp(intercept) * span_fit**1.25,
            color="red",
            lw=1.5,
            ls=":",
            label="$d_f=1.25$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ (max radius)")
    ax.set_ylabel(r"$\ell$ (loop-erased length)")
    ax.set_title(f"LERW scaling on {rows}×{cols} grid")
    ax.legend(frameon=True)

    save_plot_both_formats(fig, out_path, f"lerw_fractal_{rows}x{cols}")

    return summary
