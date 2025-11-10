"""Spectral density reconstruction via Stieltjes transform inversion."""

import math

import numpy as np
from scipy.optimize import lsq_linear


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

    Raises
    ------
    AssertionError
        If lam_grid is not 1D or has fewer than 2 points.

    Examples
    --------
    >>> lam_grid = np.linspace(0, 10, 11)
    >>> widths = compute_bin_widths(lam_grid)
    >>> len(widths) == len(lam_grid)
    True
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

    Notes
    -----
    The second difference operator approximates the second derivative,
    used for L2 smoothness regularization in density reconstruction.

    Examples
    --------
    >>> D = build_second_difference_matrix(5)
    >>> D.shape
    (3, 5)
    >>> D[0]  # First row: [1, -2, 1, 0, 0]
    array([ 1., -2.,  1.,  0.,  0.])
    """
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

    References
    ----------
    .. [1] Tikhonov regularization for ill-posed problems
    .. [2] Total variation denoising via IRLS

    Examples
    --------
    >>> # Simplified example (see full CLI for realistic usage)
    >>> q = np.logspace(-1, 1, 20)
    >>> g = np.random.rand(20)  # Mock data
    >>> lam_grid = np.linspace(0, 10, 100)
    >>> delta_lam = compute_bin_widths(lam_grid)
    >>> rho, rrmse = invert_stieltjes_density(
    ...     q, g, None, lam_grid, delta_lam, 100, 1.0, 1e4, 1e-2, 0.0, 8, 1e-6
    ... )
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

