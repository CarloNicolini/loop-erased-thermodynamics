"""Spectral density reconstruction via Stieltjes transform inversion.

This module provides low-level building blocks for inverting the Stieltjes
transform g(q) = s(q)/q into a spectral density ρ(λ), along with
higher-level helpers that integrate with Wilson sampling.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import lsq_linear

from wilson.spectral import z_from_density, z_from_spectrum
from wilson.wilson import Wilson


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


def gaver_stehfest_inversion_from_samples(
    beta: float,
    g_values: np.ndarray,
    q_abscissae: np.ndarray,
    M: int = 16,
    g_se: np.ndarray | None = None,
    cov: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute Z(beta) from samples of F(q) = s(q)/q using the Gaver-Stehfest algorithm.

    Parameters
    ----------
    beta : float
        Inverse temperature at which to compute Z(β).
    g_values : np.ndarray
        Array of F(q) = s(q)/q evaluated at the abscissae p_k = k*ln(2)/beta
        (length must be M).
    q_abscissae : np.ndarray
        The p_k abscissae corresponding to g_values (length M).
    M : int
        Even number of Stehfest terms (default 16).

    Returns
    -------
    tuple
        (Z_mean, Z_se) where Z_mean is the estimated partition function and
        Z_se is NaN (placeholder) since we do not have per-term variances here.

    Notes
    -----
    This routine expects g_values to already be evaluated at the correct
    abscissae p_k = k * ln(2) / beta. If g_values were obtained by Monte Carlo
    sampling they may carry standard errors; a separate wrapper can propagate
    those uncertainties if provided.
    """
    if M % 2 != 0:
        raise ValueError("M must be even")

    if g_values.size != M or q_abscissae.size != M:
        raise ValueError("g_values and q_abscissae must have length M")
    ln2 = math.log(2.0)

    def _stehfest_weights(M: int) -> np.ndarray:
        """
        Compute Stehfest (Salzer) weights for Gaver-Stehfest inversion.

        Parameters
        ----------
        M : int
            Even positive integer giving the degree (number of terms).

        Returns
        -------
        np.ndarray
            Array of length M with Stehfest weights V_k (k=1..M).

        Notes
        -----
        Implements the formula used by mpmath and standard references:

            V_k = (-1)^{k + M/2} * \n
                sum_{j=\lfloor (k+1)/2 \rfloor}^{min(k,M/2)}
                    j^{M/2} (2j)! / ((M/2 - j)! j! (j-1)! (k-j)! (2j-k)!)

        The algorithm requires sufficient numeric precision for large M.
        """
        if M % 2 != 0 or M <= 0:
            raise ValueError("M must be a positive even integer")

        M2 = M // 2
        V = np.zeros(M, dtype=float)
        for k in range(1, M + 1):
            s = 0.0
            jmin = (k + 1) // 2
            jmax = min(k, M2)
            for j in range(jmin, jmax + 1):
                # compute combinatorial term
                num = (j**M2) * math.factorial(2 * j)
                den = (
                    math.factorial(M2 - j)
                    * math.factorial(j)
                    * math.factorial(j - 1)
                    * math.factorial(k - j)
                    * math.factorial(2 * j - k)
                )
                s += num / den
            V[k - 1] = ((-1) ** (k + M2)) * s
        return V

    V = _stehfest_weights(M)
    # compute Z(beta) as approximation to inverse Laplace transform of g(q)=s(q)/q
    Z = (ln2 / beta) * float(np.dot(V, g_values))

    # Propagate uncertainty if provided. Support full covariance or per-abscissa se
    Z_se = float(np.nan)
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (M, M):
            raise ValueError("cov must have shape (M, M)")
        varZ = (ln2 / float(beta)) ** 2 * float(np.dot(V, cov @ V))
        Z_se = float(math.sqrt(max(varZ, 0.0)))
    elif g_se is not None:
        g_se = np.asarray(g_se, dtype=float)
        if g_se.size != M:
            raise ValueError("g_se must have length M")
        varZ = (ln2 / float(beta)) ** 2 * float(np.dot((V**2), (g_se**2)))
        Z_se = float(math.sqrt(max(varZ, 0.0)))

    return Z, Z_se


def inversion_from_wilson(
    beta: float,
    g_func: Callable,
    M: int = 16,
    method="cohen",
) -> float:
    """
    Compute Z(beta) from g(q) = s(q)/q using mpmath's inverse Laplace transform.

    Parameters
    ----------
    beta : float
        Inverse temperature at which to compute Z(β).
    g_func : callable
        Function g(q) = s(q)/q to be inverted.
    method: str
        Inversion method to use (default "cohen").
        Can be "cohen", "talbot", "dehoog", "stehfest".
    M : int
        Number of terms for the numerical inversion (default 16).

    Returns
    -------
    float
        Estimated partition function Z(β).

    Notes
    -----
    This function uses mpmath's `inverselaplace` to perform the inversion.
    The function g_func should be defined for positive real q values.
    """

    from mpmath import mp

    # mpmath's inversion routines pass complex abscissae to the provided
    # function. The Wilson sampler cannot handle complex q (it builds
    # random-walk probabilities from weights), which leads to complex-valued
    # edge weights and subsequent failures. To avoid that, wrap `g_func`
    # so that any complex abscissa is mapped to its real part before
    # calling the sampler. This is a pragmatic compromise that keeps the
    # Cohen inversion machinery while avoiding complex arithmetic in the
    # sampler.
    def _g_wrapped(p):
        return g_func(mp.re(p))

    Z_mpmath = mp.invertlaplace(_g_wrapped, beta, method="cohen")
    return float(Z_mpmath)


# ---------------------------------------------------------------------------
# High-level reconstruction from Wilson Monte Carlo data
# ---------------------------------------------------------------------------


@dataclass
class SpectrumReconstructionResult:
    """Container for spectrum and partition-function reconstruction.

    Attributes
    ----------
    G : nx.Graph
        Input graph.
    q : np.ndarray
        q-grid used for Wilson sampling.
    beta : np.ndarray
        β-grid used for Z(β) reconstruction.
    lam_grid : np.ndarray
        Discretization grid for eigenvalues.
    delta_lam : np.ndarray
        Bin widths associated with ``lam_grid``.
    n_nodes : int
        Number of graph nodes.
    rho_hat : np.ndarray
        Reconstructed spectral density on ``lam_grid``.
    Z_hat : np.ndarray
        Reconstructed partition function on ``beta``.
    Z_ci_lower, Z_ci_upper : np.ndarray or None
        Percentile-based uncertainty bands on Z(β) from bootstrap.
    Z_samples : np.ndarray or None
        Raw bootstrap samples of Z(β), shape (n_bootstrap, n_beta).
    rrmse : float
        Relative RMSE between reconstructed g(q) and Monte Carlo estimates.
    g_hat : np.ndarray
        Monte Carlo estimates of g(q).
    g_se : np.ndarray or None
        Standard errors associated with g_hat.
    Z_true : np.ndarray or None
        Spectral ground-truth Z(β) when eigenvalues are available.
    """

    G: nx.Graph
    q: np.ndarray
    beta: np.ndarray
    lam_grid: np.ndarray
    delta_lam: np.ndarray
    n_nodes: int
    rho_hat: np.ndarray
    Z_hat: np.ndarray
    Z_ci_lower: Optional[np.ndarray]
    Z_ci_upper: Optional[np.ndarray]
    Z_samples: Optional[np.ndarray]
    rrmse: float
    g_hat: np.ndarray
    g_se: Optional[np.ndarray]
    Z_true: Optional[np.ndarray] = None


def estimate_g_from_wilson(
    G: nx.Graph,
    q: np.ndarray,
    n_samples_per_q: int = 64,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate s(q) and g(q) = s(q)/q (and their std) via Wilson sampling.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    q : np.ndarray
        1D array of q values.
    n_samples_per_q : int, default 64
        Number of Wilson samples per q.
    random_state : int or None
        Seed for reproducibility; different seeds are used per q.

    Returns
    -------
    s_hat : np.ndarray
        Mean root counts s(q) over samples.
    s_std : np.ndarray
        Standard deviation of root counts over samples.
    g_hat : np.ndarray
        Mean g(q) = s(q)/q.
    g_std : np.ndarray
        Standard deviation of g(q).
    """

    q = np.asarray(q, dtype=float)
    m = q.size
    s_hat = np.empty(m, dtype=float)
    s_std = np.empty(m, dtype=float)
    g_hat = np.empty(m, dtype=float)
    g_std = np.empty(m, dtype=float)

    rng = np.random.default_rng(random_state)
    for i, qi in enumerate(q):
        # Use a different seed per q to decorrelate samples
        seed_i = int(rng.integers(0, 2**31 - 1))
        model = Wilson(q=qi, random_state=seed_i).fit(G, n_samples=n_samples_per_q)
        s_hat[i] = float(model.s_)
        s_std[i] = float(model.s_std_)
        g_hat[i] = float(model.g_)
        g_std[i] = float(model.g_std_)
    return s_hat, s_std, g_hat, g_std


def bootstrap_Z_from_wilson(
    G: nx.Graph,
    q: np.ndarray,
    lam_grid: np.ndarray,
    delta_lam: np.ndarray,
    beta: np.ndarray,
    n_nodes: int,
    mass_target: float,
    gamma_mass: float,
    tau_smooth: float,
    tau_tv: float,
    tv_iters: int,
    tv_eps: float,
    n_samples_per_q: int,
    n_bootstrap: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Bootstrap distribution of Z(β) by resampling g(q) via Wilson.

    Each bootstrap replicate re-runs Wilson(q).fit(G) for all q and then
    performs a full density inversion.
    """

    rng = np.random.default_rng(random_state)
    q = np.asarray(q, dtype=float)
    lam_grid = np.asarray(lam_grid, dtype=float)
    delta_lam = np.asarray(delta_lam, dtype=float)
    beta = np.asarray(beta, dtype=float)

    Z_samples = np.empty((n_bootstrap, beta.size), dtype=float)

    for b in range(n_bootstrap):
        # Fresh Monte Carlo data via Wilson for this replicate
        seed_b = int(rng.integers(0, 2**31 - 1))
        _, _, g_b, g_se_b = estimate_g_from_wilson(
            G, q, n_samples_per_q=n_samples_per_q, random_state=seed_b
        )
        rho_b, _ = invert_stieltjes_density(
            q=q,
            g=g_b,
            g_se=g_se_b,
            lam_grid=lam_grid,
            delta_lam=delta_lam,
            n_nodes=n_nodes,
            mass_target=mass_target,
            gamma_mass=gamma_mass,
            tau_smooth=tau_smooth,
            tau_tv=tau_tv,
            tv_iters=tv_iters,
            tv_eps=tv_eps,
        )
        Z_samples[b, :] = z_from_density(
            beta=beta,
            lam_grid=lam_grid,
            rho=rho_b,
            delta_lam=delta_lam,
            n_nodes=n_nodes,
        )

    return Z_samples


def reconstruct_spectrum_from_wilson_graph(
    G: nx.Graph,
    q: np.ndarray,
    beta: Optional[np.ndarray] = None,
    lam_grid: Optional[np.ndarray] = None,
    n_samples_per_q: int = 256,
    n_bootstrap: int = 50,
    mass_target: float = 1.0,
    gamma_mass: float = 1e4,
    tau_smooth: float = 1e-2,
    tau_tv: float = 0.0,
    tv_iters: int = 8,
    tv_eps: float = 1e-6,
    random_state: Optional[int] = None,
) -> SpectrumReconstructionResult:
    """Reconstruct spectral density and Z(β) from Wilson Monte Carlo data.

    This high-level driver wraps Wilson sampling, Stieltjes inversion and
    forward Laplace transform into a single function suitable for use
    from the CLI or notebooks.
    """

    q = np.asarray(q, dtype=float)
    if beta is None:
        beta = np.logspace(-2, 2, 50)
    else:
        beta = np.asarray(beta, dtype=float)

    # λ-grid: use exact Laplacian spectrum by default
    if lam_grid is None:
        lam_grid = nx.laplacian_spectrum(G)

    delta_lam = compute_bin_widths(lam_grid)
    n_nodes = G.number_of_nodes()

    # Central Wilson estimate of g(q) and g_std(q)
    _, _, g_hat, g_se = estimate_g_from_wilson(
        G, q, n_samples_per_q=n_samples_per_q, random_state=random_state
    )

    # Invert Stieltjes to get ρ(λ)
    rho_hat, rrmse = invert_stieltjes_density(
        q=q,
        g=g_hat,
        g_se=g_se,
        lam_grid=lam_grid,
        delta_lam=delta_lam,
        n_nodes=n_nodes,
        mass_target=mass_target,
        gamma_mass=gamma_mass,
        tau_smooth=tau_smooth,
        tau_tv=tau_tv,
        tv_iters=tv_iters,
        tv_eps=tv_eps,
    )

    # Compute Z(β) from reconstructed density
    Z_hat = z_from_density(
        beta=beta,
        lam_grid=lam_grid,
        rho=rho_hat,
        delta_lam=delta_lam,
        n_nodes=n_nodes,
    )

    # Ground truth Z(β) if spectrum is known (for diagnostics)
    try:
        lambdas_exact = np.array(nx.laplacian_spectrum(G), dtype=float)
        Z_true = z_from_spectrum(beta, lambdas_exact)
    except Exception:
        Z_true = None

    Z_ci_lower: Optional[np.ndarray] = None
    Z_ci_upper: Optional[np.ndarray] = None
    Z_samples: Optional[np.ndarray] = None
    if n_bootstrap > 0:
        Z_samples = bootstrap_Z_from_wilson(
            G=G,
            q=q,
            lam_grid=lam_grid,
            delta_lam=delta_lam,
            beta=beta,
            n_nodes=n_nodes,
            mass_target=mass_target,
            gamma_mass=gamma_mass,
            tau_smooth=tau_smooth,
            tau_tv=tau_tv,
            tv_iters=tv_iters,
            tv_eps=tv_eps,
            n_samples_per_q=n_samples_per_q,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        Z_ci_lower = np.percentile(Z_samples, 16, axis=0)
        Z_ci_upper = np.percentile(Z_samples, 84, axis=0)

    return SpectrumReconstructionResult(
        G=G,
        q=q,
        beta=beta,
        lam_grid=lam_grid,
        delta_lam=delta_lam,
        n_nodes=n_nodes,
        rho_hat=rho_hat,
        Z_hat=Z_hat,
        Z_ci_lower=Z_ci_lower,
        Z_ci_upper=Z_ci_upper,
        Z_samples=Z_samples,
        rrmse=rrmse,
        g_hat=g_hat,
        g_se=g_se,
        Z_true=Z_true,
    )
