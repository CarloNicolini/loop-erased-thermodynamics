"""Parallel execution helpers for Wilson sampling and validation."""

import math

import networkx as nx
import numpy as np

from wilson.spectral import compute_s_true
from wilson.wilson import Wilson


def sample_wilson_for_q(
    q: float,
    G: nx.Graph,
    samples_per_q: int,
    seed: int,
    lambdas: np.ndarray | None,
    compute_theory: bool,
) -> dict[str, float]:
    """
    Helper function for parallel Wilson sampling at a single q value.

    Parameters
    ----------
    q : float
        Temperature parameter.
    G : nx.Graph
        Input graph.
    samples_per_q : int
        Number of Monte Carlo samples.
    seed : int
        Random seed base.
    lambdas : np.ndarray or None
        Precomputed eigenvalues (for efficiency).
    compute_theory : bool
        Whether to compute theoretical s_true.

    Returns
    -------
    dict[str, float]
        Dictionary with q, s_mc, s_mc_se, g_mc, and optionally s_true, g_true.

    Notes
    -----
    This function is designed to be called in parallel via joblib.Parallel().
    Each call is independent and can be executed on a separate CPU core.

    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> import networkx as nx
    >>> G = nx.path_graph(10)
    >>> q_grid = [0.1, 1.0, 10.0]
    >>> results = Parallel(n_jobs=2)(
    ...     delayed(sample_wilson_for_q)(q, G, 100, 42, None, False)
    ...     for q in q_grid
    ... )
    """
    wil = Wilson(q=float(q), random_state=seed, eigenvalues=lambdas)
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
    if compute_theory and lambdas is not None:
        s_true = compute_s_true(float(q), lambdas)
        rec["s_true"] = s_true
        rec["g_true"] = s_true / float(q)
    return rec


def estimate_s_via_wilson(
    G: nx.Graph,
    q: float,
    n_samples: int,
    random_state: int,
    eigenvalues: np.ndarray | None = None,
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
    eigenvalues : np.ndarray or None
        Precomputed eigenvalues for efficiency.

    Returns
    -------
    mean : float
        Mean number of roots (estimate of s(q)).
    sem : float
        Standard error of the mean.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(10)
    >>> mean, sem = estimate_s_via_wilson(G, q=1.0, n_samples=100, random_state=42)
    """
    est = Wilson(q=q, random_state=random_state, eigenvalues=eigenvalues).fit(G)
    samples = est.transform(n_samples=n_samples, seed=random_state)
    root_counts = np.array([len(roots) for (_, roots) in samples], dtype=float)
    mean = float(root_counts.mean())
    sem = float(root_counts.std(ddof=1) / math.sqrt(max(1, n_samples)))
    return mean, sem


def estimate_s_for_validate(
    G: nx.Graph,
    q: float,
    n_samples: int,
    random_state: int,
    eigenvalues: np.ndarray | None,
) -> tuple[float, float]:
    """
    Helper for parallel Wilson estimation in validate command.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    q : float
        Temperature parameter.
    n_samples : int
        Number of MC samples.
    random_state : int
        Random seed.
    eigenvalues : np.ndarray or None
        Precomputed eigenvalues.

    Returns
    -------
    mean : float
        Mean s(q) estimate.
    sem : float
        Standard error.

    Notes
    -----
    This is a thin wrapper around estimate_s_via_wilson() designed
    for use with joblib.Parallel() in the validate command.
    """
    return estimate_s_via_wilson(G, q, n_samples, random_state, eigenvalues)

