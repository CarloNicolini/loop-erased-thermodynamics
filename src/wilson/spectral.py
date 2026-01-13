"""Spectral analysis functions for graph Laplacians and heat kernels."""

import networkx as nx
import numpy as np


def z_from_graph(beta: float, G: nx.Graph) -> float:
    """
    Compute the partition function Z(β) from the Laplacian eigenvalues of a graph.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    G : nx.Graph
        Input graph.

    Returns
    -------
    float
        Partition function Z(β) = sum_i exp(-β λ_i).

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(3)
    >>> Z = partition_function_from_graph(beta=1.0, G=G)
    """
    return z_from_spectrum(beta, nx.laplacian_spectrum(G))


def z_from_spectrum(
    beta: float | np.ndarray, lambdas: np.ndarray
) -> float | np.ndarray:
    """
    Compute the partition function Z(β) from Laplacian eigenvalues.

    Parameters
    ----------
    beta : float or np.ndarray
        Inverse temperature(s). If array, returns array of Z values.
    lambdas : np.ndarray
        Array of Laplacian eigenvalues.

    Returns
    -------
    float or np.ndarray
        Partition function Z(β) = sum_i exp(-β λ_i).
        Returns float if beta is scalar, array if beta is array.

    Examples
    --------
    >>> lambdas = np.array([0.0, 1.0, 2.0])
    >>> Z = z_from_spectrum(beta=1.0, lambdas=lambdas)
    >>> Z_array = z_from_spectrum(beta=np.array([0.5, 1.0, 2.0]), lambdas=lambdas)
    """
    beta = np.asarray(beta, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    is_scalar = beta.ndim == 0
    beta = beta.reshape(-1) if not is_scalar else beta

    # Compute Z(β) for each beta: Z = sum_i exp(-β λ_i)
    # Shape: (n_beta, n_lambdas) -> (n_beta,)
    Z = np.sum(np.exp(-np.outer(beta, lambdas)), axis=1)

    return float(Z.item()) if is_scalar else Z


def spectral_entropy(beta: float, G: nx.Graph) -> float:
    """
    Compute the spectral entropy of a graph based on its Laplacian eigenvalues.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    float
        Spectral entropy of the graph.

    Notes
    -----
    The spectral entropy is defined as:
        H = -sum(p_i * log(p_i))
    where p_i=e^{-beta * lambda_i}/Z
    """
    lambdas = nx.laplacian_spectrum(G)
    n = len(lambdas)
    if n == 0:
        return 0.0
    Z = z_from_spectrum(beta, lambdas)
    p = np.exp(-beta * lambdas) / Z
    entropy = -np.sum(p * np.log(p + 1e-15))  # Add small value to avoid log(0)
    return float(entropy)


def spectral_entropy_from_spectrum(beta: float, lambdas: np.ndarray) -> float:
    """
    Compute the spectral entropy given precomputed Laplacian eigenvalues.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    lambdas : np.ndarray
        Array of Laplacian eigenvalues.

    Returns
    -------
    float
        Spectral entropy H = -sum(p_i log p_i) where p_i = exp(-beta lambda_i)/Z.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    n = lambdas.size
    if n == 0:
        return 0.0
    Z = z_from_spectrum(beta, lambdas)
    p = np.exp(-beta * lambdas) / Z
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(entropy)


def z_from_density(
    beta: np.ndarray,
    lam_grid: np.ndarray,
    rho: np.ndarray,
    delta_lam: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Compute Z(β) from a discretized spectral density.

    Parameters
    ----------
    beta : np.ndarray
        Inverse temperature grid.
    lam_grid : np.ndarray
        Lambda grid points (eigenvalue support).
    rho : np.ndarray
        Reconstructed normalized spectral density on ``lam_grid``.
    delta_lam : np.ndarray
        Bin widths associated with ``lam_grid``.
    n_nodes : int
        Number of graph nodes (overall multiplicative factor).

    Returns
    -------
    np.ndarray
        Approximated partition function

        # Z(β) ≈ n × Σₖ [ ρₖ · exp(−β λₖ) · Δλₖ ]
    """
    beta = np.asarray(beta, dtype=float)
    lam_grid = np.asarray(lam_grid, dtype=float)
    rho = np.asarray(rho, dtype=float)
    delta_lam = np.asarray(delta_lam, dtype=float)
    kernel = np.exp(-np.outer(beta, lam_grid))  # shape (n_beta, M)
    return n_nodes * (kernel * (rho * delta_lam)).sum(axis=1)


def s_from_spectrum(q: float, lambdas: np.ndarray) -> float:
    """
    Compute s(q) from eigenvalues.

    Parameters
    ----------
    q : float
        q coefficient.
    lambdas : np.ndarray
        Laplacian eigenvalues.

    Returns
    -------
    np.ndarray
        s(q) = sum_i q / (q + λ_i).
    """
    return (q / (q + lambdas)).sum()


def g_from_spectrum(q, lambdas: np.ndarray) -> float:
    """
    Compute the quantity to be inverted in numerical laplace inversion
    Returns
    -------
    float
        g(q) = sum_i 1/(q+ lambda_i)
    """
    return s_from_spectrum(q, lambdas) / q
