"""Spectral analysis functions for graph Laplacians and heat kernels."""

import networkx as nx
import numpy as np
from numpy.polynomial.laguerre import laggauss


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

    Notes
    -----
    For large graphs (n > 1000), consider using sparse eigenvalue solvers
    from scipy.sparse.linalg instead of this dense implementation.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(4)
    >>> eigvals = laplacian_eigenvalues(G)
    >>> len(eigvals)
    4
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

    Notes
    -----
    This is the expected number of roots in a q-forest sampled via Wilson's algorithm.

    Examples
    --------
    >>> lambdas = np.array([0.0, 1.0, 2.0, 3.0])
    >>> s = compute_s_true(q=0.5, lambdas=lambdas)
    """
    return float(np.sum(q / (q + lambdas)))


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

    Notes
    -----
    The heat trace is the partition function for a quantum particle on the graph.

    Examples
    --------
    >>> beta = np.array([0.1, 1.0, 10.0])
    >>> lambdas = np.array([0.0, 1.0, 2.0])
    >>> Z = heat_trace_from_spectrum(beta, lambdas)
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

    Examples
    --------
    >>> q = np.array([0.1, 1.0, 10.0])
    >>> lambdas = np.array([0.0, 1.0, 2.0])
    >>> s = s_from_spectrum(q, lambdas)
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

    Notes
    -----
    This validates the connection between the forest partition function
    and the heat kernel via Laplace transform.

    Examples
    --------
    >>> q = np.array([0.5, 1.0, 2.0])
    >>> lambdas = np.array([0.0, 1.0, 2.0])
    >>> s = s_from_Z_via_gauss_laguerre(q, lambdas, n_nodes=64)
    """
    t_nodes, w_nodes = laggauss(n_nodes)
    s_vals = []
    for qi in q:
        beta = 1.0 / float(qi)
        Z_vals = np.exp(-np.outer(beta * t_nodes, lambdas)).sum(axis=1)
        s_vals.append(float(w_nodes @ Z_vals))
    return np.asarray(s_vals)

