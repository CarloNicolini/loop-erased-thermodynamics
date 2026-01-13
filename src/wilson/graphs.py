"""Graph construction and utility functions for Wilson algorithm experiments."""

import numpy as np
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph


def relabel_consecutive(G: nx.Graph) -> nx.Graph:
    """
    Relabel graph nodes to consecutive integers 0..n-1.

    Parameters
    ----------
    G : nx.Graph
        Input graph with arbitrary node labels.

    Returns
    -------
    nx.Graph
        Graph with nodes relabeled to 0..n-1.
    """
    mapping = {u: i for i, u in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)


def build_graph(
    kind: str,
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
) -> nx.Graph:
    """
    Build a graph from standard families.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', 'GRID', or 'SBM'.
    n : int, optional
        Number of nodes (for ER, BA, REG).
    p : float, optional
        Edge probability (for ER).
    m : int, optional
        Number of edges per new node (for BA).
    d : int, optional
        Degree (for REG).
    rows : int, optional
        Number of rows (for GRID).
    cols : int, optional
        Number of columns (for GRID).
    blocks : int, optional
        Number of equally sized communities (for SBM).
    block_size : int, optional
        Community size (for SBM).
    p_in : float, optional
        Intra-community edge probability (for SBM).
    p_out : float, optional
        Inter-community edge probability (for SBM).
    tau1 : float, optional
        Degree exponent for LFR benchmark graphs.
    tau2 : float, optional
        Community-size exponent for LFR benchmark graphs.
    mu : float, optional
        Inter-community mixing parameter for LFR benchmark graphs.
    average_degree : float, optional
        Desired average degree for LFR benchmark graphs.
    min_degree : int, optional
        Minimum degree for LFR benchmark graphs.
    max_degree : int, optional
        Maximum degree for LFR benchmark graphs.
    min_community : int, optional
        Minimum community size for LFR benchmark graphs.
    max_community : int, optional
        Maximum community size for LFR benchmark graphs.
    tol : float, optional
        Numerical tolerance passed to the LFR generator.
    max_iters : int, optional
        Maximum iterations for community/degree generation in LFR graphs.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Generated graph with consecutive integer node labels.

    Raises
    ------
    ValueError
        If required parameters are missing or graph kind is unsupported.

    Examples
    --------
    >>> G = build_graph("ER", n=100, p=0.05, seed=42)
    >>> G = build_graph("GRID", rows=10, cols=10)
    """
    kind = kind.upper()
    if kind == "ER":
        if n is None or p is None:
            raise ValueError("ER graph requires --n and --p")
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif kind == "BA":
        if n is None or m is None:
            raise ValueError("BA graph requires --n and --m")
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "REG":
        if n is None or d is None:
            raise ValueError("REG graph requires --n and --d")
        G = nx.random_regular_graph(d, n, seed=seed)
    elif kind == "GRID":
        if rows is None or cols is None:
            raise ValueError("GRID graph requires --rows and --cols")
        G = nx.grid_2d_graph(rows, cols)
    elif kind == "SBM":
        if blocks is None or block_size is None or p_in is None or p_out is None:
            raise ValueError(
                "SBM graph requires --blocks, --block-size, --p-in, and --p-out"
            )
        if blocks <= 0 or block_size <= 0:
            raise ValueError("SBM graph requires positive --blocks and --block-size")
        if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
            raise ValueError("SBM probabilities must lie in [0, 1].")
        sizes = [block_size] * blocks
        G = nx.random_partition_graph(sizes, p_in, p_out, seed=seed)
    elif kind == "LFR":
        if n is None:
            raise ValueError("LFR graph requires --n")
        if tau1 is None or tau2 is None or mu is None:
            raise ValueError("LFR graph requires --tau1, --tau2, and --mu")
        if (average_degree is None) == (min_degree is None):
            raise ValueError(
                "Specify exactly one of --lfr-average-degree or --lfr-min-degree"
            )
        if min_degree is not None and min_degree < 0:
            raise ValueError("LFR minimum degree must be non-negative.")
        if average_degree is not None and average_degree <= 0:
            raise ValueError("LFR average degree must be positive.")
        params: dict[str, float | int | None] = {
            "n": n,
            "tau1": tau1,
            "tau2": tau2,
            "mu": mu,
            "average_degree": average_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "min_community": min_community,
            "max_community": max_community,
            "tol": tol,
            "max_iters": max_iters,
            "seed": seed,
        }
        # Remove None entries to keep defaults from NetworkX
        kwargs = {k: v for k, v in params.items() if v is not None}
        base_graph = LFR_benchmark_graph(**kwargs)
        mapping = {node: idx for idx, node in enumerate(base_graph.nodes())}
        relabeled = nx.relabel_nodes(base_graph, mapping, copy=True)
        for original, idx in mapping.items():
            community = base_graph.nodes[original].get("community")
            if community is not None:
                relabeled.nodes[idx]["community"] = {mapping[v] for v in community}
        G = relabeled
    else:
        raise ValueError(f"Unsupported graph kind: {kind}")
    if kind == "LFR":
        return G
    return relabel_consecutive(G)


def graph_id(
    kind: str,
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
    tol: float | None = None,
    max_iters: int | None = None,
) -> str:
    """
    Generate a human-readable graph identifier string.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', 'GRID', or 'SBM'.
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

    Returns
    -------
    str
        Graph identifier string like 'ER_n100_p0.050' or 'GRID_10x10'.

    Examples
    --------
    >>> graph_id("ER", n=100, p=0.05)
    'ER_n100_p0.050'
    >>> graph_id("GRID", rows=10, cols=10)
    'GRID_10x10'
    """
    kind = kind.upper()
    if kind == "ER":
        return f"ER_n{n}_p{p:.3f}"
    if kind == "BA":
        return f"BA_n{n}_m{m}"
    if kind == "REG":
        return f"REG_n{n}_d{d}"
    if kind == "GRID":
        return f"GRID_{rows}x{cols}"
    if kind == "SBM":
        total = (block_size or 0) * (blocks or 0)
        return f"SBM_n{total}_k{blocks}_m{block_size}_pin{p_in:.3f}_pout{p_out:.3f}"
    if kind == "LFR":
        pieces = [
            f"LFR_n{n}",
            f"tau1{tau1:.2f}" if tau1 is not None else None,
            f"tau2{tau2:.2f}" if tau2 is not None else None,
            f"mu{mu:.2f}" if mu is not None else None,
        ]
        if average_degree is not None:
            pieces.append(f"kavg{average_degree:.1f}")
        elif min_degree is not None:
            pieces.append(f"kmin{min_degree}")
        if max_degree is not None:
            pieces.append(f"kmax{max_degree}")
        if min_community is not None:
            pieces.append(f"cmin{min_community}")
        if max_community is not None:
            pieces.append(f"cmax{max_community}")
        pieces = [p for p in pieces if p is not None]
        return "_".join(pieces)
    return kind


def degree_preserving_randomization(
    G: nx.Graph,
    *,
    n_swaps: int,
    seed: int | None = None,
) -> nx.Graph:
    """
    Generate a degree-preserving randomization via double-edge swaps.

    Parameters
    ----------
    G : nx.Graph
        Input simple graph.
    n_swaps : int
        Number of successful double-edge swaps to perform.
    seed : int, optional
        Random seed.

    Returns
    -------
    nx.Graph
        Degree-preserving randomized graph with relabeled consecutive nodes.
    """

    if n_swaps < 0:
        raise ValueError("n_swaps must be non-negative.")
    if G.number_of_edges() == 0 or n_swaps == 0:
        return relabel_consecutive(G)

    rng = np.random.default_rng(seed)
    H = G.copy()
    try:
        nx.double_edge_swap(
            H,
            nswap=n_swaps,
            max_tries=max(5 * n_swaps, 1),
            seed=rng,
        )
    except nx.NetworkXError as exc:
        raise RuntimeError("Degree-preserving randomization failed.") from exc
    return relabel_consecutive(H)
