"""Graph construction and utility functions for Wilson algorithm experiments."""

import networkx as nx


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
    seed: int | None = None,
) -> nx.Graph:
    """
    Build a graph from standard families.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', or 'GRID'.
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
    else:
        raise ValueError(f"Unsupported graph kind: {kind}")
    return relabel_consecutive(G)


def graph_id(
    kind: str,
    n: int | None = None,
    p: float | None = None,
    m: int | None = None,
    d: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
) -> str:
    """
    Generate a human-readable graph identifier string.

    Parameters
    ----------
    kind : str
        Graph family: 'ER', 'BA', 'REG', or 'GRID'.
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
    return kind

