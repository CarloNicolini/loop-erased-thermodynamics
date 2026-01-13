#!/usr/bin/env python3
"""Unified Wilson sampler implementations.

Provides two classes:
"""

from typing import Optional

import networkx as nx
import numpy as np
from wilson.spectral import s_from_spectrum


class Wilson:
    """scikit-learn-like Wilson transformer.

    Parameters
    ----------
    q : float or None
        If None, use classic unweighted Wilson sampler. If a float is provided,
        use the WilsonQ variant that attaches an extra root with weight q.
    random_state : int or None
        Seed for reproducibility.

    Notes
    -----
    After calling ``fit(G)`` the estimator exposes attributes:
    - ``G_``: the fitted graph
    - ``nv_``: number of nodes
    - ``s_``: (only when q is not None) Wilson (Monte Carlo) estimate of s(q) on G.
    - ``g_``: (only when q is not None) estimate of g(q) = s(q)/q on G.
    """

    def __init__(
        self,
        q: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.q = q
        self.random_state = random_state
        # attributes set in fit
        self.G_ = None
        self.nv_ = None
        self.s_ = None
        self.s_std_ = None
        self.g_ = None
        self.g_std_ = None

    def fit(self, X: nx.Graph, **fit_params):
        """Fit the estimator on graph X.

        If ``q`` is provided, computes Monte Carlo estimates of s(q) by
        drawing ``n_samples`` Wilson forests and storing the mean and
        standard error in ``self.s_`` and ``self.s_std_``.
        Also stores ``g_ = s_/q`` and ``g_std_``.

        Parameters
        ----------
        G : nx.Graph
            Input graph to fit.

        Keyword Arguments
        -----------------
        n_samples : int or None
            Number of Monte Carlo samples to draw for estimating s(q).
            If None, uses the estimator's ``self.n_samples`` value.

        Returns
        -------
        self
        """
        self.nv_ = X.number_of_nodes()
        # Keep original graph, but build an internal integer-labelled copy
        # for sampling routines (nodes 0..n-1). This avoids issues when the
        # input graph uses arbitrary labels (tuples, strings, ...).
        self.G_ = X
        self._G_indexed = nx.convert_node_labels_to_integers(X, first_label=0)
        # prepare RNG
        self._rng = np.random.RandomState(self.random_state)

        n_samples = fit_params.get("n_samples", 1)

        if self.q is None:
            # nothing to estimate for classic unweighted Wilson in this API
            self.s_ = None
            self.s_std_ = None
            self.g_ = None
            self.g_std_ = None
            return self

        # draw n_samples independent forests and compute root counts
        root_counts = np.empty(n_samples)
        for i in range(n_samples):
            _, roots = self.sample()
            root_counts[i] = len(roots)

        self.s_all_ = root_counts
        s_mean = root_counts.mean()
        s_std = root_counts.std()

        self.s_ = s_mean
        self.s_std_ = s_std
        # g = s / q
        try:
            self.g_ = s_mean / (self.q)
            self.g_std_ = s_std / (self.q)
        except Exception:
            self.g_ = None
            self.g_std_ = None

        return self

    # internal helper to draw a random neighbor using either G_ or H
    def _random_successor_graph(self, graph, v):
        deg = graph.degree(v)
        if deg == 0:
            return None
        nbrs = list(graph[v])
        return nbrs[self._rng.randint(0, len(nbrs))]

    def sample(self):
        """Sample a spanning tree (or forest) from the fitted graph.

        Returns
        -------
        T : iterable of edge tuples
            Edges in the sampled forest/tree (directed successor edges).
        roots : list
            Nodes that are roots (no successor).
        """
        # create RNG for this call

        # Classic Wilson (no q): operate on integer-labelled internal graph
        Gint = getattr(self, "_G_indexed", self.G_)
        # Classic Wilson (no q): simple undirected Gint
        if self.q is None:
            InTree = [False] * self.nv_
            Next = {}
            roots = []
            root = 0
            InTree[root] = True
            Next[root] = None
            for i in range(self.nv_):
                u = i
                while not InTree[u]:
                    nbrs = list(Gint[u])
                    if len(nbrs) == 0:
                        Next[u] = None
                        InTree[u] = True
                        break
                    v = nbrs[self._rng.randint(0, len(nbrs))]
                    Next[u] = v
                    u = v
                u = i
                while not InTree[u]:
                    InTree[u] = True
                    u = Next[u]
            T = []
            for i in range(self.nv_):
                v = Next.get(i)
                if v is not None:
                    T.append((i, v))
                else:
                    roots.append(i)
            return T, roots

        # q variant: build directed weighted graph H
        # Use integer-labelled graph for building H
        Gint = getattr(self, "_G_indexed", self.G_)
        H = nx.DiGraph()
        H.add_nodes_from(range(self.nv_ + 1))
        for u, v in Gint.edges():
            H.add_edge(u, v, weight=1.0)
            H.add_edge(v, u, weight=1.0)
        root = self.nv_
        for u in Gint.nodes():
            H.add_edge(u, root, weight=self.q)

        intree = [False] * H.number_of_nodes()
        successor = {}
        F = nx.DiGraph()
        roots = set()
        intree[root] = True
        successor[root] = None

        # Iterate over graph nodes in random order (root is already intree)
        order = list(range(self.nv_))
        self._rng.shuffle(order)

        for i in order:
            u = i
            while not intree[u]:
                nei = list(H.neighbors(u))
                if not nei:
                    # Defensive: isolated node in H (shouldn't normally occur),
                    # mark as root and break the walk.
                    successor[u] = None
                    intree[u] = True
                    break
                weights = np.array([H.get_edge_data(u, w)["weight"] for w in nei])
                if weights.sum() == 0:
                    v = self._rng.choice(nei)
                else:
                    probs = weights / weights.sum()
                    v = self._rng.choice(nei, p=probs.astype(float))
                successor[u] = v
                if v == root:
                    roots.add(u)
                u = v
            u = i
            while not intree[u]:
                intree[u] = True
                u = successor[u]

        for i in range(self.nv_):
            if i in successor:
                v = successor[i]
                if v is not None and v != root:
                    F.add_edge(i, v)

        return F, list(roots)


def sample_wilson(
    q: float,
    G: nx.Graph,
    n_samples: int,
    seed: int,
    lambdas: np.ndarray | None,
    compute_theory: bool = True,
) -> dict:
    """Helper function to sample s(q) for a single q value."""
    wilson = Wilson(q=q, random_state=seed)
    wilson.fit(G, n_samples=n_samples)
    record = {
        "q": q,
        "s_mc": wilson.s_,
        "s_mc_se": wilson.s_std_,
        "g_mc": wilson.g_,
        "g_mc_se": wilson.g_std_,
    }
    if compute_theory and lambdas is not None:
        s_spec = s_from_spectrum(q, lambdas)
        g_spec = s_spec / q if q != 0.0 else 0.0
        record["s_spec"] = s_spec
        record["g_spec"] = g_spec
    return record
