#!/usr/bin/env python3
"""Unified Wilson sampler implementations.

Provides two classes:
"""

import random
from typing import Optional
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


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
    - ``s_``: (only when q is not None) value of sum(q / (q + lambda_i))
    """

    def __init__(self, q: Optional[float] = None, random_state: Optional[int] = None):
        self.q = q
        self.random_state = random_state
        # attributes set in fit
        self.G_ = None
        self.nv_ = None
        self.s_ = None

    def get_params(self, deep=True):
        return {"q": self.q, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, G: nx.Graph):
        """Fit the estimator on graph G. Computes s_ when q is provided."""
        self.G_ = G
        self.nv_ = G.number_of_nodes()
        # prepare RNG
        self._rng = np.random.RandomState(self.random_state)
        if self.q is not None:
            L = nx.laplacian_matrix(G).toarray()
            lambdai = np.linalg.eigvalsh(L)
            # s = sum(q / (q + lambda_i))
            self.s_ = float(np.sum(self.q / (self.q + lambdai)))
        else:
            self.s_ = None
        return self

    # internal helper to draw a random neighbor using either G_ or H
    def _random_successor_graph(self, graph, v):
        deg = graph.degree(v)
        if deg == 0:
            return None
        nbrs = list(graph[v])
        return nbrs[self._rng.randint(0, len(nbrs))]

    def sample(self, seed: Optional[int] = None):
        """Sample a spanning tree (or forest) from the fitted graph.

        Returns
        -------
        T : iterable of edge tuples
            Edges in the sampled forest/tree (directed successor edges).
        roots : list
            Nodes that are roots (no successor).
        """
        if self.G_ is None:
            raise RuntimeError("Estimator not fitted. Call fit(G) before sampling.")

        # create RNG for this call
        if seed is None:
            rng = self._rng
        else:
            rng = np.random.RandomState(seed)

        # Classic Wilson (no q): simple undirected G_
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
                    nbrs = list(self.G_[u])
                    if len(nbrs) == 0:
                        Next[u] = None
                        InTree[u] = True
                        break
                    v = nbrs[rng.randint(0, len(nbrs))]
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
        H = nx.DiGraph()
        H.add_nodes_from(range(self.nv_ + 1))
        for u, v in self.G_.edges():
            H.add_edge(u, v, weight=1.0)
            H.add_edge(v, u, weight=1.0)
        root = self.nv_
        for u in self.G_.nodes():
            H.add_edge(u, root, weight=float(self.q))

        intree = [False] * H.number_of_nodes()
        successor = {}
        F = nx.DiGraph()
        roots = set()
        intree[root] = True
        successor[root] = None

        order = [root] + list(range(self.nv_))
        rng.shuffle(order)

        for i in order:
            u = i
            while not intree[u]:
                nei = list(H.neighbors(u))
                weights = np.array(
                    [H.get_edge_data(u, w)["weight"] for w in nei], dtype=float
                )
                if weights.sum() == 0:
                    v = rng.choice(nei)
                else:
                    probs = weights / weights.sum()
                    v = np.random.RandomState(rng.randint(2**31)).choice(nei, p=probs)
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

    # sklearn-like transform alias
    def transform(self, X=None, n_samples: int = 1, seed: Optional[int] = None):
        """Alias to sample; kept for transformer compatibility.

        Returns a list of sampled forests/trees (length n_samples).
        """
        results = []
        base = seed if seed is not None else None
        for i in range(n_samples):
            s = None if base is None else base + i
            results.append(self.sample(seed=s))
        return results


class WilsonQ:
    def __init__(self, G, q):
        self.G = G
        self.nv = G.number_of_nodes()
        self.q = float(q)
        self.H = nx.DiGraph()
        self.H.add_nodes_from(range(self.nv + 1))
        for u, v in G.edges():
            self.H.add_edge(u, v, weight=1.0)
            self.H.add_edge(v, u, weight=1.0)
        self.root = self.nv
        for u in G.nodes():
            self.H.add_edge(u, self.root, weight=self.q)

    def random_successor(self, v):
        nei = list(self.H.neighbors(v))
        weights = np.array(
            [self.H.get_edge_data(v, u)["weight"] for u in nei], dtype=float
        )
        if weights.sum() == 0:
            return random.choice(nei)
        probs = weights / weights.sum()
        return np.random.choice(nei, p=probs)

    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))
            random.seed(int(seed))

        intree = [False] * self.H.number_of_nodes()
        successor = {}
        F = nx.DiGraph()
        roots = set()
        intree[self.root] = True
        successor[self.root] = None

        order = [self.root] + list(range(self.nv))
        random.shuffle(order)

        for i in order:
            u = i
            while not intree[u]:
                successor[u] = self.random_successor(u)
                if successor[u] == self.root:
                    roots.add(u)
                u = successor[u]
            u = i
            while not intree[u]:
                intree[u] = True
                u = successor[u]

        for i in range(self.nv):
            if i in successor:
                v = successor[i]
                if v is not None and v != self.root:
                    F.add_edge(i, v)

        return F, list(roots)

    def s(self):
        L = nx.laplacian_matrix(self.G).toarray()
        lambdai = np.linalg.eigvalsh(L)
        return float(np.sum(self.q / (self.q + lambdai)))


def draw_sampling(G, T, root_nodes=None, ax=None, cmap=None, pos=None):
    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        cmap = matplotlib.cm.get_cmap("Set3")
    Tdig = nx.DiGraph()
    Tdig.add_edges_from(T)
    if pos is None:
        pos = nx.spectral_layout(G)
    if root_nodes is not None:
        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=root_nodes, node_color="r", node_size=25, ax=ax
        )
    nx.draw_networkx_nodes(G, pos=pos, node_color="k", node_size=3, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.1, edge_color="k", ax=ax)
    comps = list(nx.weakly_connected_components(Tdig))
    for i, comp in enumerate(comps):
        sub = Tdig.subgraph(comp).copy()
        e = sub.number_of_edges()
        nx.draw_networkx_edges(
            sub,
            pos=pos,
            width=2,
            edge_color=[cmap(float(i) / max(1, len(comps)))] * e,
            ax=ax,
        )
    ax.set_axis_off()
    return ax


def quad(x, y):
    pos = {}
    k = 0
    for i in range(x):
        for j in range(y):
            pos[k] = np.array([i, j])
            k += 1
    return pos
