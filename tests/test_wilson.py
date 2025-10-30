import os
import sys
import math
import numpy as np
import networkx as nx
from wilson.wilson import Wilson as Wilson1

# Ensure we can import the local src package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def canonical_edge(u, v):
    return (u, v) if u <= v else (v, u)


def test_random_successor_neighbor():
    G = nx.path_graph(5)
    W = Wilson1()
    W.fit(G)
    # sample once and ensure every sampled edge connects neighbors in G
    s_tree, roots = W.sample(seed=1)
    for u, v in s_tree:
        assert v in G[u]


def test_sample_returns_forest_and_edge_count():
    # Smaller graphs for faster tests
    graphs = [
        nx.path_graph(5),
        nx.cycle_graph(5),
        nx.erdos_renyi_graph(8, 0.4, seed=1),
    ]
    for G in graphs:
        W = Wilson1()
        W.fit(G)
        s_tree, roots = W.sample(seed=123)
        # Build graph from sampled edges
        T = nx.Graph()
        T.add_nodes_from(G.nodes())
        for e in s_tree:
            T.add_edge(e[0], e[1])
        # Should be a forest (no cycles)
        assert nx.is_forest(T)
        # Number of edges must be <= n-1
        assert T.number_of_edges() <= G.number_of_nodes() - 1


def test_sample_deterministic_with_seed():
    G = nx.path_graph(8)
    W1 = Wilson1()
    W1.fit(G)
    s1, _ = W1.sample(seed=42)
    W2 = Wilson1()
    W2.fit(G)
    s2, _ = W2.sample(seed=42)
    # With same seed the sampled (raw) s_tree representation should match
    assert set(map(tuple, s1)) == set(map(tuple, s2))


def test_complete_graph_edge_marginals_approx():
    # For complete graph K_n, effective resistance between any two distinct nodes is 2/n,
    # so p_e = 2/n for unweighted graph. We'll empirically estimate frequencies.
    # smaller complete graph to speed up test
    n = 5
    G = nx.complete_graph(n)
    W = Wilson1()
    W.fit(G)

    # fewer Monte-Carlo trials for CI-friendly unit test
    trials = 200
    counts = {canonical_edge(u, v): 0 for u, v in G.edges()}
    for i in range(trials):
        s_tree, _ = W.sample(seed=1000 + i)
        for e in s_tree:
            counts[canonical_edge(e[0], e[1])] += 1

    freqs = np.array(list(counts.values()), dtype=float) / trials
    expected = 2.0 / n
    # Check that mean absolute deviation is small (looser threshold due to fewer trials)
    mad = np.mean(np.abs(freqs - expected))
    assert mad < 0.08


def test_wilson_q_s_matches_eig_sum():
    G = nx.path_graph(5)
    q = 0.5
    Wq = Wilson1(q=q)
    Wq.fit(G)
    # compute eigenvalues of Laplacian
    L = nx.laplacian_matrix(G).toarray()
    lamb = np.linalg.eigvalsh(L)
    expected = np.sum(q / (q + lamb))
    assert math.isclose(Wq.s_, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_wilson_q_sample_returns_forest():
    G = nx.erdos_renyi_graph(8, 0.25, seed=2)
    Wq = Wilson1(q=0.1)
    Wq.fit(G)
    F, roots = Wq.sample(seed=7)
    # After removing the extra root node, F should be a forest on original nodes
    assert set(F.nodes()).issubset(set(G.nodes()))
    assert nx.is_forest(F)
