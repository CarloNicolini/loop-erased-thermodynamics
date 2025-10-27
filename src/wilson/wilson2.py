#!/usr/bin/env python3
"""
Wilson's algorithm for unweighted STs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


class Wilson:
    def __init__(self, G, q):
        self.G = G
        self.H = nx.DiGraph()
        self.nv = G.number_of_nodes()
        self.q = q
        self.L = nx.laplacian_matrix(self.G).toarray()

        # set edge attribute weight with weight 1
        self.H.add_weighted_edges_from([(u, v, 1.0) for u, v in G.edges()])
        self.H.add_weighted_edges_from([(v, u, 1.0) for u, v in G.edges()])

        # add links from all nodes in the original graph to the root with weight q
        self.root = self.nv
        self.H.add_weighted_edges_from([(u, self.root, q) for u in G.nodes()])

    # Choose an edge from v's adjacency list (randomly)
    def random_successor(self, v):
        nei = list(self.H.neighbors(v))
        weight = np.array(
            [self.H.get_edge_data(v, u)["weight"] for u in nei], dtype=float
        )
        weight /= weight.sum()
        return np.random.choice(nei, p=weight)

    def sample(self):
        intree = [False] * self.H.number_of_nodes()
        successor = {}
        # put the additional node
        F = nx.DiGraph()
        self.roots = set()
        root = self.nv
        intree[root] = True
        successor[root] = None

        from random import shuffle

        l = [root] + list(range(self.nv))
        shuffle(
            l
        )  # not necessary but nice, since the results do not depend on the order
        for i in l:
            u = i
            while not intree[u]:
                successor[u] = self.random_successor(u)
                if (
                    successor[u] == self.nv
                ):  # if the last node of the trajectory is ∆ add it to the roots
                    self.roots.add(u)
                u = successor[u]
            u = i  # come back to the node it started from
            # remove self-loops
            while not intree[u]:
                intree[u] = True
                # if u in successor:
                u = successor[u]

        # Creates the random forest
        for i in range(self.nv):
            if i in successor.keys():
                neighbor = successor[i]
                if neighbor is not None:
                    F.add_edge(i, neighbor)

        if self.nv in self.roots:
            self.roots.remove(self.root)
        # remove the root node, together with all its links
        F.remove_node(self.root)
        # save the leaves
        # self.leaves = [n for n in F.nodes() if F.degree(n)==1]
        return F, list(self.roots)

    def s(self):
        lambdai = np.linalg.eigvalsh(self.L)
        return (self.q / (self.q + lambdai)).sum()


def draw_sampling(G, T, root_nodes=None, **kwargs):
    ax = kwargs.get("ax", None)
    cmap = kwargs.get("cmap", matplotlib.cm.get_cmap("Set3"))
    T = nx.DiGraph(T)
    n_trees = nx.number_weakly_connected_components(T)
    pos = kwargs.get("pos", nx.spectral_layout(G))

    if root_nodes is not None:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=root_nodes,
            node_color="r",
            node_size=25,
            linew_width=1,
            ax=ax,
        )
        # nx.draw_networkx_labels(G, pos=pos, labels={i: i for i in range(G.number_of_nodes())})
    nx.draw_networkx_nodes(
        G, pos=pos, node_color="k", node_size=3, lines_width=0.1, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edge_style="dashed", alpha=0.1, edge_color="k", edge_width=0.01, ax=ax
    )

    for i, t in enumerate(nx.weakly_connected_component_subgraphs(T)):
        e = nx.number_of_edges(t)
        # print('|V|=%d |E|=%d' % (t.number_of_nodes(),t.number_of_edges()))
        nx.draw_networkx_edges(
            t,
            pos,
            width=4,
            edge_cmap=cmap,
            edge_color=[cmap(float(i) / n_trees)] * e,
            ax=ax,
            arrows=True,
        )
        # nx.draw_networkx_edges(t, pos, width=1, edge_color='k', arrows=True,ax=ax)
    plt.axis("off")


def trace_estimator(G):
    reps = 1
    beta_range = np.logspace(-2, 2, 100)
    L = nx.laplacian_matrix(G).toarray()
    plt.semilogx(
        beta_range,
        [
            np.mean([len(Wilson(G, 1 / beta).sample()[1]) for _ in range(reps)])
            for beta in beta_range
        ],
        label="E[|R|]",
    )
    plt.semilogx(
        beta_range,
        [Wilson(G, q=1 / beta).s() for beta in beta_range],
        label="q Tr[(qI+L)^{-1}]",
    )
    plt.legend()
    plt.grid(which="both")
    plt.show()


def sampling_example(G, pos=None):
    q = 0.1
    W = Wilson(G, q=q)
    F, roots = W.sample()
    draw_sampling(G, F, roots, pos=pos)
    plt.show()


def quad(x, y):
    pos = {}
    k = 0
    for i in range(x):
        for j in range(y):
            pos[k] = np.array([i, j])
            k = k + 1
    return pos


if __name__ == "__main__":
    G = nx.grid_2d_graph(15, 15, periodic=False)
    G = nx.from_numpy_array(nx.to_numpy_array(G))
    pos = quad(15, 15)
    # trace_estimator(G)
    sampling_example(G)
