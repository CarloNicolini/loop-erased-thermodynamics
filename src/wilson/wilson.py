#!/usr/bin/env python3
"""
Wilson's algorithm for unweighted STs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


class Wilson:
    def __init__(self, G):
        self.G = G
        self.nv = G.number_of_nodes()
        self.iteration = 0
        # Array to keep track which vertices have already been visited
        # (and in which iteration)
        self.Next = {}
        self.tree_gen = False
        self.s_tree = None
        # to sample random spanning forests with parameter q
        self.roots = []
        self.forest = []
        self.max_degree = max([d for n, d in nx.degree(G)])

    # Choose an edge from v's adjacency list (randomly)
    def random_successor(self, v):
        k_v = G.degree(v)
        return list(self.G[v])[np.random.randint(k_v)]

    def sample(self, seed=None, p=1.0, q=np.inf):
        if seed is not None:
            np.random.seed(int(seed))
        # selects a random node
        # self.root = np.random.randint(self.nv)
        # self.root = self.nv + 1
        # print(self.root)
        # self.random_tree_with_root(self.root, q)
        self.random_tree(q)
        self.extract_tree_edges()
        return self.s_tree, self.roots

    def random_tree_with_root(self, r, q=np.inf):
        InTree = [False] * self.nv
        self.Next[r] = None
        # for k in range(self.nv+1):
        #    self.Next[k] = None
        # put the root in the tree
        InTree[r] = True
        for i in range(self.nv):
            u = i
            # creates the random walk, by updating dictionary Next
            iteration = 0
            while not InTree[u]:
                self.Next[u] = self.random_successor(u)
                u = self.Next[u]
            u = i  # come back to the node it started from
            # remove self-loops
            while not InTree[u]:
                InTree[u] = True
                # if u in self.Next:
                u = self.Next[u]
        return self.Next

    def wilson(self, q=np.inf):
        print(q)
        F = nx.DiGraph()
        intree = [False] * self.nv
        successor = {}
        self.root = np.random.randint(self.nv)
        self.roots.append(self.root)
        intree[self.root] = True
        for i in range(self.nv):
            u = i
            iteration = 0
            while not intree[u] or iteration <= np.exp(q):
                successor[u] = self.random_successor(u)
                u = successor[u]
                iteration += 1
            u = i  # come back to the node it started from
            # remove self-loops
            while not intree[u]:
                intree[u] = True
                u = successor[u]
        # Creates the random forest
        for i in range(self.nv):
            if i in successor.keys():
                neighbor = successor[i]
                if neighbor is not None:
                    F.add_edge(i, neighbor)
                else:
                    self.roots.append(i)
            else:
                self.roots.append(i)

        #  print('Is a forest?', nx.is_forest(F))
        # print('Forest |V|=%d, |E|=%d' % (F.number_of_nodes(),F.number_of_edges()) )
        return F.edges(), self.roots

    def random_tree(self, q=np.inf):
        e = q
        tree = None
        while tree is None:
            e /= 2
            tree = self.attempt(e)

    def attempt(self, e):
        def chance(e):
            return np.random.random() < e

        InTree = [False] * self.nv
        num_roots = 0
        for i in range(self.nv):
            u = i
            while not InTree[u]:
                if chance(e):
                    self.Next[u] = None
                    InTree[u] = True
                    num_roots += 1
                    if num_roots == 2:
                        return None
                else:
                    self.Next[u] = self.random_successor(u)
                    u = self.Next[u]
            u = i
            while not InTree[u]:
                InTree[u] = True
                u = self.Next[u]
        return self.Next

    def extract_tree_edges(self):
        self.s_tree = []
        for i in range(self.nv):
            neighbor = self.Next[i]
            if neighbor is not None:
                self.s_tree.append((i, neighbor))
            else:
                self.root = i


def to_networkx(G, s_tree, root_nodes=None):
    T = nx.Graph()
    T.add_nodes_from(range(nx.number_of_nodes(G)))
    for e in s_tree:
        T.add_edge(e[0], e[1])

    n_trees = nx.number_connected_components(T)
    pos = nx.kamada_kawai_layout(G)
    if root_nodes is not None:
        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=root_nodes, node_color="r", node_size=25, linew_width=1
        )
        # nx.draw_networkx_labels(G, pos=pos, labels={i: i for i in range(G.number_of_nodes())})
    nx.draw_networkx_nodes(G, pos=pos, node_color="k", node_size=3, lines_width=0.1)
    nx.draw_networkx_edges(
        G, pos, edge_style="dashed", alpha=0.1, edge_color="k", edge_width=0.01
    )

    cmap = matplotlib.cm.get_cmap("viridis")
    for i, t in enumerate(nx.connected_component_subgraphs(T)):
        e = nx.number_of_edges(t)
        print("|V|=%d |E|=%d" % (t.number_of_nodes(), t.number_of_edges()))
        nx.draw_networkx_edges(
            t,
            pos,
            width=2,
            edge_cmap=plt.cm.Set2,
            edge_color=[cmap(float(i) / n_trees)] * e,
        )
    plt.axis("off")
    plt.show()
