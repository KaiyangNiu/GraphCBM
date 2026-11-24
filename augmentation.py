"""Node-aligned Mixup and the two views used in pre-training."""

from copy import copy

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from data import Graph


def append_mixup_graphs(graphs, count):
    mixed = list(graphs)
    first, second = np.random.randint(0, len(graphs), size=(2, count))
    for index, (left, right) in enumerate(zip(first, second)):
        if index % 64 == 0:
            coefficient = np.random.beta(0.5, 0.5)
            coefficient = max(coefficient, 1.0 - coefficient)
        a, b = graphs[left], graphs[right]
        alignment = (a.node_features @ b.node_features.T).softmax(dim=0)
        adj_a = to_dense_adj(a.edge_mat, max_num_nodes=len(a.g))[0].double()
        adj_b = to_dense_adj(b.edge_mat, max_num_nodes=len(b.g))[0].double()
        adj = coefficient * adj_a + (1 - coefficient) * alignment.double() @ adj_b @ alignment.double().T
        adj[adj < 0.1] = 0
        edges, _ = dense_to_sparse(adj)
        features = coefficient * a.node_features + (1 - coefficient) * alignment @ b.node_features
        mixed.append(Graph(range(features.size(0)), -1, features, edges))
    return mixed


def node_drop_view(graphs):
    view = []
    for graph in graphs:
        transformed = copy(graph)
        num_nodes = graph.node_features.size(0)
        dropped = np.random.choice(num_nodes, num_nodes // 10, replace=False)
        adj = torch.zeros((num_nodes, num_nodes))
        adj[graph.edge_mat[0], graph.edge_mat[1]] = 1
        adj[dropped, :] = 0
        adj[:, dropped] = 0
        transformed.edge_mat = adj.nonzero().T
        view.append(transformed)
    return view
