"""Graph Isomorphism Network with a sum readout at each layer."""

import torch
from torch import nn
import torch.nn.functional as F


class GINMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.Linear(hidden_size, hidden_size)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size)])

    def forward(self, features):
        hidden = F.relu(self.batch_norms[0](self.linears[0](features)))
        return self.linears[1](hidden)


class GraphCNN(nn.Module):
    def __init__(self, input_size, device, hidden_size=32, depth=4):
        super().__init__()
        self.device = device
        self.eps = nn.Parameter(torch.zeros(depth))
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(depth):
            self.mlps.append(GINMLP(input_size if layer == 0 else hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

    def _graph_pool(self, graphs, offsets):
        indices = torch.tensor([
            [index, node] for index in range(len(graphs)) for node in range(offsets[index], offsets[index + 1])
        ], dtype=torch.long).T
        values = torch.ones(offsets[-1])
        return torch.sparse_coo_tensor(indices, values, (len(graphs), offsets[-1])).to(self.device)

    def _adjacency(self, graphs, offsets):
        indices = torch.cat([graph.edge_mat + offset for graph, offset in zip(graphs, offsets)], dim=1)
        values = torch.ones(indices.size(1))
        return torch.sparse_coo_tensor(indices, values, (offsets[-1], offsets[-1])).to(self.device)

    def forward(self, graphs):
        hidden = torch.cat([graph.node_features for graph in graphs], dim=0).to(self.device)
        offsets = [0]
        for graph in graphs:
            offsets.append(offsets[-1] + len(graph.g))
        graph_pool = self._graph_pool(graphs, offsets)
        adjacency = self._adjacency(graphs, offsets)

        node_layers = []
        for layer, (mlp, norm) in enumerate(zip(self.mlps, self.batch_norms)):
            pooled = torch.spmm(adjacency, hidden)
            pooled = pooled + (1 + self.eps[layer]) * hidden
            hidden = F.relu(norm(mlp(pooled)))
            node_layers.append(hidden)
        readouts = [torch.spmm(graph_pool, nodes) for nodes in node_layers]
        return torch.cat(readouts, dim=-1), node_layers
