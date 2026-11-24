"""Shared GIN for pre-training and multimodal graph classification."""

from copy import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn.inits import glorot

from gin import GraphCNN
from layers.persistent_homology import PersistentHomologyLayer


class PretrainingModel(nn.Module):
    def __init__(self, input_size, device):
        super().__init__()
        self.gin = GraphCNN(input_size, device).to(device)
        self.projection = nn.Sequential(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 32))

    def forward(self, graphs):
        graph_features, _ = self.gin(graphs)
        return self.projection(graph_features)


def contrastive_loss(original, augmented, temperature=0.30):
    original_norm = original.norm(dim=1)
    augmented_norm = augmented.norm(dim=1)
    similarity = torch.einsum("ik,jk->ij", original, augmented) / torch.einsum("i,j->ij", original_norm, augmented_norm)
    scores = torch.exp(similarity / temperature)
    indices = range(original.size(0))
    positive = scores[indices, indices]
    return -torch.log(positive / (scores.sum(dim=1) - positive)).mean()


class GraphClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_size, 64)
        self.normalization = nn.BatchNorm1d(input_size)
        self.output = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, features):
        return self.output(self.dropout(self.hidden(self.normalization(features)).relu()))


class MultimodalClassifier(nn.Module):
    def __init__(self, input_size, num_classes, device):
        super().__init__()
        self.device = device
        self.text_projector = nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = GraphClassifier(128 + 16 + 64, num_classes)
        self.node_prompt = nn.Parameter(torch.empty(1, input_size))
        glorot(self.node_prompt)
        self.topology = nn.ModuleList(PersistentHomologyLayer() for _ in range(4))

    def structure(self, gin, graphs, feature_dropout=0.0):
        prompted = []
        for graph in graphs:
            transformed = copy(graph)
            features = graph.node_features.to(self.device)
            if feature_dropout:
                features = F.dropout(features, p=feature_dropout, training=True)
            transformed.node_features = features + self.node_prompt
            prompted.append(transformed)
        pooled, node_layers = gin(prompted)
        batch = Batch.from_data_list([
            Data(x=g.node_features, edge_index=g.edge_mat, y=g.label, batch=torch.zeros(len(g.g), dtype=torch.long))
            for g in prompted
        ]).to(self.device)
        topology = torch.stack([layer(nodes, batch) for layer, nodes in zip(self.topology, node_layers)]).mean(dim=0)
        return torch.cat([pooled, topology], dim=1)

    def forward(self, gin, graphs, text_embeddings, feature_dropout=0.0):
        structure = self.structure(gin, graphs, feature_dropout)
        text = self.text_projector(text_embeddings[[g.graph_id for g in graphs]])
        return self.classifier(torch.cat([structure, text], dim=1))
