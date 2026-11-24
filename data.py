"""Match original MUTAG graphs to text embeddings and create noisy training folds."""

from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset


@dataclass
class Graph:
    g: range  # The GIN uses the length to build its batched adjacency.
    label: int
    node_features: torch.Tensor
    edge_mat: torch.Tensor
    graph_id: int = -1
    sample_id: int = -1


def load_mutag(root: Path):
    dataset = TUDataset(str(root), name="MUTAG")
    graphs = [
        Graph(range(data.num_nodes), int(data.y.item()), data.x, data.edge_index, index)
        for index, data in enumerate(dataset)
    ]
    with np.load(root / "MUTAG_sentencebert_embeddings.npz", allow_pickle=False) as cache:
        ids = cache["graph_ids"]
        values = cache["embeddings"].astype(np.float32)
    if ids.shape != (len(graphs),) or sorted(ids.tolist()) != list(range(len(graphs))):
        raise ValueError("Text cache IDs must match the original MUTAG graph IDs")
    if values.shape != (len(graphs), 768) or not np.isfinite(values).all():
        raise ValueError("Expected one finite 768-dimensional text vector per graph")
    if not (np.linalg.norm(values, axis=1) > 0).all():
        raise ValueError("Text embeddings must be nonzero")
    return graphs, dataset.num_node_features, dataset.num_classes, torch.from_numpy(values[np.argsort(ids)])


def split_folds(graphs, seed):
    shuffled = list(graphs)
    np.random.RandomState(seed).shuffle(shuffled)
    labels = [graph.label for graph in shuffled]
    splitter = StratifiedKFold(n_splits=10, shuffle=False)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        yield [shuffled[i] for i in train], [shuffled[i] for i in test]


def add_training_noise(graphs, num_classes, rate, seed):
    transition = np.float64(rate) / np.float64(num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(transition, 1.0 - np.float64(rate))
    diagonal = np.arange(num_classes)
    transition[diagonal, diagonal] = transition[diagonal, diagonal] + 1.0 - transition.sum(0)
    rng = np.random.RandomState(seed)
    noisy_graphs = []
    for index, graph in enumerate(graphs):
        training_graph = copy(graph)
        training_graph.sample_id = index
        training_graph.label = int(rng.multinomial(1, transition[graph.label], 1)[0].argmax())
        noisy_graphs.append(training_graph)
    return noisy_graphs
