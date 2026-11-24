"""Standard 0D and 1D persistent homology with DeepSets vectorization."""

import sys
from pathlib import Path

import torch
from torch import nn

from layers.graph_equivariant_layers import DeepSetLayer0, DeepSetLayer1
from utils.utils import remove_duplicate_edges

_native = Path(__file__).resolve().parents[1] / "native"
if any(_native.glob("ph_cpu*.pyd")) or any(_native.glob("ph_cpu*.so")):
    sys.path.insert(0, str(_native))

try:
    from ph_cpu import compute_persistence_homology_batched_mt
except ImportError as exc:
    raise ImportError("Could not load ph_cpu. See the native build instructions in README.md.") from exc


class PersistentHomologyLayer(nn.Module):
    def __init__(self, hidden_size=32, filters=8, output_size=16):
        super().__init__()
        self.filters = filters
        self.filtrations = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, filters), nn.Sigmoid(),
        )
        self.deepset_fn = DeepSetLayer0(in_dim=2 * filters, out_dim=output_size)
        self.deepset_fn_dim1 = DeepSetLayer1(in_dim=2, out_dim=output_size)
        self.out = nn.Sequential(nn.Linear(output_size, output_size), nn.ReLU(), nn.Linear(output_size, output_size))
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, node_states, graph_batch):
        edges, node_slices, edge_slices, batch_ids = remove_duplicate_edges(graph_batch)
        node_values = self.filtrations(node_states)
        edge_values, _ = torch.stack((node_values[edges[0]], node_values[edges[1]])).max(dim=0)
        pers0, pers1 = compute_persistence_homology_batched_mt(
            node_values.T.cpu().contiguous(), edge_values.T.cpu().contiguous(),
            edges.T.cpu().contiguous(), node_slices.cpu().long(), edge_slices.cpu().long(),
        )
        pers0, pers1 = pers0.to(node_states.device), pers1.to(node_states.device)
        zero_dim = self.deepset_fn(pers0.permute(1, 0, 2).reshape(pers0.size(1), -1), batch_ids)
        one_dim = []
        for diagram in pers1:
            mask = ~(diagram == 0).all(dim=-1)
            one_dim.append(self.deepset_fn_dim1(diagram[mask], edge_slices, mask=mask))
        return self.out(self.bn(zero_dim + torch.stack(one_dim).mean(dim=0)))
