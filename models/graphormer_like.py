from __future__ import annotations

import torch
from torch import nn

from .common import (
    QueryReadout,
    TransformerBlock,
    batch_to_graph_slices,
    build_dense_adj,
    build_edge_type_matrix,
    shortest_path_distances,
)


class GraphormerRegressor(nn.Module):
    """Task-adapted Graphormer-style regressor.

    Inspired by Graphormer's centrality encoding and shortest-path attention bias,
    but implemented as a lightweight PyG-compatible model for graph regression.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        max_dist: int = 8,
        num_edge_types: int = 4,
        **_: dict,
    ) -> None:
        super().__init__()
        self.max_dist = max_dist
        self.num_edge_types = num_edge_types
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.degree_encoder = nn.Linear(2, hidden_channels)
        self.spd_bias = nn.Embedding(max_dist + 2, heads)
        self.edge_bias = nn.Embedding(num_edge_types + 1, heads)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_channels, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = QueryReadout(hidden_channels, out_channels=out_channels, dropout=dropout)

    def _forward_graph(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        adj = build_dense_adj(x.size(0), graph.edge_index, device=x.device)
        indeg = adj.sum(dim=0)
        outdeg = adj.sum(dim=-1)
        x = x + self.degree_encoder(torch.stack([indeg, outdeg], dim=-1))

        spd = shortest_path_distances(x.size(0), graph.edge_index, max_dist=self.max_dist)
        spd = spd.clamp_max(self.max_dist + 1)
        edge_mat = build_edge_type_matrix(x.size(0), graph.edge_index, graph.edge_type, default_type=self.num_edge_types)
        pair_bias = self.spd_bias(spd) + self.edge_bias(edge_mat)

        for layer in self.layers:
            x = layer(x, pair_bias=pair_bias)
        return self.readout(x, graph.src_id, graph.dst_id)

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
