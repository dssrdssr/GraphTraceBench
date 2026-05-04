from __future__ import annotations

import torch
from torch import nn

from .common import (
    QueryReadout,
    PairBiasedSelfAttention,
    batch_to_graph_slices,
    build_dense_adj,
    random_walk_landing_probs,
    shortest_path_distances,
)


class GRITBlock(nn.Module):
    def __init__(self, hidden_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_channels)
        self.norm_p = nn.LayerNorm(hidden_channels)
        self.attn = PairBiasedSelfAttention(hidden_channels, heads=heads, dropout=dropout)
        self.pair_to_bias = nn.Linear(hidden_channels, heads)
        self.pair_update = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.pair_to_bias(self.norm_p(pair))
        x = x + self.dropout(self.attn(self.norm_x(x), pair_bias=bias))

        xi = x.unsqueeze(1).expand_as(pair)
        xj = x.unsqueeze(0).expand_as(pair)
        pair = pair + self.dropout(self.pair_update(torch.cat([pair, xi + xj, (xi - xj).abs()], dim=-1)))
        x = x + self.dropout(self.ffn(self.norm_x(x)))
        return x, pair


class GRITRegressor(nn.Module):
    """Task-adapted GRIT-style regressor.

    GRIT uses graph inductive biases without message passing, notably random-walk
    positional encodings and learned node-pair representations. This implementation
    keeps those core ideas in a compact PyG-compatible form.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        rw_steps: int = 4,
        max_dist: int = 8,
        **_: dict,
    ) -> None:
        super().__init__()
        self.rw_steps = rw_steps
        self.max_dist = max_dist
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.degree_encoder = nn.Linear(2, hidden_channels)
        self.rw_encoder = nn.Linear(rw_steps, hidden_channels)
        self.pair_encoder = nn.Linear(3, hidden_channels)
        self.layers = nn.ModuleList(
            [GRITBlock(hidden_channels, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = QueryReadout(hidden_channels, out_channels=out_channels, dropout=dropout)

    def _forward_graph(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        adj = build_dense_adj(x.size(0), graph.edge_index, device=x.device)
        indeg = adj.sum(dim=0)
        outdeg = adj.sum(dim=-1)
        rw = random_walk_landing_probs(adj, steps=self.rw_steps)
        x = x + self.degree_encoder(torch.stack([indeg, outdeg], dim=-1)) + self.rw_encoder(rw)

        spd = shortest_path_distances(x.size(0), graph.edge_index, max_dist=self.max_dist).float()
        spd = spd.clamp_max(self.max_dist + 1) / float(self.max_dist + 1)
        pair_feats = torch.stack([adj, adj.t(), spd], dim=-1)
        pair = self.pair_encoder(pair_feats)
        for layer in self.layers:
            x, pair = layer(x, pair)
        return self.readout(x, graph.src_id, graph.dst_id)

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
