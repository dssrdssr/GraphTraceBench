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
    simple_topology_features,
    row_normalize,
)


class GraphRecalibration(nn.Module):
    def __init__(self, hidden_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, hidden_channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.net(x.mean(dim=0, keepdim=True))
        return x * gate


class TIGTBlock(nn.Module):
    def __init__(self, hidden_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)
        self.local_proj = nn.Linear(hidden_channels, hidden_channels)
        self.global_attn = PairBiasedSelfAttention(hidden_channels, heads=heads, dropout=dropout)
        self.fuse = nn.Linear(hidden_channels * 2, hidden_channels)
        self.recal = GraphRecalibration(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, p: torch.Tensor, pair_bias: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        local = self.local_proj(p @ h)
        global_h = self.global_attn(h, pair_bias=pair_bias)
        fused = self.fuse(torch.cat([local, global_h], dim=-1))
        x = x + self.dropout(fused)
        x = self.recal(x)
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x


class TIGTRegressor(nn.Module):
    """Task-adapted Topology-Informed Graph Transformer.

    The original TIGT combines topological positional embeddings, dual-path local
    and global processing, and graph-level recalibration. This compact version keeps
    those same ideas for arithmetic-path graph regression.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        topo_rw_steps: int = 4,
        max_dist: int = 8,
        **_: dict,
    ) -> None:
        super().__init__()
        self.max_dist = max_dist
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.topo_encoder = nn.Linear(3 + topo_rw_steps, hidden_channels)
        self.spd_bias = nn.Embedding(max_dist + 2, heads)
        self.layers = nn.ModuleList(
            [TIGTBlock(hidden_channels, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = QueryReadout(hidden_channels, out_channels=out_channels, dropout=dropout)
        self.topo_rw_steps = topo_rw_steps

    def _forward_graph(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        adj = build_dense_adj(x.size(0), graph.edge_index, device=x.device)
        topo = simple_topology_features(adj)
        rw = random_walk_landing_probs(adj, steps=self.topo_rw_steps)
        x = x + self.topo_encoder(torch.cat([topo, rw], dim=-1))
        p = row_normalize(adj + torch.eye(x.size(0), device=x.device))
        spd = shortest_path_distances(x.size(0), graph.edge_index, max_dist=self.max_dist).clamp_max(self.max_dist + 1)
        pair_bias = self.spd_bias(spd)
        for layer in self.layers:
            x = layer(x, p, pair_bias)
        return self.readout(x, graph.src_id, graph.dst_id)

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
