from __future__ import annotations

import torch
from torch import nn

from .common import QueryReadout, TransformerBlock, batch_to_graph_slices, build_dense_adj, row_normalize


class DIFFormerBlock(nn.Module):
    def __init__(self, hidden_channels: int, heads: int = 4, dropout: float = 0.1, diff_steps: int = 3):
        super().__init__()
        self.diff_steps = diff_steps
        self.norm = nn.LayerNorm(hidden_channels)
        self.attn = TransformerBlock(hidden_channels, heads=heads, dropout=dropout)
        self.diff_weights = nn.Parameter(torch.zeros(diff_steps))
        self.diff_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        diff = torch.zeros_like(h)
        cur = h
        coeffs = torch.softmax(self.diff_weights, dim=0)
        for k in range(self.diff_steps):
            cur = p @ cur
            diff = diff + coeffs[k] * cur
        x = x + self.dropout(self.diff_proj(diff))
        x = self.attn(x, pair_bias=None)
        return x


class DIFFormerRegressor(nn.Module):
    """Task-adapted DIFFormer-style regressor.

    The official DIFFormer is a diffusion-induced scalable graph transformer. Here
    we keep the diffusion core: each layer mixes multi-step diffusion with a global
    Transformer block, which is well-suited for long-range arithmetic-path reasoning.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        diff_steps: int = 3,
        **_: dict,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.layers = nn.ModuleList(
            [DIFFormerBlock(hidden_channels, heads=heads, dropout=dropout, diff_steps=diff_steps) for _ in range(num_layers)]
        )
        self.readout = QueryReadout(hidden_channels, out_channels=out_channels, dropout=dropout)

    def _forward_graph(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        adj = build_dense_adj(x.size(0), graph.edge_index, device=x.device)
        p = row_normalize(adj + torch.eye(x.size(0), device=x.device))
        for layer in self.layers:
            x = layer(x, p)
        return self.readout(x, graph.src_id, graph.dst_id)

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
