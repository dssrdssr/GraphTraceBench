from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    QueryReadout,
    TransformerBlock,
    batch_to_graph_slices,
    build_dense_adj,
    pool_by_assignment,
    spectral_patch_partition,
)


class PatchGTRegressor(nn.Module):
    """Task-adapted PatchGT-style regressor.

    PatchGT clusters nodes into non-trainable graph patches and runs a Transformer
    over patch representations. This implementation uses a small spectral partition
    (pure torch) and combines patch tokens with src/dst-aware readout.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        patch_ratio: float = 0.25,
        min_patches: int = 2,
        **_: dict,
    ) -> None:
        super().__init__()
        self.patch_ratio = patch_ratio
        self.min_patches = min_patches
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.patch_blocks = nn.ModuleList(
            [TransformerBlock(hidden_channels, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = QueryReadout(hidden_channels, out_channels=out_channels, dropout=dropout)

    def _forward_graph(self, graph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        n = x.size(0)
        num_patches = max(self.min_patches, int(math.ceil(n * self.patch_ratio)))
        num_patches = min(num_patches, n)

        adj = build_dense_adj(n, graph.edge_index, device=x.device)
        assignment = spectral_patch_partition(adj, num_patches)
        patch_x = pool_by_assignment(x, assignment, num_patches)
        for block in self.patch_blocks:
            patch_x = block(patch_x)

        node_x = patch_x[assignment]
        return self.readout(node_x, graph.src_id, graph.dst_id)

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
