from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "PyG is required for GraphGPSRegressor. Install torch_geometric first."
    ) from exc


class GraphGPSRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = "multihead",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn, edge_dim=hidden_dim)
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs={"dropout": dropout},
            )
            self.layers.append(gps_layer)

        pair_dim = hidden_dim * 5
        self.reg_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _src_dst_global_indices(data) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(data, "ptr"):
            src = data.src.view(-1)
            dst = data.dst.view(-1)
            return src, dst
        base = data.ptr[:-1]
        src = base + data.src.view(-1)
        dst = base + data.dst.view(-1)
        return src, dst

    def forward(self, data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for layer in self.layers:
            x = layer(x, data.edge_index, batch=data.batch, edge_attr=edge_attr)

        graph_emb = global_mean_pool(x, data.batch)
        src_idx, dst_idx = self._src_dst_global_indices(data)
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]
        pair_emb = torch.cat(
            [graph_emb, src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb],
            dim=-1,
        )
        return self.reg_head(pair_emb).view(-1)
