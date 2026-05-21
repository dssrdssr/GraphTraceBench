from __future__ import annotations

from typing import List, Set, Tuple
import random

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINEConv, global_mean_pool
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "PyG is required for ExphormerRegressor. Install torch_geometric first."
    ) from exc


class ExphormerBlock(nn.Module):
    """A self-contained Exphormer-style block.

    This is a practical, PyG-friendly sparse-attention approximation for custom
    datasets. It keeps the paper's main ideas: real edges, expander edges, and a
    virtual global connector, while staying easy to adapt.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        expander_degree: int = 2,
        use_virtual_token: bool = True,
        layer_seed: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.expander_degree = max(0, expander_degree)
        self.use_virtual_token = use_virtual_token
        self.layer_seed = layer_seed

        local_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(local_nn, edge_dim=hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_local = nn.LayerNorm(hidden_dim)
        self.norm_global = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def _build_expander_edges(self, num_nodes: int, graph_seed: int) -> Set[Tuple[int, int]]:
        edges: Set[Tuple[int, int]] = set()
        if num_nodes <= 1 or self.expander_degree <= 0:
            return edges
        rng = random.Random(graph_seed)
        nodes = list(range(num_nodes))
        for _ in range(self.expander_degree):
            perm = nodes.copy()
            rng.shuffle(perm)
            if len(perm) == 1:
                continue
            for idx in range(len(perm)):
                u = perm[idx]
                v = perm[(idx + 1) % len(perm)]
                if u == v:
                    continue
                edges.add((u, v))
                edges.add((v, u))
        return edges

    def _build_allowed_mask(
        self,
        num_nodes: int,
        edge_index_local: torch.Tensor,
        graph_seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        size = num_nodes + (1 if self.use_virtual_token else 0)
        allowed = torch.zeros((size, size), dtype=torch.bool, device=device)
        if num_nodes > 0:
            diag_idx = torch.arange(num_nodes, device=device)
            allowed[diag_idx, diag_idx] = True

        # Receiver attends to sender for each real edge u -> v.
        if edge_index_local.numel() > 0:
            u = edge_index_local[0]
            v = edge_index_local[1]
            allowed[v, u] = True

        # Add expander edges in both directions.
        for u, v in self._build_expander_edges(num_nodes, graph_seed):
            allowed[u, v] = True

        if self.use_virtual_token:
            token_idx = num_nodes
            allowed[:num_nodes, token_idx] = True
            allowed[token_idx, :num_nodes] = True
            allowed[token_idx, token_idx] = True

        attn_mask = torch.full((size, size), float("-inf"), device=device)
        attn_mask[allowed] = 0.0
        return attn_mask

    def _extract_local_edges(
        self,
        edge_index: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index.new_zeros((2, 0))
        mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        local_edge_index = edge_index[:, mask] - start
        return local_edge_index

    def _global_sparse_attention(self, x: torch.Tensor, edge_index: torch.Tensor, batch, ptr, graph_ids) -> torch.Tensor:
        out = torch.zeros_like(x)
        num_graphs = int(ptr.numel() - 1)
        for graph_pos in range(num_graphs):
            start = int(ptr[graph_pos].item())
            end = int(ptr[graph_pos + 1].item())
            x_g = x[start:end]
            if x_g.size(0) == 0:
                continue
            local_edge_index = self._extract_local_edges(edge_index, start, end)
            graph_seed = int(graph_ids[graph_pos].item()) + self.layer_seed * 100003
            if self.use_virtual_token:
                token = x_g.mean(dim=0, keepdim=True)
                seq = torch.cat([x_g, token], dim=0).unsqueeze(0)
            else:
                seq = x_g.unsqueeze(0)

            attn_mask = self._build_allowed_mask(
                num_nodes=x_g.size(0),
                edge_index_local=local_edge_index,
                graph_seed=graph_seed,
                device=x.device,
            )
            attn_out, _ = self.attn(seq, seq, seq, attn_mask=attn_mask, need_weights=False)
            out[start:end] = attn_out.squeeze(0)[: x_g.size(0)]
        return out

    def forward(self, x, edge_index, edge_attr, batch, ptr, graph_ids):
        h_local = self.local_conv(x, edge_index, edge_attr=edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm_local(h_local + x)

        h_global = self._global_sparse_attention(x, edge_index, batch, ptr, graph_ids)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        h_global = self.norm_global(h_global + x)

        out = h_local + h_global
        out = self.norm_ffn(out + self.ffn(out))
        return out


class ExphormerRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        expander_degree: int = 2,
        use_virtual_token: bool = True,
    ) -> None:
        super().__init__()
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
        self.layers = nn.ModuleList(
            [
                ExphormerBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    expander_degree=expander_degree,
                    use_virtual_token=use_virtual_token,
                    layer_seed=i,
                )
                for i in range(num_layers)
            ]
        )
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
        base = data.ptr[:-1] if hasattr(data, "ptr") else 0
        if isinstance(base, int):
            return data.src.view(-1), data.dst.view(-1)
        return base + data.src.view(-1), base + data.dst.view(-1)

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        ptr = data.ptr if hasattr(data, "ptr") else torch.tensor([0, data.num_nodes], device=x.device)
        graph_ids = data.graph_id.view(-1) if hasattr(data, "graph_id") else torch.arange(ptr.numel() - 1, device=x.device)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr, data.batch, ptr, graph_ids)

        graph_emb = global_mean_pool(x, data.batch)
        src_idx, dst_idx = self._src_dst_global_indices(data)
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]
        pair_emb = torch.cat(
            [graph_emb, src_emb, dst_emb, src_emb - dst_emb, src_emb * dst_emb],
            dim=-1,
        )
        return self.reg_head(pair_emb).view(-1)
