from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import SequenceReadout, batch_to_graph_slices


class G2PTRegressor(nn.Module):
    """Task-adapted G2PT-style regressor.

    G2PT represents graphs as node and edge sequences and applies a Transformer.
    Here we use a supervised encoder-only version for arithmetic path regression.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
        max_seq_len: int = 512,
        **_: dict,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        token_dim = in_channels * 2 + 4
        self.token_proj = nn.Linear(token_dim, hidden_channels)
        self.token_type = nn.Embedding(5, hidden_channels)  # cls, src, dst, node, edge
        self.pos_embed = nn.Embedding(max_seq_len, hidden_channels)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.readout = SequenceReadout(hidden_channels, out_channels=out_channels, dropout=dropout)

    def _serialize(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, src_id: int, dst_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat_dim = x.size(-1) * 2 + 4
        cls = x.new_zeros((1, feat_dim))
        src_tok = torch.cat([x[src_id : src_id + 1], x.new_zeros((1, 4)), x.new_zeros((1, x.size(-1)))], dim=-1)
        dst_tok = torch.cat([x[dst_id : dst_id + 1], x.new_zeros((1, 4)), x.new_zeros((1, x.size(-1)))], dim=-1)
        node_tokens = [torch.cat([x[i : i + 1], x.new_zeros((1, 4)), x.new_zeros((1, x.size(-1)))], dim=-1) for i in range(x.size(0))]
        edge_tokens = []
        for e in range(edge_index.size(1)):
            u = int(edge_index[0, e].item())
            v = int(edge_index[1, e].item())
            op = F.one_hot(edge_type[e].long(), num_classes=4).to(x.dtype).view(1, -1)
            edge_tokens.append(torch.cat([x[u : u + 1], op, x[v : v + 1]], dim=-1))
        seq = torch.cat([cls, src_tok, dst_tok] + node_tokens + edge_tokens, dim=0)
        type_ids = torch.tensor(
            [0, 1, 2] + [3] * len(node_tokens) + [4] * len(edge_tokens),
            device=x.device,
            dtype=torch.long,
        )
        return seq[: self.max_seq_len], type_ids[: self.max_seq_len]

    def _forward_graph(self, graph) -> torch.Tensor:
        seq, type_ids = self._serialize(graph.x, graph.edge_index, graph.edge_type, graph.src_id, graph.dst_id)
        tok = self.token_proj(seq)
        pos = torch.arange(tok.size(0), device=tok.device)
        tok = tok + self.token_type(type_ids) + self.pos_embed(pos)
        out = self.encoder(tok.unsqueeze(0))
        return self.readout(out[:, 0, :])

    def forward(self, data):
        outs = [self._forward_graph(g) for g in batch_to_graph_slices(data)]
        return torch.cat(outs, dim=0)
