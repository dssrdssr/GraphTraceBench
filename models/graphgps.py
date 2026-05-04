from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool

class GraphGPSRegressor(nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 128, num_layers: int = 6, heads: int = 4, dropout: float = 0.1, out_channels: int = 1, attn_type: str = 'multihead'):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(GPSConv(channels=hidden_channels, conv=GINEConv(local_nn, edge_dim=edge_dim), heads=heads, dropout=dropout, attn_type=attn_type, norm='batch_norm'))
        self.readout = nn.Sequential(nn.Linear(hidden_channels*3, hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels//2, out_channels))
    def forward(self, data):
        x = self.node_encoder(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.batch, edge_attr=data.edge_attr.to(x.dtype))
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, data.batch)
        src = data.ptr[:-1] + data.src_id.view(-1)
        dst = data.ptr[:-1] + data.dst_id.view(-1)
        return self.readout(torch.cat([pooled, x[src], x[dst]], dim=-1))
