from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class GraphSlice:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    src_id: int
    dst_id: int
    graph_id: int


def get_edge_type(data: Data) -> torch.Tensor:
    if hasattr(data, "edge_type"):
        return data.edge_type.long()
    if hasattr(data, "edge_attr"):
        return data.edge_attr.argmax(dim=-1).long()
    return torch.zeros(data.edge_index.size(1), dtype=torch.long, device=data.edge_index.device)


def batch_to_graph_slices(data: Data) -> List[GraphSlice]:
    edge_type = get_edge_type(data)
    ptr = data.ptr
    out: List[GraphSlice] = []
    for g in range(int(data.num_graphs)):
        start = int(ptr[g].item())
        end = int(ptr[g + 1].item())
        node_mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end)
        local_edge_index = data.edge_index[:, node_mask] - start
        local_edge_type = edge_type[node_mask]
        out.append(
            GraphSlice(
                x=data.x[start:end],
                edge_index=local_edge_index,
                edge_type=local_edge_type,
                src_id=int(data.src_id[g].item()),
                dst_id=int(data.dst_id[g].item()),
                graph_id=int(data.graph_id[g].item()) if hasattr(data, "graph_id") else g,
            )
        )
    return out


def build_dense_adj(num_nodes: int, edge_index: torch.Tensor, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def build_edge_type_matrix(num_nodes: int, edge_index: torch.Tensor, edge_type: torch.Tensor, default_type: int) -> torch.Tensor:
    mat = torch.full((num_nodes, num_nodes), fill_value=default_type, device=edge_index.device, dtype=torch.long)
    if edge_index.numel() > 0:
        mat[edge_index[0], edge_index[1]] = edge_type.long()
    return mat


def row_normalize(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mat.sum(dim=-1, keepdim=True).clamp_min(eps)
    return mat / denom


def shortest_path_distances(num_nodes: int, edge_index: torch.Tensor, max_dist: int = 8) -> torch.Tensor:
    device = edge_index.device
    inf = max_dist + 1
    dist = torch.full((num_nodes, num_nodes), fill_value=inf, device=device, dtype=torch.long)
    diag = torch.arange(num_nodes, device=device)
    dist[diag, diag] = 0
    if edge_index.numel() > 0:
        dist[edge_index[0], edge_index[1]] = 1
    for k in range(num_nodes):
        dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
    dist = dist.clamp_max(inf)
    return dist


def random_walk_landing_probs(adj: torch.Tensor, steps: int = 4) -> torch.Tensor:
    n = adj.size(0)
    if n == 0:
        return adj.new_zeros((0, steps))
    p = row_normalize(adj + torch.eye(n, device=adj.device, dtype=adj.dtype))
    cur = p
    outs = []
    for _ in range(steps):
        outs.append(torch.diagonal(cur, dim1=0, dim2=1))
        cur = cur @ p
    return torch.stack(outs, dim=-1)


def simple_topology_features(adj: torch.Tensor) -> torch.Tensor:
    deg_out = adj.sum(dim=-1)
    deg_in = adj.sum(dim=0)
    und = ((adj + adj.t()) > 0).float()
    tri = torch.diagonal(und @ und @ und, dim1=0, dim2=1) / 2.0
    deg_und = und.sum(dim=-1)
    clustering = tri / (deg_und * (deg_und - 1)).clamp_min(1.0)
    return torch.stack([deg_in, deg_out, clustering], dim=-1)


def kmeans_torch(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    n = x.size(0)
    if n == 0:
        return torch.zeros((0,), dtype=torch.long, device=x.device)
    k = max(1, min(k, n))
    if k == 1:
        return torch.zeros((n,), dtype=torch.long, device=x.device)
    centroids = x[:k].clone()
    for _ in range(iters):
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=-1)
        assign = dist.argmin(dim=-1)
        new_centroids = []
        for i in range(k):
            mask = assign == i
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids, dim=0)
    return assign

def safe_eigh(lap: torch.Tensor):
    device = lap.device

    # 保证严格对称
    lap = 0.5 * (lap + lap.transpose(-1, -2))

    # 用 double + CPU 做特征分解，更稳定
    lap_cpu = lap.detach().to(device="cpu", dtype=torch.float64)

    # 加一个极小扰动，缓解重复特征值/病态矩阵
    eps = 1e-6
    eye = torch.eye(lap_cpu.size(0), dtype=lap_cpu.dtype, device=lap_cpu.device)

    try:
        evals, evecs = torch.linalg.eigh(lap_cpu)
    except torch._C._LinAlgError:
        evals, evecs = torch.linalg.eigh(lap_cpu + eps * eye)

    return evals.to(device), evecs.to(device)

def spectral_patch_partition(adj: torch.Tensor, num_patches: int) -> torch.Tensor:
    n = adj.size(0)
    if n <= 1:
        return torch.zeros((n,), dtype=torch.long, device=adj.device)
    num_patches = max(1, min(num_patches, n))
    if num_patches == 1:
        return torch.zeros((n,), dtype=torch.long, device=adj.device)
    und = ((adj + adj.t()) > 0).float()
    deg = und.sum(dim=-1)
    lap = torch.diag(deg) - und
    evals, evecs = safe_eigh(lap)
    coords = evecs[:, 1 : 1 + min(num_patches, n - 1)]
    if coords.numel() == 0:
        coords = torch.arange(n, device=adj.device, dtype=adj.dtype).unsqueeze(-1)
    return kmeans_torch(coords, num_patches)


def pool_by_assignment(x: torch.Tensor, assignment: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = x.new_zeros((num_groups, x.size(-1)))
    counts = x.new_zeros((num_groups, 1))
    out.index_add_(0, assignment, x)
    counts.index_add_(0, assignment, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / counts.clamp_min(1.0)


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, hidden_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.head_dim = hidden_channels // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor | None = None) -> torch.Tensor:
        n = x.size(0)
        q = self.q_proj(x).view(n, self.heads, self.head_dim)
        k = self.k_proj(x).view(n, self.heads, self.head_dim)
        v = self.v_proj(x).view(n, self.heads, self.head_dim)
        scores = torch.einsum("ihd,jhd->hij", q, k) * self.scale
        if pair_bias is not None:
            if pair_bias.dim() == 2:
                scores = scores + pair_bias.unsqueeze(0)
            elif pair_bias.dim() == 3 and pair_bias.size(0) == self.heads:
                scores = scores + pair_bias
            elif pair_bias.dim() == 3 and pair_bias.size(-1) == self.heads:
                scores = scores + pair_bias.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported pair_bias shape: {tuple(pair_bias.shape)}")
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(n, self.hidden_channels)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attn = PairBiasedSelfAttention(hidden_channels, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), pair_bias=pair_bias))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class QueryReadout(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, src_id: int, dst_id: int) -> torch.Tensor:
        pooled = x.mean(dim=0, keepdim=True)
        src = x[src_id : src_id + 1]
        dst = x[dst_id : dst_id + 1]
        return self.mlp(torch.cat([pooled, src, dst], dim=-1))


class SequenceReadout(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, token_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(token_state)


def encode_edge_tokens(node_x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node_x.new_zeros((0, node_x.size(-1) * 2 + 4))
    onehot = F.one_hot(edge_type.long(), num_classes=4).to(node_x.dtype)
    return torch.cat([node_x[edge_index[0]], onehot, node_x[edge_index[1]]], dim=-1)


def dfs_edge_order(num_nodes: int, edge_index: torch.Tensor, src: int) -> List[int]:
    adj: Dict[int, List[tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.size(1)):
        u = int(edge_index[0, e].item())
        v = int(edge_index[1, e].item())
        adj[u].append((v, e))
    for u in adj:
        adj[u].sort(key=lambda z: z[0])
    seen_edges = set()
    order: List[int] = []

    def dfs(u: int) -> None:
        for v, eidx in adj.get(u, []):
            if eidx in seen_edges:
                continue
            seen_edges.add(eidx)
            order.append(eidx)
            dfs(v)

    dfs(src)
    for e in range(edge_index.size(1)):
        if e not in seen_edges:
            order.append(e)
    return order
