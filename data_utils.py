from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def dict_graph_to_data(g: dict) -> Data:
    node_values = torch.tensor(g["node_values"], dtype=torch.float).view(-1, 1)
    num_nodes = node_values.size(0)
    src = int(g["src"])
    dst = int(g["dst"])
    src_flag = torch.zeros((num_nodes, 1), dtype=torch.float)
    dst_flag = torch.zeros((num_nodes, 1), dtype=torch.float)
    src_flag[src, 0] = 1.0
    dst_flag[dst, 0] = 1.0
    x = torch.cat([node_values, src_flag, dst_flag], dim=1)

    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_type = torch.tensor(g.get("edge_ops", g.get("edge_type", [])), dtype=torch.long)
    if edge_type.numel() == 0 and "edge_attr" in g:
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float)
        edge_type = edge_attr.argmax(dim=-1).long()
    edge_attr = F.one_hot(edge_type, num_classes=4).float()
    path_nodes = torch.tensor(g.get("path_nodes", []), dtype=torch.long)
    path_edge_indices = set(g.get("path_edge_indices", []))
    path_edge_mask = torch.tensor([i in path_edge_indices for i in range(edge_type.numel())], dtype=torch.bool)
    path_len = len(g.get("path_nodes", [])) - 1 if g.get("path_nodes") else -1

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        y=torch.tensor([float(g["y"])], dtype=torch.float),
        src_id=torch.tensor([src], dtype=torch.long),
        dst_id=torch.tensor([dst], dtype=torch.long),
        path_nodes=path_nodes,
        path_edge_mask=path_edge_mask,
        path_len=torch.tensor([path_len], dtype=torch.long),
        graph_id=torch.tensor([int(g.get("graph_id", 0))], dtype=torch.long),
    )


def to_data_list(graphs: List) -> List[Data]:
    out: List[Data] = []
    for g in graphs:
        if isinstance(g, Data):
            out.append(g)
        elif isinstance(g, dict):
            out.append(dict_graph_to_data(g))
        else:
            raise TypeError(f"Unsupported graph type: {type(g)}")
    return out


def infer_meta(payload: dict, graphs: List[Data]) -> Dict:
    meta = payload.get("meta") or payload.get("metadata") or {}
    meta = dict(meta)
    if graphs:
        sample = graphs[0]
        meta.setdefault("num_node_features", int(sample.x.size(-1)))
        meta.setdefault("num_edge_features", int(sample.edge_attr.size(-1)))
        path_lens = [int(g.path_len.view(-1)[0].item()) for g in graphs if hasattr(g, "path_len")]
        if path_lens:
            meta.setdefault("max_path_len", max(path_lens))
    return meta


def load_dataset_splits(dataset_path: str, split_mode: str = "train_val_test") -> Tuple[List[Data], List[Data], List[Data], Dict]:
    payload = torch.load(dataset_path)
    graphs = to_data_list(payload["graphs"])
    meta = infer_meta(payload, graphs)

    if split_mode == "all_as_test":
        return [], [], graphs, meta

    if "splits" in payload:
        splits = payload["splits"]
        train_graphs = [graphs[i] for i in splits.get("train", [])]
        val_graphs = [graphs[i] for i in splits.get("val", [])]
        test_graphs = [graphs[i] for i in splits.get("test", [])]
        return train_graphs, val_graphs, test_graphs, meta

    n = len(graphs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train : n_train + n_val]
    test_graphs = graphs[n_train + n_val :]
    return train_graphs, val_graphs, test_graphs, meta


def make_loader(graphs: List[Data], batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
