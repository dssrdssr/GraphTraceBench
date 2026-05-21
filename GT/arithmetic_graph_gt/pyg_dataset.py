from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "PyG is required for ArithmeticPathPyGDataset. Install torch_geometric first."
    ) from exc


class ArithmeticPathPyGDataset(Dataset):
    """Wrap a saved arithmetic graph bundle as a PyG-ready dataset."""

    def __init__(self, bundle_path: str, split: str) -> None:
        super().__init__()
        bundle: Dict = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if split not in bundle["splits"]:
            raise KeyError(f"Unknown split={split}. Available: {list(bundle['splits'])}")
        self.bundle = bundle
        self.graphs: List[Dict] = bundle["graphs"]
        self.indices: List[int] = list(bundle["splits"][split])
        self.metadata: Dict = bundle["metadata"]
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        spec = self.graphs[self.indices[idx]]
        num_nodes = spec["num_nodes"]
        src = spec["src"]
        dst = spec["dst"]

        node_values = torch.tensor(spec["node_values"], dtype=torch.float32).unsqueeze(-1)
        is_src = torch.zeros((num_nodes, 1), dtype=torch.float32)
        is_dst = torch.zeros((num_nodes, 1), dtype=torch.float32)
        is_src[src, 0] = 1.0
        is_dst[dst, 0] = 1.0
        x = torch.cat([node_values, is_src, is_dst], dim=-1)

        edge_index = torch.tensor(spec["edge_index"], dtype=torch.long)
        op_ids = torch.tensor(spec["edge_ops"], dtype=torch.long)
        edge_attr = F.one_hot(op_ids, num_classes=4).to(torch.float32)
        y = torch.tensor([float(spec["y"])] , dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.op_id = op_ids
        data.src = torch.tensor([src], dtype=torch.long)
        data.dst = torch.tensor([dst], dtype=torch.long)
        data.graph_id = torch.tensor([spec["graph_id"]], dtype=torch.long)
        data.path_len = torch.tensor([len(spec["path_nodes"]) - 1], dtype=torch.long)
        return data
