from __future__ import annotations

from typing import Dict, List

from torch import nn

from .difformer_like import DIFFormerRegressor
from .g2pt_like import G2PTRegressor
from .graphgps import GraphGPSRegressor
from .exphormer import ExphormerRegressor
from .graphgpt_like import GraphGPTRegressor
from .graphormer_like import GraphormerRegressor
from .grit_like import GRITRegressor
from .patchgt_like import PatchGTRegressor
from .tigt_like import TIGTRegressor


REGISTRY: Dict[str, type[nn.Module]] = {
    "graphgps": GraphGPSRegressor,
    "exphormer": ExphormerRegressor,
    "graphormer": GraphormerRegressor,
    "grit": GRITRegressor,
    "difformer": DIFFormerRegressor,
    "diffformer": DIFFormerRegressor,
    "tigt": TIGTRegressor,
    "patchgt": PatchGTRegressor,
    "graphgpt": GraphGPTRegressor,
    "g2pt": G2PTRegressor,
}


def available_models() -> List[str]:
    return sorted(REGISTRY.keys())


def make_model(model_name: str, meta: Dict, args) -> nn.Module:
    name = model_name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {available_models()}")
    cls = REGISTRY[name]
    common = dict(
        in_channels=int(meta["num_node_features"]),
        hidden_channels=int(getattr(args, "hidden_channels", 128)),
        num_layers=int(getattr(args, "num_layers", 6)),
        heads=int(getattr(args, "heads", 4)),
        dropout=float(getattr(args, "dropout", 0.1)),
        out_channels=1,
    )
    kwargs = dict(common)
    if name == "graphgps":
        kwargs["edge_dim"] = int(meta.get("num_edge_features", 4))
    if name == "exphormer":
        kwargs["expander_degree"] = int(getattr(args, "expander_degree", 4))
        kwargs["num_virtual_nodes"] = int(getattr(args, "num_virtual_nodes", 1))
    if name == "graphormer":
        kwargs["max_dist"] = int(getattr(args, "max_dist", 8))
        kwargs["num_edge_types"] = int(meta.get("num_edge_features", 4))
    if name == "grit":
        kwargs["rw_steps"] = int(getattr(args, "rw_steps", 4))
        kwargs["max_dist"] = int(getattr(args, "max_dist", 8))
    if name in {"difformer", "diffformer"}:
        kwargs["diff_steps"] = int(getattr(args, "diff_steps", 3))
    if name == "tigt":
        kwargs["topo_rw_steps"] = int(getattr(args, "topo_rw_steps", 4))
        kwargs["max_dist"] = int(getattr(args, "max_dist", 8))
    if name == "patchgt":
        kwargs["patch_ratio"] = float(getattr(args, "patch_ratio", 0.25))
        kwargs["min_patches"] = int(getattr(args, "min_patches", 2))
    if name in {"graphgpt", "g2pt"}:
        kwargs["max_seq_len"] = int(getattr(args, "max_seq_len", 256))
    return cls(**kwargs)
