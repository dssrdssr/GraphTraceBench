from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from torch import nn

from data_utils import load_dataset_splits, make_loader, to_data_list, infer_meta
from models import make_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a specified dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--abs-tol", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, abs_tol: float, labels_are_integer: bool = True):
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).view(-1).detach().cpu())
        targets.append(batch.y.view(-1).detach().cpu())
    if not preds:
        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "acc_tol": math.nan, "acc_round": math.nan}
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    diff = pred - target
    mae = diff.abs().mean().item()
    mse = diff.pow(2).mean().item()
    rmse = math.sqrt(mse)
    acc_tol = (diff.abs() <= abs_tol).float().mean().item()
    acc_round = (pred.round() == target.round()).float().mean().item() if labels_are_integer else math.nan
    return {"mae": mae, "mse": mse, "rmse": rmse, "acc_tol": acc_tol, "acc_round": acc_round}


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    train_graphs, val_graphs, test_graphs, meta = load_dataset_splits(args.dataset, split_mode="train_val_test")
    if args.split == "train":
        graphs = train_graphs
    elif args.split == "val":
        graphs = val_graphs
    elif args.split == "test":
        graphs = test_graphs
    else:
        graphs = train_graphs + val_graphs + test_graphs

    meta = ckpt.get("meta", meta)
    ckpt_args = SimpleNamespace(**ckpt.get("args", {}))
    model_name = ckpt["model_name"]
    model = make_model(model_name, meta, ckpt_args)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)

    loader = make_loader(graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate(model, loader, device, args.abs_tol, labels_are_integer=True)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "split": args.split,
        "model": model_name,
        "trained_test": metrics,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
