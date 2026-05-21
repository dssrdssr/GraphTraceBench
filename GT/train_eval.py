from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_utils import load_dataset_splits, make_loader
from models import available_models, make_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate graph models on the arithmetic path regression task.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a .pt dataset file")
    parser.add_argument("--split", type=str, default="train_val_test", choices=["train_val_test", "all_as_test"])
    parser.add_argument("--models", type=str, nargs="+", required=True, choices=available_models())
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Unused in this final-only checkpoint version; training always runs full epochs.")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--abs-tol", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--expander-degree", type=int, default=4)
    parser.add_argument("--num-virtual-nodes", type=int, default=1)
    parser.add_argument("--max-dist", type=int, default=8)
    parser.add_argument("--rw-steps", type=int, default=4)
    parser.add_argument("--diff-steps", type=int, default=3)
    parser.add_argument("--topo-rw-steps", type=int, default=4)
    parser.add_argument("--patch-ratio", type=float, default=0.25)
    parser.add_argument("--min-patches", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, abs_tol: float, labels_are_integer: bool = True) -> Dict[str, float]:
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


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_graphs = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(batch).view(-1)
            target = batch.y.view(-1)
            loss = criterion(pred, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    fieldnames = sorted({k for row in history for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def checkpoint_name(max_path_len: int, epoch: int) -> str:
    return f"path_{max_path_len}_epoch_{epoch}.pt"


def run_model(args, model_name: str, train_graphs, val_graphs, test_graphs, meta: Dict, output_dir: Path) -> Dict:
    device = torch.device(args.device)
    train_loader = make_loader(train_graphs, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_graphs, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_graphs, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(model_name, meta, args).to(device)
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    untrained_test = evaluate(model, test_loader, device, args.abs_tol, labels_are_integer=True)
    untrained_val = evaluate(model, val_loader, device, args.abs_tol, labels_are_integer=True)

    if not train_graphs:
        results = {
            "model": model_name,
            "dataset": args.dataset,
            "best_epoch": None,
            "untrained_test": untrained_test,
            "untrained_val": untrained_val,
            "trained_train": {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "acc_tol": math.nan, "acc_round": math.nan},
            "trained_val": {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "acc_tol": math.nan, "acc_round": math.nan},
            "trained_test": untrained_test,
            "meta": meta,
            "args": vars(args),
        }
        results_path = run_dir / f"path_{int(meta.get('max_path_len', -1))}_epoch_{args.epochs}_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    final_epoch = args.epochs
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
        train_metrics = evaluate(model, train_loader, device, args.abs_tol, labels_are_integer=True)
        val_metrics = evaluate(model, val_loader, device, args.abs_tol, labels_are_integer=True)
        test_metrics = evaluate(model, test_loader, device, args.abs_tol, labels_are_integer=True)
        scheduler.step(val_metrics["mae"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_acc": train_metrics["acc_round"],
            "val_acc": val_metrics["acc_round"],
            "test_acc": test_metrics["acc_round"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[{model_name}] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_mae={val_metrics['mae']:.6f} test_mae={test_metrics['mae']:.6f} test_acc={test_metrics['acc_round']:.4f}"
        )

    ckpt = {
        "model_name": model_name,
        "model_state": model.state_dict(),
        "epoch": final_epoch,
        "args": vars(args),
        "meta": meta,
    }
    named_path = run_dir / checkpoint_name(int(meta.get("max_path_len", -1)), final_epoch)
    torch.save(ckpt, named_path)
    trained_train = evaluate(model, train_loader, device, args.abs_tol, labels_are_integer=True)
    trained_val = evaluate(model, val_loader, device, args.abs_tol, labels_are_integer=True)
    trained_test = evaluate(model, test_loader, device, args.abs_tol, labels_are_integer=True)

    save_history(history, run_dir / "history.csv")
    results = {
        "model": model_name,
        "dataset": args.dataset,
        "best_epoch": final_epoch,
        "untrained_test": untrained_test,
        "untrained_val": untrained_val,
        "trained_train": trained_train,
        "trained_val": trained_val,
        "trained_test": trained_test,
        "meta": meta,
        "args": vars(args),
    }
    results_path = run_dir / f"path_{int(meta.get('max_path_len', -1))}_epoch_{args.epochs}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_graphs, val_graphs, test_graphs, meta = load_dataset_splits(args.dataset, split_mode=args.split)
    all_results = []
    for model_name in args.models:
        all_results.append(run_model(args, model_name.lower(), train_graphs, val_graphs, test_graphs, meta, output_dir))

    summary = {
        r["model"]: {
            "best_epoch": r["best_epoch"],
            "untrained_test": r["untrained_test"],
            "trained_test": r["trained_test"],
        }
        for r in all_results
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
