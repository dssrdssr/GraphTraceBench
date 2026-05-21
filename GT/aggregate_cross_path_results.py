#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TRAIN_FILE_RE = re.compile(r"path_(\d+)_epoch_(\d+)_results\.json$")
TEST_FILE_RE = re.compile(r"test_path_(\d+)\.json$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="./runs")
    p.add_argument("--cross-dir", type=str, default="./results/cross_path")
    p.add_argument("--output-dir", type=str, default="./results/tables")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--round", type=int, default=3, dest="round_digits")
    return p.parse_args()


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def collect_train_results(runs_dir: Path, models: Optional[List[str]], epoch_filter: Optional[int]) -> Dict[str, Dict[int, dict]]:
    train_map: Dict[str, Dict[int, dict]] = {}
    if not runs_dir.exists():
        print(f"[WARN] runs dir not found: {runs_dir}")
        return train_map

    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if models:
        model_dirs = [d for d in model_dirs if d.name in models]

    for model_dir in sorted(model_dirs):
        model = model_dir.name
        train_map.setdefault(model, {})

        for file in sorted(model_dir.glob("path_*_epoch_*_results.json")):
            m = TRAIN_FILE_RE.search(file.name)
            if not m:
                continue
            train_path = int(m.group(1))
            epoch = int(m.group(2))
            if epoch_filter is not None and epoch != epoch_filter:
                continue

            payload = safe_read_json(file)
            if not payload:
                continue

            train_mae = payload.get("trained_train", {}).get("mae")
            train_map[model][train_path] = {
                "train_mae": train_mae,
                "epoch": epoch,
                "file": str(file),
            }

    return train_map


def collect_test_results(cross_dir: Path, models: Optional[List[str]]) -> List[dict]:
    rows: List[dict] = []
    if not cross_dir.exists():
        print(f"[WARN] cross dir not found: {cross_dir}")
        return rows

    model_dirs = [d for d in cross_dir.iterdir() if d.is_dir()]
    if models:
        model_dirs = [d for d in model_dirs if d.name in models]

    for model_dir in sorted(model_dirs):
        model = model_dir.name
        for train_dir in sorted(model_dir.glob("train_path_*")):
            m_train = re.search(r"train_path_(\d+)", train_dir.name)
            if not m_train:
                continue
            train_path = int(m_train.group(1))

            for test_file in sorted(train_dir.glob("test_path_*.json")):
                m_test = TEST_FILE_RE.search(test_file.name)
                if not m_test:
                    continue
                test_path = int(m_test.group(1))

                payload = safe_read_json(test_file)
                if not payload:
                    continue

                metrics = payload.get("trained_test", {})
                rows.append(
                    {
                        "Model": model,
                        "Train Path": train_path,
                        "Test Path": test_path,
                        "Test MAE": metrics.get("mae"),
                        "Test RMSE": metrics.get("rmse"),
                        "Acc": metrics.get("acc_round", metrics.get("acc_tol")),
                        "test_file": str(test_file),
                    }
                )
    return rows


def build_long_table(train_map: Dict[str, Dict[int, dict]], test_rows: List[dict]) -> pd.DataFrame:
    rows = []
    for row in test_rows:
        model = row["Model"]
        train_path = row["Train Path"]
        train_info = train_map.get(model, {}).get(train_path, {})
        rows.append(
            {
                "Model": model,
                "Train Path": train_path,
                "Test Path": row["Test Path"],
                "Train MAE": train_info.get("train_mae"),
                "Test MAE": row["Test MAE"],
                "Test RMSE": row["Test RMSE"],
                "Acc": row["Acc"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Model", "Train Path", "Test Path"]).reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, output_dir: Path, round_digits: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    out_df = df.copy()
    for col in ["Train MAE", "Test MAE", "Test RMSE", "Acc"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].round(round_digits)

    long_csv = output_dir / "summary_long.csv"
    table_csv = output_dir / "summary_table.csv"
    long_xlsx = output_dir / "summary_long.xlsx"
    table_xlsx = output_dir / "summary_table.xlsx"

    out_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    out_df.to_csv(table_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(long_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="summary_long")

    with pd.ExcelWriter(table_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="summary_table")

    print(f"[OK] wrote: {long_csv}")
    print(f"[OK] wrote: {table_csv}")
    print(f"[OK] wrote: {long_xlsx}")
    print(f"[OK] wrote: {table_xlsx}")


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    cross_dir = Path(args.cross_dir)
    output_dir = Path(args.output_dir)

    train_map = collect_train_results(runs_dir, args.models, args.epoch)
    test_rows = collect_test_results(cross_dir, args.models)
    df = build_long_table(train_map, test_rows)

    if df.empty:
        print("[WARN] No rows collected. Check your runs/ and results/cross_path/ structure.")
        return

    write_outputs(df, output_dir, args.round_digits)
    print("\nPreview:")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
