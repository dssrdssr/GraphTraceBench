from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

OP_TEXT = {0: "+", 1: "-", 2: "*", 3: "/"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serialize single-path or BFS arithmetic graph .pt datasets to JSONL for training/eval."
    )
    p.add_argument("--input", type=str, required=True, help="Path to .pt dataset")
    p.add_argument("--output", type=str, required=True, help="Path to output .jsonl")
    p.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--style", type=str, default="compact", choices=["compact", "verbose"])
    p.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "sft_chat", "prompt_completion"],
        help="eval for testing, sft_chat for chat fine-tuning, prompt_completion for legacy SFT",
    )
    p.add_argument("--task", type=str, default="auto", choices=["auto", "single_path", "bfs"])
    p.add_argument("--include-target", action="store_true", help="Attach target in eval output")
    p.add_argument("--include-system", action="store_true", help="Add a system message in sft_chat mode")
    p.add_argument("--answer-style", type=str, default="json", choices=["json", "plain"])
    p.add_argument("--keep-metadata", action="store_true", help="Keep metadata fields in training outputs")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def torch_load_compat(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ints(xs: Sequence[Any]) -> List[int]:
    return [int(x) for x in xs]


def detect_task(payload: Dict[str, Any], graph: Dict[str, Any], forced: str) -> str:
    if forced != "auto":
        return forced
    meta = payload.get("metadata", {})
    if meta.get("task") == "bfs_discovery_regression":
        return "bfs"
    if "bfs_tree_edge_indices" in graph or "bfs_order" in graph:
        return "bfs"
    return "single_path"


def answer_text(target: int, style: str) -> str:
    if style == "plain":
        return str(int(target))
    return json.dumps({"answer": int(target)}, ensure_ascii=False)


def node_values_text(vals: Sequence[Any]) -> str:
    return "; ".join(f"{i}:{int(v)}" for i, v in enumerate(vals))


def edges_text(edge_index: Sequence[Sequence[Any]], edge_ops: Sequence[Any]) -> str:
    us = ints(edge_index[0])
    vs = ints(edge_index[1])
    ops = ints(edge_ops)
    return "; ".join(f"{u}->{v}:{OP_TEXT[op]}" for u, v, op in zip(us, vs, ops))


def build_prompt(graph: Dict[str, Any], task: str, style: str, answer_style: str) -> str:
    nodes = node_values_text(graph["node_values"])
    edges = edges_text(graph["edge_index"], graph["edge_ops"])
    ans_req = '{"answer": integer}' if answer_style == "json" else "the integer answer only"

    if task == "single_path":
        src = int(graph["src"])
        dst = int(graph["dst"])
        hops = max(len(graph.get("path_nodes", [])) - 1, 0)
        if style == "verbose":
            return (
                "You are solving a directed arithmetic graph problem.\n"
                "There is exactly one valid directed path from src to dst.\n"
                "Initialize the accumulator with value(src).\n"
                "Traverse the unique directed path from src to dst.\n"
                "At each traversed edge, apply the edge operator to the current accumulator and the destination node value.\n"
                "Evaluate strictly left-to-right. Division is exact integer division.\n"
                f"Return ONLY {ans_req}.\n\n"
                f"num_nodes: {int(graph['num_nodes'])}\n"
                f"src: {src}\n"
                f"dst: {dst}\n"
                f"path_hops: {hops}\n"
                f"node_values: {nodes}\n"
                f"edges: {edges}\n"
            )
        return (
            "Compute the graph answer. There is exactly one directed path from src to dst. "
            "Start with value(src), traverse that unique path, and on each traversed edge apply the edge operator with the destination node value, left-to-right. "
            f"Division is exact integer division. Output ONLY {ans_req}.\n"
            f"src={src}; dst={dst}; path_hops={hops}; num_nodes={int(graph['num_nodes'])}; "
            f"node_values=[{nodes}]; edges=[{edges}]"
        )

    src = int(graph["src"])
    depth = int(graph.get("bfs_depth", len(graph.get("bfs_layers", [])) - 1))
    if style == "verbose":
        return (
            "You are solving a directed arithmetic graph problem with BFS execution.\n"
            "Run standard breadth-first search (BFS) from src up to bfs_depth layers using a FIFO queue.\n"
            "When expanding a node, inspect outgoing edges in the exact order they appear in the edge list.\n"
            "A node is discovered only the first time it is reached.\n"
            "Initialize the accumulator with value(src).\n"
            "For each newly discovered node within bfs_depth, in BFS first-discovery order, apply the operator on its discovery edge to the current accumulator and that discovered node's value.\n"
            "Division is exact integer division.\n"
            f"Return ONLY {ans_req}.\n\n"
            f"num_nodes: {int(graph['num_nodes'])}\n"
            f"src: {src}\n"
            f"bfs_depth: {depth}\n"
            f"node_values: {nodes}\n"
            f"edges: {edges}\n"
        )
    return (
        "Compute the BFS graph answer. Run standard BFS from src up to bfs_depth layers using a FIFO queue. "
        "When expanding a node, inspect outgoing edges in the exact order shown in the edge list. A node counts only at first discovery. "
        "Start with value(src). For each newly discovered node within bfs_depth, in BFS first-discovery order, apply the discovery-edge operator with that node's value. "
        f"Division is exact integer division. Output ONLY {ans_req}.\n"
        f"src={src}; bfs_depth={depth}; num_nodes={int(graph['num_nodes'])}; node_values=[{nodes}]; edges=[{edges}]"
    )


def system_prompt() -> str:
    return (
        "You are a careful graph reasoning assistant. Follow the problem instructions exactly and output only the requested answer format."
    )


def base_meta(graph: Dict[str, Any], task: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "id": int(graph["graph_id"]),
        "task_type": task,
        "num_nodes": int(graph["num_nodes"]),
        "src": int(graph["src"]),
        "target": int(graph["y"]),
    }
    if "dst" in graph:
        m["dst"] = int(graph["dst"])
    if task == "single_path":
        m["path_hops"] = max(len(graph.get("path_nodes", [])) - 1, 0)
    else:
        m["bfs_depth"] = int(graph.get("bfs_depth", len(graph.get("bfs_layers", [])) - 1))
        m["bfs_seq_len"] = max(len(graph.get("bfs_order", [])) - 1, 0)
    return m


def render_record(
    graph: Dict[str, Any],
    task: str,
    prompt: str,
    mode: str,
    answer_style: str,
    include_target: bool,
    include_system: bool,
    keep_metadata: bool,
) -> Dict[str, Any]:
    target = int(graph["y"])
    answer = answer_text(target, answer_style)
    meta = base_meta(graph, task)

    if mode == "eval":
        rec: Dict[str, Any] = {
            "id": meta["id"],
            "task_type": meta["task_type"],
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "answer_format": '{"answer": integer}' if answer_style == "json" else "integer",
        }
        for k in ["num_nodes", "src", "dst", "path_hops", "bfs_depth", "bfs_seq_len"]:
            if k in meta:
                rec[k] = meta[k]
        if include_target:
            rec["target"] = target
        return rec

    messages = []
    if include_system:
        messages.append({"role": "system", "content": system_prompt()})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": answer})

    if mode == "sft_chat":
        rec = {"messages": messages}
        if keep_metadata:
            rec.update(meta)
        return rec

    rec = {"prompt": prompt, "completion": answer}
    if keep_metadata:
        rec.update(meta)
    return rec


def select_graphs(payload: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    graphs = payload["graphs"]
    if split == "all":
        return list(graphs)
    return [graphs[int(i)] for i in payload["splits"][split]]


def main() -> None:
    args = parse_args()
    payload = torch_load_compat(args.input)
    graphs = select_graphs(payload, args.split)
    if args.max_samples is not None:
        graphs = graphs[: args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for graph in graphs:
            task = detect_task(payload, graph, args.task)
            prompt = build_prompt(graph, task, args.style, args.answer_style)
            record = render_record(
                graph=graph,
                task=task,
                prompt=prompt,
                mode=args.mode,
                answer_style=args.answer_style,
                include_target=args.include_target,
                include_system=args.include_system,
                keep_metadata=args.keep_metadata,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(graphs)} records to {out}")


if __name__ == "__main__":
    main()
