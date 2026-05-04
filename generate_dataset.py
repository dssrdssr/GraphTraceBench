from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
from collections import Counter 
import random
from typing import Dict, List, Sequence, Tuple

import torch

OPS: Tuple[str, ...] = ("add", "sub", "mul", "div")
OP_TO_ID: Dict[str, int] = {op: i for i, op in enumerate(OPS)}
ID_TO_SYMBOL: Dict[int, str] = {0: "+", 1: "-", 2: "*", 3: "/"}
SYMBOL_TO_OP: Dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
NAME_TO_SYMBOL: Dict[str, str] = {v: k for k, v in SYMBOL_TO_OP.items()}


@dataclass
class GraphSpec:
    graph_id: int
    num_nodes: int
    path_len: int
    src: int
    dst: int
    node_values: List[int]
    edge_index: List[List[int]]
    edge_ops: List[int]
    path_nodes: List[int]
    path_edge_indices: List[int]
    y: int

    def to_dict(self) -> Dict:
        return asdict(self)


class GenerationError(RuntimeError):
    pass


def parse_ops_spec(ops_spec: str) -> Tuple[int, ...]:
    normalized = ops_spec.replace(",", " ").strip()
    if not normalized:
        raise ValueError("--ops cannot be empty")

    tokens = normalized.split()
    if len(tokens) == 1 and all(ch in SYMBOL_TO_OP for ch in tokens[0]):
        raw_ops = list(tokens[0])
    else:
        raw_ops = []
        for token in tokens:
            if token in SYMBOL_TO_OP:
                raw_ops.append(token)
            elif token in OPS:
                raw_ops.append(NAME_TO_SYMBOL[token])
            else:
                raise ValueError(
                    f"Unknown op token '{token}'. Use symbols from + - * / or names from {OPS}."
                )

    allowed_op_ids: List[int] = []
    seen = set()
    for symbol in raw_ops:
        op_name = SYMBOL_TO_OP[symbol]
        op_id = OP_TO_ID[op_name]
        if op_id not in seen:
            seen.add(op_id)
            allowed_op_ids.append(op_id)

    if not allowed_op_ids:
        raise ValueError("At least one operator must be selected")

    return tuple(allowed_op_ids)


def parse_path_lens(path_len: int, path_lens: Sequence[int] | None) -> Tuple[int, ...]:
    """Resolve single/multiple path lengths from CLI arguments.

    --path-len keeps backward compatibility. If --path-lens is provided, it
    overrides --path-len and the dataset will mix those lengths.
    """
    resolved = list(path_lens) if path_lens else [path_len]
    if not resolved:
        raise ValueError("At least one path length must be provided")
    if any(length < 1 for length in resolved):
        raise ValueError("All path lengths must be >= 1")

    unique: List[int] = []
    seen = set()
    for length in resolved:
        if length not in seen:
            seen.add(length)
            unique.append(length)
    return tuple(unique)


def apply_op(lhs: int, rhs: int, op_id: int) -> int:
    if op_id == OP_TO_ID["add"]:
        return lhs + rhs
    if op_id == OP_TO_ID["sub"]:
        return lhs - rhs
    if op_id == OP_TO_ID["mul"]:
        return lhs * rhs
    if op_id == OP_TO_ID["div"]:
        if rhs == 0:
            raise ZeroDivisionError("division by zero in path generation")
        if lhs % rhs != 0:
            raise ValueError(f"unsafe integer division: {lhs} / {rhs}")
        return lhs // rhs
    raise ValueError(f"Unknown op_id={op_id}")


def count_paths(num_nodes: int, edge_index: Sequence[Sequence[int]], src: int, dst: int) -> int:
    """Count paths in a DAG-like graph via DFS with memoization.

    The generated graphs are acyclic by construction. This function is also used as a
    safety check after generation.
    """
    adjacency: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in zip(edge_index[0], edge_index[1]):
        adjacency[u].append(v)

    memo: Dict[int, int] = {}

    def dfs(node: int) -> int:
        if node == dst:
            return 1
        if node in memo:
            return memo[node]
        total = 0
        for nxt in adjacency[node]:
            total += dfs(nxt)
        memo[node] = total
        return total

    return dfs(src)


class ArithmeticGraphGenerator:
    def __init__(
        self,
        num_nodes: int,
        path_len: int,
        value_min: int = 1,
        value_max: int = 9,
        mul_abs_max: int = 3,
        max_abs_label: int = 256,
        extra_branch_edge_prob: float = 0.15,
        seed: int = 42,
        allowed_ops: Sequence[int] | None = None,
    ) -> None:
        if num_nodes < path_len + 1:
            raise ValueError("num_nodes must be >= path_len + 1")
        if value_min > value_max:
            raise ValueError("value_min must be <= value_max")
        self.num_nodes = num_nodes
        self.path_len = path_len
        self.value_min = value_min
        self.value_max = value_max
        self.mul_abs_max = max(1, mul_abs_max)
        self.max_abs_label = max_abs_label
        self.extra_branch_edge_prob = extra_branch_edge_prob
        self.rng = random.Random(seed)
        self.seed = seed

        if allowed_ops is None:
            allowed_ops = tuple(range(len(OPS)))
        self.allowed_ops = tuple(allowed_ops)
        if not self.allowed_ops:
            raise ValueError("allowed_ops cannot be empty")
        invalid_ops = [op_id for op_id in self.allowed_ops if op_id < 0 or op_id >= len(OPS)]
        if invalid_ops:
            raise ValueError(f"Invalid op ids in allowed_ops: {invalid_ops}")

        self.non_zero_candidates = [v for v in range(value_min, value_max + 1) if v != 0]
        if not self.non_zero_candidates:
            raise ValueError("Need at least one non-zero candidate node value")

    def _random_op(self) -> int:
        return self.rng.choice(self.allowed_ops)

    def _sample_safe_path_step(self, current_value: int) -> Tuple[int, int, int]:
        candidates = list(self.allowed_ops)
        self.rng.shuffle(candidates)

        # Try a healthy number of random proposals before giving up and asking for regeneration.
        for _ in range(128):
            op_id = self.rng.choice(candidates)
            if op_id == OP_TO_ID["div"]:
                divisors = [d for d in self.non_zero_candidates if current_value % d == 0]
                if not divisors:
                    continue
                rhs = self.rng.choice(divisors)
            elif op_id == OP_TO_ID["mul"]:
                mul_candidates = [
                    v for v in self.non_zero_candidates if abs(v) <= self.mul_abs_max
                ]
                if not mul_candidates:
                    continue
                rhs = self.rng.choice(mul_candidates)
            else:
                rhs = self.rng.choice(self.non_zero_candidates)

            new_value = apply_op(current_value, rhs, op_id)
            if abs(new_value) <= self.max_abs_label:
                return op_id, rhs, new_value

        allowed_symbols = [ID_TO_SYMBOL[op_id] for op_id in self.allowed_ops]
        raise GenerationError(
            "Unable to sample a safe path step under current constraints. "
            f"Selected ops={allowed_symbols}. "
            "Try reducing path_len, shrinking mul_abs_max, increasing max_abs_label, "
            "or changing the operator set."
        )

    def _assign_main_path_values(
        self, path_nodes: Sequence[int]
    ) -> Tuple[List[int], List[Tuple[int, int, int]], int]:
        node_values: List[int] = [0 for _ in range(self.num_nodes)]
        path_edges: List[Tuple[int, int, int]] = []

        src = path_nodes[0]
        node_values[src] = self.rng.choice(self.non_zero_candidates)
        current = node_values[src]

        for step in range(self.path_len):
            u = path_nodes[step]
            v = path_nodes[step + 1]
            op_id, rhs_value, current = self._sample_safe_path_step(current)
            node_values[v] = rhs_value
            path_edges.append((u, v, op_id))

        return node_values, path_edges, current

    def _attach_branches(
        self,
        path_nodes: Sequence[int],
        remaining_nodes: Sequence[int],
        node_values: List[int],
    ) -> List[Tuple[int, int, int]]:
        branch_edges: List[Tuple[int, int, int]] = []
        groups: Dict[int, List[int]] = {i: [] for i in range(self.path_len)}

        for node_id in remaining_nodes:
            anchor = self.rng.randrange(self.path_len)
            groups[anchor].append(node_id)
            node_values[node_id] = self.rng.choice(self.non_zero_candidates)

        for anchor_idx, group in groups.items():
            if not group:
                continue
            self.rng.shuffle(group)
            root = path_nodes[anchor_idx]
            created: List[int] = []
            existing_pairs = set()

            for node_id in group:
                parent = self.rng.choice([root] + created) if created else root
                op_id = self._random_op()
                branch_edges.append((parent, node_id, op_id))
                existing_pairs.add((parent, node_id))
                created.append(node_id)

            # Add extra distracting edges inside the same branch, preserving topological order.
            ordered_nodes = [root] + created
            order = {node: idx for idx, node in enumerate(ordered_nodes)}
            for parent in ordered_nodes:
                for child in created:
                    if parent == child:
                        continue
                    if order[parent] >= order[child]:
                        continue
                    if (parent, child) in existing_pairs:
                        continue
                    if self.rng.random() < self.extra_branch_edge_prob:
                        op_id = self._random_op()
                        branch_edges.append((parent, child, op_id))
                        existing_pairs.add((parent, child))

        return branch_edges

    def generate_graph(self, graph_id: int) -> GraphSpec:
        path_nodes = self.rng.sample(range(self.num_nodes), self.path_len + 1)
        src, dst = path_nodes[0], path_nodes[-1]
        remaining_nodes = [n for n in range(self.num_nodes) if n not in path_nodes]

        node_values, path_edges, y = self._assign_main_path_values(path_nodes)
        branch_edges = self._attach_branches(path_nodes, remaining_nodes, node_values)
        all_edges = path_edges + branch_edges

        edge_index = [
            [u for u, _, _ in all_edges],
            [v for _, v, _ in all_edges],
        ]
        edge_ops = [op for _, _, op in all_edges]
        path_edge_indices = list(range(len(path_edges)))

        num_paths = count_paths(self.num_nodes, edge_index, src, dst)
        if num_paths != 1:
            raise GenerationError(
                f"Expected exactly one path from src to dst, but got {num_paths}."
            )

        return GraphSpec(
            graph_id=graph_id,
            num_nodes=self.num_nodes,
            path_len=self.path_len,
            src=src,
            dst=dst,
            node_values=node_values,
            edge_index=edge_index,
            edge_ops=edge_ops,
            path_nodes=list(path_nodes),
            path_edge_indices=path_edge_indices,
            y=y,
        )


def split_indices(
    num_graphs: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Need 0 < train_ratio and train_ratio + val_ratio < 1")

    indices = list(range(num_graphs))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(num_graphs * train_ratio)
    n_val = int(num_graphs * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_bundle(
    num_graphs: int,
    num_nodes: int,
    path_len: int,
    value_min: int,
    value_max: int,
    mul_abs_max: int,
    max_abs_label: int,
    extra_branch_edge_prob: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    allowed_ops: Sequence[int] | None = None,
    path_lens: Sequence[int] | None = None,
    path_len_sampling: str = "random",
) -> Dict:
    resolved_path_lens = parse_path_lens(path_len, path_lens)
    max_path_len = max(resolved_path_lens)
    if num_nodes < max_path_len + 1:
        raise ValueError(
            f"num_nodes must be >= max(path_lens) + 1; got num_nodes={num_nodes}, "
            f"max_path_len={max_path_len}"
        )
    if path_len_sampling not in {"random", "cycle"}:
        raise ValueError("path_len_sampling must be either 'random' or 'cycle'")

    chooser_rng = random.Random(seed)
    generators: Dict[int, ArithmeticGraphGenerator] = {
        length: ArithmeticGraphGenerator(
            num_nodes=num_nodes,
            path_len=length,
            value_min=value_min,
            value_max=value_max,
            mul_abs_max=mul_abs_max,
            max_abs_label=max_abs_label,
            extra_branch_edge_prob=extra_branch_edge_prob,
            seed=seed + 1009 * length,
            allowed_ops=allowed_ops,
        )
        for length in resolved_path_lens
    }

    graphs: List[Dict] = []
    graph_id = 0
    while len(graphs) < num_graphs:
        if path_len_sampling == "cycle":
            current_path_len = resolved_path_lens[len(graphs) % len(resolved_path_lens)]
        else:
            current_path_len = chooser_rng.choice(resolved_path_lens)

        try:
            spec = generators[current_path_len].generate_graph(graph_id)
        except GenerationError:
            continue
        graphs.append(spec.to_dict())
        graph_id += 1

    splits = split_indices(num_graphs, train_ratio, val_ratio, seed)
    train_targets = [graphs[i]["y"] for i in splits["train"]]
    target_mean = float(sum(train_targets) / max(1, len(train_targets)))
    target_var = float(
        sum((y - target_mean) ** 2 for y in train_targets) / max(1, len(train_targets))
    )
    target_std = target_var ** 0.5

    selected_ops = list(allowed_ops) if allowed_ops is not None else list(range(len(OPS)))
    path_len_counts = dict(sorted(Counter(graph["path_len"] for graph in graphs).items()))
    metadata = {
        "num_graphs": num_graphs,
        "num_nodes": num_nodes,
        "path_len": resolved_path_lens[0] if len(resolved_path_lens) == 1 else None,
        "path_lens": list(resolved_path_lens),
        "path_len_sampling": path_len_sampling,
        "path_len_counts": path_len_counts,
        "value_min": value_min,
        "value_max": value_max,
        "mul_abs_max": mul_abs_max,
        "max_abs_label": max_abs_label,
        "extra_branch_edge_prob": extra_branch_edge_prob,
        "seed": seed,
        "ops": [OPS[op_id] for op_id in selected_ops],
        "op_ids": selected_ops,
        "op_symbols": [ID_TO_SYMBOL[op_id] for op_id in selected_ops],
        "target_mean": target_mean,
        "target_std": target_std,
    }

    return {
        "graphs": graphs,
        "splits": splits,
        "metadata": metadata,
    }


def save_bundle(bundle: Dict, output_path: str) -> None:
    torch.save(bundle, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate arithmetic path graph dataset")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-graphs", type=int, default=10000)
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument(
        "--path-len",
        type=int,
        default=6,
        help="Single path length. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--path-lens",
        type=int,
        nargs="+",
        default=None,
        help="Multiple path lengths to mix in one dataset, e.g. --path-lens 1 2 3 4 5 6. Overrides --path-len.",
    )
    parser.add_argument(
        "--path-len-sampling",
        type=str,
        choices=("random", "cycle"),
        default="random",
        help="How to choose a path length for each graph when --path-lens is used.",
    )
    parser.add_argument("--value-min", type=int, default=1)
    parser.add_argument("--value-max", type=int, default=9)
    parser.add_argument("--mul-abs-max", type=int, default=3)
    parser.add_argument("--max-abs-label", type=int, default=256)
    parser.add_argument("--extra-branch-edge-prob", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ops",
        type=str,
        default="+-*/",
        help="Allowed operators on edges. Examples: '+-', '+,-', '*/', 'add sub'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_ops = parse_ops_spec(args.ops)
    bundle = build_bundle(
        num_graphs=args.num_graphs,
        num_nodes=args.num_nodes,
        path_len=args.path_len,
        value_min=args.value_min,
        value_max=args.value_max,
        mul_abs_max=args.mul_abs_max,
        max_abs_label=args.max_abs_label,
        extra_branch_edge_prob=args.extra_branch_edge_prob,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        allowed_ops=allowed_ops,
        path_lens=args.path_lens,
        path_len_sampling=args.path_len_sampling,
    )
    save_bundle(bundle, args.output)

    print(f"Saved dataset to {args.output}")
    print("Split sizes:", {k: len(v) for k, v in bundle["splits"].items()})
    print("Metadata:", bundle["metadata"])


if __name__ == "__main__":
    main()
