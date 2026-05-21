from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import random
from typing import Dict, List, Sequence, Tuple

import torch

OPS: Tuple[str, ...] = ("add", "sub", "mul", "div")
OP_TO_ID: Dict[str, int] = {op: i for i, op in enumerate(OPS)}
ID_TO_SYMBOL: Dict[int, str] = {0: "+", 1: "-", 2: "*", 3: "/"}
SYMBOL_TO_OP: Dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


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
    bfs_depth: int
    bfs_layers: List[List[int]]
    bfs_order: List[int]
    bfs_tree_edge_indices: List[int]
    bfs_parent_nodes: List[int]
    layer_index: List[int]
    reachable_within_depth: List[int]
    y: int

    def to_dict(self) -> Dict:
        return asdict(self)


class GenerationError(RuntimeError):
    pass


def apply_op(lhs: int, rhs: int, op_id: int) -> int:
    if op_id == OP_TO_ID["add"]:
        return lhs + rhs
    if op_id == OP_TO_ID["sub"]:
        return lhs - rhs
    if op_id == OP_TO_ID["mul"]:
        return lhs * rhs
    if op_id == OP_TO_ID["div"]:
        if rhs == 0:
            raise ZeroDivisionError("division by zero in BFS generation")
        if lhs % rhs != 0:
            raise ValueError(f"unsafe integer division: {lhs} / {rhs}")
        return lhs // rhs
    raise ValueError(f"Unknown op_id={op_id}")


def parse_ops_spec(ops_spec: str) -> List[int]:
    text = ops_spec.strip()
    if not text:
        raise ValueError("--ops cannot be empty")

    if "," in text:
        raw_tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    elif " " in text:
        raw_tokens = [tok.strip() for tok in text.split() if tok.strip()]
    else:
        raw_tokens = list(text)

    op_names: List[str] = []
    for token in raw_tokens:
        if token in SYMBOL_TO_OP:
            op_names.append(SYMBOL_TO_OP[token])
        elif token in OP_TO_ID:
            op_names.append(token)
        else:
            raise ValueError(
                "Unsupported operator token: "
                f"{token}. Use symbols from '+-*/' or names from {list(OPS)}."
            )

    unique_names: List[str] = []
    seen = set()
    for name in op_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    if not unique_names:
        raise ValueError("No valid operators parsed from --ops")
    return [OP_TO_ID[name] for name in unique_names]


class ArithmeticBFSGraphGenerator:
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

        self.non_zero_candidates = [v for v in range(value_min, value_max + 1) if v != 0]
        if not self.non_zero_candidates:
            raise ValueError("Need at least one non-zero candidate node value")

        allowed = list(allowed_ops) if allowed_ops is not None else list(range(len(OPS)))
        if not allowed:
            raise ValueError("Need at least one allowed operator")
        self.allowed_ops = allowed

    def _random_op(self) -> int:
        return self.rng.choice(self.allowed_ops)

    def _sample_safe_step(self, current_value: int) -> Tuple[int, int, int]:
        candidates = list(self.allowed_ops)
        self.rng.shuffle(candidates)

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

        raise GenerationError(
            "Unable to sample a safe BFS step under current constraints. "
            "Try reducing path_len, shrinking mul_abs_max, increasing max_abs_label, "
            "or enabling more operators."
        )

    def _sample_layer_sizes(self) -> Tuple[List[int], int]:
        # Keep at least half of the non-spine surplus nodes reachable within BFS depth
        # so the label depends on more than a minimal chain, while still leaving room for
        # disconnected distractors.
        extra_nodes = self.num_nodes - (self.path_len + 1)
        min_reachable_extra = extra_nodes // 2
        reachable_extra = self.rng.randint(min_reachable_extra, extra_nodes) if extra_nodes > 0 else 0
        unreachable_count = extra_nodes - reachable_extra

        layer_sizes = [1] * (self.path_len + 1)
        for _ in range(reachable_extra):
            layer_sizes[self.rng.randint(1, self.path_len)] += 1

        return layer_sizes, unreachable_count

    def _sample_counts_for_parents(self, num_parents: int, num_children: int) -> List[int]:
        if num_children < 1:
            return [0] * num_parents
        counts = [0] * num_parents
        counts[0] = 1  # force the first child (the compatibility spine child) to be discovered first
        for _ in range(num_children - 1):
            counts[self.rng.randrange(num_parents)] += 1
        return counts

    def _attach_unreachable_subgraph(
        self,
        unreachable_nodes: Sequence[int],
        node_values: List[int],
    ) -> List[Tuple[int, int, int]]:
        edges: List[Tuple[int, int, int]] = []
        if not unreachable_nodes:
            return edges

        order = list(unreachable_nodes)
        self.rng.shuffle(order)
        created: List[int] = []
        existing_pairs = set()

        for node in order:
            node_values[node] = self.rng.choice(self.non_zero_candidates)
            if created and self.rng.random() < 0.85:
                parent = self.rng.choice(created)
                op_id = self._random_op()
                edges.append((parent, node, op_id))
                existing_pairs.add((parent, node))
            created.append(node)

        for i, parent in enumerate(order):
            for child in order[i + 1:]:
                if (parent, child) in existing_pairs:
                    continue
                if self.rng.random() < self.extra_branch_edge_prob * 0.5:
                    op_id = self._random_op()
                    edges.append((parent, child, op_id))
                    existing_pairs.add((parent, child))

        return edges

    def _simulate_bfs(
        self,
        edge_index: Sequence[Sequence[int]],
        src: int,
        max_depth: int,
    ) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
        adjacency: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.num_nodes)}
        for edge_idx, (u, v) in enumerate(zip(edge_index[0], edge_index[1])):
            adjacency[u].append((v, edge_idx))

        visited = {src}
        order = [src]
        layers = [[src]]
        tree_edge_indices: List[int] = []
        parent_nodes: List[int] = []
        current_layer = [src]
        depth = 0

        while current_layer and depth < max_depth:
            next_layer: List[int] = []
            for parent in current_layer:
                for child, edge_idx in adjacency[parent]:
                    if child in visited:
                        continue
                    visited.add(child)
                    next_layer.append(child)
                    order.append(child)
                    tree_edge_indices.append(edge_idx)
                    parent_nodes.append(parent)
            if not next_layer:
                break
            layers.append(next_layer)
            current_layer = next_layer
            depth += 1

        return layers, order, tree_edge_indices, parent_nodes

    def _compute_bfs_label(
        self,
        src_value: int,
        node_values: Sequence[int],
        edge_index: Sequence[Sequence[int]],
        edge_ops: Sequence[int],
        bfs_tree_edge_indices: Sequence[int],
    ) -> int:
        current = src_value
        for edge_idx in bfs_tree_edge_indices:
            child = edge_index[1][edge_idx]
            current = apply_op(current, node_values[child], edge_ops[edge_idx])
        return current

    def generate_graph(self, graph_id: int) -> GraphSpec:
        layer_sizes, unreachable_count = self._sample_layer_sizes()
        reachable_total = sum(layer_sizes)
        if reachable_total + unreachable_count != self.num_nodes:
            raise GenerationError("Internal layer size accounting error")

        all_nodes = list(range(self.num_nodes))
        self.rng.shuffle(all_nodes)

        node_values: List[int] = [0 for _ in range(self.num_nodes)]
        bfs_layers: List[List[int]] = []
        cursor = 0
        for size in layer_sizes:
            layer_nodes = all_nodes[cursor:cursor + size]
            cursor += size
            bfs_layers.append(layer_nodes)
        unreachable_nodes = all_nodes[cursor:cursor + unreachable_count]

        src = bfs_layers[0][0]
        node_values[src] = self.rng.choice(self.non_zero_candidates)
        current = node_values[src]

        edges: List[Tuple[int, int, int]] = []
        bfs_tree_edge_indices: List[int] = []
        bfs_parent_nodes: List[int] = []
        designated_parent_indices_by_layer: List[List[int]] = []
        path_nodes = [layer[0] for layer in bfs_layers]
        path_edge_indices: List[int] = []

        for depth in range(1, self.path_len + 1):
            parents = bfs_layers[depth - 1]
            children = bfs_layers[depth]
            counts = self._sample_counts_for_parents(len(parents), len(children))
            designated_parent_idx: List[int] = []
            child_cursor = 0
            for parent_idx, child_count in enumerate(counts):
                for _ in range(child_count):
                    if child_cursor >= len(children):
                        break
                    child = children[child_cursor]
                    parent = parents[parent_idx]
                    op_id, rhs_value, current = self._sample_safe_step(current)
                    node_values[child] = rhs_value
                    edges.append((parent, child, op_id))
                    edge_idx = len(edges) - 1
                    bfs_tree_edge_indices.append(edge_idx)
                    bfs_parent_nodes.append(parent)
                    designated_parent_idx.append(parent_idx)
                    if child == path_nodes[depth] and parent == path_nodes[depth - 1]:
                        path_edge_indices.append(edge_idx)
                    child_cursor += 1
            if child_cursor != len(children):
                raise GenerationError("Failed to assign all BFS-layer children to parents")
            designated_parent_indices_by_layer.append(designated_parent_idx)

        if len(path_edge_indices) != self.path_len:
            raise GenerationError(
                "Compatibility spine path was not preserved. This should not happen."
            )

        # Add extra forward edges inside each adjacent layer pair without changing first-discovery BFS order.
        for depth in range(1, self.path_len + 1):
            parents = bfs_layers[depth - 1]
            children = bfs_layers[depth]
            designated_parent_indices = designated_parent_indices_by_layer[depth - 1]
            for child_idx, child in enumerate(children):
                first_parent_idx = designated_parent_indices[child_idx]
                for parent_idx in range(first_parent_idx + 1, len(parents)):
                    if self.rng.random() < self.extra_branch_edge_prob:
                        op_id = self._random_op()
                        edges.append((parents[parent_idx], child, op_id))

        unreachable_edges = self._attach_unreachable_subgraph(unreachable_nodes, node_values)
        all_edges = edges + unreachable_edges
        edge_index = [
            [u for u, _, _ in all_edges],
            [v for _, v, _ in all_edges],
        ]
        edge_ops = [op_id for _, _, op_id in all_edges]

        sim_layers, sim_order, sim_tree_edge_indices, sim_parent_nodes = self._simulate_bfs(
            edge_index=edge_index,
            src=src,
            max_depth=self.path_len,
        )

        if sim_layers != bfs_layers:
            raise GenerationError(
                f"BFS layers mismatch: planned={bfs_layers}, simulated={sim_layers}"
            )
        if sim_order != [node for layer in bfs_layers for node in layer]:
            raise GenerationError("BFS discovery order mismatch")
        if sim_tree_edge_indices != bfs_tree_edge_indices:
            raise GenerationError("BFS tree edge indices mismatch")
        if sim_parent_nodes != bfs_parent_nodes:
            raise GenerationError("BFS parent node sequence mismatch")

        y = self._compute_bfs_label(
            src_value=node_values[src],
            node_values=node_values,
            edge_index=edge_index,
            edge_ops=edge_ops,
            bfs_tree_edge_indices=bfs_tree_edge_indices,
        )
        if abs(y) > self.max_abs_label:
            raise GenerationError("Final label magnitude exceeded max_abs_label")

        layer_index = [-1 for _ in range(self.num_nodes)]
        reachable_within_depth = [0 for _ in range(self.num_nodes)]
        for depth, layer in enumerate(bfs_layers):
            for node in layer:
                layer_index[node] = depth
                reachable_within_depth[node] = 1

        dst = path_nodes[-1]
        return GraphSpec(
            graph_id=graph_id,
            num_nodes=self.num_nodes,
            path_len=self.path_len,
            src=src,
            dst=dst,
            node_values=node_values,
            edge_index=edge_index,
            edge_ops=edge_ops,
            path_nodes=path_nodes,
            path_edge_indices=path_edge_indices,
            bfs_depth=self.path_len,
            bfs_layers=bfs_layers,
            bfs_order=sim_order,
            bfs_tree_edge_indices=sim_tree_edge_indices,
            bfs_parent_nodes=sim_parent_nodes,
            layer_index=layer_index,
            reachable_within_depth=reachable_within_depth,
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



def normalize_path_lens(path_len: int | None, path_lens: Sequence[int] | None, num_nodes: int) -> List[int]:
    """Return a validated, de-duplicated list of BFS depths/path lengths."""
    if path_lens is None or len(path_lens) == 0:
        if path_len is None:
            raise ValueError("Either --path-len or --path-lens must be provided")
        selected = [path_len]
    else:
        selected = list(path_lens)

    normalized: List[int] = []
    seen = set()
    for length in selected:
        if length < 1:
            raise ValueError("All path lengths/BFS depths must be >= 1")
        if num_nodes < length + 1:
            raise ValueError(
                f"num_nodes must be >= path_len + 1 for every selected length; "
                f"got num_nodes={num_nodes}, path_len={length}"
            )
        if length not in seen:
            seen.add(length)
            normalized.append(length)

    if not normalized:
        raise ValueError("At least one path length/BFS depth must be selected")
    return normalized


def select_path_len_for_graph(
    graph_index: int,
    path_lens: Sequence[int],
    sampling: str,
    rng: random.Random,
) -> int:
    if sampling == "cycle":
        return path_lens[graph_index % len(path_lens)]
    if sampling == "random":
        return rng.choice(list(path_lens))
    raise ValueError("path_len_sampling must be either 'random' or 'cycle'")

def build_bundle(
    num_graphs: int,
    num_nodes: int,
    path_len: int | None,
    path_lens: Sequence[int] | None,
    path_len_sampling: str,
    value_min: int,
    value_max: int,
    mul_abs_max: int,
    max_abs_label: int,
    extra_branch_edge_prob: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    ops: str = "+-*/",
) -> Dict:
    allowed_ops = parse_ops_spec(ops)
    selected_path_lens = normalize_path_lens(path_len, path_lens, num_nodes)
    if path_len_sampling not in {"random", "cycle"}:
        raise ValueError("path_len_sampling must be either 'random' or 'cycle'")

    # Outer RNG controls both random path-length choice and per-attempt generator seeds.
    # This keeps the whole dataset reproducible under one user-facing seed.
    rng = random.Random(seed)

    graphs: List[Dict] = []
    graph_id = 0
    path_len_counts: Dict[int, int] = {length: 0 for length in selected_path_lens}

    while len(graphs) < num_graphs:
        current_path_len = select_path_len_for_graph(
            graph_index=graph_id,
            path_lens=selected_path_lens,
            sampling=path_len_sampling,
            rng=rng,
        )

        # Retry the same graph_id and same selected length until a valid graph is produced.
        while True:
            attempt_seed = rng.randrange(0, 2**32)
            generator = ArithmeticBFSGraphGenerator(
                num_nodes=num_nodes,
                path_len=current_path_len,
                value_min=value_min,
                value_max=value_max,
                mul_abs_max=mul_abs_max,
                max_abs_label=max_abs_label,
                extra_branch_edge_prob=extra_branch_edge_prob,
                seed=attempt_seed,
                allowed_ops=allowed_ops,
            )
            try:
                spec = generator.generate_graph(graph_id)
            except GenerationError:
                continue
            graph_dict = spec.to_dict()
            # Redundant aliases are useful for downstream loaders that look for either name.
            graph_dict["path_len"] = current_path_len
            graph_dict["bfs_depth"] = current_path_len
            graphs.append(graph_dict)
            path_len_counts[current_path_len] += 1
            graph_id += 1
            break

    splits = split_indices(num_graphs, train_ratio, val_ratio, seed)
    train_targets = [graphs[i]["y"] for i in splits["train"]]
    target_mean = float(sum(train_targets) / max(1, len(train_targets)))
    target_var = float(
        sum((y - target_mean) ** 2 for y in train_targets) / max(1, len(train_targets))
    )
    target_std = target_var ** 0.5
    bfs_seq_lens = [len(graph["bfs_order"]) - 1 for graph in graphs]
    graph_bfs_depths = [graph["bfs_depth"] for graph in graphs]

    single_len = selected_path_lens[0] if len(selected_path_lens) == 1 else None
    metadata = {
        "num_graphs": num_graphs,
        "num_nodes": num_nodes,
        "path_len": single_len,
        "path_lens": list(selected_path_lens),
        "path_len_sampling": path_len_sampling,
        "path_len_counts": dict(path_len_counts),
        "bfs_depth": single_len,
        "bfs_depths": list(selected_path_lens),
        "value_min": value_min,
        "value_max": value_max,
        "mul_abs_max": mul_abs_max,
        "max_abs_label": max_abs_label,
        "extra_branch_edge_prob": extra_branch_edge_prob,
        "extra_forward_edge_prob": extra_branch_edge_prob,
        "seed": seed,
        "ops": [OPS[op_id] for op_id in allowed_ops],
        "op_ids": list(allowed_ops),
        "op_symbols": [ID_TO_SYMBOL[op_id] for op_id in allowed_ops],
        "task": "bfs_discovery_regression",
        "label_semantics": (
            "Start from value(src); run BFS from src up to each graph's bfs_depth/path_len; "
            "for each newly discovered node in BFS order, apply the op on the discovery edge "
            "to the current accumulator using that node's value."
        ),
        "target_mean": target_mean,
        "target_std": target_std,
        "min_path_len": min(graph_bfs_depths) if graph_bfs_depths else single_len,
        "max_path_len": max(graph_bfs_depths) if graph_bfs_depths else single_len,
        "min_bfs_depth": min(graph_bfs_depths) if graph_bfs_depths else single_len,
        "max_bfs_depth": max(graph_bfs_depths) if graph_bfs_depths else single_len,
        "max_bfs_seq_len": max(bfs_seq_lens) if bfs_seq_lens else single_len,
        "avg_bfs_seq_len": float(sum(bfs_seq_lens) / max(1, len(bfs_seq_lens))),
    }

    return {
        "graphs": graphs,
        "splits": splits,
        "metadata": metadata,
    }

def save_bundle(bundle: Dict, output_path: str) -> None:
    torch.save(bundle, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate arithmetic BFS graph dataset. "
            "Here --path-len means the BFS depth used to build the computation sequence."
        )
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-graphs", type=int, default=10000)
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument(
        "--path-len",
        type=int,
        default=6,
        help="Single BFS depth/path length. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--path-lens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Multiple BFS depths/path lengths to mix in one dataset, e.g. "
            "--path-lens 1 2 3 4 5 6. If provided, this overrides --path-len."
        ),
    )
    parser.add_argument(
        "--path-len-sampling",
        type=str,
        choices=["random", "cycle"],
        default="random",
        help=(
            "How to choose a length for each graph when --path-lens is provided. "
            "random samples uniformly; cycle makes the counts as balanced as possible."
        ),
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
        help="Allowed edge operators, e.g. '+-', '+,-', 'add sub', '*/'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle(
        num_graphs=args.num_graphs,
        num_nodes=args.num_nodes,
        path_len=args.path_len,
        path_lens=args.path_lens,
        path_len_sampling=args.path_len_sampling,
        value_min=args.value_min,
        value_max=args.value_max,
        mul_abs_max=args.mul_abs_max,
        max_abs_label=args.max_abs_label,
        extra_branch_edge_prob=args.extra_branch_edge_prob,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        ops=args.ops,
    )
    save_bundle(bundle, args.output)

    print(f"Saved dataset to {args.output}")
    print("Split sizes:", {k: len(v) for k, v in bundle["splits"].items()})
    print("Metadata:", bundle["metadata"])


if __name__ == "__main__":
    main()
