# GraphTraceBench Tool-Generation Prompt Collection（For rebuttal)

This document contains the complete prompts for generating Python solvers for GraphLR-Path and GraphLR-BFS, as well as an automatic repair prompt for cases where program generation fails.

---

## Usage

* **Path task**: Use “Prompt 1: System Prompt” together with “Prompt 2: GraphLR-Path Python Solver.”
* **BFS task**: Use “Prompt 1: System Prompt” together with “Prompt 3: GraphLR-BFS Python Solver.”
* **After program-generation failure**: Use “Prompt 4: Automatic Repair Prompt,” and fill in the diagnostic error, the surrounding error context, and the previous version of the generated code.

---

# Prompt 1：System Prompt

```text
You are an expert Python programmer specializing in graph algorithms, robust text parsing, and symbolic execution.

Your task is to generate a concise, executable Python module that exactly follows the requested interface.

Requirements:

1. Output Python source code only.
2. Do not use Markdown code fences.
3. Do not include explanations, comments outside the code, tests, examples, or alternative implementations.
4. Ensure that every string, bracket, parenthesis, and function definition is complete.
5. The generated module must be directly importable.
6. Keep the implementation concise and deterministic.
7. Do not rely on hidden annotations, labels, or external files.
8. Solve the task only from the raw graph prompt provided to the generated function.

Before finishing, internally verify that the code:
- defines the required function;
- parses all required fields;
- reconstructs the graph correctly;
- recovers the correct trace;
- executes operators strictly left-to-right;
- returns a Python integer.
```

---

# Prompt 2：GraphLR-Path Python Solver

```text
Write one complete Python 3 module for solving the GraphLR-Path task.

### Mandatory interface

The module must define exactly this public function:

def solve_prompt(prompt: str) -> int:
    ...

The evaluation harness will import the generated module and call `solve_prompt` directly.

Do not write:

- stdin or stdout handling;
- a `main()` function;
- command-line argument parsing;
- Markdown code fences;
- explanations;
- tests or example calls;
- multiple alternative programs.

### Input prompt format

The input is a raw text string similar to:

Compute the graph answer. There is exactly one directed path from src to dst. Start with value(src), traverse that unique path, and on each traversed edge apply the edge operator with the destination node value, left-to-right. Division is exact integer division. Output ONLY {"answer": integer}.
src=20; dst=4; path_hops=1; num_nodes=24; node_values=[0:2; 1:8; 2:2; 3:1; 4:3]; edges=[20->4:*; 20->3:+; 3->1:-]

The actual prompt can contain more nodes and edges.

### Required parsing

Extract the following fields from the raw string:

- `src`: source node ID;
- `dst`: destination node ID;
- `path_hops`: expected number of edges on the path;
- `num_nodes`: total number of nodes;
- `node_values`: mapping from node IDs to integer values;
- `edges`: ordered directed edges in the form `u->v:operator`.

Node IDs and node values are integers. Node values may be negative.

Operators are exactly:

- `+`
- `-`
- `*`
- `/`

Division is guaranteed to be exact integer division.

### Required graph algorithm

1. Parse every directed edge and its operator.
2. Construct the directed graph.
3. Recover the unique directed path from `src` to `dst`.
4. Do not assume that the correct path edge is the first outgoing edge.
5. Do not assume that path edges are consecutive in the serialized edge list.
6. Do not use `path_hops` as an answer or as a substitute for graph search.
7. The graph is guaranteed to contain exactly one directed path from `src` to `dst`.
8. The recovered path should contain exactly `path_hops` edges. Raise an exception if this consistency check fails.

### Required arithmetic execution

Let:

result = node_values[src]

For each directed edge `(u, v, op)` on the recovered path, in path order, update:

- `+`: `result = result + node_values[v]`
- `-`: `result = result - node_values[v]`
- `*`: `result = result * node_values[v]`
- `/`: `result = result // node_values[v]`

All operations must be executed strictly left-to-right.

Do not apply normal arithmetic precedence across multiple edges.

Return the final result as a Python `int`.

### Implementation restrictions

- Use Python standard library only.
- Allowed imports include `re`, `collections`, `typing`, `dataclasses`, `json`, `math`, and `operator`.
- Do not use `eval`, `exec`, `compile`, `open`, `input`, or `__import__`.
- Do not access files, environment variables, networks, subprocesses, or the shell.
- Do not hard-code any node ID, node value, operator sequence, answer, graph size, or path length.
- Do not use `path_nodes`, `path_edge_indices`, labels, or any oracle trace annotation.
- Keep the complete module under 150 lines.
- Avoid unnecessary classes and operator overloading.
- Prefer simple dictionaries, adjacency lists, and DFS or BFS.
- The function must work for all valid GraphLR-Path prompts following this format.

Return only the complete Python source code.
```

---

# Prompt 3：GraphLR-BFS Python Solver

```text
Write one complete Python 3 module for solving the GraphLR-BFS task.

### Mandatory interface

The module must define exactly:

def solve_prompt(prompt: str) -> int:
    ...

The evaluation harness imports the module and calls `solve_prompt` directly.

Output Python source code only. Do not output Markdown, explanations, tests, examples, stdin handling, stdout handling, or a `main()` function.

### Input prompt format

The raw text prompt contains fields such as:

src=<integer>;
bfs_depth=<integer>;
num_nodes=<integer>;
node_values=[0:9; 1:8; ...];
edges=[7->10:+; 3->4:*; ...]

Each edge has the format:

source->destination:operator

Operators are `+`, `-`, `*`, and `/`.

Division is guaranteed to be exact integer division.

### Required BFS procedure

1. Parse `src`, `bfs_depth`, `num_nodes`, `node_values`, and all directed edges.
2. Preserve the exact global edge-list order from the prompt.
3. Construct outgoing adjacency lists such that edges from the same source remain in their original global order.
4. Run standard FIFO BFS starting at `src`.
5. The source has depth 0.
6. Expand only nodes whose depth is strictly less than `bfs_depth`.
7. Discover every node at most once.
8. Mark a node as visited immediately when it is enqueued.
9. Record the discovery edge for every newly discovered node.
10. Do not use `bfs_order`, `bfs_layers`, `parent`, `bfs_tree_edge_indices`, labels, or oracle annotations.

### Required arithmetic execution

Initialize:

result = node_values[src]

Whenever BFS discovers a new node `v` through an edge `(u, v, op)`, update the accumulator immediately:

- `+`: `result = result + node_values[v]`
- `-`: `result = result - node_values[v]`
- `*`: `result = result * node_values[v]`
- `/`: `result = result // node_values[v]`

Apply operations strictly in BFS discovery order and strictly left-to-right.

Return the final result as a Python `int`.

### Restrictions

- Use standard library only.
- Do not use external files, networks, shell commands, subprocesses, environment variables, `eval`, `exec`, `compile`, `open`, `input`, or `__import__`.
- Do not hard-code answers or graph structures.
- Keep the module under 150 lines.
- Avoid unnecessary classes and operator overloading.
- The implementation must work for every valid GraphLR-BFS prompt following the specified format.

Return only the complete Python source code.
```

---

# Prompt 4：auto repair Prompt

```text
The previously generated Python module failed validation or training-set smoke tests.

Produce one complete corrected Python module. Return the full module rather than a patch.

The module must define:

def solve_prompt(prompt: str) -> int:
    ...

Do not output Markdown code fences, explanations, tests, examples, or multiple versions.

The previous failure was:

{DIAGNOSTIC_ERROR}

Relevant lines around the failure:

{ERROR_CONTEXT}

The previous generated module was:

{PREVIOUS_CODE}

Correct all syntax, interface, parsing, graph-recovery, and arithmetic-execution errors.

Important requirements:

1. The module must define a callable `solve_prompt`.
2. It must parse the raw prompt rather than use hard-coded values.
3. It must not access labels or oracle trace annotations.
4. It must recover the path or BFS order from the graph structure.
5. It must execute all operators strictly left-to-right.
6. It must return a Python integer.
7. It must be syntactically complete and directly importable.
8. Keep the implementation concise and under 150 lines.
9. Do not repeat irrelevant classes, excessive operator overloads, or duplicated code.
10. Output only the complete corrected Python source code.
```

---
