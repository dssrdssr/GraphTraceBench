# GraphTraceBench: Diagnosing Long-Range Reasoning in Graph Transformers and Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official repository for **GraphTraceBench**, a synthetic benchmark suite designed for the controlled evaluation of long-range reasoning and latent computation trace recovery over graphs.

---

## 📌 Overview

Long-range reasoning over graphs requires models to recover latent computation traces and execute multi-step operations far beyond local neighborhoods. **GraphTraceBench** provides a diagnostic testbed to study whether graph-native models and large language models (LLMs) can perform controlled, long-range execution over directed graphs.

It features two complementary tasks with independent control over graph size, reasoning depth, operator vocabulary, and distractor density:
1. **GraphLR-Path**: Execute arithmetic operators along the unique directed path from a source to a destination node.
2. **GraphLR-BFS**: Execute an arithmetic program defined over the breadth-first discovery order of a directed graph.

---

## 📂 Dataset Structure & Directory

The benchmark datasets are tailored for different input modalities and model architectures. They are split into two main directories:

```text
📁 GraphTraceBench/
├── 📁 GT/                  # Datasets for Graph-Native Transformers (Structured graph format)
│   ├── arithmetic_graph_gt/
│   ├── models/
│   ├── generation_BFS.py
│   ├── generate_dataset.py
│   └── ... ...
└── 📁 LLM/                 # Datasets for Large Language Models (Text-serialized format)
    ├── serialize_graph_reasoning_to_jsonl.py
    └── ... ...
