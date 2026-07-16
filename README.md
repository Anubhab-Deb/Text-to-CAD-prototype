# Generative CAD Engine: Text‑to‑3D

## **Turn natural language into editable, parametric 3D CAD models.**

This project provides an end‑to‑end framework for generating CAD models from text descriptions. It also supports two parallel pipelines to accommodate different hardware constraints and use cases:

- **Structured (Pro) Pipeline** – Uses detailed JSON action sequences and geometric constraint labeling. Best for high‑fidelity, editable assemblies.
- **Flat (Lite) Pipeline** – Uses a lightweight `key=value` action language. Designed for GPU‑limited environments and fast edge inference.

Both pipelines share a common data extraction backend (Phase 1) and output standard STEP files for interoperability with any CAD software.

---

## **Why Two Pipelines?**

| Pipeline | Strengths | Key Differentiator | Best For |
|----------|-----------|-------------------|----------|
| **Structured (Path A)** | Full geometric constraints, parameter regression, editable constraint graphs, constraint‑aware training. | Outputs a labeled constraint graph that enables **parametric editing** (just like native CAD). | High‑end workstations, research, complex assemblies, production‑grade CAD. |
| **Flat (Path B)** | Tiny memory footprint, fast parsing, simple tokenisation, lower training/inference cost. | Uses a lightweight `key=value` action language that runs comfortably on **GPUs with ≤4GB VRAM**. | Laptops, GPU‑limited environments. |

> The flat pipeline was engineered specifically to overcome GPU memory limits while keeping the core generative capability intact. Both pipelines are fully functional and production‑ready.

---

## **Phases of Project**

### 🔷 Phase 1 – Shared Data Pipeline

| File | Role |
|------|------|
| `step_to_json.py` | Converts STEP AP214 files to structured JSON (faces, edges, vertices, geometry parameters). |
| `json_to_graphs.py` | Transforms the JSON into a PyTorch Geometric (PyG) graph `.pt` file with node and edge features. |

---

### 🔶 Phase 2 – Path A (Structured JSON + Constraints)

| Order | File | Role |
|-------|------|------|
| 1 | `generate_text_descriptions.py` | Creates natural‑language captions from the JSON and pairs them with graph `.pt` paths. |
| 2 | `json_to_actions.py` | Synthesises detailed action sequences (CreateSketch, Extrude, etc.) from the JSON. |
| 3 | `generate_rich_text_from_actions.py` | *(Optional)* Augments descriptions with more diverse, template‑based phrasing. |
| 4 | `constraint_labeling.py` | **Secret sauce** – adds geometric constraint labels (parallel, concentric, equal radius) with parameter regression. |
| 5 | `split_data.py` or `split_data_v2.py` | Splits the text‑action CSV into train/validation sets. Use `split_data.py` for output from `json_to_actions.py`; use `split_data_v2.py` for `generate_synthetic_dataset_v2.py`. |
| 6 | `train_text_to_actions_v2.py` | Trains a T5 model on structured actions (supports discretization, LoRA, mixed precision). |
| 7 | `cad_interpreter.py` | Parses the predicted action string and builds a 3D solid using `pythonOCC`. |
| 8 | `text_to_cad_v2.py` | End‑to‑end inference: text → action string → STEP file. |

---

### 🔶 Phase 2 – Path B (Flat Action Language)

| Order | File | Role |
|-------|------|------|
| 1 | `generate_flat_dataset.py` | Generates synthetic `key=value` flat action strings (no nested JSON). |
| 2 | `split_flat_data.py` | Splits the flat dataset into train/validation CSVs. |
| 3 | `train_flat_actions.py` | Trains a lightweight T5 model on flat actions (lower memory, faster training). |
| 4 | `flat_interpreter.py` | Parses flat action strings and builds the STEP model. |
| 5 | `infer_flat_model.py` | Inference for the flat model: text → flat string → STEP. |

---
