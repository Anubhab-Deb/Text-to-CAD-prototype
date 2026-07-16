# Text-to-CAD-prototype
# Generative CAD Engine: Text‑to‑3D

**Turn natural language into editable, parametric 3D CAD models.**

This project provides an end‑to‑end framework for generating CAD models from text descriptions. It also supports two parallel pipelines to accommodate different hardware constraints and use cases:

- **Structured (Pro) Pipeline** – Uses detailed JSON action sequences and geometric constraint labeling. Best for high‑fidelity, editable assemblies.
- **Flat (Lite) Pipeline** – Uses a lightweight `key=value` action language. Designed for GPU‑limited environments and fast edge inference.

Both pipelines share a common data extraction backend (Phase 1) and output standard STEP files for interoperability with any CAD software.

---
