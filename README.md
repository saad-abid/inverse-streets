# InverseStreets – Predicting and Generating Road Layouts with Deep Learning

> **Status:** Phase 0 (setup) done. No results yet.

## Goal
Predict mobility indicators from small 2D street-grid images, and generate new layouts that reach
target indicators (inverse design). Targets can come from a text request processed by a local LLM
and checked against consistency rules. The predictor is also exported for edge devices.

## Research context
- **Inverse design with deep learning**: property prediction, conditional generation, constraint checking.
- **Smart mobility**: network accessibility, simple traffic assignment, congestion indicators.
- **Embedded AI**: ONNX export, INT8 quantization, latency and size benchmarks.

## Honesty notes
- All layouts are **synthetic**.
- Traffic indicators come from a **simplified all-or-nothing shortest-path assignment**, NOT a
  calibrated traffic simulation.
- Every number in this README comes from running the code in this repository.

## Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## How to run each phase
_To be filled as phases are completed._

## Results
_To be filled from actual runs._

## Limitations
_To be filled._

## Future work
Real OpenStreetMap tiles, calibrated traffic assignment, link with PTV Visum/Vissim,
multi-agent simulation, deployment on an NVIDIA Jetson.
