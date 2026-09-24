# MicroQML: Quantum Micro-Reservoir Features for Graph Optimization

Conditional information from quantum reservoir features beyond classical descriptors.

## Overview

This repository contains the code and results for the Quantum Micro-Teacher (QMT) research program, investigating whether quantum reservoir features — expectation values of Pauli observables on spectrally compressed graph circuits — carry conditional information about combinatorial optimization landscape structure that is not recoverable by matched classical feature transforms.

### Key Results

**Stage 1A (10,000 graphs, simulator)**: 7/7 preregistered gates passed
- Q1 ΔR² = 0.033 [0.029, 0.038], permutation p < 0.001
- Q1 > Q0 (entanglement matters): 0.033 vs 0.024
- Q1 > best classical control: 0.033 vs 0.027
- OOD size shift (3×): ΔR² = 0.020 [0.011, 0.028] — signal survives
- Entanglement sweep: non-monotone optimum at e=0.5 (ΔR² = 0.046)

**Stage 1B (100 graphs, PiastQ hardware)**: Pipeline validated end-to-end
- Full pipeline works: graph generation → solver portfolio → compression → CGQP circuits → PiastQ hardware → feature extraction → conditional tests
- Feature correlation is strong: Q1 mean r = 0.837 confirms quantum signal survives hardware noise
- ΔR² negative at 100 graphs (statistical power issue, not signal absence — Stage 1A needed 10,000 graphs)

## Repository Structure

```
qmt/                          # Quantum Micro-Teacher library
├── graphs/                   # Graph generation and descriptors
├── solvers/                  # 5-solver MaxCut portfolio
├── compression/              # Spectral graph compression
├── quantum/                  # CGQP circuits and observables
├── diagnostics/               # Statistical tests and geometry
├── experiments/              # Experiment scripts (Stage 0A, 0C, 1A)
├── results/                  # Experiment results and reports
└── RESEARCH_PAPER.md         # Full research paper

piast_q/                      # PiastQ hardware validation
├── hw_validation.py          # Hardware validation script
├── piast_common.py           # PiastQ connection helpers
├── HW_VALIDATION.md          # Hardware validation documentation
├── hw_results_dry/           # Dry run results (50 graphs, simulator)
├── hw_results_test/          # Small hardware test (20 graphs, Q1)
└── hw_results_full/          # Full hardware validation (100 graphs, Q1+Q0)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For hardware validation (PiastQ)
pip install -r piast_q/requirements.txt
```

## Running Experiments

### Stage 1A (Simulator)

```bash
PYTHONPATH=. python -m qmt.experiments.exp04_stage1a \
    --graphs 10000 --k 8 --probe P2 \
    --outdir qmt/results/stage1a \
    --seeds 10 --permutations 1000 --bootstrap 1000
```

### Stage 1B (Hardware Validation)

```bash
# Dry run (simulator only)
cd piast_q/
PYTHONPATH=../qmt .venv/bin/python hw_validation.py \
    --graphs 100 --k 8 --probe P2 --dry-run

# Full hardware run
PYTHONPATH=../qmt .venv/bin/python hw_validation.py \
    --graphs 100 --k 8 --probe P2 \
    --conditions Q1 Q0 --shots 200 \
    --outdir hw_results_full
```

## Hardware

- **PiastQ**: 20-qubit trapped-ion quantum computer
- **Native gates**: {RZ, R(=RY), RXX} — exact match for CGQP circuits
- **Connectivity**: All-to-all
- **Shots**: 200 per circuit (hard limit)

## License

Research code. See individual files for details.

## Citation

If you use this code, please cite the accompanying research paper.
