# Quantum Micro-Teacher Benchmark

Four simulator-first research tracks for testing whether a tiny quantum computation can teach a 25-100 parameter classical model a rule that generalizes to larger combinatorial problems.

Tracks:
1. `qwalk.py`: CTQW / XY quantum-walk micro-solver for MIS branching.
2. `grover_sat.py`: partial Grover response as a micro-teacher for SAT polarity/branching.
3. `spin.py`: short transverse-field quantum escape probe for spin-glass local search.
4. `hardness.py`: fixed QAOA response curve for solver selection / hardness prediction.

Core rule: run exact NumPy/SciPy simulation for millions of <=8-qubit teacher examples. Use Qiskit only to verify selected circuits, apply trapped-ion noise, and prepare the PIAST-Q implementation.

Quick start:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
pytest -q
```

The shared student `MicroDistillNet(12,4,task_dim,4)` has 77 trainable parameters for a scalar task. The auxiliary quantum head is used only during training; inference may use classical features only.
