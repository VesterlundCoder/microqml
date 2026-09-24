"""Run the canonical Bell-state job on PIAST-Q (200 shots).

Mirrors the example from the PCSS onboarding email, using the shared helpers.
A correct 2-qubit Bell state should yield counts concentrated on '00' and '11'.

    python bell_test.py
"""
from __future__ import annotations

import sys

from qiskit import QuantumCircuit, transpile

from piast_common import SHOTS_MAX, get_backend, load_token, login


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def main() -> int:
    try:
        login(load_token())
        backend = get_backend(direct=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Setup failed: {type(exc).__name__}: {exc}")
        return 1

    qc = bell_circuit()
    print(qc.draw(output="text"))

    # AQT trapped-ion backends expose a native basis {rz, r, rxx}; transpile
    # the logical circuit down to it before submitting.
    tqc = transpile(qc, backend)
    print("Transpiled to backend basis:")
    print(tqc.draw(output="text"))
    print(f"Submitting Bell state, shots={SHOTS_MAX} (PIAST-Q per-execution limit)...")

    try:
        job = backend.run(tqc, shots=SHOTS_MAX)
        result = job.result()
        counts = result.get_counts()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Job failed: {type(exc).__name__}: {exc}")
        return 1

    print("\nCounts:")
    for bitstring, n in sorted(counts.items()):
        bar = "#" * int(40 * n / SHOTS_MAX)
        print(f"  {bitstring}: {n:4d}  {bar}")

    correlated = counts.get("00", 0) + counts.get("11", 0)
    frac = correlated / max(1, sum(counts.values()))
    print(f"\nBell correlation (|00>+|11>) fraction: {frac:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
