"""Frozen quantum probe bank.

Four fixed probe regimes with different (gamma, beta) parameters.
For Experiment 0A, use one probe (P2 medium).
"""
from __future__ import annotations

PROBES = {
    "P1": {
        "name": "weak",
        "gamma": 0.20,
        "beta": 0.25,
        "layers": 1,
    },
    "P2": {
        "name": "medium",
        "gamma": 0.45,
        "beta": 0.60,
        "layers": 1,
    },
    "P3": {
        "name": "strong",
        "gamma": 0.80,
        "beta": 0.35,
        "layers": 1,
    },
    "P4": {
        "name": "two_layer",
        "gamma": [0.30, 0.65],
        "beta": [0.25, 0.55],
        "layers": 2,
    },
}


def get_probe(name: str) -> dict:
    """Get probe parameters by name."""
    if name not in PROBES:
        raise ValueError(f"Unknown probe: {name}. Available: {list(PROBES.keys())}")
    return PROBES[name]


def get_all_probes() -> dict[str, dict]:
    """Return all probes."""
    return PROBES.copy()
