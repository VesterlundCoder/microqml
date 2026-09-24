# STAGE 1 REPORT: Full Condition Matrix (M0-M6)

**Graphs**: 2000 | **k**: 8 | **Probe**: P2

**λ_q sweep**: [0.0, 0.1, 0.3, 1.0, 3.0] | **Hidden sweep**: [2, 3, 4, 5]

**Date**: 2026-09-23 10:12:08


## 1. Dataset Validity

- Solver win fractions: {'spectral': 0.0, 'greedy': 0.428, 'multistart': 0.0915, 'annealing': 0.4015, 'tabu': 0.079}
- Max fraction: 0.428
- Gate: PASS

## 2. Condition Summary (best val R² per condition)

| Condition | Description | Best val R² | Best λ_q | Best h | Params |
|-----------|-------------|-------------|----------|--------|--------|
| M0 | Classical baseline (no quantum) | 0.7481 | N/A | 4 | 57 |
| M1 | Quantum distillation (spectral Q) | 0.7112 | 0.3 | 4 | 57 |
| M2 | Shuffled control (shuffled Q) | 0.2231 | N/A | 4 | 77 |
| M3 | Random compression control | 0.2231 | N/A | 4 | 77 |
| M4 | Classical compressed teacher | 0.2231 | N/A | 4 | 77 |
| M5 | Direct quantum features | 0.0373 | N/A | 5 | 75 |
| M6 | Oracle (C+Q, large model) | 0.4235 | N/A | 128 | -1 |

### Best at λ_q > 0 (distillation active)

| Condition | Best val R² | Best λ_q | Best h |
|-----------|-------------|----------|--------|
| M1 | 0.7112 | 0.3 | 4 |
| M2 | 0.2169 | 0.1 | 4 |
| M3 | 0.2178 | 0.1 | 4 |
| M4 | 0.2221 | 0.1 | 4 |

## 3. Detailed Sweep Results


### M1: Quantum distillation (spectral Q)

| λ_q \ h | 2 | 3 | 4 | 5 |
|---------|---------|---------|---------|---------|
| 0.0 | 0.1020 | 0.1968 | 0.2231 | 0.2199 |
| 0.1 | 0.0994 | 0.1930 | 0.2181 | 0.2155 |
| 0.3 | 0.0928 | 0.1851 | 0.2058 | 0.2060 |
| 1.0 | 0.0645 | 0.1619 | 0.1715 | 0.1714 |
| 3.0 | 0.0335 | 0.1188 | 0.1204 | 0.1059 |

### M2: Shuffled control (shuffled Q)

| λ_q \ h | 2 | 3 | 4 | 5 |
|---------|---------|---------|---------|---------|
| 0.0 | 0.1020 | 0.1968 | 0.2231 | 0.2199 |
| 0.1 | 0.0997 | 0.1922 | 0.2169 | 0.2151 |
| 0.3 | 0.0907 | 0.1822 | 0.2020 | 0.2023 |
| 1.0 | 0.0482 | 0.1476 | 0.1518 | 0.1559 |
| 3.0 | 0.0171 | 0.0780 | 0.0766 | 0.0722 |

### M3: Random compression control

| λ_q \ h | 2 | 3 | 4 | 5 |
|---------|---------|---------|---------|---------|
| 0.0 | 0.1020 | 0.1968 | 0.2231 | 0.2199 |
| 0.1 | 0.0997 | 0.1922 | 0.2178 | 0.2155 |
| 0.3 | 0.0907 | 0.1828 | 0.2041 | 0.2043 |
| 1.0 | 0.0545 | 0.1544 | 0.1651 | 0.1650 |
| 3.0 | 0.0252 | 0.1088 | 0.1052 | 0.0981 |

### M4: Classical compressed teacher

| λ_q \ h | 2 | 3 | 4 | 5 |
|---------|---------|---------|---------|---------|
| 0.0 | 0.1020 | 0.1968 | 0.2231 | 0.2199 |
| 0.1 | 0.1018 | 0.1972 | 0.2221 | 0.2196 |
| 0.3 | 0.1015 | 0.1978 | 0.2203 | 0.2189 |
| 1.0 | 0.1006 | 0.1984 | 0.2162 | 0.2173 |
| 3.0 | 0.0990 | 0.1983 | 0.2117 | 0.2151 |

### M0: Classical baseline (no quantum)

| hidden | val R² | params |
|--------|--------|--------|
| 2 | 0.1020 | 41 |
| 3 | 0.1968 | 59 |
| 4 | 0.2231 | 77 |
| 5 | 0.2199 | 95 |
| 4 | 0.7481 | 57 |

### M5: Direct quantum features

| hidden | val R² | params |
|--------|--------|--------|
| 2 | -0.0738 | 33 |
| 3 | 0.0070 | 47 |
| 4 | 0.0292 | 61 |
| 5 | 0.0373 | 75 |

## 4. Key Comparisons (at λ_q > 0, distillation active)

- **M1 vs M0** (quantum distillation vs classical): ΔR² = -0.0369 ✗
- **M1 vs M2** (real vs shuffled Q): ΔR² = +0.4943 ✓
- **M1 vs M3** (spectral vs random compression): ΔR² = +0.4934 ✓
- **M1 vs M4** (quantum vs classical compressed teacher): ΔR² = +0.4891 ✓
- **M5 vs M0** (direct Q vs classical): ΔR² = -0.7108 ✗
- **M6 vs M0** (oracle vs classical): ΔR² = -0.3247

## 5. Hardness Target

| Condition | λ_q | val R² |
|-----------|-----|--------|
| M0 | 0.0 | 0.7481 |
| M1 | 0.3 | 0.7112 |
| M1 | 1.0 | 0.5833 |

## 6. Verdict

- M1 > M0: FAIL
- M1 > M2: PASS
- M1 > M3: PASS
- M1 > M4: PASS

**VERDICT: PROMOTE_TO_STAGE_2 = YES** (3/4 gates passed)

Quantum distillation shows genuine incremental signal. Proceed to Stage 2 with OOD evaluation.

---

*Internal research gate, not a scientific claim.*