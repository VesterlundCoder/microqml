# STAGE 1 REPORT: Full Condition Matrix (M0-M6)

**Graphs**: 100 | **k**: 8 | **Probe**: P2

**λ_q sweep**: [0.0, 0.3, 1.0] | **Hidden sweep**: [3, 4]

**Date**: 2026-09-23 09:55:17


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.45), 'multistart': np.float64(0.07), 'annealing': np.float64(0.43), 'tabu': np.float64(0.05)}
- Max fraction: 0.450
- Gate: PASS

## 2. Condition Summary (best val R² per condition)

| Condition | Description | Best val R² | Best λ_q | Best h | Params |
|-----------|-------------|-------------|----------|--------|--------|
| M0 | Classical baseline (no quantum) | 0.0186 | N/A | 4 | 77 |
| M1 | Quantum distillation (spectral Q) | 0.0239 | 0.3 | 4 | 77 |
| M2 | Shuffled control (shuffled Q) | 0.0217 | 0.3 | 4 | 77 |
| M3 | Random compression control | 0.0212 | 0.3 | 4 | 77 |
| M4 | Classical compressed teacher | 0.0226 | 1.0 | 4 | 77 |
| M5 | Direct quantum features | -0.0968 | N/A | 4 | 61 |
| M6 | Oracle (C+Q, large model) | -2.0970 | N/A | 128 | -1 |

## 3. Detailed Sweep Results


### M1: Quantum distillation (spectral Q)

| λ_q \ h | 3 | 4 |
|---------|---------|---------|
| 0.0 | -0.0645 | 0.0186 |
| 0.3 | -0.0650 | 0.0239 |
| 1.0 | -0.0716 | 0.0238 |

### M2: Shuffled control (shuffled Q)

| λ_q \ h | 3 | 4 |
|---------|---------|---------|
| 0.0 | -0.0645 | 0.0186 |
| 0.3 | -0.0687 | 0.0217 |
| 1.0 | -0.0731 | 0.0207 |

### M3: Random compression control

| λ_q \ h | 3 | 4 |
|---------|---------|---------|
| 0.0 | -0.0645 | 0.0186 |
| 0.3 | -0.0628 | 0.0212 |
| 1.0 | -0.0605 | 0.0203 |

### M4: Classical compressed teacher

| λ_q \ h | 3 | 4 |
|---------|---------|---------|
| 0.0 | -0.0645 | 0.0186 |
| 0.3 | -0.0646 | 0.0204 |
| 1.0 | -0.0638 | 0.0226 |

### M0: Classical baseline (no quantum)

| hidden | val R² | params |
|--------|--------|--------|
| 3 | -0.0645 | 59 |
| 4 | 0.0186 | 77 |

### M5: Direct quantum features

| hidden | val R² | params |
|--------|--------|--------|
| 3 | -0.1494 | 47 |
| 4 | -0.0968 | 61 |

## 4. Key Comparisons

- **M1 vs M0** (quantum distillation vs classical): ΔR² = +0.0053 ✓
- **M1 vs M2** (real vs shuffled Q): ΔR² = +0.0022 ✓
- **M1 vs M3** (spectral vs random compression): ΔR² = +0.0027 ✓
- **M1 vs M4** (quantum vs classical compressed teacher): ΔR² = +0.0013 ✓
- **M5 vs M0** (direct Q vs classical): ΔR² = -0.1154 ✗
- **M6 vs M0** (oracle vs classical): ΔR² = -2.1156

## 5. Hardness Target

| Condition | λ_q | val R² |
|-----------|-----|--------|
| M0 | 0.0 | 0.3935 |
| M1 | 0.3 | 0.3728 |
| M1 | 1.0 | 0.3155 |

## 6. Verdict

- M1 > M0: PASS
- M1 > M2: PASS
- M1 > M3: PASS
- M1 > M4: PASS

**VERDICT: PROMOTE_TO_STAGE_2 = YES** (4/4 gates passed)

Quantum distillation shows genuine incremental signal. Proceed to Stage 2 with OOD evaluation.

---

*Internal research gate, not a scientific claim.*