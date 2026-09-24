# STAGE 0C REPORT: Falsification Experiment

**Graphs**: 100 | **k**: 8 | **Probe**: P2

**Seeds**: 2 | **Permutations**: 20 | **Bootstrap**: 20

**Date**: 2026-09-23 10:24:17


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.45), 'multistart': np.float64(0.07), 'annealing': np.float64(0.43), 'tabu': np.float64(0.05)}
- Max fraction: 0.450
- Gate: PASS

## 2. Cross-Validated Conditional Test (5-fold × 2 seeds)

| Feature | ΔR² mean | ΔR² std | 95% CI | R²(C) | R²(C+Q) | R² residual |
|---------|----------|---------|--------|-------|---------|-------------|
| **Q1 (entangled)** | -0.0709 | 0.0534 | [-0.1500, -0.0145] | 0.0797 | 0.0088 | -0.1075 |
| Q0 (separable) | 0.0059 | 0.0558 | [-0.0608, 0.1141] | 0.0797 | 0.0857 | -0.0550 |
| Q2 (scrambled conn.) | -0.0805 | 0.0690 | [-0.1801, -0.0070] | 0.0797 | -0.0008 | -0.1223 |
| Q3 (scrambled input) | -0.0335 | 0.0475 | [-0.1028, 0.0374] | 0.0797 | 0.0462 | -0.0838 |
| Q4_e0.0 | 0.0059 | 0.0558 | [-0.0608, 0.1141] | 0.0797 | 0.0857 | -0.0550 |
| Q4_e0.25 | 0.0269 | 0.0783 | [-0.0938, 0.1748] | 0.0797 | 0.1066 | -0.0531 |
| Q4_e0.5 | -0.0240 | 0.1617 | [-0.3891, 0.1053] | 0.0797 | 0.0557 | -0.0777 |
| Q4_e0.75 | -0.0649 | 0.0718 | [-0.1748, 0.0244] | 0.0797 | 0.0148 | -0.0875 |
| Q4_e1.0 | -0.0709 | 0.0534 | [-0.1500, -0.0145] | 0.0797 | 0.0088 | -0.1075 |
| A_gaussian | -0.0063 | 0.0138 | [-0.0378, 0.0043] | 0.0797 | 0.0735 | -0.0634 |
| B_rff | -0.1648 | 0.1515 | [-0.5078, -0.0281] | 0.0797 | -0.0851 | -0.1595 |
| C_reservoir | -0.0535 | 0.0665 | [-0.2064, -0.0112] | 0.0797 | 0.0262 | -0.0809 |
| Q1 (hardness) | 0.0634 | 0.0544 | [-0.0059, 0.1736] | 0.7800 | 0.8434 | 0.1286 |

## 3. Key Comparisons (Gates)

- Gate A: Q1 ΔR² > 0: FAIL (ΔR² = -0.0709)
- Gate B: Q1 > best classical: FAIL (Q1=-0.0709 vs best_cl=-0.0063)
- Gate C: Q1 > Q0 (entanglement): FAIL (Q1=-0.0709 vs Q0=0.0059)
- Gate D: Q1 > Q2 (scrambled conn.): PASS (Q1=-0.0709 vs Q2=-0.0805)
- Gate E: Q1 > Q3 (scrambled input): FAIL (Q1=-0.0709 vs Q3=-0.0335)

## 4. Permutation Test (H₀: I(Q;Y|C) = 0)

- Observed ΔR²: -0.0710
- Null distribution: mean=-0.1301, std=0.0525
- p-value: 0.1000
- Significant (p<0.01): NO

## 5. Bootstrap Confidence Interval

- ΔR² = -0.0346 [-0.1164, 0.0629]_95%
- CI excludes zero: NO

## 6. Entanglement Sweep

| e | ΔR² |
|---|------|
| 0.00 | 0.0059 |
| 0.25 | 0.0269 |
| 0.50 | -0.0240 |
| 0.75 | -0.0649 |
| 1.00 | -0.0709 |

Monotone increasing: NO

## 7. Representation Geometry

- Classical effective rank: 8.01
- Quantum effective rank: 6.96
- CKA(C, Q) linear: 0.2078
- CKA(C, Q) rbf: 0.1646
- CKA(Q, Y) linear: 0.0436
- CKA(C, Y) linear: 0.0273
- Kernel alignment Q→Y: 0.1789
- Kernel alignment C→Y: 0.0114
- Principal angles C↔Q: ['21.2°', '45.3°', '53.4°', '57.7°', '63.1°', '72.3°', '79.2°', '83.5°']

## 8. Verdict

Passed 1/5 gates.

**VERDICT: SIGNAL FALSIFIED**

The quantum signal does not survive classical or quantum controls.

→ The pilot signal was likely nonlinear feature expansion, not quantum-specific.

---

*Internal research gate, not a scientific claim.*