# STAGE 1A REPORT: Full IID Study + OOD Generalization

**Graphs**: 100 IID + 50 OOD-size + 50 OOD-family
 | **k**: 8 | **Probe**: P2

**Seeds**: 2 | **Permutations**: 20 | **Bootstrap**: 20

**Date**: 2026-09-23 11:07:40


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

## 3. Hypothesis Tests (Gates)

- H1: Q1 ΔR² > 0: FAIL (ΔR² = -0.0709)
- H2: Q1 > best classical: FAIL (Q1=-0.0709 vs best_cl=-0.0063)
- H3: Q1 > Q0 (entanglement): FAIL (Q1=-0.0709 vs Q0=0.0059)
- H3: Q1 > Q2 (scrambled conn.): PASS (Q1=-0.0709 vs Q2=-0.0805)
- H3: Q1 > Q3 (scrambled input): FAIL (Q1=-0.0709 vs Q3=-0.0335)

## 4. Permutation Test (H₀: I(Q;Y|C) = 0)

- Observed ΔR²: -0.0710
- Null distribution: mean=-0.1301, std=0.0525
- p-value: 0.1000
- Significant (p<0.01): NO

## 5. Bootstrap Confidence Interval

- ΔR² = -0.0346 [-0.1164, 0.0629]_95%
- CI excludes zero: NO

## 6. Entanglement Sweep (H3)

| e | ΔR² |
|---|------|
| 0.00 | 0.0059 |
| 0.25 | 0.0269 |
| 0.50 | -0.0240 |
| 0.75 | -0.0649 |
| 1.00 | -0.0709 |

Monotone increasing: NO
Optimal entanglement: e=0.25 (ΔR²=0.0269)

## 7. Representation Geometry (H5)

- Classical effective rank: 8.01
- Quantum effective rank: 6.96
- CKA(C, Q) linear: 0.2078
- CKA(C, Q) rbf: 0.1646
- CKA(Q, Y) linear: 0.0436
- CKA(C, Y) linear: 0.0273
- Kernel alignment Q→Y: 0.1789
- Kernel alignment C→Y: 0.0114
- Principal angles C↔Q: ['21.2°', '45.3°', '53.4°', '57.7°', '63.1°', '72.3°', '79.2°', '83.5°']

## 8. Feature Attribution

- Mean per-graph gain: -0.0005
- Median per-graph gain: -0.0005
- Positive gains (Q helped): 38
- Negative gains (Q hurt): 62
- Attribution R² (gain ~ graph properties): 0.1392

### Top property correlations with gain:

| Property | Correlation |
|----------|------------|
| degree_variance | -0.1285 |
| diameter | 0.1134 |
| n_components | -0.1118 |
| degree_std | -0.0964 |
| n_edges | -0.0872 |
| degree_mean | -0.0723 |
| n_nodes | 0.0469 |
| edge_density | -0.0386 |

### Regression coefficients:

| Property | Coefficient |
|----------|------------|
| degree_variance | -0.0024 |
| n_nodes | 0.0021 |
| modularity | -0.0016 |
| n_edges | -0.0015 |
| clustering_mean | 0.0011 |
| edge_density | 0.0009 |
| degree_std | 0.0006 |
| degree_mean | -0.0006 |

## 9. Solver-Boundary Analysis

- Correlation (margin, gain): 0.0940
- Hypothesis: ΔR² ↓ when margin → 0
- Hypothesis supported: NO

| Category | N | Mean gain | Std | Frac positive |
|----------|---|-----------|-----|---------------|
| boundary | 25 | -0.0006 | 0.0032 | 0.400 |
| medium | 25 | -0.0012 | 0.0038 | 0.240 |
| easy | 50 | -0.0002 | 0.0052 | 0.440 |

## 10. OOD Generalization (H4)

### OOD-Size (train N∈[30,80], test N∈[150,250])

- N: 50
- ΔR² = -1.4237 ± 3.8137 [-10.1054, 0.0455]
- R²(C) = 0.2580
- R²(C+Q) = -1.1657
- ΔR² > 0: FAIL
- CI excludes zero: FAIL

### OOD-Family (train {ER, regular, WS}, test {BA, geometric, config})

- N: 48
- Families: ['ba', 'geometric', 'config']
- ΔR² = -0.1560 ± 0.1210 [-0.3748, 0.0272]
- R²(C) = 0.2341
- R²(C+Q) = 0.0781
- ΔR² > 0: FAIL
- CI excludes zero: FAIL

#### Per-family breakdown:

| Family | N | ΔR² | R²(C) | R²(C+Q) |
|--------|---|-----|-------|---------|

## 11. Verdict

- H1: ΔR² > 0: FAIL
- H2: Q > classical: FAIL
- H3: Q1 > Q0: FAIL
- Permutation p<0.01: FAIL
- Bootstrap CI > 0: FAIL
- H4: OOD-size ΔR² > 0: FAIL
- H4: OOD-family ΔR² > 0: FAIL

Passed 0/7 gates.

**VERDICT: SIGNAL DOES NOT SURVIVE AT SCALE**

The 2k-graph signal was a finite-sample effect.

---

*Internal research gate, not a scientific claim.*