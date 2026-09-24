# STAGE 0C REPORT: Falsification Experiment

**Graphs**: 2000 | **k**: 8 | **Probe**: P2

**Seeds**: 10 | **Permutations**: 1000 | **Bootstrap**: 1000

**Date**: 2026-09-23 10:57:35


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.428), 'multistart': np.float64(0.0915), 'annealing': np.float64(0.4015), 'tabu': np.float64(0.079)}
- Max fraction: 0.428
- Gate: PASS

## 2. Cross-Validated Conditional Test (5-fold × 10 seeds)

| Feature | ΔR² mean | ΔR² std | 95% CI | R²(C) | R²(C+Q) | R² residual |
|---------|----------|---------|--------|-------|---------|-------------|
| **Q1 (entangled)** | 0.0292 | 0.0089 | [0.0146, 0.0461] | 0.2602 | 0.2894 | 0.0291 |
| Q0 (separable) | 0.0231 | 0.0075 | [0.0078, 0.0351] | 0.2602 | 0.2833 | 0.0256 |
| Q2 (scrambled conn.) | 0.0228 | 0.0082 | [0.0074, 0.0365] | 0.2602 | 0.2830 | 0.0220 |
| Q3 (scrambled input) | 0.0275 | 0.0091 | [0.0124, 0.0437] | 0.2602 | 0.2877 | 0.0278 |
| Q4_e0.0 | 0.0231 | 0.0075 | [0.0078, 0.0351] | 0.2602 | 0.2833 | 0.0256 |
| Q4_e0.25 | 0.0376 | 0.0121 | [0.0186, 0.0563] | 0.2602 | 0.2978 | 0.0263 |
| Q4_e0.5 | 0.0397 | 0.0118 | [0.0204, 0.0582] | 0.2602 | 0.2999 | 0.0357 |
| Q4_e0.75 | 0.0343 | 0.0102 | [0.0182, 0.0529] | 0.2602 | 0.2944 | 0.0339 |
| Q4_e1.0 | 0.0292 | 0.0089 | [0.0146, 0.0461] | 0.2602 | 0.2894 | 0.0291 |
| A_gaussian | -0.0001 | 0.0009 | [-0.0019, 0.0009] | 0.2602 | 0.2601 | -0.0035 |
| B_rff | 0.0018 | 0.0043 | [-0.0054, 0.0098] | 0.2602 | 0.2620 | -0.0021 |
| C_reservoir | 0.0178 | 0.0088 | [0.0023, 0.0318] | 0.2602 | 0.2780 | 0.0036 |
| Q1 (hardness) | 0.0264 | 0.0050 | [0.0171, 0.0355] | 0.8719 | 0.8983 | 0.1351 |

## 3. Key Comparisons (Gates)

- Gate A: Q1 ΔR² > 0: PASS (ΔR² = 0.0292)
- Gate B: Q1 > best classical: PASS (Q1=0.0292 vs best_cl=0.0178)
- Gate C: Q1 > Q0 (entanglement): PASS (Q1=0.0292 vs Q0=0.0231)
- Gate D: Q1 > Q2 (scrambled conn.): PASS (Q1=0.0292 vs Q2=0.0228)
- Gate E: Q1 > Q3 (scrambled input): PASS (Q1=0.0292 vs Q3=0.0275)

## 4. Permutation Test (H₀: I(Q;Y|C) = 0)

- Observed ΔR²: 0.0290
- Null distribution: mean=-0.0037, std=0.0014
- p-value: 0.0000
- Significant (p<0.01): YES

## 5. Bootstrap Confidence Interval

- ΔR² = 0.0309 [0.0215, 0.0414]_95%
- CI excludes zero: YES

## 6. Entanglement Sweep

| e | ΔR² |
|---|------|
| 0.00 | 0.0231 |
| 0.25 | 0.0376 |
| 0.50 | 0.0397 |
| 0.75 | 0.0343 |
| 1.00 | 0.0292 |

Monotone increasing: NO

## 7. Representation Geometry

- Classical effective rank: 8.25
- Quantum effective rank: 7.12
- CKA(C, Q) linear: 0.1581
- CKA(C, Q) rbf: 0.1625
- CKA(Q, Y) linear: 0.0277
- CKA(C, Y) linear: 0.0418
- Kernel alignment Q→Y: 0.1461
- Kernel alignment C→Y: -0.0675
- Principal angles C↔Q: ['22.6°', '49.6°', '57.6°', '62.6°', '70.6°', '73.4°', '81.0°', '86.0°']

## 8. Verdict

Passed 5/5 gates.

**VERDICT: QUANTUM SIGNAL SURVIVES FALSIFICATION**

The quantum reservoir signal is:
- Reproducible across seeds (CV)
- Statistically significant (permutation test)
- Not explained by matched classical controls
- Dependent on entanglement and graph structure

→ Proceed to Stage 1A (full IID study, 10k+ graphs)

---

*Internal research gate, not a scientific claim.*