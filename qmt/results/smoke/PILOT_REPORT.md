# PILOT REPORT: Experiment 0A — Global Quantum Micro-Reservoir

**Graphs**: 50 | **k values**: [8] | **Probe**: P2

**Date**: 2026-09-23 09:03:01


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.06), 'multistart': np.float64(0.08), 'annealing': np.float64(0.02), 'tabu': np.float64(0.84)}
- Max fraction: 0.840
- Gate: FAIL

## Results for k=8

- Compression: spectral
- Q shape: [50, 8]
- Effective rank: 5.46
- Constant features: 0

### Conditional Residual Test (I(Q;Y|C))
- R^2 classical: 0.5876
- R^2 residual: 0.1226
- Delta rho: nan
- R^2 combined: 0.6622
- R^2 quantum only: 0.2132
- Gate: PASS

### Conditional Residual on Hardness
- R^2 classical: -95.3548
- R^2 residual: 0.6731
- Delta rho: 0.0116
- Gate: PASS

### Oracle Diagnostic
- Oracle R^2 (Q): 0.7232
- Oracle R^2 (C): 0.8155
- Oracle R^2 (C+Q): 0.8178
- Incremental (oracle): 0.0023
- Incremental (ridge): 0.0325

## Recommendation

**PROMOTE_TO_FULL_STAGE_1 = YES**

At least one k value shows incremental quantum signal (R^2_residual >= 0.02 or delta_rho >= 0.03).
Proceed to full Stage 1 with all conditions (M0-M6), multiple compressors, and full parameter sweep.

---

*This is an internal pilot gate, not a scientific claim.*