# PILOT REPORT: Experiment 0A — Global Quantum Micro-Reservoir

**Graphs**: 2000 | **k values**: [8, 12] | **Probe**: P2

**Date**: 2026-09-23 09:26:46


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.318), 'multistart': np.float64(0.067), 'annealing': np.float64(0.0515), 'tabu': np.float64(0.5635)}
- Max fraction: 0.564
- Gate: FAIL

## Results for k=8

- Compression: spectral
- Q shape: [2000, 8]
- Effective rank: 5.79
- Constant features: 0

### Conditional Residual Test (I(Q;Y|C))
- R^2 classical: 0.2608
- R^2 residual: 0.0402
- Delta rho: 0.0081
- R^2 combined: 0.2976
- R^2 quantum only: 0.0604
- Gate: PASS

### Conditional Residual on Hardness
- R^2 classical: -3734.8338
- R^2 residual: 0.6867
- Delta rho: 0.0334
- Gate: PASS

### Oracle Diagnostic
- Oracle R^2 (Q): 0.2668
- Oracle R^2 (C): 0.5128
- Oracle R^2 (C+Q): 0.5540
- Incremental (oracle): 0.0412
- Incremental (ridge): 0.0187

## Results for k=12

- Compression: spectral
- Q shape: [2000, 8]
- Effective rank: 5.21
- Constant features: 0

### Conditional Residual Test (I(Q;Y|C))
- R^2 classical: 0.2608
- R^2 residual: 0.0440
- Delta rho: 0.0091
- R^2 combined: 0.3048
- R^2 quantum only: 0.0775
- Gate: PASS

### Conditional Residual on Hardness
- R^2 classical: -3734.8338
- R^2 residual: 0.7095
- Delta rho: 0.0312
- Gate: PASS

### Oracle Diagnostic
- Oracle R^2 (Q): 0.2596
- Oracle R^2 (C): 0.5128
- Oracle R^2 (C+Q): 0.5479
- Incremental (oracle): 0.0351
- Incremental (ridge): 0.0226

## Recommendation

**PROMOTE_TO_FULL_STAGE_1 = YES**

At least one k value shows incremental quantum signal (R^2_residual >= 0.02 or delta_rho >= 0.03).
Proceed to full Stage 1 with all conditions (M0-M6), multiple compressors, and full parameter sweep.

---

*This is an internal pilot gate, not a scientific claim.*