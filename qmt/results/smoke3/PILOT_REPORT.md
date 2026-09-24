# PILOT REPORT: Experiment 0A — Global Quantum Micro-Reservoir

**Graphs**: 100 | **k values**: [8] | **Probe**: P2

**Date**: 2026-09-23 09:05:55


## 1. Dataset Validity

- Solver win fractions: {'spectral': np.float64(0.0), 'greedy': np.float64(0.05), 'multistart': np.float64(0.0), 'annealing': np.float64(0.0), 'tabu': np.float64(0.95)}
- Max fraction: 0.950
- Gate: FAIL

## Results for k=8

- Compression: spectral
- Q shape: [100, 8]
- Effective rank: 5.56
- Constant features: 0

### Conditional Residual Test (I(Q;Y|C))
- R^2 classical: 0.4897
- R^2 residual: 0.0954
- Delta rho: 0.0275
- R^2 combined: 0.5538
- R^2 quantum only: 0.1147
- Gate: PASS

### Conditional Residual on Hardness
- R^2 classical: -183.4046
- R^2 residual: 0.6774
- Delta rho: 0.0768
- Gate: PASS

### Oracle Diagnostic
- Oracle R^2 (Q): 0.7860
- Oracle R^2 (C): 0.8723
- Oracle R^2 (C+Q): 0.9337
- Incremental (oracle): 0.0613
- Incremental (ridge): 0.0361

## Recommendation

**PROMOTE_TO_FULL_STAGE_1 = YES**

At least one k value shows incremental quantum signal (R^2_residual >= 0.02 or delta_rho >= 0.03).
Proceed to full Stage 1 with all conditions (M0-M6), multiple compressors, and full parameter sweep.

---

*This is an internal pilot gate, not a scientific claim.*