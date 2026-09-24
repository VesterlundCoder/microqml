# Pilot 0A Freeze Manifest

**Frozen**: 2026-09-23
**Status**: HISTORICAL — do not modify or re-run after parameter changes

## Parameters

| Parameter | Value |
|-----------|-------|
| n_graphs | 2000 |
| k values | [8, 12] |
| probe | P2 (medium) |
| solver_budget | 50 |
| seed | 42 |
| graph_families | ER, random regular, Watts-Strogatz, SBM |
| graph_size_range | N=30-80 |
| compression | spectral (normalized Laplacian + k-means) |
| quantum_backend | Qiskit Aer EstimatorV2 (exact) |
| observables | mu_Z, sigma_Z, mu_X, sigma_X, C_XX, C_ZZ, C_contrast, V_H (8 features) |
| classical_model | Ridge(alpha=1.0) |
| conditional_test | in-sample (no CV) |

## Solver Budget Allocation (at time of freeze)

```
spectral: budget=b (deterministic, free)
greedy: budget=b
multistart: budget=b, n_starts=5
annealing: budget=b*2
tabu: budget=max(b//3, 10), tabu_size=min(7, n//4+1)
```

## Results Summary

| Metric | k=8 | k=12 |
|--------|-----|------|
| R² classical | 0.2608 | 0.2608 |
| R² residual | 0.0402 | 0.0440 |
| Delta rho | 0.0081 | 0.0091 |
| R² combined | 0.2976 | 0.3048 |
| R² quantum only | 0.0604 | 0.0775 |
| Gate | PASS | PASS |
| Hardness R² classical | -3734.83 | -3734.83 |
| Hardness R² residual | 0.6867 | 0.7095 |
| Hardness Delta rho | 0.0334 | 0.0312 |
| Oracle R² (C+Q) | 0.5540 | 0.5479 |
| Oracle incremental | 0.0412 | 0.0351 |

## Solver Win Fractions

```
spectral: 0.0%
greedy: 31.8%
multistart: 6.7%
annealing: 5.2%
tabu: 56.4%
```

## Known Issues

1. **Hardness R²_C = -3734.83**: Catastrophic — likely target outliers + no train/test split
2. **Solver dominance**: tabu 56.4% > 55% threshold (gate failed)
3. **In-sample evaluation**: No cross-validation; all metrics are optimistic
4. **No permutation test**: No null distribution; gate threshold is arbitrary

## Files

- `qmt/results/PILOT_REPORT.md` — final report
- `qmt/results/exp0a_results.json` — raw results
- `qmt/results/Q_k8_P2.npy` — quantum features k=8
- `qmt/results/Q_k12_P2.npy` — quantum features k=12
- `qmt/results/X_classical.npy` — classical descriptors
- `qmt/results/Y_normalized.npy` — solver performance vectors
- `qmt/results/Y_hardness.npy` — hardness targets
- `qmt/results/Y_best_solver.npy` — best solver indices

## Environment

```
Python 3.10.12
Qiskit 2.5.2
Qiskit-Aer (EstimatorV2)
NumPy, SciPy, NetworkX, scikit-learn, PyTorch
```
