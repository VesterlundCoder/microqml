# Preregistered Hypotheses

**Registered**: 2026-09-23
**Before**: Stage 0C falsification experiments

## H1: Conditional quantum signal
$$H_1: R^2(C+Q) > R^2(C)$$

The quantum reservoir features Q contain incremental information about solver portfolio performance Y beyond cheap classical descriptors C.

**Primary measure**: $\Delta R^2_Q = R^2(C+Q) - R^2(C)$
**Pilot estimate**: $\Delta R^2 \approx 0.037 - 0.044$
**Gate**: Median $\Delta R^2_Q \geq 0.02$ across seeds, $CI_{95\%} > 0$

## H2: Quantum-specificity (not just nonlinear expansion)
$$H_2: R^2(C+Q) > R^2(C + R_{\text{classical}})$$

The quantum signal is not reproducible by matched classical feature transforms (Gaussian random projection, Random Fourier Features, random classical reservoir, trainable nonlinear baseline).

**Gate**: $\Delta R^2_Q - \Delta R^2_{\text{best classical}} > 0$ (CI of the difference)

## H3: Entanglement contribution
$$H_3: Q_{\text{entangled}} > Q_{\text{separable}}$$

Entangling gates (RXX) contribute to the signal beyond single-qubit dynamics.

**Measure**: $\Delta R^2$ for entangled vs separable circuits at matched depth/qubit count
**Sweep**: entanglement strength $e \in \{0, 0.25, 0.5, 0.75, 1.0\}$

## H4: OOD generalization
$$H_4: \Delta R^2_Q > 0 \text{ under distribution shift}$$

The quantum signal survives shifts in graph size, density, and family.

**Tests**:
- Size OOD: train N∈[50,100], test N∈[150,250]
- Family OOD: train {ER,BA,WS}, test {SBM}
- Density OOD: train p∈[0.1,0.3], test p∈[0.4,0.6]

## H5: Representation geometry predicts utility
$$H_5: \text{quantum representation geometry predicts downstream utility}$$

Properties of the Q representation (effective rank, CKA with C, kernel alignment with Y) predict when the reservoir provides conditional information.

**Measures**: $r_{\text{eff}}$, $CKA(C,Q)$, kernel alignment $K_Q$ vs $K_Y$
**Hypothesis**: There exists an optimum — "lagom komprimerad men geometriskt annorlunda" — not simply more rank = better.

## Statistical Protocol

- **Cross-validation**: 5-fold CV × 10 seeds
- **Permutation test**: B=1000 permutations of Q (or Y) within strata
- **Bootstrap CIs**: Over graph instances and experiment seeds
- **All transformations fitted inside fold**: scaling, PCA, regression, residualization

## Falsification Chain

$$\text{Pilot signal} \rightarrow \text{falsification} \rightarrow \text{quantum-specificity} \rightarrow \text{OOD} \rightarrow \text{representation geometry} \rightarrow \text{hardware}$$

## Possible Outcomes

| Result | Interpretation |
|--------|---------------|
| Q ≈ RFF/classical | Signal was nonlinear feature expansion, not quantum-specific |
| Q > classical, Q_ent ≈ Q_sep | Quantum-inspired feature map, but entanglement not the mechanism |
| Q_ent > Q_sep, Q > classical | Strong quantum-specific signal |
| + OOD + hardware | Compelling QML paper |
