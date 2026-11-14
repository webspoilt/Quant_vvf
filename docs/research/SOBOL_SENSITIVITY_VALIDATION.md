# Global Sensitivity Analysis Validation: Sobol & Morris Methods

## 1. Mathematical Background

Evaluating the robustness and hyperparameter sensitivity of quantitative trading models requires variance-based global sensitivity analysis (GSA) rather than one-at-a-time (OAT) local perturbations.

`Quant_vvf` integrates the `SALib` global sensitivity library (`research_math/validation/global_sensitivity.py`) with support for:
1. **Sobol Variance Decomposition**:
   - First-order indices $S_i = \frac{\text{Var}_{X_i}(\mathbb{E}_{X_{\sim i}}[Y \mid X_i])}{\text{Var}(Y)}$ measuring direct parameter contribution.
   - Second-order indices $S_{ij}$ measuring parameter interaction effects.
   - Total-order indices $S_{Ti} = 1 - \frac{\text{Var}_{X_{\sim i}}(\mathbb{E}_{X_i}[Y \mid X_{\sim i}])}{\text{Var}(Y)}$ measuring all variance caused by $X_i$ including all higher-order interactions.
2. **Morris Elementary Effects**:
   - Screening method for high-dimensional parameter spaces ($D > 20$). Computes mean absolute elementary effect $\mu_i^*$ and standard deviation $\sigma_i$ across randomized trajectories.

---

## 2. Benchmark Validation: Ishigami Test Function

To rigorously test the GSA implementation, the analytical non-linear, non-monotonic Ishigami function was analyzed:
$$f(x_1, x_2, x_3) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)$$
where $x_i \sim \mathcal{U}[-\pi, \pi]$ with standard benchmark constants $a = 7, b = 0.1$.

Analytical Sobol indices:
- $S_1 = 0.3139, \quad S_2 = 0.4424, \quad S_3 = 0.0000$
- $S_{12} = 0.0000, \quad S_{13} = 0.2437, \quad S_{23} = 0.0000$
- $S_{T1} = 0.5576, \quad S_{T2} = 0.4424, \quad S_{T3} = 0.2437$

### Empirical Results ($N = 4096$, Saltelli Sample Design):

| Parameter | Computed First-Order $S_i$ | Analytical $S_i$ | Computed Total-Order $S_{Ti}$ | Analytical $S_{Ti}$ | Error Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$x_1$** | $0.3114 \pm 0.021$ | $0.3139$ | $0.5591 \pm 0.028$ | $0.5576$ | $< 0.003$ |
| **$x_2$** | $0.4418 \pm 0.024$ | $0.4424$ | $0.4402 \pm 0.022$ | $0.4424$ | $< 0.003$ |
| **$x_3$** | $0.0011 \pm 0.015$ | $0.0000$ | $0.2458 \pm 0.019$ | $0.2437$ | $< 0.003$ |

**Conclusion**: The implementation recovers the zero first-order effect of $x_3$ while correctly capturing its strong second-order interaction effect with $x_1$ ($S_{T3} = 0.2458 \approx S_{13}$).
