# Quant_vvf Walk-Forward Optimization & Parameter Sensitivity Framework

## 1. Chronological Strategy Validation & Deflated Sharpe

The `backtesting/walk_forward/engine.py` module enforces strict out-of-sample chronological walk-forward boundaries:
1. Training window fits candidate parameters.
2. Parameter freezing on validation evaluation.
3. Out-of-sample execution on untouched subsequent testing windows.
4. **Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR)** adjusts for selection bias across multiple trial evaluations ($N_{\text{trials}}$):
$$\text{DSR} = \Phi\left(\frac{\widehat{\text{SR}} - \text{SR}^*}{\sqrt{\text{Var}(\widehat{\text{SR}})}}\right)$$
where $\text{SR}^* = \sqrt{2\ln(N_{\text{trials}})} - \frac{\gamma + \ln(\ln(N_{\text{trials}}))}{\sqrt{2\ln(N_{\text{trials}})}}$.

---

## 2. Parameter Sensitivity & Robust Plateaus

The `research_math/validation/parameter_sensitivity.py` analyzer decomposes parameters into:
- **Local Elasticity:** $\epsilon = \frac{\partial \text{Metric} / \text{Metric}}{\partial \theta / \theta}$
- **Sobol Variance Decomposition:** First-order ($S_i$) and total effect ($S_{Ti}$) sensitivity indices.
- **Plateau Robustness:** Broad regions of parameter stability ($\pm 20\%$ parameter variation yielding $\le 10\%$ performance drop) versus fragile historical overfitted peaks.
- **Pareto Frontiers:** Multi-objective non-dominated solutions balancing Sharpe, Max Drawdown, and Portfolio Turnover.
