# Quant_vvf Advanced Monte Carlo, Pricing & Quantitative Simulation Architecture

## 1. Mathematical Foundations & Stochastic Models

The `research_math/monte_carlo/` engine provides reusable, vectorized path simulation across continuous and discontinuous stochastic differential equations (SDEs):

### 1.1 Geometric Brownian Motion (GBM)
$$\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\,dW_t^{\mathbb{Q}}$$

### 1.2 Heston Stochastic Volatility
$$\frac{dS_t}{S_t} = (r - q)\,dt + \sqrt{v_t}\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \quad \mathbb{E}[dW_t^S dW_t^v] = \rho\,dt$$
- Discretization: **Andersen (2008) Quadratic-Exponential (QE)** scheme with switching variance threshold $\psi_c = 1.5$ and Broadie-Kaya predictor-corrector asset propagation.

### 1.3 Merton Jump Diffusion
$$\frac{dS_t}{S_{t-}} = (r - q - \lambda k)\,dt + \sigma\,dW_t + (Y - 1)\,dN_t$$
where $\ln(Y) \sim \mathcal{N}(\mu_j, \sigma_j^2)$ and $k = \exp(\mu_j + \frac{1}{2}\sigma_j^2) - 1$.

### 1.4 Kou Double Exponential Jump Diffusion
Asymmetric upward/downward log jumps:
$$f_Y(y) = p\,\eta_1 e^{-\eta_1 y}\mathbf{1}_{\{y \ge 0\}} + (1 - p)\,\eta_2 e^{\eta_2 y}\mathbf{1}_{\{y < 0\}}$$
$$k = p\frac{\eta_1}{\eta_1 - 1} + (1 - p)\frac{\eta_2}{\eta_2 + 1} - 1, \quad \eta_1 > 1$$

### 1.5 Bates Model
Unifies Heston stochastic variance with compound Poisson Merton jumps.

---

## 2. Deribit Inverse BTC/USD Derivatives Valuation

For crypto derivatives settled in Bitcoin, payoff functions invert the fiat denomination:
- **Inverse Call Payoff:** $\frac{\max(S_T - K, 0)}{S_T}$ (settled in BTC)
- **Inverse Put Payoff:** $\frac{\max(K - S_T, 0)}{S_T}$ (settled in BTC)

---

## 3. Variance Reduction Benchmark Summary

| Method | Pricing Estimate | Standard Error | Variance Reduction Factor | Efficiency Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Standard MC** | 6312.4 | 42.1 | **1.0x** | 1.0x |
| **Antithetic Variates** | 6311.9 | 28.4 | **2.2x** | 1.9x |
| **Moment Matching** | 6311.6 | 18.2 | **5.3x** | 4.8x |
| **Control Variates** | 6311.5 | 4.1 | **105.6x** | **84.2x** |
| **Scrambled Sobol QMC**| 6311.5 | 6.2 | **46.1x** | 39.4x |
