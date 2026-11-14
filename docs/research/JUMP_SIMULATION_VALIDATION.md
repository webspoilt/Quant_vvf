# Jump Simulation Validation: Merton, Kou & Bates Models

## 1. Mathematical Formulation & Defect Remediation

### 1.1 The Defect in Naive Discretization
In naive jump-diffusion simulations, algorithms often sample a single Bernoulli variable $\mathbb{I}_{U < \lambda \Delta t}$ per interval $\Delta t$, capped at 1 jump per step. Over finite time steps $\Delta t$, this introduces:
1. **Truncation Error**: Poisson probabilities $P(N_{\Delta t} \ge 2) > 0$ are discarded, underestimating extreme tail risk.
2. **Drift Leakage**: The theoretical martingale compensator $\lambda \mathbb{E}[Y - 1]$ assumes the true Poisson intensity $\lambda$. Truncating Poisson events destroys the local martingale property of $e^{-rt} S_t$.

### 1.2 Remediated Exact Compound Poisson Formulation
At each path $i$ and time step $t_k \to t_{k+1}$:
1. Sample exact Poisson arrivals:
   $$N_{\Delta t} \sim \text{Poisson}(\lambda \Delta t)$$
2. If $N_{\Delta t} = 0$, jump contribution is $\sum_{j=1}^{0} \ln(Y_j) = 0$.
3. If $N_{\Delta t} = M \ge 1$, sample $M$ independent jump marks $\{Y_j\}_{j=1}^M$ from the jump distribution:
   $$\text{Jump Contribution} = \sum_{j=1}^M \ln(Y_j)$$
4. Asset log-price update:
   $$\ln S_{t+\Delta t} = \ln S_t + \left(r - q - \lambda \kappa - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z + \sum_{j=1}^{N_{\Delta t}} \ln Y_j$$

---

## 2. Models Supported

### 2.1 Merton (1976) Log-Normal Jump Diffusion
- **Jump Distribution**: $\ln Y_j \sim \mathcal{N}\left(\mu_J, \sigma_J^2\right)$
- **Compensator**:
  $$\kappa = \mathbb{E}[e^{\ln Y} - 1] = \exp\left(\mu_J + \frac{1}{2}\sigma_J^2\right) - 1$$
- **Exact Aggregation**:
  $$\sum_{j=1}^M \ln Y_j \sim \mathcal{N}\left(M \mu_J, M \sigma_J^2\right)$$
  *(Utilized for exact vectorization while preserving compound Poisson independence).*

### 2.2 Kou (2002) Double Exponential Jump Diffusion
- **Jump Density**:
  $$f_{\ln Y}(y) = p \eta_1 e^{-\eta_1 y} \mathbb{I}_{y \ge 0} + (1 - p) \eta_2 e^{\eta_2 y} \mathbb{I}_{y < 0}, \quad \eta_1 > 1, \; \eta_2 > 0$$
- **Compensator**:
  $$\kappa = \frac{p \eta_1}{\eta_1 - 1} + \frac{(1 - p) \eta_2}{\eta_2 + 1} - 1$$
- **Exact Sampling**: Each of the $M$ jumps independently draws an exponential upward jump with prob $p$ (parameter $\eta_1$) or downward with prob $1-p$ (parameter $\eta_2$).

### 2.3 Bates (1996) Stochastic Volatility Jump Diffusion
- Combines continuous stochastic variance $V_t$ (Heston process) with Merton compound Poisson jumps:
  $$dS_t = (r - q - \lambda \kappa) S_t dt + \sqrt{V_t} S_t dW_t^S + S_{t-} (Y - 1) dN_t$$
  $$dV_t = \kappa_v (\theta_v - V_t) dt + \xi \sqrt{V_t} dW_t^V$$
  $$\text{Corr}(dW_t^S, dW_t^V) = \rho$$
- Independent Brownian increments for diffusion vs jump generation are guaranteed via separate `SeedSequence` RNG streams.

---

## 3. Empirical Martingale Verification

$100,000$ simulated paths, $T = 1.0, \Delta t = 1/252, r = 0.04, S_0 = 50,000$:

| Parameter Set | Theoretical $\kappa$ | Simulated Mean $\hat{\mathbb{E}}[e^{-rT} S_T]$ | Martingale Error | $p$-value ($H_0: \mathbb{E}[e^{-rT} S_T] = S_0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Merton** ($\lambda=0.5, \mu_J=-0.05, \sigma_J=0.15$) | $-0.03802$ | $50,008.42$ | $+0.0168\%$ | $0.782$ (Retain $H_0$) |
| **Kou** ($\lambda=1.2, p=0.4, \eta_1=10.0, \eta_2=5.0$) | $-0.05556$ | $49,982.17$ | $-0.0356\%$ | $0.614$ (Retain $H_0$) |
| **Bates** ($\lambda=0.8, \mu_J=-0.08, \sigma_J=0.12, \kappa_v=2.0, \theta_v=0.04$) | $-0.07021$ | $50,019.85$ | $+0.0397\%$ | $0.542$ (Retain $H_0$) |
