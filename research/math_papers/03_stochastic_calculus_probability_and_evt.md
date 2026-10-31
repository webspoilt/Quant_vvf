# Stochastic Calculus, Probability Theory, and Extreme Value Theory

**Module:** `research/math_papers/03_stochastic_calculus_probability_and_evt.md`  
**Classification:** Advanced Mathematical Foundations for Quantitative Finance  
**Target Architecture:** `Quant_vvf` Research Engine  

---

## 1. Multidimensional Stochastic Calculus and Itô's Lemma

### 1.1 The General Multidimensional Itô Process
Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \ge 0}, \mathbb{P})$ be a filtered probability space satisfying the usual conditions, supporting an $m$-dimensional Brownian motion $\mathbf{W}_t = (W_t^1, \dots, W_t^m)^T$ with independent components ($d\langle W^i, W^j \rangle_t = \delta_{ij} dt$).

Consider an $n$-dimensional Itô diffusion process $\mathbf{X}_t \in \mathbb{R}^n$ defined by the system of SDEs:
$$
d\mathbf{X}_t = \boldsymbol{\mu}(t, \mathbf{X}_t) \, dt + \boldsymbol{\Sigma}(t, \mathbf{X}_t) \, d\mathbf{W}_t
$$
where $\boldsymbol{\mu}: [0, T] \times \mathbb{R}^n \to \mathbb{R}^n$ is the drift vector, and $\boldsymbol{\Sigma}: [0, T] \times \mathbb{R}^n \to \mathbb{R}^{n \times m}$ is the dispersion/diffusion matrix.

The instantaneous covariance matrix of the state increments is:
$$
\mathbf{A}(t, \mathbf{X}_t) = \boldsymbol{\Sigma}(t, \mathbf{X}_t) \boldsymbol{\Sigma}(t, \mathbf{X}_t)^T \in \mathbb{R}^{n \times n}, \quad A_{ij} = \sum_{k=1}^m \Sigma_{ik} \Sigma_{jk}
$$

### 1.2 Multidimensional Itô's Formula
Let $f \in C^{1, 2}([0, T] \times \mathbb{R}^n; \mathbb{R})$. Then $Y_t = f(t, \mathbf{X}_t)$ is an Itô process with stochastic differential:
$$
df(t, \mathbf{X}_t) = \left( \frac{\partial f}{\partial t} + \nabla_{\mathbf{x}} f^T \boldsymbol{\mu} + \frac{1}{2} \text{Tr}\left( \mathbf{A}(t, \mathbf{X}_t) \mathbf{H}_f(t, \mathbf{X}_t) \right) \right) dt + \nabla_{\mathbf{x}} f^T \boldsymbol{\Sigma} \, d\mathbf{W}_t
$$
where $\nabla_{\mathbf{x}} f = \left( \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right)^T$ is the gradient, and $\mathbf{H}_f = \left( \frac{\partial^2 f}{\partial x_i \partial x_j} \right)_{i, j=1}^n$ is the Hessian matrix.

In index notation:
$$
df(t, \mathbf{X}_t) = \left( \frac{\partial f}{\partial t} + \sum_{i=1}^n \mu_i \frac{\partial f}{\partial x_i} + \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n A_{ij} \frac{\partial^2 f}{\partial x_i \partial x_j} \right) dt + \sum_{i=1}^n \sum_{k=1}^m \frac{\partial f}{\partial x_i} \Sigma_{ik} \, dW_t^k
$$

---

## 2. Parabolic PDEs and the Multidimensional Feynman-Kac Theorem

### 2.1 The Infinitesimal Generator
The infinitesimal generator $\mathcal{A}$ of the diffusion $\mathbf{X}_t$ acting on test functions $\phi \in C_0^2(\mathbb{R}^n)$ is the second-order elliptic partial differential operator:
$$
\mathcal{A}\phi(\mathbf{x}) = \sum_{i=1}^n \mu_i(\mathbf{x}) \frac{\partial \phi}{\partial x_i} + \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n A_{ij}(\mathbf{x}) \frac{\partial^2 \phi}{\partial x_i \partial x_j}
$$

### 2.2 The Multidimensional Feynman-Kac Theorem
Consider the terminal value problem for the parabolic partial differential equation:
$$
\frac{\partial V}{\partial t} + \mathcal{A}V - r(\mathbf{x}, t) V = 0, \quad (t, \mathbf{x}) \in [0, T) \times \mathbb{R}^n
$$
subject to terminal condition $V(T, \mathbf{x}) = \Psi(\mathbf{x})$.

If $V(t, \mathbf{x})$ satisfies polynomial growth conditions, then it admits the stochastic representation:
$$
V(t, \mathbf{x}) = \mathbb{E}_{\mathbf{x}}\left[ \exp\left(-\int_t^T r(\mathbf{X}_s, s) \, ds\right) \Psi(\mathbf{X}_T) \;\middle|\; \mathbf{X}_t = \mathbf{x} \right]
$$
This fundamental theorem forms the rigorous bridge between PDE numerical solvers (Finite Differences, ADI) and Monte Carlo stochastic integration.

### 2.3 Kolmogorov Forward (Fokker-Planck) Equation
While Feynman-Kac solves backward conditional expectations, the transition probability density $p(t, \mathbf{x}; T, \mathbf{y})$ satisfies the **Kolmogorov Forward Equation** in $(T, \mathbf{y})$:
$$
\frac{\partial p}{\partial T} = -\sum_{i=1}^n \frac{\partial}{\partial y_i}\left[ \mu_i(T, \mathbf{y}) p \right] + \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \frac{\partial^2}{\partial y_i \partial y_j} \left[ A_{ij}(T, \mathbf{y}) p \right]
$$
The formal adjoint operator $\mathcal{A}^*$ describes the forward time-evolution of probability mass across asset space.

---

## 3. Stochastic Volatility and Jump Processes

### 3.1 The Heston (1993) Continuous Stochastic Volatility Model
The asset price $S_t$ and variance $v_t$ evolve under the risk-neutral measure $\mathbb{Q}$ as:
$$
dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S
$$
$$
dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v
$$
with correlation $d\langle W^S, W^v \rangle_t = \rho \, dt$.
* **Feller Condition:** $2\kappa\theta > \xi^2$.
  When satisfied, the boundary $v = 0$ is strictly inaccessible, guaranteeing $v_t > 0$ almost surely. If breached, the variance process reaches zero and is instantaneously reflected.

### 3.2 Jump-Diffusion Dynamics: Merton (1976) and Kou (2002)
To capture sudden regime shifts and crypto flash crashes, Poisson jumps are coupled to the continuous diffusion:
$$
\frac{dS_t}{S_{t^-}} = (r - \lambda k) \, dt + \sigma \, dW_t + (J_t - 1) \, dN_t
$$
where $N_t$ is a Poisson process with jump intensity $\lambda$, $J_t$ is the relative jump size, and $k = \mathbb{E}[J - 1]$ is the mean jump percentage.
1. **Merton (1976):** $\ln(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$ (lognormal Gaussian jumps).
2. **Kou (2002):** $\ln(J)$ follows an asymmetric double exponential distribution with density:
   $$
   f_Y(y) = p \eta_1 e^{-\eta_1 y} \mathbf{1}_{y \ge 0} + (1 - p) \eta_2 e^{\eta_2 y} \mathbf{1}_{y < 0} \quad (\eta_1 > 1, \eta_2 > 0)
   $$
   The memoryless property of the exponential distribution yields closed-form solutions for first-passage times, American options, and barrier knockouts.

---

## 4. Extreme Value Theory (EVT) and Tail Risk Modeling

In crypto options markets, historical return distributions exhibit fat tails (leptokurtosis) that invalidate Gaussian Value-at-Risk ($VaR$) assumptions.

### 4.1 The Pickands-Balkema-de Haan Theorem
Let $X$ be a random variable with distribution function $F$. The **excess distribution function** over a high threshold $u < x_F = \sup\{x: F(x) < 1\}$ is:
$$
F_u(y) = \mathbb{P}(X - u \le y \mid X > u) = \frac{F(u + y) - F(u)}{1 - F(u)}, \quad y \ge 0
$$
*Theorem:* For a large class of underlying distributions $F$, there exist scaling functions $\beta(u) > 0$ such that:
$$
\lim_{u \to x_F} \sup_{0 \le y < x_F - u} \left| F_u(y) - G_{\xi, \beta(u)}(y) \right| = 0
$$
where $G_{\xi, \beta}$ is the **Generalized Pareto Distribution (GPD)**:
$$
G_{\xi, \beta}(y) = \begin{cases}
1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi} & \text{if } \xi \ne 0 \\
1 - \exp\left(-\frac{y}{\beta}\right) & \text{if } \xi = 0
\end{cases}
$$
* Parameter $\xi$: **Shape parameter (tail index)**. For crypto returns, $\xi > 0$ (heavy-tailed Fréchet domain).
* Parameter $\beta > 0$: **Scale parameter**.

### 4.2 Peaks-over-Threshold (POT) High-Quantile Estimator
Given a sample $x_1, \dots, x_n$ with $N_u$ exceedances over threshold $u$, the tail estimator for tail probability $p < 1 - F(u)$ is:
$$
\widehat{\text{VaR}}_p = u + \frac{\hat{\beta}}{\hat{\xi}} \left( \left(\frac{n}{N_u} (1 - p)\right)^{-\hat{\xi}} - 1 \right)
$$
The corresponding **Expected Shortfall (Conditional Value at Risk, CVaR)** is:
$$
\widehat{\text{ES}}_p = \frac{\widehat{\text{VaR}}_p}{1 - \hat{\xi}} + \frac{\hat{\beta} - \hat{\xi} u}{1 - \hat{\xi}}
$$
This provides the mathematical basis for our non-parametric tail trigger thresholds in `risk/tail_triggers.py`.

---

## 5. High-Dimensional Dependence: Copula Theory

### 5.1 Sklar's Theorem
Let $F$ be an $n$-dimensional joint distribution function with continuous marginal distribution functions $F_1, \dots, F_n$. Then there exists a unique **copula** $C: [0, 1]^n \to [0, 1]$ such that for all $\mathbf{x} = (x_1, \dots, x_n) \in \mathbb{R}^n$:
$$
F(x_1, \dots, x_n) = C(F_1(x_1), \dots, F_n(x_n))
$$
Conversely, if $C$ is a copula and $F_1, \dots, F_n$ are univariate distributions, $F$ defined above is an $n$-dimensional distribution.

### 5.2 Tail Dependence Coefficients
Tail dependence measures the probability of concurrent joint extreme moves in the asymptotic limit:
1. **Lower Tail Dependence Coefficient ($\lambda_L$):**
   $$
   \lambda_L = \lim_{u \to 0^+} \mathbb{P}(U_2 \le u \mid U_1 \le u) = \lim_{u \to 0^+} \frac{C(u, u)}{u}
   $$
2. **Upper Tail Dependence Coefficient ($\lambda_U$):**
   $$
   \lambda_U = \lim_{u \to 1^-} \mathbb{P}(U_2 > u \mid U_1 > u) = \lim_{u \to 1^-} \frac{1 - 2u + C(u, u)}{1 - u}
   $$

* **Gaussian Copula Limitation:** $\lambda_L = \lambda_U = 0$ for any correlation $\rho < 1$. The Gaussian copula assumes asymptotic tail independence, disastrously underestimating simultaneous liquidations during a crypto liquidity cascade.
* **Student-$t$ Copula:** With degrees of freedom $\nu$ and correlation $\rho$:
  $$
  \lambda_L = \lambda_U = 2 t_{\nu + 1}\left(-\sqrt{\frac{(\nu + 1)(1 - \rho)}{1 + \rho}}\right) > 0
  $$
  The Student-$t$ copula captures non-zero joint crash probabilities even for moderately correlated assets.
