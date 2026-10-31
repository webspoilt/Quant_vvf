# High-Dimensional Numerical Algorithms: Fourier Transform, COS Method, and Sparse Grids

**Module:** `research/math_papers/05_high_dimensional_algorithms_and_fourier_methods.md`  
**Classification:** Advanced Mathematical Foundations for Quantitative Finance  
**Target Architecture:** `Quant_vvf` Research Engine  

---

## 1. Carr-Madan (1999) Fast Fourier Transform (FFT) Pricing

### 1.1 The Theoretical Bottleneck
When pricing under non-Gaussian models (e.g. Heston stochastic volatility or Merton jump-diffusion), the risk-neutral transition probability density $p(s_T)$ is unknown in closed form, but the **characteristic function** $\phi_T(u) = \mathbb{E}^\mathbb{Q}[e^{iu \ln S_T}]$ is explicitly known.
Direct numerical integration for each individual strike $K$ is computationally slow ($O(M)$ operations per strike). Carr and Madan (1999) reformulated option pricing into an inverse Fourier transform that prices an entire continuum of strikes concurrently using the Fast Fourier Transform (FFT) in $O(N \log N)$ time.

### 1.2 Mathematical Formulation and Dampening Factor
Let $s = \ln(S_T)$ and $k = \ln(K)$. The call option price is:
$$
C_T(k) = e^{-rT} \int_k^\infty (e^s - e^k) p(s) \, ds
$$
Because $C_T(k) \to S_0$ as $k \to -\infty$, $C_T(k)$ is not square-integrable in $L^1(\mathbb{R})$, preventing direct Fourier transformation.
Carr and Madan introduced a **dampening factor** $\alpha > 0$:
$$
c_T(k) \equiv e^{\alpha k} C_T(k)
$$
For suitable $\alpha > 0$ (typically $\alpha \in [0.75, 1.5]$), $c_T(k) \in L^1(\mathbb{R})$ and decays to zero as $k \to \pm \infty$.

The continuous Fourier transform of $c_T(k)$ with respect to $k$ is:
$$
\psi_T(v) = \int_{-\infty}^\infty e^{ivk} c_T(k) \, dk = \frac{e^{-rT} \phi_T(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v}
$$
Inverting the transform recovers the option price:
$$
C_T(k) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk} \psi_T(v) \, dv
$$

### 1.3 Discrete FFT Formulation and Simpson's Rule
Discretizing the integral using trapezoidal rule with step size $\eta$ and $N$ points ($v_j = j\eta, \; j = 0, \dots, N-1$):
$$
C_T(k) \approx \frac{e^{-\alpha k}}{\pi} \sum_{j=0}^{N-1} e^{-i v_j k} \psi_T(v_j) w_j \eta
$$
where $w_j = \frac{1}{3}(3 + (-1)^j) - \frac{1}{3}\delta_{j, 0}$ are Simpson's quadrature weights.

Let log-strikes be discretized on a uniform grid: $k_u = -b + u\lambda$ for $u = 0, \dots, N-1$, with grid spacing constraint:
$$
\lambda \eta = \frac{2\pi}{N}
$$
Substituting gives the standard Discrete Fourier Transform:
$$
C_T(k_u) \approx \frac{e^{-\alpha k_u}}{\pi} \sum_{j=0}^{N-1} e^{-i \frac{2\pi}{N} j u} \left[ e^{i b v_j} \psi_T(v_j) w_j \eta \right]
$$
This sum evaluates simultaneously for all $N$ strikes in $O(N \log N)$ via the Cooley-Tukey FFT algorithm.

---

## 2. Fang & Oosterlee (2008) COS Method: Fourier-Cosine Series Expansion

The COS method represents the state-of-the-art alternative to the FFT, achieving **exponential convergence** for smooth densities with $O(N)$ linear computational complexity and zero damping parameter tuning.

### 2.1 The Fourier-Cosine Series Representation
Let $x = \ln(S_T / K)$ on a truncated integration domain $[a, b] \subset \mathbb{R}$.
The probability density function $f(x)$ is expanded in a Fourier-cosine series:
$$
f(x) = \sideset{}{'}\sum_{k=0}^{N-1} A_k \cos\left(k\pi \frac{x - a}{b - a}\right)
$$
where $\sum'$ denotes that the $k=0$ term is weighted by $1/2$, and the Fourier-cosine coefficients are:
$$
A_k = \frac{2}{b - a} \int_a^b f(x) \cos\left(k\pi \frac{x - a}{b - a}\right) dx
$$

### 2.2 The Key Analytical Connection
Because $\cos(\theta) = \text{Re}(e^{i\theta})$, the coefficients $A_k$ are directly extracted from the **characteristic function** $\phi(u)$ without numerical integration:
$$
A_k \approx \frac{2}{b - a} \text{Re}\left\{ \phi\left( \frac{k\pi}{b - a} \right) \exp\left( -i \frac{k\pi a}{b - a} \right) \right\}
$$

### 2.3 Option Pricing Formula
For a European call option with payoff $V(x) = [K(e^x - 1)]^+$:
$$
V(S_0, T) = e^{-rT} K \sideset{}{'}\sum_{k=0}^{N-1} A_k U_k
$$
where $U_k$ are the exact analytical cosine coefficients of the payoff:
$$
U_k = \frac{2}{b - a} \int_0^b (e^x - 1) \cos\left(k\pi \frac{x - a}{b - a}\right) dx = \chi_k(0, b) - \psi_k(0, b)
$$
with explicit analytical forms:
$$
\chi_k(c, d) = \frac{1}{1 + \left(\frac{k\pi}{b-a}\right)^2} \left[ \cos\left(k\pi \frac{d-a}{b-a}\right) e^d - \cos\left(k\pi \frac{c-a}{b-a}\right) e^c + \frac{k\pi}{b-a} \sin\left(k\pi \frac{d-a}{b-a}\right) e^d - \frac{k\pi}{b-a} \sin\left(k\pi \frac{c-a}{b-a}\right) e^c \right]
$$
$$
\psi_k(c, d) = \begin{cases}
d - c & \text{if } k = 0 \\
\frac{b - a}{k\pi} \left[ \sin\left(k\pi \frac{d-a}{b-a}\right) - \sin\left(k\pi \frac{c-a}{b-a}\right) \right] & \text{if } k > 0
\end{cases}
$$

*Convergence Rate:* For infinitely differentiable densities (e.g. Heston or Black-Scholes), error decays as:
$$
|\text{Error}| \le C \exp(-r N)
$$
Machine precision ($10^{-14}$) is achieved with as few as $N = 64$ or $N = 128$ terms!

---

## 3. Smolyak Sparse Grid Algorithm for Multi-Asset Integration

### 3.1 Overcoming the Curse of Dimensionality
When pricing multi-asset options in $d$ dimensions (e.g. $d = 10$ underlying cryptos):
* A standard full grid with $M = 20$ points per dimension requires $M^d = 20^{10} \approx 1.02 \times 10^{13}$ function evaluations (computationally impossible in real time).
* The **Smolyak construction** retains only multi-indices whose sum satisfies an accuracy level $L$:
  $$
  |\mathbf{i}| = i_1 + \dots + i_d \le L + d - 1
  $$
  Total grid points scale as:
  $$
  N(L, d) \approx \frac{2^L}{L!} d^L = O\left( M (\log M)^{d-1} \right)
  $$
  For $d = 10, L = 3$, evaluations drop from $10^{13}$ to $\sim 10^3$, enabling sub-millisecond multi-asset pricing.

---

## 4. Quasi-Monte Carlo (QMC) and Low-Discrepancy Sequences

### 4.1 Star Discrepancy and the Koksma-Hlawka Inequality
Let $P_N = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset [0, 1)^d$ be a point set. The **star discrepancy** is:
$$
D_N^*(P_N) = \sup_{B \in \mathcal{J}^*} \left| \frac{\#(P_N \cap B)}{N} - \text{vol}(B) \right|
$$
where $\mathcal{J}^*$ is the family of sub-intervals $\prod_{i=1}^d [0, u_i)$.

The **Koksma-Hlawka Inequality** provides the deterministic error bound:
$$
\left| \frac{1}{N} \sum_{i=1}^N f(\mathbf{x}_i) - \int_{[0, 1)^d} f(\mathbf{u}) \, d\mathbf{u} \right| \le V_{\text{HK}}(f) \cdot D_N^*(P_N)
$$
where $V_{\text{HK}}(f)$ is the total variation of $f$ in the sense of Hardy and Krause.

* For pseudo-random Monte Carlo: $D_N^* \sim O(N^{-1/2})$.
* For Sobol low-discrepancy sequences: $D_N^* \sim O\left( \frac{(\log N)^d}{N} \right) \approx O(N^{-1})$ asymptotically.

### 4.2 Brownian Bridge Dimension Reduction
In multi-step path simulation, standard sequential sampling spreads variance equally across all time increments.
The **Brownian Bridge construction** recursively samples midpoint coordinates:
$$
W_{t_{\text{mid}}} \mid (W_{t_a}, W_{t_b}) \sim \mathcal{N}\left( \frac{W_{t_a} + W_{t_b}}{2}, \frac{t_b - t_a}{4} \right)
$$
This packs $> 85\%$ of total path variance into the first 3 to 4 dimensions, aligning optimal low-discrepancy Sobol coordinates with the most influential path features and accelerating Monte Carlo convergence by up to $100\times$.
