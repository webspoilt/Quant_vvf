# Advanced Linear Algebra, Spectral Theory, and Correlation Manifolds

**Module:** `research/math_papers/04_advanced_linear_algebra_and_spectral_methods.md`  
**Classification:** Advanced Mathematical Foundations for Quantitative Finance  
**Target Architecture:** `Quant_vvf` Research Engine  

---

## 1. Spectral Theory and Positive Semi-Definite Matrix Cones

### 1.1 The Spectral Decomposition Theorem
Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a real symmetric matrix ($\mathbf{A} = \mathbf{A}^T$). Then:
1. All eigenvalues $\lambda_1, \dots, \lambda_n$ of $\mathbf{A}$ are real.
2. There exists an orthonormal basis of eigenvectors $\mathbf{q}_1, \dots, \mathbf{q}_n \in \mathbb{R}^n$ such that:
$$
\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T
$$
where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_n]$ is an orthogonal matrix ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$), and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$.

### 1.2 The Courant-Fischer Minimax Theorem
Let $\mathbf{A} \in \mathcal{S}^n$ with sorted eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$. Then for any $k \in \{1, \dots, n\}$:
$$
\lambda_k = \max_{\substack{V \subset \mathbb{R}^n \\ \dim(V) = k}} \min_{\substack{\mathbf{x} \in V \\ \mathbf{x} \ne \mathbf{0}}} \frac{\mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}^T \mathbf{x}} = \min_{\substack{W \subset \mathbb{R}^n \\ \dim(W) = n - k + 1}} \max_{\substack{\mathbf{x} \in W \\ \mathbf{x} \ne \mathbf{0}}} \frac{\mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$
This characterizes the principal axes of volatility risk dispersion across multi-strike books.

### 1.3 The Cone of Symmetric Positive Semi-Definite Matrices ($\mathcal{S}_+^n$)
The set of positive semi-definite matrices:
$$
\mathcal{S}_+^n = \{ \mathbf{X} \in \mathcal{S}^n : \mathbf{v}^T \mathbf{X} \mathbf{v} \ge 0 \; \forall \mathbf{v} \in \mathbb{R}^n \}
$$
forms a closed, convex cone in the Euclidean space $(\mathcal{S}^n, \langle \cdot, \cdot \rangle_F)$ equipped with the Frobenius inner product:
$$
\langle \mathbf{X}, \mathbf{Y} \rangle_F = \text{Tr}(\mathbf{X}^T \mathbf{Y}) = \sum_{i,j=1}^n X_{ij} Y_{ij}, \quad \|\mathbf{X}\|_F = \sqrt{\langle \mathbf{X}, \mathbf{X} \rangle_F}
$$
The orthogonal projection of a symmetric matrix $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$ onto $\mathcal{S}_+^n$ is uniquely given by:
$$
\mathcal{P}_{\mathcal{S}_+^n}(\mathbf{A}) = \mathbf{Q} \boldsymbol{\Lambda}_+ \mathbf{Q}^T
$$
where $\boldsymbol{\Lambda}_+ = \text{diag}(\max(0, \lambda_1), \dots, \max(0, \lambda_n))$.

---

## 2. Higham (2002) Nearest Correlation Matrix Algorithm

### 2.1 The Problem Formulation
Empirical correlation matrices estimated from asynchronous crypto tick data, missing observations, or historical regime shifts are frequently non-positive semi-definite (possessing negative eigenvalues). Such corrupted matrices cannot be used in Cholesky factorization for Monte Carlo or portfolio variance calculation.

The **Nearest Correlation Matrix problem** is formulated as:
$$
\min_{\mathbf{X} \in \mathcal{S}^n} \frac{1}{2} \|\mathbf{A} - \mathbf{X}\|_F^2 \quad \text{subject to } \mathbf{X} \in \mathcal{S}_+^n \cap \mathcal{U}
$$
where $\mathcal{U} = \{ \mathbf{X} \in \mathcal{S}^n : X_{ii} = 1, \; i = 1, \dots, n \}$ is the affine space of unit-diagonal symmetric matrices.

### 2.2 Dykstra's Alternating Projection Algorithm
Because both $\mathcal{S}_+^n$ and $\mathcal{U}$ are closed convex sets, Dykstra's cyclical projection algorithm with Moreau correction sequences converges globally to the unique minimizer $\mathbf{X}^*$.

**Projection Operators:**
1. **Projection onto $\mathcal{U}$ ($\mathcal{P}_{\mathcal{U}}$):**
   $$
   [\mathcal{P}_{\mathcal{U}}(\mathbf{Y})]_{ij} = \begin{cases} 1 & \text{if } i = j \\ Y_{ij} & \text{if } i \ne j \end{cases}
   $$
2. **Projection onto $\mathcal{S}_+^n$ ($\mathcal{P}_{\mathcal{S}_+}$):**
   $$
   \mathcal{P}_{\mathcal{S}_+}(\mathbf{Y}) = \mathbf{Q} \text{diag}(\max(0, \lambda_i)) \mathbf{Q}^T
   $$

**Algorithmic Iteration (Higham 2002):**
Initialize $\mathbf{X}_0 = \mathbf{A}$, correction matrix $\Delta\mathbf{S}_0 = \mathbf{0}$.
For $k = 1, 2, \dots$ until convergence ($\|\mathbf{X}_k - \mathbf{X}_{k-1}\|_F < \epsilon$):
1. $\mathbf{R}_k = \mathbf{X}_{k-1} - \Delta\mathbf{S}_{k-1}$
2. $\mathbf{Y}_k = \mathcal{P}_{\mathcal{S}_+}(\mathbf{R}_k)$ (spectral projection onto PSD cone)
3. $\Delta\mathbf{S}_k = \mathbf{Y}_k - \mathbf{R}_k$ (Dykstra correction update)
4. $\mathbf{X}_k = \mathcal{P}_{\mathcal{U}}(\mathbf{Y}_k)$ (diagonal reset to ones)

The resulting matrix is guaranteed to be a mathematically valid correlation matrix with strictly non-negative eigenvalues and unit diagonals.

---

## 3. Matrix Exponentials and Continuous-Time Markov Transition Semigroups

### 3.1 Transition Semigroups and Infinitesimal Generators
Let $(Z_t)_{t \ge 0}$ be a continuous-time Markov chain on a finite state space $\{1, \dots, K\}$ representing discrete market volatility regimes (e.g., Quiet, Mean-Reverting VRP, Flash Crash).
The transition probability matrix $\mathbf{P}(t) \in \mathbb{R}^{K \times K}$, where $P_{ij}(t) = \mathbb{P}(Z_t = j \mid Z_0 = i)$, satisfies the Chapman-Kolmogorov semigroup equation:
$$
\mathbf{P}(t + s) = \mathbf{P}(t) \mathbf{P}(s), \quad \mathbf{P}(0) = \mathbf{I}
$$
The transition rate matrix (infinitesimal generator) $\mathbf{Q} = \mathbf{P}'(0)$ satisfies:
$$
Q_{ij} \ge 0 \quad (i \ne j), \quad Q_{ii} = -\sum_{j \ne i} Q_{ij}
$$
The unique solution to Kolmogorov's backward and forward differential equations is the matrix exponential:
$$
\mathbf{P}(t) = \exp(t\mathbf{Q}) = \sum_{k=0}^\infty \frac{t^k \mathbf{Q}^k}{k!}
$$

### 3.2 Computation via Scaling and Squaring with Padé Approximants
Direct series summation suffers from severe numerical instability. We employ the Higham (2005) scaling and squaring algorithm:
1. Compute scale factor $m \in \mathbb{N}$ such that $\|2^{-m} t \mathbf{Q}\|_F \le \frac{1}{2}$.
2. Evaluate the $[p/p]$ diagonal Padé approximant $R_{pp}(2^{-m} t \mathbf{Q}) = [D_{pp}]^{-1} N_{pp}$.
3. Square $m$ times: $\mathbf{P}(t) = [R_{pp}(2^{-m} t \mathbf{Q})]^{2^m}$.

---

## 4. Lie Groups and Invariant Symmetries in Option Pricing

The Black-Scholes-Merton PDE:
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$
admits a 6-dimensional Lie algebra of infinitesimal symmetry vector fields $\mathfrak{g} = \text{span}\{v_1, \dots, v_6\}$.

Under change of variables $x = \ln(S / K)$, $\tau = T - t$, $u = e^{r\tau} V$, the equation maps to the 1D heat equation $\frac{\partial u}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2}$.
The underlying symmetry group is isomorphic to the projective special linear group $SL(2, \mathbb{R}) \ltimes \mathcal{H}_3$ (the Heisenberg-Weyl group).
In modern implied volatility surface geometry, these Lie algebraic symmetries govern the asymptotic slope boundaries (Lee's Moment Formulas) and the invariance of SVI slices under moneyness affine transformations.
