# High-Level Multivariable Calculus: Double, Triple, and N-Dimensional Integration

**Module:** `research/math_papers/01_multivariable_calculus_and_integration.md`  
**Classification:** Advanced Mathematical Foundations for Quantitative Finance  
**Target Architecture:** `Quant_vvf` Research Engine  

---

## 1. Rigorous Foundation of Multidimensional Riemann and Lebesgue Integration

### 1.1 Fubini-Tonelli Theorem on Product Spaces
Let $(X, \mathcal{A}, \mu)$ and $(Y, \mathcal{B}, \nu)$ be $\sigma$-finite measure spaces. Let $(X \times Y, \mathcal{A} \otimes \mathcal{B}, \mu \times \nu)$ denote the product measure space.

**Tonelli's Theorem:** If $f: X \times Y \to [0, +\infty]$ is non-negative and $\mathcal{A} \otimes \mathcal{B}$-measurable, then:
$$
\int_{X \times Y} f(x, y) \, d(\mu \times \nu) = \int_X \left( \int_Y f(x, y) \, d\nu(y) \right) d\mu(x) = \int_Y \left( \int_X f(x, y) \, d\mu(x) \right) d\nu(y)
$$
All iterated integrals exist, are measurable, and coincide in $[0, +\infty]$.

**Fubini's Theorem:** If $f: X \times Y \to \mathbb{R}$ is integrable with respect to $\mu \times \nu$ (i.e. $\int_{X \times Y} |f| \, d(\mu \times \nu) < \infty$), then for almost every $x \in X$, the section $y \mapsto f(x, y)$ is integrable, and:
$$
\int_{X \times Y} f(x, y) \, d(\mu \times \nu) = \int_X \left( \int_Y f(x, y) \, d\nu(y) \right) d\mu(x) = \int_Y \left( \int_X f(x, y) \, d\mu(x) \right) d\nu(y)
$$

### 1.2 Financial and Risk Interpretation
In multi-asset volatility harvesting and derivative pricing, evaluation of multi-asset options (e.g., basket, spread, or rainbow options on Deribit BTC and ETH volatility) requires integrating discontinuous payoff kernels over multi-dimensional domains $\Omega \subset \mathbb{R}^d$:
$$
\mathbb{E}^\mathbb{Q}[\Phi(S_1(T), \dots, S_d(T))] = \int_{\mathbb{R}^d} \Phi(s_1, \dots, s_d) \, p(s_1, \dots, s_d; T) \, ds_1 \dots ds_d
$$
Fubini-Tonelli guarantees that cross-sectional conditionings and sequential strike integrations can be interchanged without path-dependent order artifacts.

---

## 2. Jacobian Transformations and Change of Variables in $\mathbb{R}^n$

### 2.1 The Multidimensional Change of Variables Theorem
Let $U \subset \mathbb{R}^n$ be an open set and let $\mathbf{\Phi}: U \to \mathbf{\Phi}(U) \subset \mathbb{R}^n$ be a $C^1$-diffeomorphism (injective, continuously differentiable with invertible derivative $\mathbf{D\Phi}(\mathbf{u})$ everywhere on $U$).

Let $\mathbf{J}_{\mathbf{\Phi}}(\mathbf{u}) = \mathbf{D\Phi}(\mathbf{u})$ denote the $n \times n$ Jacobian matrix:
$$
\mathbf{J}_{\mathbf{\Phi}}(\mathbf{u}) = \begin{pmatrix}
\frac{\partial \Phi_1}{\partial u_1} & \dots & \frac{\partial \Phi_1}{\partial u_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \Phi_n}{\partial u_1} & \dots & \frac{\partial \Phi_n}{\partial u_n}
\end{pmatrix}
$$
The Jacobian determinant is defined as $J(\mathbf{u}) = \det(\mathbf{J}_{\mathbf{\Phi}}(\mathbf{u}))$.

For any Lebesgue-integrable function $f: \mathbf{\Phi}(U) \to \mathbb{R}$:
$$
\int_{\mathbf{\Phi}(U)} f(\mathbf{x}) \, d\mathbf{x} = \int_U f(\mathbf{\Phi}(\mathbf{u})) \, \left| \det \mathbf{J}_{\mathbf{\Phi}}(\mathbf{u}) \right| \, d\mathbf{u}
$$

### 2.2 Double Integration: Polar and Hyperbolic Coordinate Transformations
In planar variance dispersion and bivariate correlation modeling:
1. **Polar Coordinates:** $\mathbf{\Phi}(r, \theta) = (r \cos\theta, r \sin\theta)$
   $$
   \det \mathbf{J}(r, \theta) = \det \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix} = r(\cos^2\theta + \sin^2\theta) = r
   $$
   $$
   \iint_{\Omega} f(x, y) \, dx \, dy = \iint_{\mathbf{\Phi}^{-1}(\Omega)} f(r\cos\theta, r\sin\theta) \, r \, dr \, d\theta
   $$

2. **Hyperbolic Coordinates (Used in SVI Volatility Skew Calibration):**
   Let $u = xy$ and $v = y/x$ for $x, y > 0$. Then $x = \sqrt{u/v}$, $y = \sqrt{uv}$.
   $$
   \mathbf{J}(u, v) = \begin{pmatrix} \frac{1}{2\sqrt{uv}} & -\frac{1}{2v}\sqrt{\frac{u}{v}} \\ \frac{1}{2}\sqrt{\frac{v}{u}} & \frac{1}{2}\sqrt{\frac{u}{v}} \end{pmatrix} \implies \det \mathbf{J}(u, v) = \frac{1}{2v}
   $$

### 2.3 Triple Integration: Spherical and Cylindrical Transformations
1. **Cylindrical Coordinates:** $(r, \theta, z)$ with $x = r\cos\theta, y = r\sin\theta, z = z$:
   $$
   dxdydz = r \, dr \, d\theta \, dz
   $$
2. **Spherical Coordinates:** $(\rho, \phi, \theta)$ where $\rho \in [0, \infty)$, $\phi \in [0, \pi]$ (colatitude), $\theta \in [0, 2\pi]$ (azimuth):
   $$
   x = \rho \sin\phi \cos\theta, \quad y = \rho \sin\phi \sin\theta, \quad z = \rho \cos\phi
   $$
   The Jacobian determinant is:
   $$
   \det \mathbf{J}(\rho, \phi, \theta) = \rho^2 \sin\phi \implies dxdydz = \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta
   $$

### 2.4 $N$-Dimensional Gaussian Orthogonalization (Cholesky Jacobian)
In correlated asset baskets $(S_1, \dots, S_n)$ with covariance matrix $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$:
Under transformation $\mathbf{x} = \mathbf{L}\mathbf{z} + \boldsymbol{\mu}$, the Jacobian is constant:
$$
\mathbf{J} = \mathbf{L} \implies \left|\det \mathbf{J}\right| = \prod_{i=1}^n L_{ii} = \sqrt{\det \boldsymbol{\Sigma}}
$$
Thus the multivariate normal density integrates as:
$$
\int_{\mathbb{R}^n} \frac{1}{(2\pi)^{n/2} \sqrt{\det \boldsymbol{\Sigma}}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right) d\mathbf{x} = \int_{\mathbb{R}^n} \frac{1}{(2\pi)^{n/2}} \exp\left(-\frac{1}{2}\|\mathbf{z}\|^2\right) d\mathbf{z} = 1
$$

---

## 3. Vector Calculus: Green's, Stokes', and Divergence Theorems

### 3.1 Differential Forms Perspective
Let $\omega$ be a smooth $(k-1)$-form on an oriented $k$-dimensional manifold $M$ with boundary $\partial M$. The **Generalized Stokes' Theorem** states:
$$
\int_{\partial M} \omega = \int_M d\omega
$$
where $d\omega$ is the exterior derivative of $\omega$.

### 3.2 Green's Theorem in the Plane
Let $C = \partial D$ be a positively oriented, piecewise smooth, simple closed curve in $\mathbb{R}^2$, enclosing a compact region $D$. If $\mathbf{F} = (P(x, y), Q(x, y))$ has continuous partial derivatives on $D$:
$$
\oint_{\partial D} (P \, dx + Q \, dy) = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dx \, dy
$$
**Application to Option Pricing:** By setting $P = 0$ and $Q = x$, the planar area of an exercise region $D$ is given by contour line integral $\text{Area}(D) = \oint_{\partial D} x \, dy$.

### 3.3 The Divergence (Gauss-Ostrogradsky) Theorem in $\mathbb{R}^3$ and $\mathbb{R}^n$
Let $V \subset \mathbb{R}^n$ be a compact domain with piecewise smooth boundary $S = \partial V$, and $\mathbf{n}$ the outward unit normal field. For any $C^1$ vector field $\mathbf{F}$:
$$
\iiint_V (\nabla \cdot \mathbf{F}) \, dV = \iint_{\partial V} (\mathbf{F} \cdot \mathbf{n}) \, dS
$$
where $\nabla \cdot \mathbf{F} = \sum_{i=1}^n \frac{\partial F_i}{\partial x_i}$.

### 3.4 Physical Application: Probability Current Flux in Fokker-Planck PDEs
The transition probability density $p(\mathbf{x}, t)$ of a multi-asset diffusion satisfies the Fokker-Planck (Kolmogorov Forward) equation:
$$
\frac{\partial p}{\partial t} + \nabla \cdot \mathbf{J} = 0
$$
where $\mathbf{J}(\mathbf{x}, t) = \boldsymbol{\mu}(\mathbf{x}, t) p(\mathbf{x}, t) - \frac{1}{2} \nabla \cdot (\mathbf{D}(\mathbf{x}, t) p(\mathbf{x}, t))$ is the **probability flux vector field**.
By the Divergence Theorem, the rate of change of total probability mass within an unliquidated solvency domain $\Omega_{\text{solvent}}$ equals the flux through the liquidation boundary $\partial \Omega$:
$$
\frac{d}{dt} \int_{\Omega} p \, dV = \int_{\Omega} \frac{\partial p}{\partial t} \, dV = -\int_{\Omega} (\nabla \cdot \mathbf{J}) \, dV = -\iint_{\partial \Omega} (\mathbf{J} \cdot \mathbf{n}) \, dS
$$
The surface integral directly computes the **instantaneous liquidation hazard rate** of the leveraged options book.

---

## 4. Multi-Dimensional High-Order Numerical Quadrature Algorithms

### 4.1 Tensor-Product Gauss-Legendre Quadrature
For an integral over hypercube $[-1, 1]^d$:
$$
\int_{[-1, 1]^d} f(\mathbf{x}) \, d\mathbf{x} \approx \sum_{i_1=1}^{M} \dots \sum_{i_d=1}^{M} w_{i_1} \dots w_{i_d} f(x_{i_1}, \dots, x_{i_d})
$$
Nodes $x_i$ are roots of the $M$-th Legendre polynomial $P_M(x)$, and weights are:
$$
w_i = \frac{2}{(1 - x_i^2) [P'_M(x_i)]^2}
$$
*Curse of Dimensionality:* Complexity is $O(M^d)$, making $d \ge 5$ intractable.

### 4.2 Smolyak Sparse Grid Algorithm
By pruning tensor products whose multi-indices exceed an accuracy level $L$:
$$
\mathcal{A}(L, d) = \sum_{L - d + 1 \le |\mathbf{i}| \le L} (-1)^{L - |\mathbf{i}|} \binom{d - 1}{L - |\mathbf{i}|} \left( \bigotimes_{j=1}^d \mathcal{U}_{i_j} \right)
$$
Total grid points grow as $O(M \cdot (\log M)^{d-1})$, breaking the exponential barrier.
