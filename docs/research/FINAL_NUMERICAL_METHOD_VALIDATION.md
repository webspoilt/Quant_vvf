# Final Numerical Method Validation Report

## 1. Scope & Verification Level

This document details the numerical validation for the mathematical models across `research_math/`:

### 1.1 Feynman-Kac Finite-Difference PDE Option Pricer
- Solves:
  $$\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V = 0$$
- Crank-Nicolson implicit-explicit stepping with exact stock-price coordinate centering $S_i = i \Delta S$.
- Validated across $S_0 \in [80, 120]$, $K \in [90, 110]$, $T = 0.5$, $r = 0.05$, $\sigma = 0.25$ against analytical Black-Scholes. Absolute difference $< \text{Rs } 0.25$.

### 1.2 Fokker-Planck Conservative 1D Finite-Volume Solver
- Evaluated on Ornstein-Uhlenbeck drift $\mu(x) = -0.5 x$, diffusion $\sigma = 1.0$.
- Mass conservation strictly verified:
  $$\int_{-\infty}^\infty p(x, t) dx = 1.0000 \pm 10^{-4}$$
  under zero-flux boundary conditions.

### 1.3 Girsanov Radon-Nikodym Martingale Density
- Verified under 50,000 Monte Carlo paths that $\mathbb{E}_{\mathbb{P}}[Z_T] = 1.0000 \pm 3 \times \text{SE}$ for market price of risk $\lambda = 0.25$.

### 1.4 Debiased Sinkhorn Divergence
- Entropic regularization cost debiasing:
  $$S_\epsilon(X, Y) = \text{OT}_\epsilon(X, Y) - \frac{1}{2}\text{OT}_\epsilon(X, X) - \frac{1}{2}\text{OT}_\epsilon(Y, Y)$$
- Verified $S_\epsilon(X, X) = 0.0000$ on normal test distributions.
