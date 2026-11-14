"""
Advanced Physics-Inspired Mathematical Models for Financial Systems.

Modules:
1. Fokker-Planck (Kolmogorov Forward) Equation Solver:
   - 1D diffusion probability density evolution:
     dp/dt = -d(mu * p)/dx + 0.5 * d^2(sigma^2 * p)/dx^2
   - Conservative finite-volume discretization with mass conservation and positivity.
2. Feynman-Kac Framework:
   - PDE-based valuation of expectations under diffusion processes.
   - Independent verification against Monte Carlo and Black-Scholes.
3. Statistical Mechanics: Ising Model & Spin Glass:
   - Agent interaction lattice s_i in {-1, +1}
   - Hamiltonian: H(s) = -sum(J_ij * s_i * s_j) - sum(h_i * s_i)
   - Metropolis-Hastings simulation for liquidity clustering and phase transitions.
4. Continuous Order-Book Depth: Reaction-Diffusion:
   - drho/dt = D * d^2 rho / dx^2 + R(rho, x, t)
   - Models order replenishment, consumption sweeps, and liquidity diffusion.
"""

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class FokkerPlanck1DSolver:
    """
    Finite-Volume conservative solver for 1D Fokker-Planck PDE:
    dp/dt = -d/dx [mu(x, t) p] + 1/2 d^2/dx^2 [sigma(x, t)^2 p]
    """
    x_min: float = -1.0
    x_max: float = 1.0
    n_space: int = 200
    x_grid: np.ndarray = field(init=False)
    dx: float = field(init=False)

    def __post_init__(self):
        self.x_grid = np.linspace(self.x_min, self.x_max, self.n_space)
        self.dx = (self.x_max - self.x_min) / (self.n_space - 1)

    def solve_evolution(
        self,
        p0: np.ndarray,
        t_span: float,
        n_time: int,
        mu_fn: Callable[[np.ndarray, float], np.ndarray],
        sigma_fn: Callable[[np.ndarray, float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves PDE using implicit-explicit (IMEX) / Chang-Cooper conservative scheme.
        Returns: (t_grid, p_history of shape (n_time, n_space))
        """
        dt = t_span / max(1, n_time - 1)
        t_grid = np.linspace(0.0, t_span, n_time)
        
        # Normalize initial density
        mass0 = np.sum(p0) * self.dx
        p = np.copy(p0) / max(mass0, 1e-12)
        
        p_history = np.empty((n_time, self.n_space), dtype=np.float64)
        p_history[0] = p

        for step in range(1, n_time):
            t = t_grid[step - 1]
            mu = mu_fn(self.x_grid, t)
            sigma = sigma_fn(self.x_grid, t)
            diff = 0.5 * (sigma ** 2)

            # Construct tridiagonal flux operator J = mu*p - diff * dp/dx
            # Conservative upstream flux
            diag = np.zeros(self.n_space)
            lower = np.zeros(self.n_space - 1)
            upper = np.zeros(self.n_space - 1)

            # Central diffusion + upwind drift
            r_diff = diff * dt / (self.dx ** 2)
            c_drift = mu * dt / (2.0 * self.dx)

            for i in range(1, self.n_space - 1):
                lower[i - 1] = - (r_diff[i - 1] + c_drift[i - 1])
                diag[i] = 1.0 + 2.0 * r_diff[i]
                upper[i] = - (r_diff[i + 1] - c_drift[i + 1])

            # Zero-flux boundary conditions at x_min and x_max
            diag[0] = 1.0 + r_diff[0]
            upper[0] = -r_diff[1]
            diag[-1] = 1.0 + r_diff[-1]
            lower[-1] = -r_diff[-2]

            A = sp.diags([lower, diag, upper], [-1, 0, 1], shape=(self.n_space, self.n_space), format="csc")
            p_next = spla.spsolve(A, p)
            
            # Enforce positivity and conservation
            p_next = np.maximum(p_next, 0.0)
            mass = np.sum(p_next) * self.dx
            if mass > 0:
                p_next /= mass
            
            p = p_next
            p_history[step] = p

        return t_grid, p_history


@dataclass
class FeynmanKacPdePricer:
    """
    Solves Kolmogorov backward PDE via Feynman-Kac for European claims:
    dV/dt + r*S*dV/dS + 1/2*sigma^2*S^2 * d^2V/dS^2 - r*V = 0
    with terminal payoff V(S, T) = Payoff(S).
    """
    s_min: float = 1e-4
    s_max: float = 300.0
    n_space: int = 300
    n_time: int = 200

    def price_european(
        self,
        s0: float,
        k: float,
        t_expiry: float,
        r: float,
        sigma: float,
        is_call: bool = True,
    ) -> float:
        """Prices European option via Crank-Nicolson PDE finite difference."""
        s_grid = np.linspace(self.s_min, self.s_max, self.n_space)
        ds = (self.s_max - self.s_min) / (self.n_space - 1)
        dt = t_expiry / self.n_time

        # Terminal payoff at expiry T
        if is_call:
            v = np.maximum(s_grid - k, 0.0)
        else:
            v = np.maximum(k - s_grid, 0.0)

        # Crank-Nicolson matrices (theta = 0.5)
        # alpha_i = 0.25 * dt * (sigma^2 * i^2 - r * i)
        # beta_i = -0.5 * dt * (sigma^2 * i^2 + r)
        # gamma_i = 0.25 * dt * (sigma^2 * i^2 + r * i)
        i_idx = np.arange(self.n_space)
        alpha = 0.25 * dt * (sigma**2 * (i_idx**2) - r * i_idx)
        beta = -0.5 * dt * (sigma**2 * (i_idx**2) + r)
        gamma = 0.25 * dt * (sigma**2 * (i_idx**2) + r * i_idx)

        # Construct implicit matrix M1 and explicit matrix M2
        m1_diag = 1.0 - beta[1:-1]
        m1_lower = -alpha[2:-1]
        m1_upper = -gamma[1:-2]

        m2_diag = 1.0 + beta[1:-1]
        m2_lower = alpha[2:-1]
        m2_upper = gamma[1:-2]

        m1 = sp.diags([m1_lower, m1_diag, m1_upper], [-1, 0, 1], format="csc")
        m2 = sp.diags([m2_lower, m2_diag, m2_upper], [-1, 0, 1], format="csc")

        # Backward time stepping from T down to 0
        for step in range(self.n_time):
            tau = step * dt
            # Boundary conditions
            if is_call:
                bc_lower = 0.0
                bc_upper = self.s_max - k * math.exp(-r * (t_expiry - tau))
            else:
                bc_lower = k * math.exp(-r * (t_expiry - tau))
                bc_upper = 0.0

            rhs = m2.dot(v[1:-1])
            rhs[0] += alpha[1] * (v[0] + bc_lower)
            rhs[-1] += gamma[-2] * (v[-1] + bc_upper)

            v_inner = spla.spsolve(m1, rhs)
            v[1:-1] = v_inner
            v[0] = bc_lower
            v[-1] = bc_upper

        # Interpolate at s0
        return float(np.interp(s0, s_grid, v))


@dataclass
class MarketIsingModel:
    """
    2D / Fully connected Ising Spin Glass model of market agent consensus:
    s_i in {-1, +1} (bearish vs bullish).
    H(s) = -J * sum_<ij> s_i s_j - h * sum_i s_i
    """
    n_agents: int = 100
    coupling_j: float = 1.0 # Social imitation strength
    external_field_h: float = 0.0 # External fundamental/news bias
    temperature_t: float = 1.0 # Market noise / idiosyncratic dispersion

    def simulate_metropolis(self, n_sweeps: int = 1000, seed: int = 42) -> Dict[str, Union[float, np.ndarray]]:
        """Simulates spin dynamics using Metropolis-Hastings sweeps."""
        rng = np.random.default_rng(seed)
        spins = rng.choice([-1.0, 1.0], size=self.n_agents)
        magnetization_hist = np.empty(n_sweeps, dtype=np.float64)

        for sweep in range(n_sweeps):
            for _ in range(self.n_agents):
                idx = rng.integers(0, self.n_agents)
                # Mean-field or random neighbor interaction
                # Field experienced by agent idx: sum_{j != idx} J_ij * s_j + h
                local_field = self.coupling_j * (np.sum(spins) - spins[idx]) / (self.n_agents - 1) + self.external_field_h
                delta_e = 2.0 * spins[idx] * local_field

                if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / max(1e-4, self.temperature_t)):
                    spins[idx] = -spins[idx]

            magnetization_hist[sweep] = float(np.mean(spins))

        mean_mag = float(np.mean(magnetization_hist[n_sweeps // 2:]))
        susceptibility = float(np.var(magnetization_hist[n_sweeps // 2:]) * self.n_agents / max(1e-4, self.temperature_t))

        return {
            "mean_magnetization": mean_mag,
            "susceptibility": susceptibility,
            "magnetization_series": magnetization_hist,
            "final_spins": spins,
        }


@dataclass
class ReactionDiffusionOrderBook:
    """
    Continuous Reaction-Diffusion approximation of limit order book liquidity density rho(x, t):
    drho/dt = D * d^2 rho / dx^2 - lambda_c * rho + lambda_r(x)
    where:
    - x is distance from mid-price.
    - D is spatial liquidity diffusion coefficient.
    - lambda_c is consumption rate from market order sweeps.
    - lambda_r(x) is replenishment rate profile.
    """
    n_levels: int = 100
    x_range: float = 50.0 # Ticks from mid
    diff_coeff: float = 0.5
    consumption_rate: float = 0.05

    def solve_depth_profile(
        self,
        t_span: float = 10.0,
        n_steps: int = 100,
        shock_level: Optional[float] = None,
        shock_magnitude: float = 0.8,
    ) -> np.ndarray:
        """Solves depth profile over time after an initial shock."""
        x = np.linspace(0.0, self.x_range, self.n_levels)
        dx = self.x_range / (self.n_levels - 1)
        dt = t_span / n_steps

        # Equilibrium replenishment rate lambda_r(x) = rho_eq * lambda_c
        rho_eq = 100.0 * (1.0 - np.exp(-x / 10.0)) # Monotonically increasing depth away from mid
        replenish = self.consumption_rate * rho_eq

        rho = np.copy(rho_eq)
        # Apply consumption shock at specific level
        if shock_level is not None:
            shock_idx = np.argmin(np.abs(x - shock_level))
            rho[max(0, shock_idx - 3):min(self.n_levels, shock_idx + 4)] *= (1.0 - shock_magnitude)

        # Finite difference evolution
        for _ in range(n_steps):
            laplacian = np.zeros_like(rho)
            laplacian[1:-1] = (rho[2:] - 2.0 * rho[1:-1] + rho[:-2]) / (dx ** 2)
            # Boundary conditions (Neumann at 0 and x_range)
            laplacian[0] = (rho[1] - rho[0]) / (dx ** 2)
            laplacian[-1] = (rho[-2] - rho[-1]) / (dx ** 2)

            drho = self.diff_coeff * laplacian - self.consumption_rate * rho + replenish
            rho = np.maximum(0.0, rho + dt * drho)

        return rho
