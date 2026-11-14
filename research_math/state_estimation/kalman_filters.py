"""
QUANT_VVF — State Estimation, Kalman Filtering & Latent Hedge Ratios.
References:
- Kalman (1960): A New Approach to Linear Filtering and Prediction Problems
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


class KalmanHedgeRatioEstimator:
    """
    Dynamic state-space estimation of cointegration hedge ratio (Kalman 1960):
    State transition: beta_t = beta_{t-1} + w_t,  w_t ~ N(0, Q)
    Measurement:      y_t    = x_t * beta_t + v_t, v_t ~ N(0, R)
    """

    def __init__(self, initial_state: np.ndarray, initial_cov: np.ndarray, process_noise: np.ndarray, measurement_noise: float) -> None:
        self.state = np.asarray(initial_state, dtype=np.float64).reshape(-1, 1)  # [intercept, beta]^T
        self.P = np.asarray(initial_cov, dtype=np.float64)  # (2, 2)
        self.Q = np.asarray(process_noise, dtype=np.float64)  # (2, 2)
        self.R = float(measurement_noise)  # scalar variance

    def update(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Ingest new measurement pair (x_t, y_t) and return updated (intercept, beta, innovation).
        """
        # 1. Prediction step
        # State: x_{t|t-1} = x_{t-1|t-1}
        # Covariance: P_{t|t-1} = P_{t-1|t-1} + Q
        self.P = self.P + self.Q

        # 2. Measurement matrix H = [1.0, x]
        H = np.array([[1.0, float(x)]], dtype=np.float64)

        # Innovation: y - H * state
        y_pred = float((H @ self.state).item())
        innovation = float(y) - y_pred

        # Innovation covariance: S = H * P * H^T + R
        S = float((H @ self.P @ H.T).item()) + self.R

        # Kalman gain: K = P * H^T / S
        K = (self.P @ H.T) / S

        # State update: state = state + K * innovation
        self.state = self.state + K * innovation

        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

        intercept = float(self.state[0, 0])
        beta = float(self.state[1, 0])
        return intercept, beta, innovation


class LatentVolatilityFilter:
    """
    1D Kalman filter for tracking latent true log-volatility under microstructure noise:
    State:       h_t = phi * h_{t-1} + w_t,  w_t ~ N(0, sigma_w^2)
    Observation: y_t = h_t + v_t,            v_t ~ N(0, sigma_v^2)
    """

    def __init__(self, phi: float = 0.98, q_var: float = 0.05, r_var: float = 0.10) -> None:
        self.phi = float(phi)
        self.q = float(q_var)
        self.r = float(r_var)
        self.h = 0.0
        self.p = 1.0

    def filter_step(self, obs_log_vol: float) -> float:
        # Predict
        h_pred = self.phi * self.h
        p_pred = (self.phi ** 2) * self.p + self.q

        # Update
        k = p_pred / (p_pred + self.r)
        self.h = h_pred + k * (obs_log_vol - h_pred)
        self.p = (1.0 - k) * p_pred

        return float(self.h)
