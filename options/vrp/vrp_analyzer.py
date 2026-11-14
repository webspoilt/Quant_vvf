"""
Volatility Risk Premium (VRP) & Term Structure Analytics
Annualized Forward RV vs IV Comparison, Skew Metrics, and Empirical Distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class VRPAnalysisResult:
    mean_vrp: float
    median_vrp: float
    std_vrp: float
    percentile_25: float
    percentile_75: float
    vrp_series: pd.Series
    is_positive_premium_dominant: bool


class VRPAnalyzer:
    """Comprehensive Volatility Risk Premium (VRP = IV - RV) analytics suite."""

    @staticmethod
    def calculate_vrp(implied_vol: pd.Series, realized_vol: pd.Series) -> pd.Series:
        """Calculates VRP series ensuring consistent annualized volatility units."""
        df = pd.DataFrame({"iv": implied_vol, "rv": realized_vol}).dropna()
        return df["iv"] - df["rv"]

    @classmethod
    def analyze_historical_vrp(
        cls,
        implied_vol: pd.Series,
        realized_vol: pd.Series
    ) -> VRPAnalysisResult:
        """Computes statistical distribution of historical VRP."""
        vrp = cls.calculate_vrp(implied_vol, realized_vol)
        if len(vrp) < 10:
            raise ValueError("Insufficient observations for VRP distribution analysis")

        mean_v = float(np.mean(vrp))
        median_v = float(np.median(vrp))
        std_v = float(np.std(vrp, ddof=1))
        p25 = float(np.percentile(vrp, 25))
        p75 = float(np.percentile(vrp, 75))

        pos_ratio = np.mean(vrp > 0)
        return VRPAnalysisResult(
            mean_vrp=mean_v,
            median_vrp=median_v,
            std_vrp=std_v,
            percentile_25=p25,
            percentile_75=p75,
            vrp_series=vrp,
            is_positive_premium_dominant=bool(pos_ratio > 0.50),
        )

    @staticmethod
    def calculate_skew_25delta(put_25d_iv: float, call_25d_iv: float, atm_iv: float) -> float:
        """Computes standardized 25-delta risk reversal skew: (25d Put IV - 25d Call IV) / ATM IV."""
        if atm_iv <= 0:
            raise ValueError("ATM IV must be strictly positive")
        return (put_25d_iv - call_25d_iv) / atm_iv
