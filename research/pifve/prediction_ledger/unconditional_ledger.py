"""
Quant_vvf Predictive Intelligence & Forward Validation Engine (PIFVE)
Subsystem 1: Unconditional Prediction Ledger (UPL)

Core Tenet:
Every eligible model forecast is immutably recorded BEFORE downstream trading policy decisions.
Evaluates model skill unconditionally, eliminating policy filtering selection bias.
Also logs operational failures (timeouts, NaN outputs, API dropouts) as explicit events.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


class DecisionPolicyAction(Enum):
    EXECUTED = "executed"
    REJECTED_EXPECTED_VALUE = "rejected_negative_ev"
    REJECTED_RISK_LIMIT = "rejected_risk_gate"
    REJECTED_OTR_LIMIT = "rejected_otr_threshold"
    REJECTED_SPREAD_SLIPPAGE = "rejected_excessive_spread"
    SHADOW_OBSERVATION_ONLY = "shadow_observation_only"
    FAILED_OPERATIONAL = "failed_operational"


class OutcomeBarrier(Enum):
    UNMATURED = "unmatured"
    UPPER_TOUCHED = "upper_barrier_touched"    # +1
    LOWER_TOUCHED = "lower_barrier_touched"    # -1
    VERTICAL_TOUCHED = "vertical_barrier_touched" # 0 (Expiry)
    AMBIGUOUS_BAR = "ambiguous_same_bar"       # Both crossed in same bar


@dataclass
class ForecastRecord:
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_id: str = ""
    model_version: str = "1.0.0"
    git_commit: str = ""
    market: str = "DERIBIT" # DERIBIT, NSE, BSE, NSE_IX
    instrument: str = "BTC-PERPETUAL"
    forecast_horizon_seconds: int = 7200 # e.g. 2 hours
    feature_snapshot_id: str = ""
    
    # Probabilities: [P(Upper/+1), P(Lower/-1), P(Vertical/0)]
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    calibrated_probabilities: Dict[str, float] = field(default_factory=dict)
    historical_base_rates: Dict[str, float] = field(default_factory=dict)
    
    # Trading Decision Metadata
    policy_action: DecisionPolicyAction = DecisionPolicyAction.SHADOW_OBSERVATION_ONLY
    rejection_reason: Optional[str] = None
    trade_attempted: bool = False
    
    # Operational Status
    is_operational_failure: bool = False
    error_message: Optional[str] = None

    # Matured Outcome (populated when horizon completes)
    matured_at_utc: Optional[str] = None
    realized_barrier: OutcomeBarrier = OutcomeBarrier.UNMATURED
    realized_return_pct: Optional[float] = None
    realized_pnl_bps: Optional[float] = None


class UnconditionalPredictionLedger:
    """
    Durable, queryable in-memory and append-only ledger for model predictions.
    Is isolated from latency-critical C++ engine paths.
    """
    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self._records: Dict[str, ForecastRecord] = {}

    def log_forecast(
        self,
        model_id: str,
        model_version: str,
        market: str,
        instrument: str,
        forecast_horizon_seconds: int,
        raw_probabilities: Dict[str, float],
        calibrated_probabilities: Optional[Dict[str, float]] = None,
        historical_base_rates: Optional[Dict[str, float]] = None,
        feature_snapshot_id: str = "",
        git_commit: str = "",
        policy_action: DecisionPolicyAction = DecisionPolicyAction.SHADOW_OBSERVATION_ONLY,
        rejection_reason: Optional[str] = None,
        trade_attempted: bool = False,
    ) -> ForecastRecord:
        """
        Unconditionally logs a valid prediction before trading policy filtering.
        """
        rec = ForecastRecord(
            model_id=model_id,
            model_version=model_version,
            git_commit=git_commit,
            market=market,
            instrument=instrument,
            forecast_horizon_seconds=forecast_horizon_seconds,
            feature_snapshot_id=feature_snapshot_id,
            raw_probabilities=raw_probabilities,
            calibrated_probabilities=calibrated_probabilities or raw_probabilities.copy(),
            historical_base_rates=historical_base_rates or {"upper": 0.333, "lower": 0.333, "vertical": 0.334},
            policy_action=policy_action,
            rejection_reason=rejection_reason,
            trade_attempted=trade_attempted,
            is_operational_failure=False,
        )
        self._records[rec.prediction_id] = rec
        self._persist_record(rec)
        return rec

    def log_operational_failure(
        self,
        model_id: str,
        market: str,
        instrument: str,
        error_message: str,
        feature_snapshot_id: str = "",
    ) -> ForecastRecord:
        """
        Logs an operational fault (timeout, NaN output, missing feature) as a first-class event.
        """
        rec = ForecastRecord(
            model_id=model_id,
            market=market,
            instrument=instrument,
            feature_snapshot_id=feature_snapshot_id,
            policy_action=DecisionPolicyAction.FAILED_OPERATIONAL,
            is_operational_failure=True,
            error_message=error_message,
        )
        self._records[rec.prediction_id] = rec
        self._persist_record(rec)
        return rec

    def resolve_outcome(
        self,
        prediction_id: str,
        realized_barrier: OutcomeBarrier,
        realized_return_pct: float,
        realized_pnl_bps: float,
        matured_at_utc: Optional[str] = None,
    ) -> bool:
        """
        Attaches the realized ground-truth barrier outcome once the window matures.
        """
        if prediction_id not in self._records:
            return False

        rec = self._records[prediction_id]
        rec.realized_barrier = realized_barrier
        rec.realized_return_pct = realized_return_pct
        rec.realized_pnl_bps = realized_pnl_bps
        rec.matured_at_utc = matured_at_utc or datetime.now(timezone.utc).isoformat()
        self._persist_record(rec)
        return True

    def get_records_by_model(self, model_id: str) -> List[ForecastRecord]:
        return [r for r in self._records.values() if r.model_id == model_id]

    def get_matured_records(self, model_id: Optional[str] = None) -> List[ForecastRecord]:
        recs = [r for r in self._records.values() if r.realized_barrier != OutcomeBarrier.UNMATURED and not r.is_operational_failure]
        if model_id:
            recs = [r for r in recs if r.model_id == model_id]
        return recs

    def _persist_record(self, rec: ForecastRecord):
        if not self.persistence_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)
        with open(self.persistence_path, "a", encoding="utf-8") as f:
            data = asdict(rec)
            # Enums to string
            data["policy_action"] = rec.policy_action.value
            data["realized_barrier"] = rec.realized_barrier.value
            f.write(json.dumps(data) + "\n")
