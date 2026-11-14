"""
QUANT_VVF — Central Model Registry & JARVIS Multi-Agent Interface.
Governs model provenance, validation status, inference freshness, and sandboxed advisory integration.
"""

from __future__ import annotations
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


class ModelLifecycleState:
    RESEARCH = "RESEARCH"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    APPROVED_FOR_BACKTEST = "APPROVED_FOR_BACKTEST"
    APPROVED_FOR_PAPER = "APPROVED_FOR_PAPER"
    DISABLED = "DISABLED"


@dataclass
class ModelMetadata:
    model_name: str
    research_reference: str
    architecture: str
    version: str
    feature_schema: List[str]
    training_period: str
    validation_metrics: Dict[str, float]
    artifact_checksum: str
    deployment_status: str
    limitations: str
    last_updated_epoch: float


@dataclass
class ModelPrediction:
    timestamp: float
    model_name: str
    model_version: str
    forecast_horizon: int
    predicted_value: Any
    uncertainty: float
    input_freshness_seconds: float
    is_valid: bool
    status_reason: str


class ModelRegistry:
    """Central registry tracking approved model versions and metadata."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelMetadata] = {}

    def register_model(self, metadata: ModelMetadata) -> None:
        self._models[metadata.model_name] = metadata

    def get_model(self, model_name: str) -> Optional[ModelMetadata]:
        return self._models.get(model_name)

    def list_models(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self._models.values()]

    def update_lifecycle_state(self, model_name: str, new_state: str) -> None:
        if model_name in self._models:
            self._models[model_name].deployment_status = new_state
            self._models[model_name].last_updated_epoch = time.time()


class JarvisModelAdvisor:
    """
    JARVIS AI Multi-Agent Advisory Interface.
    Advisory only: can inspect predictions, compare models, and alert trading desk.
    Strictly zero authority to place or alter deterministic exchange orders.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def evaluate_prediction(
        self, prediction: ModelPrediction, max_allowed_staleness_sec: float = 60.0
    ) -> Dict[str, Any]:
        """Verify model output validity and freshness before strategy consumption."""
        model_meta = self.registry.get_model(prediction.model_name)
        if not model_meta:
            return {"approved": False, "reason": f"Unknown model {prediction.model_name}"}

        # Check approval state
        if model_meta.deployment_status not in [
            ModelLifecycleState.APPROVED_FOR_BACKTEST,
            ModelLifecycleState.APPROVED_FOR_PAPER,
        ]:
            return {
                "approved": False,
                "reason": f"Model {prediction.model_name} status is {model_meta.deployment_status}, not approved for active strategy use",
            }

        # Check freshness
        if prediction.input_freshness_seconds > max_allowed_staleness_sec:
            return {
                "approved": False,
                "reason": f"Input data stale: {prediction.input_freshness_seconds:.1f}s > {max_allowed_staleness_sec:.1f}s limit",
            }

        if not prediction.is_valid:
            return {"approved": False, "reason": f"Invalid prediction: {prediction.status_reason}"}

        return {
            "approved": True,
            "model_version": prediction.model_version,
            "predicted_value": prediction.predicted_value,
            "uncertainty": prediction.uncertainty,
            "advisory_note": f"Advisory evaluation passed with {prediction.uncertainty:.4f} uncertainty.",
        }
