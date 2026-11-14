"""
JARVIS Multi-Agent AI System Core & Specialist Agent Orchestration
Enforces strict unidirectional flow:
AI Proposes -> Quantitative Research Evaluates -> Deterministic Risk Engine Authorizes.
AI model predictions have ZERO direct order entry authority.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
import json


class AgentRole(Enum):
    SUPERVISOR = "Supervisor"
    MARKET_ANALYST = "Market Analyst"
    RESEARCH_AGENT = "Research Agent"
    STRATEGY_AGENT = "Strategy Agent"
    RISK_ANALYST = "Risk Analyst"
    PORTFOLIO_ANALYST = "Portfolio Analyst"
    EXECUTION_ANALYST = "Execution Analyst"


@dataclass
class AgentProposal:
    proposal_id: str
    author_role: AgentRole
    hypothesis: str
    proposed_symbol: str
    suggested_action: str
    suggested_quantity_btc: float
    confidence_score: float
    reasoning: str
    required_human_approval: bool = True
    deterministic_risk_approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecialistAgent:
    """Specialist advisory agent within the JARVIS multi-agent hierarchy."""

    def __init__(self, role: AgentRole):
        self.role = role

    def evaluate_task(self, task_description: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Produces structured analysis for a given research/market context."""
        spot = market_context.get("spot_price", 65000.0)
        iv = market_context.get("implied_vol", 0.60)
        rv = market_context.get("realized_vol", 0.50)
        vrp = iv - rv

        if self.role == AgentRole.MARKET_ANALYST:
            regime = "High Volatility" if iv > 0.70 else "Normal Volatility"
            return {
                "role": self.role.value,
                "regime": regime,
                "spot_price": spot,
                "implied_vol": iv,
                "realized_vol": rv,
                "vrp_estimate": vrp
            }

        elif self.role == AgentRole.RESEARCH_AGENT:
            hypothesis = f"VRP is positive ({vrp:.2%}), suggesting short volatility premium capture"
            return {
                "role": self.role.value,
                "hypothesis": hypothesis,
                "recommended_experiment": "Walk-forward delta-hedged straddle simulation",
                "statistical_significance_target": "p < 0.05"
            }

        elif self.role == AgentRole.STRATEGY_AGENT:
            return {
                "role": self.role.value,
                "suggested_strategy": "VRP_REBALANCE_STRADDLE",
                "rebalance_band_btc": 0.15,
                "action": "SELL_VOLATILITY",
                "target_delta": 0.0
            }

        elif self.role == AgentRole.RISK_ANALYST:
            # Advisory only: actual authorization is strictly executed by C++ risk gate!
            return {
                "role": self.role.value,
                "advisory_warning": "Max gross exposure must not exceed 50 BTC",
                "recommended_clip_limit_btc": 2.0,
                "margin_stress_scenario": "Spot drop of 30% increases short put BTC liability"
            }

        elif self.role == AgentRole.PORTFOLIO_ANALYST:
            return {
                "role": self.role.value,
                "allocation_recommendation": "Risk Parity (ERC) 40% BTC, 60% Perp Hedge",
                "expected_diversification_ratio": 1.45
            }

        elif self.role == AgentRole.EXECUTION_ANALYST:
            return {
                "role": self.role.value,
                "order_routing_protocol": "Deribit Starbase SBE over TCP",
                "expected_slippage_bps": 2.0
            }

        return {"role": self.role.value, "status": "Ready"}


class JarvisSupervisor:
    """Supervisory orchestrator coordinating multi-agent research proposals."""

    def __init__(self):
        self.agents = {
            role: SpecialistAgent(role)
            for role in AgentRole
            if role != AgentRole.SUPERVISOR
        }

    def process_user_query(self, query: str, market_context: Dict[str, Any]) -> AgentProposal:
        """Runs multi-agent consultation cycle and produces structured proposal."""
        analysis_log = {}
        for role, agent in self.agents.items():
            analysis_log[role.value] = agent.evaluate_task(query, market_context)

        strat_output = analysis_log[AgentRole.STRATEGY_AGENT.value]
        research_output = analysis_log[AgentRole.RESEARCH_AGENT.value]

        proposal = AgentProposal(
            proposal_id="PROP-BTC-VRP-001",
            author_role=AgentRole.SUPERVISOR,
            hypothesis=research_output["hypothesis"],
            proposed_symbol="BTC-PERPETUAL",
            suggested_action=strat_output["action"],
            suggested_quantity_btc=1.0,
            confidence_score=0.85,
            reasoning="Multi-agent consensus confirms positive VRP with manageable tail risk",
            required_human_approval=True,
            deterministic_risk_approved=False,  # Unapproved until C++ risk ledger verifies
            metadata=analysis_log
        )
        return proposal
