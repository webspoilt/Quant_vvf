"""
Unit Tests for JARVIS Multi-Agent Architecture.
Ensures AI proposals require deterministic risk validation and human approval.
"""

import pytest
from ai_orchestrator.core import JarvisSupervisor, AgentRole, AgentProposal


class TestJarvisMultiAgentLayer:

    def test_jarvis_supervisor_proposal_cycle(self):
        supervisor = JarvisSupervisor()
        context = {
            "spot_price": 65000.0,
            "implied_vol": 0.65,
            "realized_vol": 0.48,
        }

        proposal = supervisor.process_user_query("Analyze BTC volatility strategy", context)
        assert isinstance(proposal, AgentProposal)
        assert proposal.author_role == AgentRole.SUPERVISOR
        assert proposal.proposed_symbol == "BTC-PERPETUAL"
        assert proposal.suggested_action == "SELL_VOLATILITY"
        assert 0.0 <= proposal.confidence_score <= 1.0

        # Mandatory Safety Invariants
        assert proposal.required_human_approval is True
        assert proposal.deterministic_risk_approved is False, (
            "AI agents must NEVER have pre-authorized live execution status!"
        )

        # Specialist consultation log presence
        meta = proposal.metadata
        assert AgentRole.MARKET_ANALYST.value in meta
        assert AgentRole.RESEARCH_AGENT.value in meta
        assert AgentRole.RISK_ANALYST.value in meta
        assert AgentRole.PORTFOLIO_ANALYST.value in meta
