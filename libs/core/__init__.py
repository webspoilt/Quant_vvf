"""Core numerical library: instrument math shared by research, sim, and prod code.

Research-to-production slippage is eliminated by construction: the simulation
walk-forward engine and the live Nautilus actor must import these exact
functions (:mod:`inverse_payoffs`, and from Phase 1 :mod:`svi_surface`).
"""
