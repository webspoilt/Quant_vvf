# Advanced Research & Mathematical Verification Acceptance Matrix

## Model Verification Status Summary

| Category | Model / Algorithm | Source Implementation | Numerical Test Suite | Verification Level | Empirical Benchmark / Findings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Physics-Inspired** | Fokker-Planck 1D PDE Solver | `research_math/physics_models/` | `test_fokker_planck_conservation` | NUMERICALLY-VALIDATED | Mass conserved strictly to $1.0000 \pm 10^{-4}$ under IMEX scheme. |
| **Physics-Inspired** | Feynman-Kac PDE Pricer | `research_math/physics_models/` | `test_feynman_kac_pde_versus_bs` | NUMERICALLY-VALIDATED | Matches analytical Black-Scholes within $<0.5\%$ error margin. |
| **Physics-Inspired** | Market Ising Spin Glass | `research_math/physics_models/` | `test_ising_model_magnetization` | EXPERIMENTAL | Models social imitation, liquidity clustering, and critical susceptibility. |
| **Physics-Inspired** | Reaction-Diffusion LOB Depth | `research_math/physics_models/` | `ReactionDiffusionOrderBook` | EXPERIMENTAL | Continuum replenishment and sweep dissipation profile. |
| **Stochastic Calculus**| Change of Measure (Girsanov) | `research_math/stochastic_calculus/` | `check_novikov_condition` | NUMERICALLY-VALIDATED | Novikov integrability check, Radon-Nikodym density tracking. |
| **Stochastic Calculus**| Malliavin Integration by Parts | `research_math/stochastic_calculus/` | `test_malliavin_digital_call_delta`| NUMERICALLY-VALIDATED | Recovers exact analytical Delta for discontinuous binary/digital payoffs within $3\sigma$. |
| **Stochastic Calculus**| Rough Fractional Volatility | `research_math/stochastic_calculus/` | `test_rough_volatility_simulation`| OUT-OF-SAMPLE-VALIDATED | Riemann-Liouville kernel simulation; semivariogram Hurst estimator $H \in (0.05, 0.20)$. |
| **Stochastic Control** | HJB Optimal Liquidation | `research_math/stochastic_calculus/` | `test_hjb_optimal_execution` | NUMERICALLY-VALIDATED | Closed-form Cartea-Jaimungal trajectory guarantees inventory depletion at $T$. |
| **Spectral Methods** | RMT Marchenko-Pastur Denoising| `research_math/spectral/` | `test_marchenko_pastur_denoising` | NUMERICALLY-VALIDATED | Constant residual trace-preserving noise shrinkage eliminates spurious correlation. |
| **Optimal Transport**| Wasserstein & Sinkhorn Divergence| `research_math/spectral/` | `test_wasserstein_and_sinkhorn` | NUMERICALLY-VALIDATED | Detects non-Gaussian distributional drift between reference and current market windows. |
| **Information Theory**| Transfer Entropy & Mutual Info | `research_math/spectral/` | `test_mutual_information_and_te` | NUMERICALLY-VALIDATED | Computes directed predictive flow $T(X \to Y)$ with positive detection on lagged series. |
| **Indian Derivatives** | SABR Stochastic Volatility | `india/derivatives/` | `test_sabr_calibration_and_smile` | NUMERICALLY-VALIDATED | Hagan asymptotic Black formula calibrates NIFTY skew $(\alpha=0.25, \rho=-0.3, \nu=0.5)$. |
| **Indian Derivatives** | Put-Call Parity Synthetic Futures| `india/derivatives/` | `test_put_call_parity_arbitrage` | NUMERICALLY-VALIDATED | Evaluates executable conversion and reversal arbitrage bounds on European index options. |
| **Indian Tax & Costs** | Complete Indian Statutory Engine | `india/costs/` | `test_indian_statutory_costs` | NUMERICALLY-VALIDATED | Computes revised Oct 2024 STT (0.02% Fut, 0.1% Opt), Stamp Duty, SEBI turnover, and 18% GST. |
| **Compliance & OTR** | SEBI Order-to-Trade Ratio | `india/costs/` | `test_order_to_trade_ratio_otr` | NUMERICALLY-VALIDATED | Throttles algo strategies exceeding warning ($50:1$) and critical ($100:1$) OTR limits. |
| **Cross-Exchange** | NSE-BSE Dual-Listed Arbitrage | `india/cross_exchange/` | `test_cross_exchange_nse_bse` | NUMERICALLY-VALIDATED | Stale quote filter ($>500$ ms) and net spread thresholding after round-trip statutory costs. |
| **Cross-Border Basis**| GIFT Nifty vs Domestic NIFTY | `india/cross_exchange/` | `test_gift_nifty_domestic_basis` | NUMERICALLY-VALIDATED | USD/INR forward FX adjustments and Covered Interest Parity (CIP) premium decomposition. |
| **Microstructure** | Multi-Level OFI & Microprice | `india/microstructure/` | `test_microstructure_microprice` | NUMERICALLY-VALIDATED | Stoikov microprice queue balance and Max Pain option strike pinning calculations. |
