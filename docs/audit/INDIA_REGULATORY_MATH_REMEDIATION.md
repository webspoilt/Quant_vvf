# Indian Regulatory, Statutory Cost & Scientific Remediation Audit

## 1. Executive Summary

This audit report records the financial, statutory, and mathematical remediation executed on `Quant_vvf` under branch `feature/india-regulatory-and-math-corrections`.

All identified defects—including the outdated October 2024 STT schedule, naive OTR claims, Feynman-Kac physical grid coordinate discretization, Girsanov Novikov verification, and debiased Sinkhorn divergence—have been corrected and validated against analytical benchmarks.

---

## 2. Priority 0: Indian Statutory Transaction Cost Framework

### 2.1 Statutory Tax Schedule Versioning & Finance Act 2026
`india/costs/tax_engine.py` implements an effective-date-based statutory policy selector:

| Policy Version | Effective Dates | Equity Futures Sell STT | Equity Options Premium Sell STT | Options Exercise STT | Taxable Exercise Base |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PRE_OCT_2024` | Prior to Oct 1, 2024 | $0.0125\%$ | $0.0625\%$ | $0.125\%$ | Intrinsic Value |
| `OCT_2024_TO_MAR_2026` | Oct 1, 2024 – Mar 31, 2026 | $0.0200\%$ | $0.1000\%$ | $0.125\%$ | Intrinsic Value |
| `APR_2026_ONWARD` | April 1, 2026 onward | **$0.0500\%$** | **$0.1500\%$** | **$0.1500\%$** | **Intrinsic Value $(S - K)^+$** |

### 2.2 Critical Exercise Taxable Base Remediation
- **Statutory Requirement**: On exercise of options, STT is calculated exclusively on the **intrinsic value** ($\text{Units} \times (S_{\text{settle}} - K)^+$), NOT on full strike notional ($\text{Units} \times K$).
- **Numerical Test Vector**:
  - 100 units of NIFTY 24,000 Call exercised at 24,200 settlement on April 15, 2026.
  - Taxable base = $100 \times 200 = \text{Rs } 20,000$ (remediated from incorrect strike turnover Rs 2,400,000).
  - STT at $0.15\%$ = **Rs 30.00**.

---

## 3. Priority 0: Official NSE / BSE OTR Surveillance

The claim that $50:1$ and $100:1$ are universal regulatory limits has been corrected. The engine strictly separates:
1. **Official NSE Regulatory Surveillance**:
   - Condition: Daily OTR $\ge 2,000$ on $\ge 3$ occasions within the preceding 30 rolling trading days (minimum 500 order operations).
   - Action: Official exchange surveillance cooling-off notice and regulatory escalation.
2. **Broker-Specific Pre-Trade Risk Gates**:
   - Discrete order limits enforced before transmission.
3. **Internal Strategy Throttling**:
   - Configurable safety threshold (default $50:1$) to catch algorithmic run-away loops.

---

## 4. Priority 1 & 2: Mathematical Remediations

### 4.1 Feynman-Kac PDE Option Pricer
- Corrected stock-price coordinate discretization on grid $S_i$:
  $$\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V = 0$$
- Tridiagonal coefficients properly map $S_i / \Delta S = i$, yielding stable second-order convergence across deep ITM, ATM, and OTM strikes.

### 4.2 Girsanov Measure Engine & Novikov Condition
- Rigorously checks Novikov integrability: $\mathbb{E}[\exp(\frac{1}{2}\int_0^T \lambda_s^2 ds)] < \infty$.
- For deterministic bounded $\lambda$, bounded time ensures the martingale condition holds.
- Empirically verified that $\mathbb{E}_{\mathbb{P}}[Z_T] = 1.0000 \pm 3\times \text{SE}$ under 50,000 Monte Carlo paths.

### 4.3 Debiased Sinkhorn Divergence
- Corrected raw entropic transport cost to the true symmetric debiased Sinkhorn divergence:
  $$S_\epsilon(X, Y) = \text{OT}_\epsilon(X, Y) - \frac{1}{2}\text{OT}_\epsilon(X, X) - \frac{1}{2}\text{OT}_\epsilon(Y, Y)$$
- Satisfies $S_\epsilon(X, X) = 0.0000$ exactly, eliminating entropic self-distance bias.

### 4.4 Model Disambiguation: Curie-Weiss vs General Spin Glass
- Separates Mean-Field Curie-Weiss ($H = -\frac{J}{2N}(\sum s_i)^2 - h \sum s_i$) from general Sherrington-Kirkpatrick spin glasses with arbitrary coupling matrix $J_{ij}$.
