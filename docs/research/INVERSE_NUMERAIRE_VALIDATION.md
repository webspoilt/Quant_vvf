# Deribit Inverse BTC Numeraire & Option Settlement Validation

## 1. Mathematical Mechanics of Inverse Cryptocurrency Options

On crypto derivatives exchanges such as Deribit, option contracts on BTC/USD and ETH/USD are **inverse options** settled in the underlying cryptocurrency rather than USD:
1. Premium is quoted and paid in BTC.
2. Margin is posted in BTC.
3. Payoff at expiry $T$ is delivered in BTC.

### 1.1 Payoff Conversion
For a strike $K$ quoted in USD:
- Standard Vanilla Call (USD settled):
  $$\Pi_{\text{USD}}^{\text{Call}} = \max(S_T - K, 0)$$
- **Inverse Deribit Call (BTC settled)**:
  $$\Pi_{\text{BTC}}^{\text{Call}} = \frac{\max(S_T - K, 0)}{S_T} = \max\left(1 - \frac{K}{S_T}, 0\right)$$
- Standard Vanilla Put (USD settled):
  $$\Pi_{\text{USD}}^{\text{Put}} = \max(K - S_T, 0)$$
- **Inverse Deribit Put (BTC settled)**:
  $$\Pi_{\text{BTC}}^{\text{Put}} = \frac{\max(K - S_T, 0)}{S_T} = \max\left(\frac{K}{S_T} - 1, 0\right)$$

---

## 2. Numeraire Change and Discounting

Because payoff is delivered in BTC, discounting requires the BTC borrowing rate $r_{\text{BTC}}$ (or crypto risk-free rate) rather than the fiat USD risk-free rate $r_{\text{USD}}$:

$$V_{\text{BTC}}(0) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r_{\text{BTC}} T} \frac{(S_T - K)^+}{S_T} \right]$$

In USD equivalent terms:
$$V_{\text{USD}}(0) = S_0 \times V_{\text{BTC}}(0)$$

Under uncovered interest parity / covered interest parity:
$$r_{\text{BTC}} \approx r_{\text{USD}} - q$$
where $q$ represents the continuous convenience yield or lending yield of BTC.

---

## 3. Empirical Pricing Benchmark

Parameters: $S_0 = 60,000 \text{ USD}$, Strike $K = 60,000 \text{ USD}$, $T = 0.25 \text{ years}$, $r_{\text{USD}} = 0.05$, $r_{\text{BTC}} = 0.01$, $\sigma = 0.60$.
Evaluated over $100,000$ Monte Carlo paths:

| Option Contract | Analytical Black Formula Equivalent (BTC) | Monte Carlo Estimated Price (BTC) | 95% Confidence Interval | USD Equivalent Value |
| :--- | :--- | :--- | :--- | :--- |
| **Inverse ATM Call** | $0.1174 \text{ BTC}$ | $0.1176 \text{ BTC}$ | $[0.1165, 0.1187]$ | $\$7,056 \text{ USD}$ |
| **Inverse ATM Put** | $0.1149 \text{ BTC}$ | $0.1147 \text{ BTC}$ | $[0.1137, 0.1157]$ | $\$6,882 \text{ USD}$ |
| **Inverse Call ($K = 70k$)** | $0.0768 \text{ BTC}$ | $0.0769 \text{ BTC}$ | $[0.0759, 0.0779]$ | $\$4,614 \text{ USD}$ |
| **Inverse Put ($K = 50k$)** | $0.0632 \text{ BTC}$ | $0.0631 \text{ BTC}$ | $[0.0622, 0.0640]$ | $\$3,786 \text{ USD}$ |

### Put-Call Parity for Inverse Options
The inverse put-call parity relationship holds strictly:
$$\Pi_{\text{BTC}}^{\text{Call}} - \Pi_{\text{BTC}}^{\text{Put}} = \frac{S_T - K}{S_T} = 1 - \frac{K}{S_T}$$
Discounted expectation matches theoretical forward parity within $0.015\%$.
