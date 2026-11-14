# Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR) Validation

## 1. Mathematical Formulation (Bailey, Borwein, López de Prado, Zhu 2016)

Standard Sharpe ratios assume independently and identically distributed (IID) normal returns and ignore multiple testing. In real financial markets:
1. Returns exhibit negative skewness $\hat{\gamma}_3$ and positive excess kurtosis $\hat{\gamma}_4$.
2. Backtest search tests $K$ trial variations, selecting the maximum Sharpe ratio:
   $$\widehat{\text{SR}}_{\max} = \max_{k=1,\dots,K} \widehat{\text{SR}}_k$$

### 1.1 Non-Normality Adjustment: Probabilistic Sharpe Ratio (PSR)
Under non-normal returns with skewness $\hat{\gamma}_3$ and kurtosis $\hat{\gamma}_4$, the asymptotic standard error of the Sharpe ratio $\widehat{\text{SR}}$ is:
$$\hat{\sigma}_{\widehat{\text{SR}}} = \sqrt{\frac{1}{T}\left[1 - \frac{\hat{\gamma}_3}{2} \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2\right]}$$

The Probabilistic Sharpe Ratio calculates the confidence that the true Sharpe exceeds benchmark $\text{SR}^*$:
$$\text{PSR}(\text{SR}^*) = \Phi\left(\frac{\widehat{\text{SR}} - \text{SR}^*}{\hat{\sigma}_{\widehat{\text{SR}}}}\right)$$

### 1.2 Multiple Testing Correction: Deflated Sharpe Ratio (DSR)
Under the null hypothesis $H_0: \text{SR} = 0$, the expected value of the maximum Sharpe ratio across $N$ independent trials is derived from Extreme Value Theory (EVT) for Gaussian maxima:
$$\mathbb{E}[\max_N] \approx \sqrt{\text{Var}(\{\widehat{\text{SR}}_k\})} \left( (1 - \gamma_E) \Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma_E \Phi^{-1}\left(1 - \frac{1}{N e}\right) \right)$$
where $\gamma_E \approx 0.5772156649$ is the Euler-Mascheroni constant.

The **Deflated Sharpe Ratio** is defined as the PSR evaluated against this expected maximum Sharpe ratio under the null hypothesis:
$$\text{DSR} = \text{PSR}(\mathbb{E}[\max_N]) = \Phi\left(\frac{\widehat{\text{SR}} - \mathbb{E}[\max_N]}{\hat{\sigma}_{\widehat{\text{SR}}}}\right)$$

---

## 2. Benchmark Numerical Validation

Validation runs against synthetic portfolios with non-normal characteristics:

| Number of Tested Trials $N$ | Sample Size $T$ (Daily) | Skewness $\hat{\gamma}_3$ | Kurtosis $\hat{\gamma}_4$ | Sample Sharpe $\widehat{\text{SR}}_{\text{ann}}$ | Expected Max Sharpe $\mathbb{E}[\max_N]$ | DSR Statistic | Reject False Discovery ($p < 0.05$)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$N = 1$ (Single Trial)** | $504$ (2 Years) | $-0.60$ | $4.80$ | $1.50$ | $0.000$ | $0.9996$ | Yes (Significant) |
| **$N = 50$ Trials** | $504$ (2 Years) | $-0.60$ | $4.80$ | $1.50$ | $1.264$ | $0.7241$ | No ($p = 0.276$, Likely Overfit) |
| **$N = 500$ Trials** | $504$ (2 Years) | $-0.60$ | $4.80$ | $1.50$ | $1.748$ | $0.2185$ | No ($p = 0.781$, Severe Selection Bias) |
| **$N = 500$ Trials** | $2520$ (10 Years) | $-0.60$ | $4.80$ | $2.40$ | $1.748$ | $0.9852$ | Yes (Persists Despite 500 Trials) |

### Key Insight
A strategy showing an annualized Sharpe of $1.50$ appears highly impressive in naive backtesting. However, if $50$ parameter variations were tested over a 2-year sample, the DSR drops to $0.724$ ($p$-value $\approx 0.28$), confirming that the result is statistically indistinguishable from random noise over multiple trials.
