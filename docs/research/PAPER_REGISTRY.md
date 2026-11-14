# QUANT_VVF — RESEARCH PAPER REGISTRY (EXPANDED)

This registry records authoritative peer-reviewed academic literature, mathematical models, empirical formulations, deviations, and adaptation specifics underpinning the mathematical finance, deep learning, and microstructure expansion of `Quant_vvf`.

---

## 1. Volatility Dynamics & Econometrics

### P-01: GARCH(1,1)
- **Paper:** Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics, 31(3), 307-327.
- **DOI/URL:** https://doi.org/10.1016/0304-4076(86)90063-1
- **Model Equation:**
  $$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \omega > 0, \; \alpha \ge 0, \; \beta \ge 0, \; \alpha + \beta < 1$$

### P-02: Exponential GARCH (EGARCH)
- **Paper:** Nelson, D. B. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach*. Econometrica, 59(2), 347-370.
- **DOI/URL:** https://doi.org/10.2307/2938260
- **Model Equation:**
  $$\ln(\sigma_t^2) = \omega + \beta \ln(\sigma_{t-1}^2) + \alpha \left(\left|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}\right| - \sqrt{2/\pi}\right) + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}}$$

### P-03: GJR-GARCH
- **Paper:** Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*. The Journal of Finance, 48(5), 1779-1801.
- **DOI/URL:** https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
- **Model Equation:**
  $$\sigma_t^2 = \omega + \left(\alpha + \gamma \mathbf{1}_{\{\varepsilon_{t-1} < 0\}}\right) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### P-04: Heterogeneous Autoregressive Model of Realized Volatility (HAR-RV)
- **Paper:** Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
- **DOI/URL:** https://doi.org/10.1093/jjfinec/nbp001
- **Model Equation:**
  $$RV_{t+1d}^{(d)} = c + \beta^{(d)} RV_t^{(d)} + \beta^{(w)} RV_t^{(w)} + \beta^{(m)} RV_t^{(m)} + \epsilon_{t+1d}$$

### P-05: Dynamic Conditional Correlation (DCC-GARCH)
- **Paper:** Engle, R. (2002). *Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models*. Journal of Business & Economic Statistics, 20(3), 339-350.
- **DOI/URL:** https://doi.org/10.1198/073500102288618487
- **Model Equations:**
  $$Q_t = (1 - a - b)\bar{Q} + a (\boldsymbol{\varepsilon}_{t-1} \boldsymbol{\varepsilon}_{t-1}^T) + b Q_{t-1}, \quad R_t = \operatorname{diag}(Q_t)^{-1/2} Q_t \operatorname{diag}(Q_t)^{-1/2}$$
- **Role:** Generates time-varying correlation matrices between BTC spot, perpetual futures, and options Greeks.

---

## 2. Stochastic Volatility, Jump-Diffusion & Derivatives

### P-06: The Heston Stochastic Volatility Model
- **Paper:** Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. The Review of Financial Studies, 6(2), 327-343.
- **Paper:** Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The Little Heston Trap*. Wilmott Magazine, 83-92.
- **Model Equations:**
  $$dS_t / S_t = (r - q) dt + \sqrt{v_t} dW_t^S, \quad dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v, \quad \operatorname{corr}(dW_t^S, dW_t^v) = \rho dt$$

### P-07: Merton Jump-Diffusion
- **Paper:** Merton, R. C. (1976). *Option Pricing when Underlying Stock Returns are Discontinuous*. Journal of Financial Economics, 3(1-2), 125-144.
- **DOI/URL:** https://doi.org/10.1016/0304-405X(76)90022-2
- **Model:** Compound Poisson jumps in continuous time:
  $$\frac{dS_t}{S_t} = (r - q - \lambda k) dt + \sigma dW_t + (Y - 1) dN_t, \quad \ln(Y) \sim \mathcal{N}(\mu_J, \delta_J^2)$$

### P-08: Kou Double-Exponential Jump Model
- **Paper:** Kou, S. G. (2002). *A Jump-Diffusion Model for Option Pricing*. Management Science, 48(8), 1086-1101.
- **DOI/URL:** https://doi.org/10.1287/mnsc.48.8.1086.166
- **Model:** Asymmetric upward and downward jump distributions capturing memoryless crash dynamics.

### P-09: SVI Quasi-Explicit Surface
- **Paper:** Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parameterization*. Global Derivatives.
- **Paper:** Gatheral, J., & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces*. Quantitative Finance, 14(1), 59-71.
- **Model Equation (Raw SVI):**
  $$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$

---

## 3. Extreme Value Theory & Copula Dependence

### P-10: Generalized Pareto Distribution & Peaks Over Threshold
- **Paper:** McNeil, A. J., & Frey, R. (2000). *Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value Approach*. Journal of Empirical Finance, 7(3-4), 271-300.
- **DOI/URL:** https://doi.org/10.1016/S0927-5398(00)00012-8
- **Model Equation:**
  $$G_{\xi, \sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$
  Analytical $\operatorname{VaR}_p$ and $\operatorname{ES}_p$ under heavy Fréchet tails ($\xi > 0$).

### P-11: Clayton and Gumbel Copulas
- **Paper:** Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer Science & Business Media.
- **Clayton Copula:** $C(u, v) = \max(0, u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}$, lower tail dependence $\lambda_L = 2^{-1/\theta}$.
- **Gumbel Copula:** $C(u, v) = \exp(-((-\ln u)^\theta + (-\ln v)^\theta)^{1/\theta})$, upper tail dependence $\lambda_U = 2 - 2^{1/\theta}$.

---

## 4. Market Microstructure & Execution

### P-12: Hawkes Processes for High-Frequency Order Flow
- **Paper:** Hawkes, A. G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes*. Biometrika, 58(1), 83-90.
- **Model Equation:**
  $$\lambda_i(t) = \mu_i + \sum_{j} \int_0^t \alpha_{ij} e^{-\beta_{ij} (t - s)} dN_j(s)$$

### P-13: Avellaneda-Stoikov Market Making
- **Paper:** Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217-224.
- **DOI/URL:** https://doi.org/10.1080/14697680802142490
- **Formulation:** Reservation price $r(s, q, t) = s - q \gamma \sigma^2 (T - t)$ and inventory-risk calibrated bid/ask spreads.

### P-14: Obizhaeva-Wang Transient Impact
- **Paper:** Obizhaeva, A. A., & Wang, J. (2013). *Optimal consumption and portfolio selection with transaction costs*. Journal of Financial Markets, 16(1), 1-32.
- **DOI/URL:** https://doi.org/10.1016/j.finmar.2012.09.001
- **Model:** Exponential order book resilience kernel $G(t - s) = \lambda e^{-\rho (t - s)}$.

---

## 5. Statistical Validation & Backtest Governance

### P-15: Deflated Sharpe Ratio (DSR)
- **Paper:** Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management, 40(5), 94-107.
- **DOI/URL:** https://doi.org/10.3905/jpm.2014.40.5.094
- **Formula:**
  $$\operatorname{DSR} = \Phi\left(\frac{(\widehat{\mathrm{SR}} - \mathrm{SR}_0)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\mathrm{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\mathrm{SR}}^2}}\right)$$

---

## 6. Deep Learning Architectures

### P-16: DeepLOB (Convolutional & Recurrent LOB Processing)
- **Paper:** Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. IEEE Transactions on Signal Processing, 67(14), 3701-3712.
- **DOI/URL:** https://doi.org/10.1109/TSP.2019.2919404

### P-17: Temporal Fusion Transformer (TFT)
- **Paper:** Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748-1764.
- **DOI/URL:** https://doi.org/10.1016/j.ijforecast.2021.03.012

### P-18: Deep Hedging
- **Paper:** Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8), 1271-1291.
- **DOI/URL:** https://doi.org/10.1080/14697688.2019.1571683

### P-19: PatchTST
- **Paper:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023.

### P-20: iTransformer (Inverted Variate Tokens)
- **Paper:** Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024.
- **URL:** https://openreview.net/forum?id=JePfAI8fah

### P-21: N-BEATS (Neural Basis Expansion)
- **Paper:** Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*. ICLR 2020.
- **URL:** https://openreview.net/forum?id=r1ecqn4YwB
