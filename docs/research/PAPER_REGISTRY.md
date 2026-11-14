# QUANT_VVF — RESEARCH PAPER REGISTRY

This registry records authoritative peer-reviewed academic literature, mathematical models, empirical formulations, deviations, and adaptation specifics underpinning the mathematical finance and deep learning expansion of `Quant_vvf`.

---

## 1. Volatility Dynamics & Econometrics

### P-01: GARCH(1,1)
- **Paper:** Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics, 31(3), 307-327.
- **DOI/URL:** https://doi.org/10.1016/0304-4076(86)90063-1
- **Model Equation:**
  $$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \omega > 0, \; \alpha \ge 0, \; \beta \ge 0, \; \alpha + \beta < 1$$
- **Core Assumptions:** Weak stationarity, conditionally normal innovations, symmetric impact of positive and negative shocks.
- **Crypto / BTC Adaptation:** Enforced stationarity boundary ($\alpha + \beta < 0.9999$) with Student's $t$ heavy-tail likelihood option to handle Bitcoin volatility clustering.

### P-02: Exponential GARCH (EGARCH)
- **Paper:** Nelson, D. B. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach*. Econometrica, 59(2), 347-370.
- **DOI/URL:** https://doi.org/10.2307/2938260
- **Model Equation:**
  $$\ln(\sigma_t^2) = \omega + \beta \ln(\sigma_{t-1}^2) + \alpha \left(\left|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}\right| - \sqrt{2/\pi}\right) + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}}$$
- **Core Assumptions:** Asymmetric leverage effect without requiring artificial non-negativity parameter constraints.
- **Crypto / BTC Adaptation:** Captures inverse crypto leverage effects where upward price momentum often accelerates implied volatility.

### P-03: GJR-GARCH
- **Paper:** Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*. The Journal of Finance, 48(5), 1779-1801.
- **DOI/URL:** https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
- **Model Equation:**
  $$\sigma_t^2 = \omega + \left(\alpha + \gamma I_{t-1}\right) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad I_{t-1} = \mathbf{1}_{\{\varepsilon_{t-1} < 0\}}$$
- **Core Assumptions:** Threshold indicator captures differential response between negative returns and positive shocks.

### P-04: Heterogeneous Autoregressive Model of Realized Volatility (HAR-RV)
- **Paper:** Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
- **DOI/URL:** https://doi.org/10.1093/jjfinec/nbp001
- **Model Equation:**
  $$RV_{t+1d}^{(d)} = c + \beta^{(d)} RV_t^{(d)} + \beta^{(w)} RV_t^{(w)} + \beta^{(m)} RV_t^{(m)} + \epsilon_{t+1d}$$
  where $RV_t^{(w)} = \frac{1}{5}\sum_{i=0}^4 RV_{t-i}^{(d)}$ and $RV_t^{(m)} = \frac{1}{22}\sum_{i=0}^{21} RV_{t-i}^{(d)}$.
- **Crypto / BTC Adaptation:** Continuous 24/7/365 trading; daily component uses 1-day (24h), weekly uses 7-day (168h), and monthly uses 30-day (720h) subsampled rolling realized variance.

---

## 2. Stochastic Volatility & Derivatives Pricing

### P-05: The Heston Stochastic Volatility Model
- **Paper:** Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. The Review of Financial Studies, 6(2), 327-343.
- **DOI/URL:** https://doi.org/10.1093/rfs/6.2.327
- **Model Equations:**
  $$dS_t / S_t = (r - q) dt + \sqrt{v_t} dW_t^S$$
  $$dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v$$
  $$\operatorname{corr}(dW_t^S, dW_t^v) = \rho dt$$
- **Analytical Solution:** Characteristic function Fourier inversion with Albrecher et al. (2007) stable formulation preventing branch cut discontinuities:
  $$C(S_0, K, T) = S_0 e^{-qT} P_1 - K e^{-rT} P_2$$
- **Feller Condition:** $2\kappa\theta > \xi^2$ guarantees variance stays strictly positive.
- **Crypto / BTC Adaptation:** Integrated with Deribit coin-margined inverse transformation where payoff in BTC equals $\max(0, S_T - K)/S_T$.

### P-06: SVI Quasi-Explicit Surface
- **Paper:** Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parameterization*. Global Derivatives.
- **Paper:** Gatheral, J., & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces*. Quantitative Finance, 14(1), 59-71.
- **DOI/URL:** https://doi.org/10.1080/14697688.2013.819994
- **Model Equation (Raw SVI):**
  $$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$
- **Arbitrage Diagnostics:** Durrleman butterfly non-arbitrage density $g(k) \ge 0$ and calendar spread monotonicity $w(k, \tau_1) \le w(k, \tau_2)$.

---

## 3. Market Microstructure

### P-07: Hawkes Processes for High-Frequency Order Flow
- **Paper:** Hawkes, A. G. (1971). *Spectra of Some Self-Exciting and Mutually Exciting Point Processes*. Biometrika, 58(1), 83-90.
- **Paper:** Bacry, E., Delattre, S., Hoffmann, M., & Muzy, J. F. (2013). *Some Properties of the Mutually Exciting Hawkes Processes with Applications to Financial Data*. Stochastic Processes and their Applications, 123(7), 2475-2499.
- **DOI/URL:** https://doi.org/10.1016/j.spa.2013.04.007
- **Model Equation:**
  $$\lambda_i(t) = \mu_i + \sum_{j} \int_0^t \alpha_{ij} e^{-\beta_{ij} (t - s)} dN_j(s)$$
- **Stability Condition:** Spectral radius of matrix $\Gamma = [\alpha_{ij} / \beta_{ij}]$ satisfies $\rho(\Gamma) < 1$.
- **Crypto / BTC Adaptation:** Bivariate buy-aggressive vs. sell-aggressive market orders and quote cancellations on Deribit order book.

### P-08: Order Book Imbalance, Microprice & Execution Costs
- **Paper:** Stoikov, S. (2018). *The Micro-Price: a High-Frequency Estimator of Future Prices*. Quantitative Finance, 18(12), 1959-1966.
- **DOI/URL:** https://doi.org/10.1080/14697688.2018.1489132
- **Paper:** Almgren, R., & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions*. Journal of Risk, 3(2), 5-40.
- **Formulas:**
  $$\text{Imbalance} = \frac{Q_b - Q_a}{Q_b + Q_a}, \quad P_{\text{micro}} = P_a \frac{Q_b}{Q_b + Q_a} + P_b \frac{Q_a}{Q_b + Q_a}$$

---

## 4. State Estimation & Dynamic Filtering

### P-09: Kalman Filtering for Dynamic State Estimation & Hedge Ratios
- **Paper:** Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering, 82(1), 35-45.
- **DOI/URL:** https://doi.org/10.1115/1.3662552
- **State-Space System:**
  $$x_t = F_t x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)$$
  $$y_t = H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)$$
- **Application:** Real-time tracking of time-varying cointegration hedge ratios and latent volatility state under microstructure noise.

---

## 5. Portfolio Mathematics & Risk Optimization

### P-10: Conditional Value-at-Risk (CVaR) Optimization
- **Paper:** Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21-42.
- **DOI/URL:** https://doi.org/10.21314/JOR.2000.038
- **Loss Threshold Formulation:**
  $$\min_{\mathbf{w}, \eta} \left[ \eta + \frac{1}{1 - \alpha} \frac{1}{S} \sum_{s=1}^S \max(L_s(\mathbf{w}) - \eta, 0) \right]$$
  subject to $\sum_i w_i = 1, \; w_i \ge 0, \; \mathbf{w}^T \boldsymbol{\mu} \ge R_{\text{target}}$.

### P-11: Covariance Shrinkage & Hierarchical Risk Parity
- **Paper:** Ledoit, O., & Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis, 88(2), 365-411.
- **Paper:** López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. The Journal of Portfolio Management, 42(4), 59-69.
- **DOI/URL:** https://doi.org/10.3905/jpm.2016.42.4.059

---

## 6. Deep Learning Architectures

### P-12: DeepLOB (Convolutional & Recurrent LOB Processing)
- **Paper:** Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. IEEE Transactions on Signal Processing, 67(14), 3701-3712.
- **DOI/URL:** https://doi.org/10.1109/TSP.2019.2919404
- **Architecture:** 3 Conv2D + LeakyReLU layers across price/volume spatial features $\to$ Inception module $\to$ LSTM recurrent sequence processing $\to$ Fully connected Softmax (3-class: Up, Stationary, Down).
- **Crypto / BTC Adaptation:** Ingests top 10 levels of Deribit LOB with volume normalized by rolling average depth; output incorporates prediction uncertainty via entropy.

### P-13: Temporal Fusion Transformer (TFT)
- **Paper:** Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748-1764.
- **DOI/URL:** https://doi.org/10.1016/j.ijforecast.2021.03.012
- **Components:** Variable Selection Networks (VSN), Gated Residual Networks (GRN), static context enrichment, multi-head temporal self-attention, and quantile regression ($10\%, 50\%, 90\%$).
- **Crypto / BTC Adaptation:** Multi-horizon forecasting of realized variance and implied volatility with interpretability attention weights.

### P-14: Deep Hedging
- **Paper:** Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8), 1271-1291.
- **DOI/URL:** https://doi.org/10.1080/14697688.2019.1571683
- **Formulation:** Recurrent policy network $\delta_t = \pi_\theta(S_t, v_t, \delta_{t-1}, \tau_t)$ minimizing convex risk measure $\rho(- \Pi_T)$ subject to proportional transaction costs $c \cdot S_t |\Delta \delta_t|$.
- **Crypto / BTC Adaptation:** Inverse $(S_0/S_1)^2$ delta scaling and non-linear BTC numeraire payoff constraints embedded directly in loss function.

### P-15: PatchTST
- **Paper:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. International Conference on Learning Representations (ICLR 2023).
- **URL:** https://openreview.net/forum?id=Jbdc0vTOcol
- **Architecture:** Channel-independent sub-series patching, linear embedding, standard Transformer encoder with self-attention, and flattening linear projection.
- **Role:** Direct challenger to TFT for high-dimensional multivariate market volatility forecasting.
