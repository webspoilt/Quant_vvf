# Multi-Market Mathematical Architecture: Deribit, NSE, BSE & GIFT City (NSE IX)

## 1. Architectural Overview

`Quant_vvf` implements an institutional-grade, multi-market quantitative research and execution architecture unifying four distinct derivatives and spot trading domains:

```mermaid
flowchart TD
    subgraph Market_A["Market A: Deribit (Crypto)"]
        D1["BTC / ETH Options & Perpetuals"]
        D2["Inverse Settlement (BTC Numeraire r_BTC)"]
        D3["24/7 Continuous Trading Calendar"]
    end

    subgraph Market_B["Market B: NSE India (Domestic)"]
        N1["NIFTY 50 / BANKNIFTY / FINNIFTY"]
        N2["INR Cash, Futures & European Options"]
        N3["STT, Stamp Duty, GST, OTR Controller"]
    end

    subgraph Market_C["Market C: BSE India (Domestic)"]
        B1["SENSEX / BANKEX Derivatives"]
        B2["Dual-Listed Arbitrage & Executable BPS Spreads"]
        B3["Friday Expiry Schedule Synchronization"]
    end

    subgraph Market_D["Market D: NSE IX / GIFT City (IFSC)"]
        G1["GIFT Nifty USD-Denominated Contracts"]
        G2["Cross-Border Currency Basis (USD/INR Forward)"]
        G3["Extended Multi-Session Global Hours"]
    end

    Market_A & Market_B & Market_C & Market_D --> RiskEngine["Deterministic Risk Engine & Kill-Switch"]
    RiskEngine --> ResearchCore["Unified Research Core: Physics, Stochastic Calculus & Spectral Algebra"]
```

---

## 2. Market Domain Specifications & Segregation Matrix

| Attribute | Market A: Deribit | Market B: NSE India | Market C: BSE India | Market D: GIFT City (NSE IX) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Instruments** | BTC / ETH Options & Inverse Perps | NIFTY, BANKNIFTY, Stock F&O | SENSEX, BANKEX F&O | GIFT Nifty Futures & Options |
| **Settlement Currency** | BTC / ETH (Inverse Numeraire) | INR (Rupees) | INR (Rupees) | USD (US Dollars) |
| **Trading Sessions** | 24/7/365 Continuous | 09:15 – 15:30 IST | 09:15 – 15:30 IST | 06:30 – 15:40 & 16:35 – 02:45 IST |
| **Contract Multiplier** | 1 Contract = 1 USD (or 1 BTC) | 25 (NIFTY), 15 (BANKNIFTY) | 10 (SENSEX), 15 (BANKEX) | 2 USD $\times$ Index Points |
| **Statutory Tax Framework**| Exchange Fee (0.03% taker) | STT (0.02% Fut, 0.1% Opt), Stamp, GST | STT, Stamp Duty, GST, Exchange | Zero STT/Stamp (IFSC Exemptions) |
| **Surveillance / OTR** | Rate Limits (req/sec) | SEBI OTR (Penalty > 50:1) | SEBI OTR (Penalty > 50:1) | IFSCA Algorithmic Rules |
| **Primary Pricing Engine** | Heston QE-M, Bates Jump, SVI | SABR ($\beta=1.0$), Black-76, Parity | SABR, Dual-Listed Carry Arbitrage | Cross-Border CIP Currency Basis |
