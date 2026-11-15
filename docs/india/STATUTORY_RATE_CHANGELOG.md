# Statutory Rate Changelog & Authoritative Exchange Schedule Matrix

**Reference Authority:** Indian Parliament Finance Acts, Central Board of Direct Taxes (CBDT), National Stock Exchange of India (NSE), Bombay Stock Exchange (BSE), and International Financial Services Centres Authority (IFSCA).  
**Repository Branch:** `feature/india-cost-otr-execution-final`  
**Engine Implementation:** `india/costs/tax_engine.py` (`IndiaStatutoryCostEngine`)

---

## 1. Securities Transaction Tax (STT) Progression

| Effective Boundary | Statutory Authority | Equity Futures (Sale) | Equity Options (Sale on Premium) | Exercised Options (Purchaser) | Taxable Base (Exercise) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prior to Sept 1, 2019** | Finance (No. 2) Act, 2004, Chapter VII | 0.0125% | 0.0625% | 0.1250% | **Settlement Value ($Q \times S_{\text{settle}}$)** |
| **Sept 1, 2019 – Sept 30, 2024** | Finance (No. 2) Act, 2019 (Section 98(3) Amendment) | 0.0125% | 0.0625% | 0.1250% | **Intrinsic Value ($Q \times (S - K)^+$)** |
| **Oct 1, 2024 – Mar 31, 2026** | Finance (No. 2) Act, 2024 | 0.0200% | 0.1000% | 0.1250% | **Intrinsic Value ($Q \times (S - K)^+$)** |
| **April 1, 2026 Onward** | Finance Act, 2026 | **0.0500%** | **0.1500%** | **0.1500%** | **Intrinsic Value ($Q \times (S - K)^+$)** |

### Official Gazette & Circular Citations:
1. **Finance (No. 2) Act, 2019 (Effective Sept 1, 2019)**:
   - Amends Chapter VII Section 98 of Finance (No. 2) Act, 2004.
   - Mandates that where option is exercised, STT payable by the purchaser is computed strictly on the intrinsic price ($\max(S_{\text{settle}} - K, 0)$ for calls and $\max(K - S_{\text{settle}}, 0)$ for puts), eradicating the excessive tax on full settlement notional.
2. **Finance (No. 2) Act, 2024 (Effective Oct 1, 2024)**:
   - Escalates Equity Futures sell rate from 0.0125% to 0.020%.
   - Escalates Equity Options premium sell rate from 0.0625% to 0.100%.
3. **Finance Act, 2026 (Effective April 1, 2026)**:
   - Escalates Equity Futures sell rate from 0.020% to 0.050%.
   - Escalates Equity Options premium sell rate from 0.100% to 0.150%.
   - Escalates Exercised Options rate from 0.125% to 0.150% on intrinsic value.

---

## 2. Exchange Transaction Charges & IPFT Schedule

### 2.1 National Stock Exchange of India (NSE)

| Segment | Pre-March 1, 2026 Tariff (per crore) | Pre-March 1, 2026 IPFT (per crore) | March 1, 2026 Onward Tariff (NSE/FA/73061) | March 1, 2026 Onward IPFT |
| :--- | :--- | :--- | :--- | :--- |
| **Capital Market (Cash)** | ₹297.00 (0.00297%) | ₹0.05 (0.0000005%) | **₹306.99 (0.0030699%)** | **₹0.01 (0.0000001%)** |
| **Equity Futures** | ₹173.00 (0.00173%) | ₹0.05 (0.0000005%) | **₹182.99 (0.0018299%)** | **₹0.01 (0.0000001%)** |
| **Equity Options** | ₹3,503.00 (0.03503% on prem) | ₹0.05 (0.0000005% on prem) | **₹3,552.00 (0.03552% on prem)** | **₹0.01 (0.0000001% on prem)** |

*Authority:* NSE Circular **NSE/FA/73061** issued February 2026, effective March 1, 2026.

### 2.2 Bombay Stock Exchange (BSE)

| Segment | Transaction Charge (per crore) | IPFT (per crore) | Verification Notes |
| :--- | :--- | :--- | :--- |
| **Equity Cash** | ₹275.00 (0.00275%) | ₹0.01 (0.0000001%) | Active Equity cash schedule |
| **Equity Futures** | ₹150.00 (0.00150%) | ₹0.01 (0.0000001%) | Standard derivatives tariff (excluding promotional zero-fee bands) |
| **Equity Options** | ₹3,250.00 (0.03250% on prem) | ₹0.01 (0.0000001% on prem) | SENSEX/BANKEX premium turnover fee |

---

## 3. Regulatory Fees & Centralized Stamp Duties

1. **SEBI Regulatory Fee**:
   - Flat ₹10.00 per crore (0.0000010) on aggregate traded notional / premium value across all domestic exchange segments.
2. **Indian Stamp Duty (Indian Stamp Rules 2019, Central Schedule)**:
   - **Buyer leg exclusively**:
     - Equity Delivery: 0.0150% (₹1,500 per crore)
     - Equity Intraday: 0.0030% (₹300 per crore)
     - Equity Futures: 0.0020% (₹200 per crore)
     - Equity Options: 0.0030% (₹300 per crore on premium)
   - All sell legs: 0.0000%.
3. **Goods and Services Tax (GST)**:
   - 18.00% levied on taxable service charges:
     $$\text{GST} = 0.18 \times (\text{Brokerage} + \text{Exchange Charges} + \text{IPFT} + \text{SEBI Fee})$$
   - STT and Stamp Duty are non-taxable sovereign duties and excluded from the GST base.

---

## 4. GIFT City (NSE IX / IFSC) Tariff & Exemption Framework

- **Entity Eligibility**: Exemption applies strictly to entities verified as `IFSC_REGISTERED_UNIT` operating within the GIFT SEZ.
- **Domestic Taxes**: Fully exempt from domestic STT, domestic Stamp Duty, and domestic GST (treated as zero-rated export of financial services under Section 16 of IGST Act).
- **Exchange Fees**: Standard international schedule (1.0 bps default).
- **Fallback Rule**: Unverified entities or ambiguous jurisdictions return `legal_classification_status = LEGAL_CLASSIFICATION_REQUIRED`.
