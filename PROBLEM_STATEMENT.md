# Problem Statement — Stock Market Overbought/Oversold Detection System
### Indian Equity Markets (NSE / BSE)

---

## 1. Project Overview

This project builds a data science system that detects **overbought and oversold conditions** in Indian stocks listed on the NSE and BSE. The system uses technical indicators and machine learning models (built with Python / scikit-learn) to classify whether a stock is currently in an extreme price zone — helping retail investors make more informed entry and exit decisions.

---

## 2. Problem Statement

Retail investors in Indian equity markets often buy stocks after a significant price rally (overbought) or panic-sell during deep corrections (oversold), leading to poor trade timing. Traditional tools require manual chart reading.

**This project answers:**
> *"Given the recent price and volume history of an NSE/BSE listed stock, can we automatically detect whether it is currently overbought or oversold — and with what confidence?"*

---

## 3. Goals & Objectives

| # | Objective | Type |
|---|-----------|------|
| 1 | Detect overbought / oversold zones using technical indicators | Primary |
| 2 | Build an ML classifier to predict the condition from raw OHLCV features | Primary |
| 3 | Visualise signals on an interactive Streamlit dashboard | Secondary |
| 4 | Backtest signal accuracy over historical NSE/BSE data | Secondary |

---

## 4. Success Metrics

These are the measurable targets the project must hit before it is considered complete:

| Metric | Target |
|--------|--------|
| Classification accuracy (overbought / oversold / neutral) | ≥ 65% on held-out test set |
| Precision on overbought class | ≥ 60% |
| Precision on oversold class | ≥ 60% |
| RSI signal agreement with model prediction | ≥ 75% alignment |
| Backtested signal win rate | ≥ 55% |

---

## 5. Scope

### In scope
- NSE/BSE listed large-cap and mid-cap stocks (Nifty 50 + Nifty Midcap 100)
- Historical OHLCV data (minimum 3 years, daily timeframe)
- Technical indicator computation: RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator
- Supervised ML classification using scikit-learn (Random Forest, XGBoost)
- Streamlit dashboard for signal visualisation
- Walk-forward validation to prevent data leakage

### Out of scope
- Real-time live trading or order execution
- Options, futures, or derivatives
- Intraday (tick / minute-level) data
- Deep learning models (LSTM, Transformer) — reserved for v2
- Fundamental analysis (P/E, EPS, financials)

---

## 6. Data Sources

| Source | Library / API | Data Provided |
|--------|--------------|---------------|
| NSE/BSE historical prices | `yfinance` (`.NS` / `.BO` suffix) | OHLCV daily data |
| NSE official data | `nsepy` or `jugaad-trader` | NSE-specific adjusted prices |
| Technical indicators | `pandas-ta` | RSI, MACD, Bollinger, ATR |
| Market calendar | `pandas_market_calendars` | Trading day alignment |

**Sample tickers:** `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `WIPRO.NS`

---

## 7. Target Variable Definition

The ML model predicts one of three classes:

| Label | Condition | Based On |
|-------|-----------|----------|
| `2 — Overbought` | RSI > 70 AND price above upper Bollinger Band | Technical rules |
| `0 — Oversold` | RSI < 30 AND price below lower Bollinger Band | Technical rules |
| `1 — Neutral` | Neither condition met | Default |

Labels are generated programmatically from indicator thresholds, making this a **rule-assisted supervised learning** problem.

---

## 8. Constraints & Assumptions

- **No look-ahead bias:** All features use only past data available at time `t`. Future data is never used to compute features at `t`.
- **Walk-forward validation only:** The train/test split rolls forward in time — no random shuffle splits.
- **Adjusted prices used:** Corporate actions (splits, dividends) are accounted for via adjusted close prices from `yfinance`.
- **Daily timeframe only:** The model targets end-of-day signals for next-day decision making.
- **No real money involved:** This is a research and portfolio project — not financial advice.

---

## 9. Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11 |
| Data collection | `yfinance`, `nsepy` |
| Data processing | `pandas`, `numpy` |
| Technical indicators | `pandas-ta` |
| Machine learning | `scikit-learn`, `xgboost` |
| Experiment tracking | `MLflow` |
| Visualisation | `plotly`, `matplotlib` |
| Dashboard | `Streamlit` |
| Version control | Git + GitHub |
| Environment | `venv` + `requirements.txt` |

---

## 10. Project Timeline

| Phase | Description | Duration |
|-------|-------------|----------|
| Phase 1 | Problem definition (this document) | Week 1 |
| Phase 2 | Project setup — repo, folder structure, environment | Week 1–2 |
| Phase 3 | Data collection + EDA notebook | Week 2–3 |
| Phase 4 | Feature engineering + ML modelling | Week 3–6 |
| Phase 5 | Streamlit dashboard + FastAPI endpoint | Week 6–9 |
| Phase 6 | Deployment + README + documentation | Week 9–12 |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Data gaps for smaller NSE stocks | Medium | Focus on Nifty 50 + Midcap 100 only |
| Class imbalance (more neutral days) | High | Use SMOTE or class-weight balancing |
| Overfitting on technical indicators | Medium | Walk-forward CV + feature importance pruning |
| `yfinance` API downtime | Low | Cache raw data locally as Parquet files |

---

## 12. Deliverables

- [ ] `PROBLEM_STATEMENT.md` — this document ✅
- [ ] `notebooks/01_eda.ipynb` — exploratory data analysis
- [ ] `notebooks/02_features.ipynb` — feature engineering
- [ ] `notebooks/03_modelling.ipynb` — model training + MLflow logs
- [ ] `app/dashboard.py` — Streamlit dashboard
- [ ] `README.md` — project overview with architecture diagram
- [ ] Deployed app URL (Streamlit Cloud or Render)

---

*Document version: 1.0 | Author: [Your Name] | Date: May 2026*
*This is a personal data science portfolio project. Nothing in this project constitutes financial advice.*
