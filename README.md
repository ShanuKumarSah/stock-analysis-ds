# Stock Overbought / Oversold Detector — NSE/BSE

A data science project that detects overbought and oversold conditions in Indian equity markets using technical indicators and machine learning.

---

## Problem

Retail investors in Indian markets often make poor timing decisions — buying after rallies (overbought) or panic-selling during corrections (oversold). This system automatically detects these extreme conditions using RSI, Bollinger Bands, and a trained Random Forest classifier.

## Approach

1. Download NSE/BSE OHLCV data via `yfinance`
2. Compute technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic
3. Generate labels: Overbought (2) / Neutral (1) / Oversold (0)
4. Train a Random Forest classifier with SMOTE to handle class imbalance
5. Track experiments with MLflow
6. Visualise signals on an interactive Streamlit dashboard

## Project Structure

```
stock-analysis-ds/
├── data/
│   ├── raw/              ← downloaded Parquet files (gitignored)
│   └── processed/        ← feature-engineered Parquet files (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb      ← exploratory data analysis
│   ├── 02_features.ipynb ← feature engineering walkthrough
│   └── 03_modelling.ipynb← model training + evaluation
├── src/
│   ├── data_ingestion.py ← yfinance download + caching
│   ├── features.py       ← indicator computation + label generation
│   ├── model.py          ← training + MLflow logging
│   └── utils.py          ← shared helpers
├── app/
│   └── dashboard.py      ← Streamlit dashboard
├── tests/
│   └── test_features.py  ← pytest unit tests
├── .github/workflows/
│   └── ci.yml            ← GitHub Actions CI
├── requirements.txt
├── .env.example
└── PROBLEM_STATEMENT.md
```

## Setup

```bash
git clone https://github.com/yourusername/stock-analysis-ds.git
cd stock-analysis-ds
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

```bash
# Download stock data
python src/data_ingestion.py

# Run dashboard
streamlit run app/dashboard.py

# Run tests
pytest tests/ -v

# View MLflow experiments
mlflow ui
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 65% |
| Precision — overbought | ≥ 60% |
| Precision — oversold | ≥ 60% |
| RSI signal agreement | ≥ 75% |

## Tech Stack

Python 3.11 · pandas · scikit-learn · XGBoost · pandas-ta · MLflow · Streamlit · Plotly · yfinance

## Disclaimer

This is a personal portfolio project for educational purposes only. Nothing here constitutes financial advice.
