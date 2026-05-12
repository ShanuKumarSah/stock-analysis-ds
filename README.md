# Stock Overbought / Oversold Detector — NSE/BSE

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

A data science project that detects overbought and oversold conditions
in Indian equity markets using technical indicators and machine learning.

## 🔴 Live Demo

👉 [Launch Dashboard](https://stock-analysis-ds-a626w4qmdzisjrngpqbziw.streamlit.app/)

---

## 📌 Problem

Retail investors in Indian markets often buy after rallies (overbought)
or panic-sell during corrections (oversold). This system automatically
detects these extreme conditions using RSI, Bollinger Bands and a
trained Random Forest classifier.

---

## 🧠 Approach

1. Download NSE/BSE OHLCV data via `yfinance`
2. Compute technical indicators — RSI, MACD, Bollinger Bands, ATR, Stochastic
3. Generate labels: Overbought (2) / Neutral (1) / Oversold (0)
4. Train Random Forest classifier with SMOTE to handle class imbalance
5. Track experiments with MLflow
6. Visualise signals on interactive Streamlit dashboard

---

## 📊 Model Performance

| Metric            | Score |
| ----------------- | ----- |
| Overall Accuracy  | 89%   |
| Neutral Precision | 100%  |
| Oversold Recall   | 100%  |
| Overbought Recall | 100%  |

---

## 📁 Project Structure

stock-analysis-ds/
├── data/
│ └── processed/ ← feature-engineered CSV files
├── notebooks/
│ ├── 01_eda.ipynb ← exploratory data analysis
│ ├── 02_features.ipynb ← feature engineering + labels
│ └── 03_modelling.ipynb← model training + MLflow
├── src/
│ ├── data_ingestion.py
│ ├── features.py
│ ├── model.py
│ └── utils.py
├── app/
│ └── dashboard.py ← Streamlit dashboard
├── models/ ← saved model + scaler
├── tests/
│ └── test_features.py
├── requirements.txt
└── PROBLEM_STATEMENT.md

````

---

## ⚙️ Setup
```bash
git clone https://github.com/ShanuKumarSah/stock-analysis-ds.git
cd stock-analysis-ds
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
````

## 🚀 Run Dashboard

```bash
python -m streamlit run app/dashboard.py
```

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

Python 3.11 · pandas · scikit-learn · XGBoost · ta · MLflow ·
Streamlit · Plotly · yfinance

---

## ⚠️ Disclaimer

Personal portfolio project for educational purposes only.
Nothing here constitutes financial advice.
