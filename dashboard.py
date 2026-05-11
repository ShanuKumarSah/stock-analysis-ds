"""
dashboard.py
------------
Streamlit dashboard for the overbought/oversold detection system.
Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_ingestion import download_stock
from src.features import compute_indicators, generate_labels

st.set_page_config(page_title="Stock OB/OS Detector", layout="wide")
st.title("Stock Overbought / Oversold Detector — NSE/BSE")
st.caption("A data science portfolio project. Not financial advice.")

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]

with st.sidebar:
    st.header("Settings")
    ticker  = st.selectbox("Select stock", TICKERS)
    period  = st.selectbox("Data period", ["1y", "2y", "3y", "5y"], index=1)
    rsi_ob  = st.slider("Overbought RSI threshold", 60, 80, 70)
    rsi_os  = st.slider("Oversold RSI threshold",   20, 40, 30)

@st.cache_data(ttl=3600)
def get_data(ticker, period):
    df = download_stock(ticker, period=period)
    df = compute_indicators(df)
    df = generate_labels(df)
    return df

with st.spinner(f"Loading {ticker} ..."):
    df = get_data(ticker, period)

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current RSI",   f"{latest['RSI_14']:.1f}")
col2.metric("Close price",   f"Rs.{latest['Close']:.2f}")
col3.metric("ATR (14)",      f"{latest['ATRr_14']:.2f}")
signal_map = {0: "Oversold", 1: "Neutral", 2: "Overbought"}
col4.metric("Signal", signal_map[int(latest["label"])])

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="OHLCV"
))
fig.add_trace(go.Scatter(x=df.index, y=df["BBU_20_2.0"],
    line=dict(color="rgba(100,100,255,0.4)", dash="dash"), name="Upper BB"))
fig.add_trace(go.Scatter(x=df.index, y=df["BBL_20_2.0"],
    line=dict(color="rgba(100,100,255,0.4)", dash="dash"), name="Lower BB"))

ob  = df[df["label"] == 2]
os_ = df[df["label"] == 0]
fig.add_trace(go.Scatter(x=ob.index,  y=ob["High"]  * 1.01,
    mode="markers", marker=dict(color="red",   symbol="triangle-down", size=10), name="Overbought"))
fig.add_trace(go.Scatter(x=os_.index, y=os_["Low"] * 0.99,
    mode="markers", marker=dict(color="green", symbol="triangle-up",   size=10), name="Oversold"))

fig.update_layout(title=f"{ticker} — Price with OB/OS Signals",
                  xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

st.subheader("RSI — 14 day")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI", line=dict(color="#7F77DD")))
rsi_fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red",   annotation_text="Overbought")
rsi_fig.add_hline(y=rsi_os, line_dash="dash", line_color="green", annotation_text="Oversold")
rsi_fig.update_layout(height=250, yaxis=dict(range=[0, 100]))
st.plotly_chart(rsi_fig, use_container_width=True)

with st.expander("View raw data"):
    st.dataframe(df.tail(50))
