import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import ta
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock OB/OS Detector", layout="wide")
st.title("Stock Overbought / Oversold Detector — NSE/BSE")
st.caption("A data science portfolio project. Not financial advice.")

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Select stock", TICKERS)
    period = st.selectbox("Data period", ["1y", "2y", "3y"], index=1)
    rsi_ob = st.slider("Overbought RSI threshold", 60, 80, 70)
    rsi_os = st.slider("Oversold RSI threshold",   20, 40, 30)

@st.cache_data(ttl=3600)
def get_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    df["RSI_14"]    = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    bb              = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_lower"]  = bb.bollinger_lband()
    df.dropna(inplace=True)

    conditions = [
        (df["RSI_14"] > 70) & (df["Close"] > df["BB_upper"]),
        (df["RSI_14"] < 30) & (df["Close"] < df["BB_lower"]),
    ]
    df["label"] = np.select(conditions, [2, 0], default=1)
    return df

with st.spinner(f"Loading {ticker} ..."):
    df = get_data(ticker, period)

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current RSI",  f"{latest['RSI_14']:.1f}")
col2.metric("Close Price",  f"Rs.{latest['Close']:.2f}")
col3.metric("Upper BB",     f"Rs.{latest['BB_upper']:.2f}")
signal_map = {0: "🟢 Oversold", 1: "⚪ Neutral", 2: "🔴 Overbought"}
col4.metric("Signal", signal_map[int(latest["label"])])

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="OHLCV"
))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"],
    line=dict(color="rgba(100,100,255,0.4)", dash="dash"), name="Upper BB"))
fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"],
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

st.subheader("RSI — 14 Day")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"],
    name="RSI", line=dict(color="#7F77DD")))
rsi_fig.add_hline(y=rsi_ob, line_dash="dash", line_color="red",   annotation_text="Overbought")
rsi_fig.add_hline(y=rsi_os, line_dash="dash", line_color="green", annotation_text="Oversold")
rsi_fig.update_layout(height=250, yaxis=dict(range=[0, 100]))
st.plotly_chart(rsi_fig, use_container_width=True)

with st.expander("View raw data"):
    st.dataframe(df.tail(50))