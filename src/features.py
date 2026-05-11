"""
features.py
-----------
Computes technical indicators and generates overbought/oversold labels.
All computations use only past data — no look-ahead bias.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, ATR, and Stochastic to OHLCV dataframe."""
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.dropna(inplace=True)
    return df


def generate_labels(df: pd.DataFrame,
                    rsi_col: str = "RSI_14",
                    upper_bb: str = "BBU_20_2.0",
                    lower_bb: str = "BBL_20_2.0",
                    overbought_rsi: float = 70,
                    oversold_rsi: float = 30) -> pd.DataFrame:
    """
    Create target labels based on RSI + Bollinger Band conditions.
      2 = Overbought  (RSI > 70 AND Close > Upper BB)
      0 = Oversold    (RSI < 30 AND Close < Lower BB)
      1 = Neutral     (everything else)
    """
    df = df.copy()
    conditions = [
        (df[rsi_col] > overbought_rsi) & (df["Close"] > df[upper_bb]),
        (df[rsi_col] < oversold_rsi)   & (df["Close"] < df[lower_bb]),
    ]
    df["label"] = np.select(conditions, [2, 0], default=1)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model input features from the enriched dataframe."""
    feature_cols = [
        "RSI_14",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0",
        "ATRr_14",
        "STOCHk_14_3_3", "STOCHd_14_3_3",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    return df[existing + ["label"]].dropna()
