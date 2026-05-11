"""
data_ingestion.py
-----------------
Downloads and caches OHLCV data for NSE/BSE stocks using yfinance.
All raw data is saved as Parquet files under data/raw/.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "WIPRO.NS", "KOTAKBANK.NS",
]


def download_stock(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data for a single NSE/BSE ticker."""
    print(f"Downloading {ticker} ...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    out_path = RAW_DIR / f"{ticker.replace('.', '_')}.parquet"
    df.to_parquet(out_path)
    print(f"  Saved to {out_path} — {len(df)} rows")
    return df


def download_all(tickers: list = NIFTY50_TICKERS, period: str = "5y") -> None:
    """Download OHLCV data for all tickers in the list."""
    for ticker in tickers:
        try:
            download_stock(ticker, period=period)
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")


def load_stock(ticker: str) -> pd.DataFrame:
    """Load a previously downloaded stock from Parquet cache."""
    path = RAW_DIR / f"{ticker.replace('.', '_')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {ticker}. Run download_stock() first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    download_all()
