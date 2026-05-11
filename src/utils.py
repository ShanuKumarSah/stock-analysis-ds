"""
utils.py
--------
Shared helper functions used across the project.
"""

import pandas as pd
from pathlib import Path


def load_processed(ticker: str) -> pd.DataFrame:
    path = Path("data/processed") / f"{ticker.replace('.', '_')}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found for {ticker}.")
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, ticker: str) -> None:
    path = Path("data/processed") / f"{ticker.replace('.', '_')}_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Saved processed data: {path}")


def label_name(label: int) -> str:
    return {0: "Oversold", 1: "Neutral", 2: "Overbought"}.get(label, "Unknown")
