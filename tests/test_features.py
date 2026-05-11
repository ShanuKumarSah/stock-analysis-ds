"""
test_features.py
----------------
Unit tests for feature engineering and label generation.
Run with: pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.features import generate_labels


def make_dummy_df():
    n = 100
    np.random.seed(42)
    return pd.DataFrame({
        "Close":       np.random.uniform(100, 400, n),
        "RSI_14":      np.random.uniform(35, 65, n),
        "BBU_20_2.0":  np.full(n, 500.0),
        "BBL_20_2.0":  np.full(n, 80.0),
    })


def test_label_has_three_possible_classes():
    df = make_dummy_df()
    df.loc[0, "RSI_14"] = 80; df.loc[0, "Close"] = 600; df.loc[0, "BBU_20_2.0"] = 500
    df.loc[1, "RSI_14"] = 20; df.loc[1, "Close"] = 60;  df.loc[1, "BBL_20_2.0"] = 90
    labeled = generate_labels(df)
    assert set(labeled["label"].unique()).issubset({0, 1, 2})


def test_overbought_label():
    df = make_dummy_df()
    df.loc[0, "RSI_14"] = 80
    df.loc[0, "Close"]  = 600
    df.loc[0, "BBU_20_2.0"] = 500
    labeled = generate_labels(df)
    assert labeled.loc[0, "label"] == 2


def test_oversold_label():
    df = make_dummy_df()
    df.loc[0, "RSI_14"] = 20
    df.loc[0, "Close"]  = 60
    df.loc[0, "BBL_20_2.0"] = 90
    labeled = generate_labels(df)
    assert labeled.loc[0, "label"] == 0


def test_neutral_is_default():
    df = make_dummy_df()
    labeled = generate_labels(df)
    assert (labeled["label"] == 1).all()
