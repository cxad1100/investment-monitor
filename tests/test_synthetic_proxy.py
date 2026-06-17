import numpy as np
import pandas as pd
import pytest

from tools.synthetic_proxy import to_eur, median_turnover_eur


def test_to_eur_divides_by_fx():
    idx = pd.bdate_range("2020-01-01", periods=5)
    usd = pd.Series(108.0, index=idx)
    fx = pd.Series(1.08, index=idx)                      # USD per EUR
    assert np.allclose(to_eur(usd, fx).values, 100.0)    # 108 USD / 1.08 = 100 EUR


def test_to_eur_gap_fills_fx():
    idx = pd.bdate_range("2020-01-01", periods=4)
    usd = pd.Series(120.0, index=idx)
    fx = pd.Series([1.20, np.nan, 1.50, np.nan], index=idx)
    eur = to_eur(usd, fx)
    assert eur.iloc[1] == pytest.approx(120 / 1.20)      # ffilled rate
    assert eur.iloc[2] == pytest.approx(120 / 1.50)
    assert eur.iloc[3] == pytest.approx(120 / 1.50)      # ffilled forward


def test_median_turnover_eur():
    idx = pd.bdate_range("2020-01-01", periods=3)
    eur = pd.Series([10.0, 10.0, 10.0], index=idx)
    vol = pd.Series([1000, 2000, 3000], index=idx)
    assert median_turnover_eur(vol, eur, tail=3) == pytest.approx(20000.0)  # median(10k,20k,30k)
