"""Pairs-engine math: cointegration testing, pair selection, z-score signals.

Pure functions — no I/O, no network. The walk-forward contract:
every parameter (alpha, beta, mu, sigma) is estimated on a formation
window and FROZEN before being applied to a later trading window.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def engle_granger(a: pd.Series, b: pd.Series) -> dict:
    """Engle-Granger two-step on log prices, both orientations; keep the
    lower p-value.

    Step 1: OLS log(y) = alpha + beta*log(x); residual = spread.
    Step 2: statsmodels `coint` — ADF on the residuals with the CORRECT
    Engle-Granger critical values (plain `adfuller` on estimated residuals
    uses the wrong distribution).
    """
    best = None
    for y, x in ((a, b), (b, a)):
        ly, lx = np.log(y), np.log(x)
        ols = sm.OLS(ly, sm.add_constant(lx)).fit()
        alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
        pval = float(coint(ly, lx)[1])
        spread = ly - (alpha + beta * lx)
        if best is None or pval < best["pvalue"]:
            best = dict(y=str(y.name), x=str(x.name), alpha=alpha, beta=beta,
                        pvalue=pval, spread=spread)
    return best


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life in trading days from an AR(1) fit:
    Δs_t = c + ρ·s_{t-1} + ε  →  HL = −ln2/ρ. Non-reverting (ρ ≥ 0) → inf."""
    ds = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    rho = float(sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1])
    return float("inf") if rho >= 0 else float(-np.log(2.0) / rho)
