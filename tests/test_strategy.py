import re

import numpy as np
import pandas as pd

import build_strategy_report as bs
from tools.momentum import run_momentum
from tools.momentum_grid import _stats_slice


def _fake_d():
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500)))
                       for i in range(15)}, index=idx)
    slip = {t: 10 for t in px.columns}
    res = run_momentum(px, slip, k=5, lookback=200, skip=10, cost_mults=(1.0,))
    eq, tr = res["runs"][1.0]["equity"], res["runs"][1.0]["trades"]
    te = pd.Timestamp("2019-06-30")
    return dict(prices=px, res=res, benchmarks=pd.DataFrame(index=idx), capital=10_000.0,
                meta={t: dict(name=t, local_id="000", country="X", sector="Y") for t in px.columns},
                strategy=bs.STRATEGY,
                train=_stats_slice(eq, tr, eq.index[0], te, 10_000.0),
                val=_stats_slice(eq, tr, te + pd.Timedelta(days=1), eq.index[-1], 10_000.0),
                n_dead=42)


def test_strategy_page_builds():
    html = bs.build(_fake_d(), public=False)
    assert "<html" in html.lower() and "strategy" in html.lower()
    assert "Validation" in html and bs.STRATEGY.code in html


def test_strategy_public_no_euro_amounts():
    html = bs.build(_fake_d(), public=True)
    euros = re.findall(r"€[0-9][0-9.,]*", html)
    assert all(e == "€1" for e in euros), euros
