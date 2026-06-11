# Pairs Trading Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Statistical-arbitrage pairs-trading engine: Engle-Granger cointegration scan over a curated LS-Exchange universe, walk-forward z-score backtest with Trade Republic-style costs, rendered as a static HTML page (`local/pairs.html` + `docs/pairs.html`).

**Architecture:** Three new pure-logic modules under `tools/` (universe, engine math, backtester) + one orchestrator script `build_pairs_report.py` that mirrors `build_report.py`'s section-function style. Shared HTML helpers get extracted to `tools/report_html.py` (DRY with `build_report.py`). All engine functions are pure (no I/O); only `fetch_prices` and the report script touch the network/disk.

**Tech Stack:** Python 3.13, pandas/numpy, statsmodels (new dep — `coint` + OLS), plotly + existing `tools/theme.py`. Custom vectorized backtester (no Backtrader/VectorBT). Price cache is CSV, not parquet (avoids the pyarrow dependency; daily closes lose nothing in CSV — conscious deviation from spec).

**Spec:** `docs/superpowers/specs/2026-06-11-pairs-trading-engine-design.md`

**Conventions for the executor:**
- Always run Python/pytest via `.venv/bin/python` / `.venv/bin/pytest` (never system Python; IDE import errors are false positives).
- Walk-forward contract (the whole point of this project): every parameter applied in a trading window (α, β, μ, σ) is estimated on the formation window before it and **frozen**. A signal computed on close *t* executes at close *t+1*.
- Two seeded-randomness tests (`test_rejects_independent_walks`, `test_run_backtest_structure`) are deterministic for a fixed seed but the seed was chosen blind. If one fails, bump the seed (independent random walks give a uniform p-value — we just need one above 0.05) and note it in the commit.

---

### Task 1: Dependency + universe module

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `tools/pairs_universe.py`
- Test: `tests/test_pairs_engine.py` (new file, first tests)

- [ ] **Step 1: Add statsmodels to requirements and install**

Append to `requirements.txt`:

```
statsmodels>=0.14
```

Run: `uv pip install -r requirements.txt --python .venv`
Expected: statsmodels installed without error.

- [ ] **Step 2: Add price cache to .gitignore**

Append to `.gitignore`:

```
data/pairs_prices.csv
```

- [ ] **Step 3: Write failing tests for the universe**

Create `tests/test_pairs_engine.py`:

```python
"""Unit tests for the pairs-trading universe and engine math."""

import numpy as np
import pandas as pd
import pytest

from tools.pairs_universe import UNIVERSE, candidate_pairs


def test_universe_entries_complete():
    for tk, meta in UNIVERSE.items():
        assert set(meta) == {"sector", "currency", "slippage_bps"}, tk
        assert meta["currency"] in ("USD", "EUR")
        assert meta["slippage_bps"] in (5, 10, 15)


def test_candidate_pairs_same_sector_and_currency():
    pairs = candidate_pairs()
    assert ("BAC", "JPM") in pairs            # same sector, same currency
    for a, b in pairs:
        assert UNIVERSE[a]["sector"] == UNIVERSE[b]["sector"]
        assert UNIVERSE[a]["currency"] == UNIVERSE[b]["currency"]
    # never cross-sector or cross-currency
    assert ("NVDA", "SIE.DE") not in pairs
    assert ("JPM", "DBK.DE") not in pairs


def test_candidate_pairs_count_reasonable():
    n = len(candidate_pairs())
    assert 30 <= n <= 80                      # ~52 with the curated universe
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.pairs_universe'`

- [ ] **Step 5: Implement the universe module**

Create `tools/pairs_universe.py`:

```python
"""Curated stock universe for the pairs-trading engine.

Every name is tradeable on Lang & Schwarz Exchange (Trade Republic).
Candidate pairs are restricted to same sector AND same currency —
cross-currency pairs leak the FX trend into the spread and produce
spurious cointegration.

slippage_bps = assumed half-spread cost per traded leg:
5 bps US mega-caps, 10 bps DAX names, 15 bps mid-liquidity names.
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent

UNIVERSE: dict[str, dict] = {
    # US banks
    "JPM": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "BAC": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "C":   dict(sector="US Banks", currency="USD", slippage_bps=5),
    "WFC": dict(sector="US Banks", currency="USD", slippage_bps=5),
    "GS":  dict(sector="US Banks", currency="USD", slippage_bps=5),
    "MS":  dict(sector="US Banks", currency="USD", slippage_bps=5),
    # US semiconductors
    "NVDA": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "AMD":  dict(sector="US Semis", currency="USD", slippage_bps=5),
    "AVGO": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "TXN":  dict(sector="US Semis", currency="USD", slippage_bps=5),
    "INTC": dict(sector="US Semis", currency="USD", slippage_bps=5),
    "MU":   dict(sector="US Semis", currency="USD", slippage_bps=5),
    # US consumer staples
    "KO":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    "PEP": dict(sector="US Staples", currency="USD", slippage_bps=5),
    "PG":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    "CL":  dict(sector="US Staples", currency="USD", slippage_bps=5),
    # US payments
    "V":   dict(sector="US Payments", currency="USD", slippage_bps=5),
    "MA":  dict(sector="US Payments", currency="USD", slippage_bps=5),
    "AXP": dict(sector="US Payments", currency="USD", slippage_bps=5),
    # US energy
    "XOM": dict(sector="US Energy", currency="USD", slippage_bps=5),
    "CVX": dict(sector="US Energy", currency="USD", slippage_bps=5),
    "COP": dict(sector="US Energy", currency="USD", slippage_bps=5),
    # German autos
    "BMW.DE":  dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "MBG.DE":  dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "VOW3.DE": dict(sector="DE Autos", currency="EUR", slippage_bps=10),
    "P911.DE": dict(sector="DE Autos", currency="EUR", slippage_bps=15),
    # German industrials / chemicals
    "SIE.DE":  dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    "BAS.DE":  dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    "BAYN.DE": dict(sector="DE Industrial", currency="EUR", slippage_bps=10),
    # German banks
    "DBK.DE": dict(sector="DE Banks", currency="EUR", slippage_bps=10),
    "CBK.DE": dict(sector="DE Banks", currency="EUR", slippage_bps=10),
}


def candidate_pairs() -> list[tuple[str, str]]:
    """All ticker pairs sharing sector AND currency."""
    out = []
    for a, b in combinations(sorted(UNIVERSE), 2):
        ua, ub = UNIVERSE[a], UNIVERSE[b]
        if ua["sector"] == ub["sector"] and ua["currency"] == ub["currency"]:
            out.append((a, b))
    return out


def fetch_prices(tickers: list[str] | None = None, years: int = 5,
                 cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes for the universe, cached as CSV in data/.

    The cache is reused when it covers all requested tickers and its last
    row is at most 3 days old. Interior gaps are forward-filled; leading
    NaN (late IPOs like P911.DE) are kept — per-pair alignment happens in
    select_pairs / run_backtest.
    """
    tickers = list(tickers or UNIVERSE)
    cache = cache or ROOT / "data" / "pairs_prices.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        age = (pd.Timestamp.today().normalize() - df.index[-1]).days
        if set(tickers) <= set(df.columns) and age <= 3:
            return df[tickers]
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.dropna(how="all").ffill()
    cache.parent.mkdir(exist_ok=True)
    close.to_csv(cache)
    return close[tickers]
```

(`fetch_prices` is network I/O — no unit test; it gets verified end-to-end in Task 9.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: 3 PASSED

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore tools/pairs_universe.py tests/test_pairs_engine.py
git commit -m "Add pairs-trading universe module + statsmodels dep

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Engle-Granger cointegration test

**Files:**
- Create: `tools/pairs_engine.py`
- Test: `tests/test_pairs_engine.py` (append)

- [ ] **Step 1: Write failing tests with synthetic data helpers**

Append to `tests/test_pairs_engine.py`:

```python
from tools.pairs_engine import engle_granger


def make_cointegrated(n=750, seed=42, beta=0.8, alpha=0.5, phi=0.7, noise=0.02):
    """Price pair sharing a random-walk driver:
    log x = random walk; log y = alpha + beta*log x + AR(1) noise."""
    rng = np.random.default_rng(seed)
    lx = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = phi * eps[i - 1] + rng.normal(0, noise)
    ly = alpha + beta * lx + eps
    idx = pd.bdate_range("2022-01-03", periods=n)
    return (pd.Series(np.exp(ly), idx, name="YYY"),
            pd.Series(np.exp(lx), idx, name="XXX"))


def make_independent(n=750, seed=7):
    """Two unrelated random walks — correlated by luck at most, never cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    a = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="AAA")
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)), idx, name="BBB")
    return a, b


def test_detects_cointegrated_pair():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    assert r["pvalue"] < 0.05
    assert r["beta"] > 0
    assert {r["y"], r["x"]} == {"YYY", "XXX"}
    assert set(r) >= {"y", "x", "alpha", "beta", "pvalue", "spread"}


def test_rejects_independent_walks():
    a, b = make_independent()
    r = engle_granger(a, b)
    assert r["pvalue"] >= 0.05


def test_spread_is_ols_residual():
    y, x = make_cointegrated()
    r = engle_granger(y, x)
    # residual of OLS with constant has (near-)zero mean by construction
    assert abs(r["spread"].mean()) < 1e-10
    assert len(r["spread"]) == len(y)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.pairs_engine'`

- [ ] **Step 3: Implement engle_granger**

Create `tools/pairs_engine.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_engine.py tests/test_pairs_engine.py
git commit -m "Add Engle-Granger two-step cointegration test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Spread half-life

**Files:**
- Modify: `tools/pairs_engine.py`
- Test: `tests/test_pairs_engine.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairs_engine.py`:

```python
from tools.pairs_engine import half_life


def test_half_life_known_ar1():
    # AR(1) with phi=0.9: Δs = (phi-1)·s + ε, so HL ≈ ln2/0.1 ≈ 6.9 days
    rng = np.random.default_rng(3)
    s = np.zeros(5000)
    for i in range(1, 5000):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(s, pd.bdate_range("2010-01-04", periods=5000)))
    assert 5.5 < hl < 8.5


def test_half_life_random_walk_is_huge():
    rng = np.random.default_rng(4)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)),
                   pd.bdate_range("2012-01-02", periods=3000))
    assert half_life(rw) > 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v -k half_life`
Expected: FAIL with `ImportError: cannot import name 'half_life'`

- [ ] **Step 3: Implement half_life**

Append to `tools/pairs_engine.py`:

```python
def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life in trading days from an AR(1) fit:
    Δs_t = c + ρ·s_{t-1} + ε  →  HL = −ln2/ρ. Non-reverting (ρ ≥ 0) → inf."""
    ds = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    rho = float(sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1])
    return float("inf") if rho >= 0 else float(-np.log(2.0) / rho)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_engine.py tests/test_pairs_engine.py
git commit -m "Add spread half-life estimator (AR(1) fit)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Walk-forward windows + pair selection

**Files:**
- Modify: `tools/pairs_engine.py`
- Test: `tests/test_pairs_engine.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairs_engine.py`:

```python
from tools.pairs_engine import select_pairs, walkforward_windows


def test_walkforward_windows_roll_without_overlap():
    idx = pd.bdate_range("2022-01-03", periods=400)
    wins = walkforward_windows(idx, formation_days=252, trading_days=63)
    assert len(wins) == 3
    for f, t in wins:
        assert len(f) == 252
        assert f[-1] < t[0]                     # formation strictly before trading
    # consecutive trading windows must not overlap
    for (_, t1), (_, t2) in zip(wins, wins[1:]):
        assert t1[-1] < t2[0]


def test_select_pairs_picks_cointegrated_only():
    y, x = make_cointegrated()
    a, b = make_independent()
    prices = pd.concat([y, x, a, b], axis=1)
    res = select_pairs(prices, [("YYY", "XXX"), ("AAA", "BBB")])
    assert res["n_tested"] == 2
    assert len(res["selected"]) == 1
    sel = res["selected"][0]
    assert {sel["y"], sel["x"]} == {"YYY", "XXX"}
    # frozen params present; raw window data must NOT leak out
    assert set(sel) >= {"alpha", "beta", "pvalue", "half_life", "mu", "sigma"}
    assert "spread" not in sel
    assert sel["sigma"] > 0


def test_select_pairs_respects_min_obs():
    y, x = make_cointegrated(n=50)
    prices = pd.concat([y, x], axis=1)
    res = select_pairs(prices, [("YYY", "XXX")], min_obs=100)
    assert res["n_tested"] == 0
    assert res["selected"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v -k "walkforward or select"`
Expected: FAIL with `ImportError: cannot import name 'select_pairs'`

- [ ] **Step 3: Implement**

Append to `tools/pairs_engine.py`:

```python
def walkforward_windows(index: pd.DatetimeIndex, formation_days: int = 252,
                        trading_days: int = 63,
                        ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Formation/trading window pairs, rolled forward by trading_days.
    Trading windows never overlap; a partial last window needs ≥ 5 days."""
    out, start = [], 0
    while start + formation_days + 5 <= len(index):
        f = index[start:start + formation_days]
        t = index[start + formation_days:start + formation_days + trading_days]
        out.append((f, t))
        start += trading_days
    return out


def select_pairs(prices: pd.DataFrame, candidates: list[tuple[str, str]],
                 p_max: float = 0.05, hl_min: float = 2.0, hl_max: float = 60.0,
                 top_n: int = 10, min_obs: int = 100) -> dict:
    """Test every candidate on the formation window; return the top pairs.

    Filters: cointegration p < p_max, positive hedge ratio, half-life in
    [hl_min, hl_max] trading days. Capped at top_n by p-value to bound
    multiple-testing damage; n_tested is reported so the report can show
    the multiple-comparisons caveat honestly.
    """
    found, tested = [], 0
    for a, b in candidates:
        df = prices[[a, b]].dropna()
        if len(df) < min_obs:
            continue
        tested += 1
        r = engle_granger(df[a], df[b])
        if r["pvalue"] >= p_max or r["beta"] <= 0:
            continue
        hl = half_life(r["spread"])
        if not (hl_min <= hl <= hl_max):
            continue
        r["half_life"] = hl
        r["mu"] = float(r["spread"].mean())
        r["sigma"] = float(r["spread"].std())
        del r["spread"]            # frozen params only; no window data leaks out
        found.append(r)
    found.sort(key=lambda r: r["pvalue"])
    return {"selected": found[:top_n], "n_tested": tested}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_engine.py tests/test_pairs_engine.py
git commit -m "Add walk-forward windows + cointegrated pair selection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Z-score + signal state machine

**Files:**
- Modify: `tools/pairs_engine.py`
- Test: `tests/test_pairs_engine.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairs_engine.py`:

```python
from tools.pairs_engine import generate_signals, pair_zscore


def _z(vals):
    return pd.Series(vals, pd.bdate_range("2024-01-01", periods=len(vals)), dtype=float)


def test_pair_zscore_uses_frozen_params():
    idx = pd.bdate_range("2024-01-01", periods=3)
    py = pd.Series([np.e] * 3, idx)            # log = 1
    px = pd.Series([1.0] * 3, idx)             # log = 0
    pair = dict(alpha=0.0, beta=1.0, mu=0.0, sigma=0.5)
    z = pair_zscore(py, px, pair)
    assert np.allclose(z.values, 2.0)          # (1 - 0)/0.5


def test_signals_long_entry_and_exit():
    sig = generate_signals(_z([0, -2.5, -1, 0.1, 0]))
    assert sig.tolist() == [0, 1, 1, 0, 0]


def test_signals_short_entry_and_exit():
    sig = generate_signals(_z([0, 2.5, 1, -0.1]))
    assert sig.tolist() == [0, -1, -1, 0]


def test_signals_stop_loss_bans_reentry():
    sig = generate_signals(_z([0, 2.5, 3.6, 2.5, 2.5]))
    assert sig.tolist() == [0, -1, 0, 0, 0]    # stopped at 3.6, no re-entry at 2.5


def test_signals_no_lookahead():
    rng = np.random.default_rng(11)
    z = _z(np.cumsum(rng.normal(0, 0.6, 200)))
    full = generate_signals(z)
    for k in (10, 50, 150):
        assert generate_signals(z.iloc[:k]).tolist() == full.iloc[:k].tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v -k "zscore or signals"`
Expected: FAIL with `ImportError: cannot import name 'generate_signals'`

- [ ] **Step 3: Implement**

Append to `tools/pairs_engine.py`:

```python
def pair_zscore(py: pd.Series, px: pd.Series, pair: dict) -> pd.Series:
    """z-score of the spread using FROZEN formation params (alpha, beta, mu, sigma)."""
    spread = np.log(py) - (pair["alpha"] + pair["beta"] * np.log(px))
    return (spread - pair["mu"]) / pair["sigma"]


def generate_signals(z: pd.Series, entry: float = 2.0, exit_band: float = 0.0,
                     stop: float = 3.5) -> pd.Series:
    """Desired spread position per close: +1 long spread, -1 short, 0 flat.

    Sequential state machine — the position at t depends only on z up to t
    (no look-ahead). Long spread = spread cheap (z ≤ −entry): long Y, short X.
    Exit when z reverts through the exit band; stop when |z| ≥ stop —
    cointegration treated as broken, NO re-entry for the rest of the window.
    """
    pos, banned, out = 0, False, []
    for zt in z.to_numpy():
        if np.isnan(zt):
            out.append(pos)
            continue
        if pos == 0 and not banned:
            if zt <= -entry:
                pos = 1
            elif zt >= entry:
                pos = -1
        elif pos == 1:
            if zt <= -stop:
                pos, banned = 0, True
            elif zt >= -exit_band:
                pos = 0
        elif pos == -1:
            if zt >= stop:
                pos, banned = 0, True
            elif zt <= exit_band:
                pos = 0
        out.append(pos)
    return pd.Series(out, index=z.index, dtype=float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pairs_engine.py -v`
Expected: 16 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_engine.py tests/test_pairs_engine.py
git commit -m "Add frozen-param z-score + signal state machine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: simulate_pair — exact P&L with costs

**Files:**
- Create: `tools/pairs_backtest.py`
- Test: `tests/test_pairs_backtest.py` (new file)

- [ ] **Step 1: Write failing tests with hand-computed P&L**

Create `tests/test_pairs_backtest.py`:

```python
"""Backtester tests: exact P&L arithmetic and walk-forward orchestration."""

import numpy as np
import pandas as pd

from tools.pairs_backtest import backtest_stats, run_backtest, simulate_pair
from tests.test_pairs_engine import make_cointegrated, make_independent


def _frame():
    idx = pd.bdate_range("2024-01-01", periods=5)
    py = pd.Series([100.0, 100.0, 100.0, 110.0, 110.0], idx)
    px = pd.Series([100.0] * 5, idx)
    sig = pd.Series([0, 1, 1, 0, 0], idx, dtype=float)
    return py, px, sig


def test_simulate_pair_exact_pnl():
    """Hand-computed: beta=1, capital 2000 → N_y = N_x = 1000.
    signal=[0,1,1,0,0] → held=[0,0,1,1,0] (t+1 execution).
    Gross: day3 = 1000*10% - 1000*0 = 100; day4 held but flat prices = 0.
    Costs per turn = (1€ + 10bps*1000) * 2 legs = 4€; turns on day2 and day4 → 8€.
    Net = 92."""
    py, px, sig = _frame()
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10, fee_eur=1.0, cost_mult=1.0)
    assert abs(res["gross"].sum() - 100.0) < 1e-9
    assert abs(res["costs"].sum() - 8.0) < 1e-9
    assert abs(res["pnl"].sum() - 92.0) < 1e-9
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["entry"] == py.index[2] and t["exit"] == py.index[4]
    assert t["side"] == 1 and t["days"] == 2
    assert abs(t["net"] - 92.0) < 1e-9


def test_simulate_pair_zero_cost_mult():
    py, px, sig = _frame()
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10, fee_eur=1.0, cost_mult=0.0)
    assert abs(res["pnl"].sum() - 100.0) < 1e-9


def test_simulate_pair_force_close_at_window_end():
    py, px, _ = _frame()
    sig = pd.Series([0, 1, 1, 1, 1], py.index, dtype=float)   # never exits by itself
    res = simulate_pair(py, px, sig, beta=1.0, pair_capital=2000.0,
                        slip_y_bps=10, slip_x_bps=10)
    assert len(res["trades"]) == 1                            # closed by force
    assert res["trades"][0]["exit"] == py.index[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.pairs_backtest'`

- [ ] **Step 3: Implement simulate_pair (stub the rest)**

Create `tools/pairs_backtest.py`:

```python
"""Vectorized pairs backtester with Trade Republic-style costs.

Execution model: a signal computed on the close of day t is executed at
the close of day t+1; P&L accrues from t+1 onward. Costs per traded leg:
fixed fee (€1) + slippage as a half-spread in bps of leg notional.
Shorting is simulated — Trade Republic offers no shorting.
"""

import numpy as np
import pandas as pd

from tools.pairs_engine import (
    generate_signals,
    pair_zscore,
    select_pairs,
    walkforward_windows,
)


def simulate_pair(py: pd.Series, px: pd.Series, signal: pd.Series, beta: float,
                  pair_capital: float, slip_y_bps: float, slip_x_bps: float,
                  fee_eur: float = 1.0, cost_mult: float = 1.0,
                  z: pd.Series | None = None) -> dict:
    """Daily net P&L of one pair over one trading window, plus a trade ledger.

    Long spread (+1) = long Y, short X, beta-weighted notionals:
    N_y = pair_capital/(1+beta), N_x = beta*N_y.
    """
    n_y = pair_capital / (1.0 + beta)
    n_x = beta * n_y
    held = signal.shift(1).fillna(0.0)          # t+1 execution
    held.iloc[-1] = 0.0                         # force-close at window end
    r_y = py.pct_change().fillna(0.0)
    r_x = px.pct_change().fillna(0.0)
    gross = held.shift(1).fillna(0.0) * (n_y * r_y - n_x * r_x)
    turns = held.diff().abs().fillna(0.0)
    per_turn = (fee_eur + slip_y_bps / 1e4 * n_y) + (fee_eur + slip_x_bps / 1e4 * n_x)
    costs = turns * per_turn * cost_mult
    pnl = gross - costs

    trades, open_t, open_i = [], None, None
    hv = held.to_numpy()
    for i, d in enumerate(held.index):
        if hv[i] != 0 and (i == 0 or hv[i - 1] == 0):
            open_t = dict(entry=d, side=int(hv[i]), gross=0.0, costs=0.0,
                          z_entry=None if z is None else float(z.iloc[i - 1]))
            open_i = i
        if open_t is not None:
            open_t["gross"] += float(gross.iloc[i])
            open_t["costs"] += float(costs.iloc[i])
        if open_t is not None and hv[i] == 0 and i > 0 and hv[i - 1] != 0:
            open_t.update(exit=d, days=i - open_i,
                          net=open_t["gross"] - open_t["costs"])
            trades.append(open_t)
            open_t = None
    return {"pnl": pnl, "gross": gross, "costs": costs, "trades": trades}
```

- [ ] **Step 4: Add stubs so the test module imports, then run**

The test file imports `backtest_stats` and `run_backtest` at the top (implemented in Task 7). Append stubs to `tools/pairs_backtest.py` so the import succeeds:

```python
def backtest_stats(equity, trades, capital):
    raise NotImplementedError


def run_backtest(*args, **kwargs):
    raise NotImplementedError
```

Run: `.venv/bin/pytest tests/test_pairs_backtest.py -v -k simulate_pair`
Expected: 3 PASSED (the two `run_backtest` tests arrive in Task 7)

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_backtest.py tests/test_pairs_backtest.py
git commit -m "Add per-pair trade simulator with fees + slippage

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: run_backtest — walk-forward orchestration + stats

**Files:**
- Modify: `tools/pairs_backtest.py`
- Test: `tests/test_pairs_backtest.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pairs_backtest.py`:

```python
def _synthetic_universe(n=400):
    y, x = make_cointegrated(n=n, seed=42)
    a, b = make_independent(n=n, seed=7)
    prices = pd.concat([y, x, a, b], axis=1)
    candidates = [("YYY", "XXX"), ("AAA", "BBB")]
    slippage = {c: 5 for c in prices.columns}
    return prices, candidates, slippage


def test_run_backtest_structure_and_cost_monotonicity():
    prices, cands, slip = _synthetic_universe()
    bt = run_backtest(prices, cands, slip, capital=10_000.0,
                      formation_days=120, trading_days=40)
    assert set(bt["runs"]) == {0.0, 1.0, 2.0}
    base = bt["runs"][1.0]
    eq = base["equity"]
    assert eq.index.equals(prices.index)
    assert eq.iloc[0] == 10_000.0
    assert len(base["trades"]) > 0                      # seeded — see plan header
    for t in base["trades"]:
        assert {"pair", "capital", "entry", "exit", "net", "days"} <= set(t)
    st = base["stats"]
    assert {"net_return", "sharpe", "max_drawdown", "n_trades",
            "win_rate", "avg_days", "total_costs"} <= set(st)
    # identical trades, only costs differ → frictionless ≥ realistic ≥ pessimistic
    n0 = bt["runs"][0.0]["stats"]["net_return"]
    n2 = bt["runs"][2.0]["stats"]["net_return"]
    assert n0 >= bt["runs"][1.0]["stats"]["net_return"] >= n2
    assert bt["runs"][0.0]["stats"]["total_costs"] == 0.0
    assert len(bt["windows"]) > 0
    for w in bt["windows"]:
        assert {"formation_end", "trade_start", "n_tested", "n_selected"} <= set(w)


def test_run_backtest_no_lookahead():
    """Truncating future data must not change past equity. Both runs roll
    windows from index 0, so all windows fully inside the truncation are
    identical."""
    prices, cands, slip = _synthetic_universe(n=400)
    full = run_backtest(prices, cands, slip, capital=10_000.0,
                        formation_days=120, trading_days=40,
                        cost_mults=(1.0,))["runs"][1.0]["equity"]
    trunc = run_backtest(prices.iloc[:250], cands, slip, capital=10_000.0,
                         formation_days=120, trading_days=40,
                         cost_mults=(1.0,))["runs"][1.0]["equity"]
    # windows inside first 250 rows: trading days 120-160, 160-200, 200-240
    assert np.allclose(full.iloc[:240].values, trunc.iloc[:240].values)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pairs_backtest.py -v -k run_backtest`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement (replace both stubs)**

Replace the two stub functions in `tools/pairs_backtest.py` with:

```python
def backtest_stats(equity: pd.Series, trades: list[dict], capital: float) -> dict:
    rets = equity.pct_change().dropna()
    sd = float(rets.std())
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())
    wins = sum(1 for t in trades if t["net"] > 0)
    return dict(
        net_return=float(equity.iloc[-1] / capital - 1.0),
        sharpe=sharpe,
        max_drawdown=dd,
        n_trades=len(trades),
        win_rate=wins / len(trades) if trades else None,
        avg_days=float(np.mean([t["days"] for t in trades])) if trades else None,
        total_costs=float(sum(t["costs"] for t in trades)),
    )


def run_backtest(prices: pd.DataFrame, candidates: list[tuple[str, str]],
                 slippage_bps: dict, capital: float = 10_000.0,
                 formation_days: int = 252, trading_days: int = 63,
                 p_max: float = 0.05, top_n: int = 10, entry: float = 2.0,
                 stop: float = 3.5, fee_eur: float = 1.0,
                 cost_mults: tuple = (0.0, 1.0, 2.0)) -> dict:
    """Walk-forward backtest. Pair selection and signals are computed once
    (they don't depend on costs); each cost multiple re-prices the same
    trades — that's the cost-sensitivity table.
    """
    windows = walkforward_windows(prices.index, formation_days, trading_days)
    legs, window_log = [], []
    for form_idx, trade_idx in windows:
        sel = select_pairs(prices.loc[form_idx], candidates,
                           p_max=p_max, top_n=top_n)
        pairs = sel["selected"]
        window_log.append(dict(formation_end=form_idx[-1], trade_start=trade_idx[0],
                               n_tested=sel["n_tested"], n_selected=len(pairs)))
        if not pairs:
            continue
        slice_cap = capital / len(pairs)
        for pr in pairs:
            py = prices.loc[trade_idx, pr["y"]]
            px = prices.loc[trade_idx, pr["x"]]
            if py.isna().any() or px.isna().any():
                continue
            z = pair_zscore(py, px, pr)
            sig = generate_signals(z, entry=entry, stop=stop)
            legs.append((trade_idx, pr, py, px, sig, z, slice_cap))

    runs = {}
    for m in cost_mults:
        daily = pd.Series(0.0, index=prices.index)
        trades = []
        for trade_idx, pr, py, px, sig, z, cap in legs:
            res = simulate_pair(py, px, sig, pr["beta"], cap,
                                slippage_bps[pr["y"]], slippage_bps[pr["x"]],
                                fee_eur=fee_eur, cost_mult=m, z=z)
            daily.loc[trade_idx] = daily.loc[trade_idx] + res["pnl"]
            for t in res["trades"]:
                trades.append({**t, "pair": f'{pr["y"]}/{pr["x"]}', "capital": cap})
        equity = capital + daily.cumsum()
        runs[m] = dict(equity=equity, trades=trades,
                       stats=backtest_stats(equity, trades, capital))
    return {"runs": runs, "windows": window_log,
            "start": str(windows[0][1][0].date()) if windows else None}
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all PASSED (existing optimizer tests + 16 engine + 5 backtest)

- [ ] **Step 5: Commit**

```bash
git add tools/pairs_backtest.py tests/test_pairs_backtest.py
git commit -m "Add walk-forward backtest with cost-sensitivity runs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Shared HTML helpers (DRY with build_report.py)

**Files:**
- Create: `tools/report_html.py`
- Modify: `tools/theme.py` (add `a` link color to REPORT_CSS)
- Modify: `build_report.py` (use the shared helpers)

- [ ] **Step 1: Create the shared module**

Create `tools/report_html.py`:

```python
"""Shared HTML building blocks for the static reports."""

import plotly.graph_objects as go

from tools import theme


def fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def pct(x, signed=True, nd=1) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls} mono">{s}</span>'


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div></div>'


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{theme.REPORT_CSS}</style>
</head><body><main>{body}</main></body></html>"""
```

- [ ] **Step 2: Add link styling to theme**

In `tools/theme.py`, inside `REPORT_CSS`, after the `summary {{ ... }}` line add:

```
a {{ color: {ACCENT}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
```

- [ ] **Step 3: Refactor build_report.py to use the shared helpers**

In `build_report.py`:

1. Add import after the `from tools import theme` line:
   ```python
   from tools.report_html import fig_html, pct, card, page
   ```
2. Delete the local `_fig_html`, `_pct`, `_card` function definitions (lines ~64-78) and replace them with aliases right where they were:
   ```python
   _fig_html, _pct, _card = fig_html, pct, card
   ```
   (Aliases keep the diff minimal — every call site stays untouched.)
3. In `build()`, replace the trailing `return f"""<!DOCTYPE html>...</html>"""` block with:
   ```python
   return page(title, body)
   ```

- [ ] **Step 4: Verify nothing broke**

Run: `.venv/bin/python -c "import build_report; import tools.report_html"`
Expected: no output, exit 0.
Run: `.venv/bin/pytest tests/ -q`
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add tools/report_html.py tools/theme.py build_report.py
git commit -m "Extract shared HTML helpers into tools/report_html.py

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Pairs report page + nav link

**Files:**
- Create: `build_pairs_report.py`
- Modify: `build_report.py` (one nav link)

- [ ] **Step 1: Write the report script**

Create `build_pairs_report.py`:

```python
"""Generate the static pairs-trading HTML report.

  python build_pairs_report.py            # writes local/pairs.html + docs/pairs.html
  python build_pairs_report.py --refresh  # force re-download of price data

Statistical-arbitrage showcase: Engle-Granger cointegration scan over a
curated LS-Exchange universe, walk-forward z-score backtest with
Trade Republic-style costs. Paper simulation — shorting is simulated.
Public build shows percentages only.
"""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tools import theme
from tools.report_html import fig_html, pct as _pct, card as _card, page
from tools.pairs_universe import UNIVERSE, candidate_pairs, fetch_prices
from tools.pairs_engine import select_pairs, pair_zscore
from tools.pairs_backtest import run_backtest

# ── Settings (one place) ──────────────────────────────────────────────────────
CAPITAL        = 10_000.0   # paper account, EUR
FORMATION_DAYS = 252        # 12 months
TRADING_DAYS   = 63         # 3 months
P_MAX          = 0.05
TOP_N          = 10
ENTRY_Z        = 2.0
STOP_Z         = 3.5
FEE_EUR        = 1.0        # Trade Republic per-order fee
COST_MULTS     = (0.0, 1.0, 2.0)

ROOT = Path(__file__).parent


# ── Data assembly ─────────────────────────────────────────────────────────────

def gather(refresh: bool = False) -> dict:
    prices = fetch_prices(refresh=refresh)
    cands = candidate_pairs()
    slip = {t: UNIVERSE[t]["slippage_bps"] for t in UNIVERSE}

    bt = run_backtest(prices, cands, slip, capital=CAPITAL,
                      formation_days=FORMATION_DAYS, trading_days=TRADING_DAYS,
                      p_max=P_MAX, top_n=TOP_N, entry=ENTRY_Z, stop=STOP_Z,
                      fee_eur=FEE_EUR, cost_mults=COST_MULTS)

    # Live snapshot: latest 12 months as formation window
    form = prices.tail(FORMATION_DAYS)
    sel = select_pairs(form, cands, p_max=P_MAX, top_n=TOP_N)
    live = []
    for pr in sel["selected"]:
        pair_px = form[[pr["y"], pr["x"]]].dropna()
        z = pair_zscore(pair_px[pr["y"]], pair_px[pr["x"]], pr)
        live.append({**pr, "z_now": float(z.iloc[-1]), "z_series": z})
    return dict(prices=prices, bt=bt, live=live, n_tested=sel["n_tested"],
                n_candidates=len(cands))


# ── Sections ──────────────────────────────────────────────────────────────────

def sec_intro() -> str:
    return f"""
<div class="note">
<b>Statistical arbitrage demo</b> — scans {len(UNIVERSE)} LS-Exchange-tradeable
stocks for cointegrated same-sector, same-currency pairs (Engle-Granger two-step),
then trades the spread on z-score signals in a walk-forward backtest with
Trade Republic-style costs (€{FEE_EUR:.0f}/order + 5–15 bps slippage per leg).
Paper simulation: shorting is simulated — Trade Republic offers no shorting.
Not financial advice.
</div>"""


def sec_snapshot(d: dict) -> str:
    out = ["<h2>Current snapshot</h2>"]
    out.append(f"""<p class='dim'>Latest {FORMATION_DAYS}-trading-day formation window.
{d['n_tested']} same-sector/same-currency candidates tested, {len(d['live'])}
cointegrated (p &lt; {P_MAX}) with a tradeable half-life. Multiple-testing caveat:
at p &lt; {P_MAX}, roughly {round(P_MAX * d['n_tested'])} of {d['n_tested']} tests
pass by pure chance.</p>""")
    n_signal = sum(1 for p in d["live"] if abs(p["z_now"]) >= ENTRY_Z)
    cards = [
        _card("Universe", str(len(UNIVERSE))),
        _card("Candidate pairs", str(d["n_candidates"])),
        _card("Tested", str(d["n_tested"])),
        _card("Cointegrated now", str(len(d["live"]))),
        _card("Signals now", str(n_signal)),
    ]
    out.append(f'<div class="cards">{"".join(cards)}</div>')
    rows = []
    for p in d["live"]:
        zcls = "neg" if abs(p["z_now"]) >= ENTRY_Z else ""
        sig = "—"
        if p["z_now"] <= -ENTRY_Z:
            sig = f"LONG {p['y']} / SHORT {p['x']}"
        elif p["z_now"] >= ENTRY_Z:
            sig = f"SHORT {p['y']} / LONG {p['x']}"
        rows.append(
            f"<tr><td class='mono'>{p['y']}/{p['x']}</td>"
            f"<td>{UNIVERSE[p['y']]['sector']}</td>"
            f"<td class='num mono'>{p['pvalue']:.3f}</td>"
            f"<td class='num mono'>{p['beta']:.2f}</td>"
            f"<td class='num mono'>{p['half_life']:.0f}d</td>"
            f"<td class='num mono {zcls}'>{p['z_now']:+.2f}</td>"
            f"<td class='mono'>{sig}</td></tr>")
    out.append("<table><tr><th>Pair</th><th>Sector</th><th class='num'>p-value</th>"
               "<th class='num'>β</th><th class='num'>Half-life</th>"
               "<th class='num'>z now</th><th>Signal</th></tr>"
               + "".join(rows) + "</table>")
    return "".join(out)


def sec_pair_charts(d: dict) -> str:
    live = d["live"][:3]
    if not live:
        return ""
    out = ["<h2>Spread z-scores (top pairs)</h2>",
           "<p class='dim'>z = (spread − μ) / σ with μ, σ, β frozen from the "
           f"formation window. Enter beyond ±{ENTRY_Z:.0f}, exit at 0, stop "
           f"beyond ±{STOP_Z:.1f}.</p>"]
    for p in live:
        z = p["z_series"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z.values, name="z",
                                 line=dict(color=theme.ACCENT, width=1.6)))
        for lvl, col in ((0.0, theme.FG_DIM), (ENTRY_Z, theme.YELLOW),
                         (-ENTRY_Z, theme.YELLOW), (STOP_Z, theme.RED),
                         (-STOP_Z, theme.RED)):
            fig.add_hline(y=lvl, line_color=col, line_width=1,
                          line_dash="dash" if lvl == 0 else "dot")
        fig.update_layout(
            height=260, yaxis_title="z-score", showlegend=False,
            margin=dict(t=34),
            title=dict(text=f"{p['y']} / {p['x']} · p={p['pvalue']:.3f} · "
                            f"β={p['beta']:.2f} · HL={p['half_life']:.0f}d",
                       font=dict(size=13)))
        out.append(f"<div class='chart'>{fig_html(fig)}</div>")
    return "".join(out)


def sec_backtest(d: dict, public: bool) -> str:
    bt = d["bt"]
    runs = bt["runs"]
    base = runs[1.0]
    st = base["stats"]
    out = [f"<h2>Walk-forward backtest (since {bt['start']})</h2>"]
    out.append(f"""<p class='dim'>Every {TRADING_DAYS} trading days: re-run
Engle-Granger on the trailing {FORMATION_DAYS} days (strictly before the trading
window — no look-ahead), select up to {TOP_N} pairs, freeze β/μ/σ, trade z-score
signals with t+1 execution. Capital is split equally across that window's pairs.</p>""")

    fig = go.Figure()
    colors = {0.0: theme.GREEN, 1.0: theme.ACCENT, 2.0: theme.RED}
    names = {0.0: "0× costs (frictionless)", 1.0: "1× costs (realistic)",
             2.0: "2× costs (pessimistic)"}
    for m in sorted(runs):
        roi = (runs[m]["equity"] / CAPITAL - 1.0) * 100
        fig.add_trace(go.Scatter(x=roi.index, y=roi.values,
                                 name=names.get(m, f"{m}× costs"),
                                 line=dict(color=colors.get(m, "#aaa"),
                                           width=2.4 if m == 1.0 else 1.4)))
    fig.add_hline(y=0, line_dash="dash", line_color=theme.FG_DIM, line_width=1)
    fig.update_layout(height=400, yaxis=dict(title="Cumulative return (%)",
                                             ticksuffix="%"),
                      hovermode="x unified", legend=dict(x=0.01, y=0.99))
    out.append(f"<div class='chart'>{fig_html(fig)}</div>")

    cards = [
        _card("Net return", _pct(st["net_return"] * 100)),
        _card("Sharpe", f"{st['sharpe']:.2f}"),
        _card("Max drawdown", _pct(st["max_drawdown"] * 100)),
        _card("Trades", str(st["n_trades"])),
        _card("Win rate", _pct(st["win_rate"] * 100, signed=False)
              if st["win_rate"] is not None else "—"),
        _card("Avg holding", f"{st['avg_days']:.0f}d"
              if st["avg_days"] is not None else "—"),
    ]
    if not public:
        cards.append(_card("Net P&L", f"€{st['net_return'] * CAPITAL:+,.0f}"))
        cards.append(_card("Costs paid", f"€{st['total_costs']:,.0f}"))
    else:
        cards.append(_card("Costs / capital",
                           _pct(st["total_costs"] / CAPITAL * 100, signed=False)))
    out.append(f'<div class="cards">{"".join(cards)}</div>')

    trades = sorted(base["trades"], key=lambda t: t["entry"])[-25:]
    rows = []
    for t in trades:
        ret_pct = t["net"] / t["capital"] * 100
        z_e = f"{t['z_entry']:+.2f}" if t["z_entry"] is not None else "—"
        rows.append(f"<tr><td class='mono'>{t['pair']}</td>"
                    f"<td>{'long' if t['side'] == 1 else 'short'} spread</td>"
                    f"<td class='mono'>{t['entry'].date()}</td>"
                    f"<td class='mono'>{t['exit'].date()}</td>"
                    f"<td class='num mono'>{t['days']}</td>"
                    f"<td class='num mono'>{z_e}</td>"
                    f"<td class='num'>{_pct(ret_pct)}</td>"
                    + ("" if public else f"<td class='num mono'>€{t['net']:+,.0f}</td>")
                    + "</tr>")
    head = ("<tr><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th>"
            "<th class='num'>Days</th><th class='num'>z entry</th>"
            "<th class='num'>Return</th>"
            + ("" if public else "<th class='num'>Net P&amp;L</th>") + "</tr>")
    out.append(f"<h3>Trade ledger (last {len(trades)})</h3>"
               f"<table>{head}{''.join(rows)}</table>")

    wrows = "".join(
        f"<tr><td class='mono'>{w['trade_start'].date()}</td>"
        f"<td class='num mono'>{w['n_tested']}</td>"
        f"<td class='num mono'>{w['n_selected']}</td></tr>"
        for w in bt["windows"])
    out.append("<details><summary>Walk-forward windows: tested vs selected</summary>"
               "<table><tr><th>Trading window start</th><th class='num'>Tested</th>"
               f"<th class='num'>Selected</th></tr>{wrows}</table></details>")
    return "".join(out)


def sec_costs(d: dict) -> str:
    runs = d["bt"]["runs"]
    rows = []
    for m in sorted(runs):
        st = runs[m]["stats"]
        rows.append(f"<tr><td class='mono'>{m:.0f}×</td>"
                    f"<td class='num'>{_pct(st['net_return'] * 100)}</td>"
                    f"<td class='num mono'>{st['sharpe']:.2f}</td>"
                    f"<td class='num'>{_pct(st['max_drawdown'] * 100)}</td>"
                    f"<td class='num mono'>{st['n_trades']}</td>"
                    f"<td class='num'>{_pct(st['total_costs'] / CAPITAL * 100, signed=False)}</td></tr>")
    return ("<h2>Cost sensitivity</h2>"
            "<p class='dim'>Identical signals re-priced at 0×, 1× and 2× the assumed "
            f"frictions (€{FEE_EUR:.0f}/order + 5–15 bps slippage per leg). A strategy "
            "that only works at 0× has no edge; the gap between rows is the cost drag.</p>"
            "<table><tr><th>Costs</th><th class='num'>Net return</th>"
            "<th class='num'>Sharpe</th><th class='num'>Max DD</th>"
            "<th class='num'>Trades</th><th class='num'>Costs / capital</th></tr>"
            + "".join(rows) + "</table>"
            "<div class='note warn'><b>Honest caveats</b> — (1) Multiple testing: "
            "scanning ~50 pairs at p&lt;0.05 finds ~2-3 false positives per window by "
            "chance; the half-life filter and sector restriction mitigate but don't "
            "eliminate this. (2) Shorting is simulated — Trade Republic offers no "
            "shorting, so half of every pair trade is hypothetical. (3) Survivorship: "
            "the curated universe contains today's large caps. (4) Daily closes only — "
            "intraday spread behaviour and borrow costs are ignored.</div>")


def sec_method() -> str:
    return f"""
<h2>How it works</h2>
<details open><summary>Correlation is not cointegration (the core idea)</summary>
<p>Two stocks can be highly <i>correlated</i> (daily moves in the same direction)
while drifting apart forever — correlation says nothing about the <i>level</i> of
the spread. <b>Cointegration</b> is the property that some combination
log(P<sub>Y</sub>) − β·log(P<sub>X</sub>) is stationary: it oscillates around a
stable mean instead of wandering off. That stationary spread is the tradeable
object — divergence is expected to revert.</p></details>
<details><summary>Engle-Granger two-step</summary>
<p>Step 1: OLS log(P<sub>Y</sub>) = α + β·log(P<sub>X</sub>) gives the hedge ratio β;
the residual is the spread. Step 2: a unit-root test on those residuals. Subtlety:
because the residuals come from an estimated regression, plain ADF critical values
are wrong — this engine uses <code>statsmodels.tsa.stattools.coint</code>, which
applies the correct Engle-Granger distribution. Both orientations (Y on X, X on Y)
are tested; the lower p-value wins.</p></details>
<details><summary>Pair selection filters</summary>
<p>p &lt; {P_MAX}, hedge ratio β &gt; 0, and spread half-life between 2 and 60
trading days from an AR(1) fit (HL = −ln 2 / ρ). The half-life filter drops pairs
that revert too slowly to trade inside a {TRADING_DAYS}-day window. At most {TOP_N}
pairs per window, ranked by p-value.</p></details>
<details><summary>Signals &amp; execution</summary>
<p>z-score of the spread with μ, σ, β <b>frozen from the formation window</b>.
Enter when |z| ≥ {ENTRY_Z:.0f} (long the cheap leg, short the rich leg, β-weighted,
dollar-neutral); exit when z crosses 0; stop out when |z| ≥ {STOP_Z:.1f}
(cointegration treated as broken — no re-entry that window). A signal on day t
executes at the close of day t+1.</p></details>
<details><summary>No look-ahead, by construction</summary>
<p>Every parameter used in a trading window is estimated strictly before it; the
t+1 execution lag removes same-close bias; unit tests assert that truncating
future data leaves past signals and past equity unchanged.</p></details>
"""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build(d: dict, public: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Pairs Trading Lab" + ("" if public else " — private")
    back = "index.html" if public else "report.html"
    badge = "public build — percentages only" if public else "private build"
    body = "".join([
        f"<h1>{title}</h1>",
        f"<p class='dim'>generated {now} · {badge} · "
        f"<a href='{back}'>← portfolio monitor</a></p>",
        sec_intro(),
        sec_snapshot(d),
        sec_pair_charts(d),
        sec_backtest(d, public),
        sec_costs(d),
        sec_method(),
    ])
    return page(title, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the local report")
    ap.add_argument("--refresh", action="store_true", help="force price re-download")
    args = ap.parse_args()

    print("gathering pairs data (yfinance)...")
    d = gather(refresh=args.refresh)

    local = ROOT / "local/pairs.html"
    local.parent.mkdir(exist_ok=True)
    local.write_text(build(d, public=False))
    print(f"wrote {local}")

    pub = ROOT / "docs/pairs.html"
    pub.parent.mkdir(exist_ok=True)
    pub.write_text(build(d, public=True))
    print(f"wrote {pub}  (percentages only)")

    if args.open:
        webbrowser.open(local.as_uri())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add nav link in the main report**

In `build_report.py`, in `build()`, change the subtitle line of `body` from:

```python
        f"<h1>{title}</h1><p class='dim'>generated {now} · {badge}</p>",
```

to:

```python
        f"<h1>{title}</h1><p class='dim'>generated {now} · {badge} · "
        f"<a href='pairs.html'>Pairs Trading Lab →</a></p>",
```

(`pairs.html` is a sibling file in both `local/` and `docs/`, so the same relative href works for both builds.)

- [ ] **Step 3: Run the report end-to-end**

Run: `.venv/bin/python build_pairs_report.py`
Expected output (first run downloads ~5y × 31 tickers, takes ~1-3 min incl. the
walk-forward Engle-Granger scan — ~52 pairs × ~16 windows):

```
gathering pairs data (yfinance)...
wrote /Users/cxmc/code/investment-monitor/local/pairs.html
wrote /Users/cxmc/code/investment-monitor/docs/pairs.html  (percentages only)
```

- [ ] **Step 4: Verify the outputs**

```bash
ls -la local/pairs.html docs/pairs.html data/pairs_prices.csv
grep -c "Engle-Granger" docs/pairs.html        # expected ≥ 2
grep -c "Cost sensitivity" docs/pairs.html     # expected 1
grep -c 'Net P&amp;L' docs/pairs.html          # expected 0 (private-only column)
grep -c 'Net P&amp;L' local/pairs.html         # expected ≥ 1
```

Also open `local/pairs.html` in a browser and eyeball: dark theme matches the main
report, snapshot table populated, equity chart shows three cost lines, ledger
non-empty. If the live snapshot has zero cointegrated pairs that's plausible
(markets aren't obligated to cooperate) — the backtest section must still render.

- [ ] **Step 5: Commit**

```bash
git add build_pairs_report.py build_report.py docs/pairs.html
git commit -m "Add Pairs Trading Lab report page

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(`local/` and `data/pairs_prices.csv` are gitignored — only `docs/pairs.html` deploys.)

---

### Task 10: Docs + final verification

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

In the Architecture section of `CLAUDE.md`, extend the tree:

```
build_pairs_report.py             ← pairs-trading engine → local/pairs.html + docs/pairs.html
 ├─ tools/pairs_universe.py       ← curated LS-Exchange universe + price cache (CSV in data/)
 ├─ tools/pairs_engine.py         ← Engle-Granger, half-life, z-score signals (pure functions)
 ├─ tools/pairs_backtest.py       ← walk-forward backtester, costs + slippage, cost sensitivity
 └─ tools/report_html.py          ← shared HTML helpers (used by both reports)
```

And add a short section after "Calculation invariants":

```markdown
## Pairs engine invariants

- Walk-forward only: α/β/μ/σ are estimated on the formation window and frozen;
  signals on day t execute at close t+1. Tests assert truncating future data
  changes nothing in the past.
- Cointegration p-values come from `statsmodels.tsa.stattools.coint`
  (correct Engle-Granger critical values), never plain `adfuller` on residuals.
- Candidate pairs: same sector AND same currency only (FX leaks cause spurious
  cointegration).
- Cost-sensitivity (0×/1×/2×) re-prices identical signals — selection and
  signals must never depend on the cost multiplier.
```

- [ ] **Step 2: Full test suite + both reports build**

```bash
.venv/bin/pytest tests/ -q
.venv/bin/python build_pairs_report.py
```

Expected: all tests pass; report rebuilds from cache in seconds (no re-download).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document pairs engine in CLAUDE.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review checklist (run after writing, before execution)

- Spec coverage: universe ✓ (T1), Engle-Granger both orientations ✓ (T2), half-life filter ✓ (T3/T4), top-N + n_tested visibility ✓ (T4), frozen params + stop/ban + t+1 lag ✓ (T5/T6), €1 + tiered bps costs, force-close, ledger ✓ (T6), walk-forward + cost sensitivity ✓ (T7), CSV cache + refresh flag ✓ (T1/T9), report sections 1-5 ✓ (T9), nav link ✓ (T9), public/private split ✓ (T9), all spec'd tests ✓ (T2-T7).
- Known deviations from spec (intentional): CSV cache instead of parquet (no pyarrow dep); tests split into two files; `tools/report_html.py` added for DRY.
