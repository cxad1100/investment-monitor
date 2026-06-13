"""
Markowitz mean-variance optimization.

Pure functions (numpy / pandas / scipy). Efficient frontier, max-Sharpe and
min-variance portfolios under long-only / max-weight / min-weight / sector-cap
constraints, plus a rolling-window dynamic-rebalance backtest.

Reuses:
- tools.portfolio_tools.TICKER_MAP   — TR ticker → yfinance price ticker
- tools.portfolio_meta.sector_exposure_matrix — sector-cap constraint matrix
- tools.portfolio_analytics.compute_quant_metrics — score each backtest strategy
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from tools.portfolio_tools import TICKER_MAP
from tools.portfolio_meta import sector_exposure_matrix

TRADING_DAYS = 252


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_price_history(tr_tickers: list[str], start: str | None = None,
                        period: str = "5y") -> pd.DataFrame:
    """
    Download adjusted Close for the given Trade Republic tickers (mapped to their
    yfinance price tickers). Returns a DataFrame indexed by date with one column
    per *TR* ticker (renamed back from the yfinance ticker).
    """
    yf_map = {tk: TICKER_MAP.get(tk, tk) for tk in tr_tickers}
    yf_tickers = list(dict.fromkeys(yf_map.values()))   # unique, order-preserving

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"start": start} if start else {"period": period}
        raw = yf.download(yf_tickers, auto_adjust=True, progress=False, **kw)
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=yf_tickers[0])
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    # Map yfinance columns back to TR tickers (first TR ticker wins on collisions)
    cols = {}
    for tk in tr_tickers:
        yft = yf_map[tk]
        if yft in close.columns and tk not in cols:
            cols[tk] = close[yft]
    return pd.DataFrame(cols).sort_index()


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, rows with any missing asset dropped (aligned panel)."""
    return prices.pct_change().dropna(how="any")


def annualize(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean-return vector and covariance matrix."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


# ── Portfolio math ────────────────────────────────────────────────────────────

def portfolio_perf(w: np.ndarray, mean_ann: np.ndarray, cov_ann: np.ndarray,
                   rf: float = 0.045) -> tuple[float, float, float]:
    """Return (annual_return, annual_vol, sharpe) for weight vector w."""
    ret = float(w @ mean_ann)
    vol = float(np.sqrt(w @ cov_ann @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _bounds(n: int, long_only: bool, max_w: float, min_w: float):
    lo = min_w if long_only else -max_w
    return tuple((lo, max_w) for _ in range(n))


def _sector_constraints(tickers: list[str], sector_caps: dict[str, float]) -> list[dict]:
    """Linear inequality constraints: sector exposure S @ w <= cap."""
    if not sector_caps:
        return []
    sectors, matrix = sector_exposure_matrix(tickers)
    cons = []
    for sec, row in zip(sectors, matrix):
        cap = sector_caps.get(sec)
        if cap is None or cap >= 0.999:
            continue
        r = np.array(row)
        cons.append({"type": "ineq", "fun": (lambda w, r=r, cap=cap: cap - float(r @ w))})
    return cons


def _solve(objective, n: int, bounds, constraints, x0=None) -> np.ndarray | None:
    if x0 is None:
        x0 = np.repeat(1.0 / n, n)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        return None
    w = res.x
    w[np.abs(w) < 1e-6] = 0.0
    s = w.sum()
    return w / s if s != 0 else None


def optimize(mean_ann: pd.Series, cov_ann: pd.DataFrame, *, objective: str = "sharpe",
             rf: float = 0.045, long_only: bool = True, max_w: float = 1.0,
             min_w: float = 0.0, sector_caps: dict[str, float] | None = None) -> np.ndarray | None:
    """Solve for optimal weights. objective: 'sharpe' (max) or 'min_var'."""
    tickers = list(mean_ann.index)
    n = len(tickers)
    mu, sig = mean_ann.values, cov_ann.values
    bounds = _bounds(n, long_only, max_w, min_w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    cons += _sector_constraints(tickers, sector_caps or {})

    if objective == "min_var":
        fn = lambda w: float(w @ sig @ w)
    else:
        fn = lambda w: -portfolio_perf(w, mu, sig, rf)[2]   # negative Sharpe
    return _solve(fn, n, bounds, cons)


def risk_contributions(w, cov_ann) -> np.ndarray:
    """Fraction of total portfolio variance contributed by each asset (sums to 1)."""
    w = np.asarray(w, float)
    sig = cov_ann.values if hasattr(cov_ann, "values") else np.asarray(cov_ann)
    pv = float(w @ sig @ w)
    if pv <= 0:
        return np.zeros_like(w)
    return (w * (sig @ w)) / pv


def risk_parity(cov_ann: pd.DataFrame, *, max_w: float = 1.0) -> np.ndarray | None:
    """
    Equal Risk Contribution portfolio: weights so every asset contributes the same
    share of total risk. Needs only the covariance matrix — no return forecast.
    Long-only, weights sum to 1.
    """
    sig = cov_ann.values if hasattr(cov_ann, "values") else np.asarray(cov_ann)
    n = sig.shape[0]
    target = 1.0 / n

    def obj(w):
        pv = float(w @ sig @ w)
        if pv <= 0:
            return 1e6
        rc = (w * (sig @ w)) / pv
        return float(np.sum((rc - target) ** 2))

    bounds = tuple((1e-4, max_w) for _ in range(n))   # tiny floor keeps risk-contrib defined
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    return _solve(obj, n, bounds, cons, x0=np.repeat(1.0 / n, n))


def hrp(cov_ann: pd.DataFrame) -> np.ndarray | None:
    """Hierarchical Risk Parity (López de Prado): allocate risk top-down across
    a correlation-clustered tree. Covariance-only, no return forecast, long-only,
    weights sum to 1. Unlike Risk-Parity it has no per-asset cap — clustering
    controls concentration naturally (correlated mega-caps are treated as one
    risk unit), which is the whole point.

    Returns weights as an ndarray in cov_ann.index order (matches risk_parity).
    """
    sig = cov_ann.values if hasattr(cov_ann, "values") else np.asarray(cov_ann, float)
    n = sig.shape[0]
    if n == 0:
        return None
    if n == 1:
        return np.array([1.0])

    # correlation → distance d_ij = sqrt(0.5*(1 - corr_ij))
    std = np.sqrt(np.diag(sig))
    corr = sig / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))

    # 1. tree clustering on the condensed distance matrix
    link = linkage(squareform(dist, checks=False), method="single")

    # 2. quasi-diagonalization: recover the leaf order from the dendrogram
    order = _hrp_quasi_diag(link, n)

    # 3. recursive bisection: split risk budget down the tree (inverse-variance)
    ivp_var = np.diag(sig)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[half:] if j else c[:half]            # bisect each cluster
                    for c in clusters if len(c) > 1
                    for half in (len(c) // 2,) for j in (0, 1)]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left = _hrp_cluster_var(sig, ivp_var, left)
            v_right = _hrp_cluster_var(sig, ivp_var, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha

    return w.reindex(range(n)).values    # back to cov_ann.index order


def _hrp_quasi_diag(link: np.ndarray, n: int) -> list[int]:
    """Leaf order of the dendrogram so correlated assets sit adjacent."""
    link = link.astype(int)
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:                                  # expand merged nodes
        new = []
        for node in order:
            if node < n:
                new.append(node)
            else:
                m = node - n
                new.extend([link[m, 0], link[m, 1]])
        order = new
    return order


def _hrp_cluster_var(sig: np.ndarray, ivp_var: np.ndarray, idx: list[int]) -> float:
    """Variance of a cluster under inverse-variance weights (the bisection metric)."""
    sub = sig[np.ix_(idx, idx)]
    iv = 1.0 / ivp_var[idx]
    w = iv / iv.sum()
    return float(w @ sub @ w)


def implied_equilibrium_returns(cov_ann: pd.DataFrame, market_weights, delta: float = 2.5) -> pd.Series:
    """
    Black-Litterman reverse optimization: the excess returns the market must expect
    to hold its current cap-weighted mix. Π = δ · Σ · w_market.
    """
    sig = cov_ann.values if hasattr(cov_ann, "values") else np.asarray(cov_ann)
    w = np.asarray(market_weights, float)
    return pd.Series(delta * (sig @ w), index=cov_ann.index)


def black_litterman(cov_ann: pd.DataFrame, market_weights, *, delta: float = 2.5,
                    tau: float = 0.05, views: list[dict] | None = None) -> pd.Series:
    """
    Black-Litterman posterior expected EXCESS returns.

    Starts from market-implied equilibrium (Π = δΣw_mkt) — the market's collective
    forecast, not trailing momentum — and blends in optional subjective views.
    No views → returns Π unchanged.

    views: [{"assets": {ticker: weight}, "ret": annual_excess, "confidence": 0..1}]
           e.g. absolute view "NVDA returns 15%/yr": {"assets": {"NVD.F": 1}, "ret": 0.15, "confidence": 0.6}
    """
    idx = list(cov_ann.index)
    sig = cov_ann.values if hasattr(cov_ann, "values") else np.asarray(cov_ann)
    pi = delta * (sig @ np.asarray(market_weights, float))
    if not views:
        return pd.Series(pi, index=idx)

    P, Q, conf = [], [], []
    for v in views:
        row = np.zeros(len(idx))
        for tk, wt in v["assets"].items():
            if tk in idx:
                row[idx.index(tk)] = wt
        P.append(row); Q.append(v["ret"]); conf.append(v.get("confidence", 0.5))
    P, Q = np.array(P), np.array(Q)
    tau_sig = tau * sig
    # Ω: view uncertainty scaled inversely by confidence
    omega = np.diag(np.diag(P @ tau_sig @ P.T) / np.clip(conf, 1e-3, 1.0))
    A = np.linalg.inv(tau_sig)
    Oi = np.linalg.inv(omega)
    mu = np.linalg.inv(A + P.T @ Oi @ P) @ (A @ pi + P.T @ Oi @ Q)
    return pd.Series(mu, index=idx)


def fetch_market_caps(tr_tickers: list[str]) -> dict[str, float]:
    """Single-stock market cap per ticker (Frankfurt listings carry caps too).

    Deliberately ignores ETF ``totalAssets``: an ETF's AUM (tens of billions) is
    dwarfed by trillion-euro single-stock caps, so including it crushes a
    diversified ETF to ~0% in the Black-Litterman prior. ETFs return no cap here
    and the caller falls back to position value instead.
    """
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tk in tr_tickers:
            yft = TICKER_MAP.get(tk, tk)
            try:
                info = yf.Ticker(yft).info
                if info.get("quoteType") in ("ETF", "MUTUALFUND", "INDEX"):
                    continue                      # fund — caller uses position value
                cap = info.get("marketCap")
                if cap and cap > 0:
                    out[tk] = float(cap)
            except Exception:
                pass
    return out


def max_return_at_vol(mean_ann: pd.Series, cov_ann: pd.DataFrame, vol_cap: float, *,
                      long_only: bool = True, max_w: float = 1.0, min_w: float = 0.0,
                      sector_caps: dict[str, float] | None = None) -> np.ndarray | None:
    """
    Maximize expected return subject to portfolio volatility <= vol_cap.
    Answers: "same risk as now, how much more expected return can the mix deliver?"
    """
    tickers = list(mean_ann.index)
    n = len(tickers)
    mu, sig = mean_ann.values, cov_ann.values
    bounds = _bounds(n, long_only, max_w, min_w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "ineq", "fun": lambda w: vol_cap ** 2 - float(w @ sig @ w)}]
    cons += _sector_constraints(tickers, sector_caps or {})
    return _solve(lambda w: -float(w @ mu), n, bounds, cons)


def efficient_frontier(mean_ann: pd.Series, cov_ann: pd.DataFrame, *, n_points: int = 40,
                       long_only: bool = True, max_w: float = 1.0, min_w: float = 0.0,
                       sector_caps: dict[str, float] | None = None) -> list[dict]:
    """For a sweep of target returns, minimize variance. Returns [{ret,vol,weights}]."""
    tickers = list(mean_ann.index)
    n = len(tickers)
    mu, sig = mean_ann.values, cov_ann.values
    bounds = _bounds(n, long_only, max_w, min_w)
    base = _sector_constraints(tickers, sector_caps or {})

    lo, hi = float(mu.min()), float(mu.max())
    out = []
    for target in np.linspace(lo, hi, n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": (lambda w, t=target: float(w @ mu) - t)}] + base
        w = _solve(lambda w: float(w @ sig @ w), n, bounds, cons)
        if w is None:
            continue
        ret, vol, _ = portfolio_perf(w, mu, sig)
        out.append({"ret": ret, "vol": vol, "weights": w})
    return out


def random_portfolios(mean_ann: pd.Series, cov_ann: pd.DataFrame, n: int = 3000,
                      rf: float = 0.045, seed: int = 0) -> pd.DataFrame:
    """Dirichlet-sampled long-only portfolios for the frontier scatter cloud."""
    rng = np.random.default_rng(seed)
    mu, sig = mean_ann.values, cov_ann.values
    k = len(mu)
    W = rng.dirichlet(np.ones(k), size=n)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, sig, W))
    sharpe = np.where(vols > 0, (rets - rf) / vols, 0.0)
    return pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpe})


# ── Rolling-window backtest ───────────────────────────────────────────────────

def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """First trading day of each month ('M') or quarter ('Q')."""
    per = "Q" if freq.upper().startswith("Q") else "M"
    first = pd.Series(index, index=index).groupby(index.to_period(per)).first()
    return [pd.Timestamp(x) for x in first]


def rolling_backtest(prices: pd.DataFrame, *, lookback_days: int = 365,
                     rebalance_freq: str = "M", objective: str = "sharpe",
                     rf: float = 0.045, long_only: bool = True, max_w: float = 1.0,
                     min_w: float = 0.0, sector_caps: dict[str, float] | None = None,
                     benchmark_ticker: str = "CSPX.AS") -> dict:
    """
    Walk-forward: at each rebalance date estimate (mean,cov) from the trailing
    `lookback_days`, solve weights, hold them out-of-sample until the next
    rebalance. Build Optimized, Equal-Weight and benchmark equity curves (all
    start at 1.0 on the same date).

    Returns {"equity": {strategy: Series}, "weights_history": [...],
             "start": date, "tickers": [...]}.
    """
    rets = to_returns(prices)
    if len(rets) < lookback_days // 2 + 20 or rets.shape[1] < 2:
        return {}

    tickers = list(rets.columns)
    n = len(tickers)
    idx = rets.index
    reb_dates = [d for d in _rebalance_dates(idx, rebalance_freq)
                 if d >= idx[0] + pd.Timedelta(days=lookback_days)]
    if not reb_dates:
        return {}

    opt_w = np.repeat(1.0 / n, n)
    eq_w = np.repeat(1.0 / n, n)
    weights_history = []

    opt_curve, eq_curve, dates = [1.0], [1.0], [reb_dates[0]]
    reb_set = set(reb_dates)

    bounds_args = dict(objective=objective, rf=rf, long_only=long_only,
                       max_w=max_w, min_w=min_w, sector_caps=sector_caps)

    active = rets.loc[reb_dates[0]:]
    for dt, row in active.iterrows():
        if dt in reb_set:
            # Estimation window: trailing calendar days, strictly BEFORE dt —
            # no look-ahead (dt's own return must not inform dt's weights).
            window = rets.loc[(rets.index >= dt - pd.Timedelta(days=lookback_days))
                              & (rets.index < dt)]
            if len(window) >= 40:
                mean_ann, cov_ann = annualize(window)
                w = optimize(mean_ann, cov_ann, **bounds_args)
                if w is not None:
                    opt_w = w
                    weights_history.append({"date": str(dt.date()),
                                            "weights": dict(zip(tickers, np.round(w, 4)))})
        if dt == dates[0]:
            continue
        opt_curve.append(opt_curve[-1] * (1 + float(row.values @ opt_w)))
        eq_curve.append(eq_curve[-1] * (1 + float(row.values @ eq_w)))
        dates.append(dt)

    equity = {
        "Optimized":    pd.Series(opt_curve, index=pd.to_datetime(dates)),
        "Equal-Weight": pd.Series(eq_curve, index=pd.to_datetime(dates)),
    }

    # Benchmark (e.g. S&P 500 EUR-listed) over the same window
    try:
        bench = fetch_price_history([benchmark_ticker], start=str(dates[0].date()))
        if not bench.empty:
            bser = bench.iloc[:, 0].reindex(pd.to_datetime(dates), method="ffill")
            bret = bser.pct_change().fillna(0.0)
            equity["S&P 500"] = (1 + bret).cumprod()
    except Exception:
        pass

    return {"equity": equity, "weights_history": weights_history,
            "start": str(dates[0].date()), "tickers": tickers}


def equity_to_roi(series: pd.Series) -> pd.Series:
    """Equity curve (starts ~1.0) → cumulative ROI % series for compute_quant_metrics."""
    if series.empty:
        return series
    return (series / series.iloc[0] - 1) * 100
