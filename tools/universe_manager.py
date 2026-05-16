"""Full TR asset universe: build, validate, and refresh weekly."""

import io
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

UNIVERSE_PATH = "data/universe.csv"

SEED_EUROPEAN_STOCKS = [
    {"isin": "DE0007164600", "name": "SAP SE", "yf_ticker": "SAP.DE", "type": "stock", "sector": "Technology", "region": "DE"},
    {"isin": "NL0010273215", "name": "ASML Holding", "yf_ticker": "ASML.AS", "type": "stock", "sector": "Technology", "region": "NL"},
    {"isin": "FR0000131104", "name": "BNP Paribas", "yf_ticker": "BNP.PA", "type": "stock", "sector": "Financials", "region": "FR"},
    {"isin": "FR0000120321", "name": "L'Oreal", "yf_ticker": "OR.PA", "type": "stock", "sector": "Consumer Staples", "region": "FR"},
    {"isin": "DE0005140008", "name": "Deutsche Bank", "yf_ticker": "DBK.DE", "type": "stock", "sector": "Financials", "region": "DE"},
    {"isin": "DE0007100000", "name": "Mercedes-Benz", "yf_ticker": "MBG.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0005552004", "name": "Deutsche Telekom", "yf_ticker": "DTE.DE", "type": "stock", "sector": "Communication Services", "region": "DE"},
    {"isin": "FR0000121014", "name": "LVMH", "yf_ticker": "MC.PA", "type": "stock", "sector": "Consumer Discretionary", "region": "FR"},
    {"isin": "NL0000009165", "name": "Heineken", "yf_ticker": "HEIA.AS", "type": "stock", "sector": "Consumer Staples", "region": "NL"},
    {"isin": "GB0002374006", "name": "Diageo", "yf_ticker": "DGE.L", "type": "stock", "sector": "Consumer Staples", "region": "GB"},
    {"isin": "GB00B10RZP78", "name": "Unilever", "yf_ticker": "ULVR.L", "type": "stock", "sector": "Consumer Staples", "region": "GB"},
    {"isin": "GB0031348658", "name": "HSBC Holdings", "yf_ticker": "HSBA.L", "type": "stock", "sector": "Financials", "region": "GB"},
    {"isin": "GB0007188757", "name": "GSK", "yf_ticker": "GSK.L", "type": "stock", "sector": "Healthcare", "region": "GB"},
    {"isin": "CH0012221716", "name": "ABB Ltd", "yf_ticker": "ABBN.SW", "type": "stock", "sector": "Industrials", "region": "CH"},
    {"isin": "DE0005190003", "name": "BMW AG", "yf_ticker": "BMW.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0007037129", "name": "Rheinmetall", "yf_ticker": "RHM.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
    {"isin": "SE0000115446", "name": "Ericsson", "yf_ticker": "ERIC-B.ST", "type": "stock", "sector": "Technology", "region": "SE"},
    {"isin": "FI0009000681", "name": "Nokia", "yf_ticker": "NOKIA.HE", "type": "stock", "sector": "Technology", "region": "FI"},
    {"isin": "ES0113211835", "name": "Banco Santander", "yf_ticker": "SAN.MC", "type": "stock", "sector": "Financials", "region": "ES"},
    {"isin": "IT0003128367", "name": "Enel", "yf_ticker": "ENEL.MI", "type": "stock", "sector": "Utilities", "region": "IT"},
    {"isin": "DE0006231004", "name": "Infineon Technologies", "yf_ticker": "IFX.DE", "type": "stock", "sector": "Technology", "region": "DE"},
    {"isin": "NL0009434992", "name": "Airbus SE", "yf_ticker": "AIR.PA", "type": "stock", "sector": "Industrials", "region": "FR"},
    {"isin": "GB00BH4HKS39", "name": "Ryanair", "yf_ticker": "RYA.L", "type": "stock", "sector": "Industrials", "region": "IE"},
    {"isin": "DE0008232125", "name": "Deutsche Lufthansa", "yf_ticker": "LHA.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
]

SEED_ETFS = [
    {"isin": "IE00B4L5Y983", "name": "iShares Core MSCI World", "yf_ticker": "IWDA.AS", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B3XXRP09", "name": "Vanguard S&P 500", "yf_ticker": "VUSA.AS", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00B52MJY50", "name": "iShares Core MSCI EM", "yf_ticker": "IEMG", "type": "etf", "sector": "Blend", "region": "EM"},
    {"isin": "IE00BKM4GZ66", "name": "iShares Core MSCI Europe", "yf_ticker": "IEUA.AS", "type": "etf", "sector": "Blend", "region": "EU"},
    {"isin": "IE00B5BMR087", "name": "iShares Core S&P 500", "yf_ticker": "CSPX.L", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00B3RBWM25", "name": "Vanguard FTSE All-World", "yf_ticker": "VWRL.L", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B4L5Y983", "name": "iShares MSCI Germany", "yf_ticker": "EWG", "type": "etf", "sector": "Blend", "region": "DE"},
    {"isin": "US4642874659", "name": "iShares MSCI South Korea", "yf_ticker": "EWY", "type": "etf", "sector": "Blend", "region": "KR"},
    {"isin": "US4642873859", "name": "iShares MSCI Japan", "yf_ticker": "EWJ", "type": "etf", "sector": "Blend", "region": "JP"},
    {"isin": "US4642872349", "name": "iShares MSCI China", "yf_ticker": "MCHI", "type": "etf", "sector": "Blend", "region": "CN"},
    {"isin": "US46434G1031", "name": "iShares MSCI Brazil", "yf_ticker": "EWZ", "type": "etf", "sector": "Blend", "region": "BR"},
    {"isin": "US78378X2036", "name": "S&P Global Clean Energy ETF", "yf_ticker": "ICLN", "type": "etf", "sector": "Energy", "region": "Global"},
    {"isin": "US46432F3428", "name": "iShares Global Defense & Aerospace", "yf_ticker": "ITA", "type": "etf", "sector": "Industrials", "region": "US"},
    {"isin": "US46090E5030", "name": "iShares Semiconductor ETF", "yf_ticker": "SOXX", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US46137V4085", "name": "iShares Robotics and AI ETF", "yf_ticker": "IRBO", "type": "etf", "sector": "Technology", "region": "Global"},
    {"isin": "US33740Q1085", "name": "First Trust NASDAQ Cybersecurity", "yf_ticker": "CIBR", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US00214Q1040", "name": "ARK Innovation ETF", "yf_ticker": "ARKK", "type": "etf", "sector": "Technology", "region": "US"},
    {"isin": "US78462F1030", "name": "SPDR Gold Shares", "yf_ticker": "GLD", "type": "etf", "sector": "Commodities", "region": "Global"},
    {"isin": "US9229087690", "name": "Vanguard FTSE Europe ETF", "yf_ticker": "VGK", "type": "etf", "sector": "Blend", "region": "EU"},
    {"isin": "US46434G8473", "name": "iShares MSCI India ETF", "yf_ticker": "INDA", "type": "etf", "sector": "Blend", "region": "IN"},
    {"isin": "US46432F8491", "name": "iShares Global Clean Energy ETF", "yf_ticker": "INRG.L", "type": "etf", "sector": "Energy", "region": "Global"},
]


def download_sp500_tickers() -> list[dict]:
    """Download S&P 500 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; universe-manager/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), header=0)
    df = tables[0]
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol", "")).replace(".", "-")
        result.append({
            "isin": str(row.get("ISIN", "")),
            "name": str(row.get("Security", "")),
            "yf_ticker": ticker,
            "type": "stock",
            "sector": str(row.get("GICS Sector", "Unknown")),
            "region": "US",
            "added_date": datetime.now().isoformat(),
        })
    return result


def fetch_seed_etfs() -> list[dict]:
    """Return curated list of ETFs available on Trade Republic."""
    return [{**etf, "added_date": datetime.now().isoformat()} for etf in SEED_ETFS]


def fetch_seed_european_stocks() -> list[dict]:
    """Return curated list of European stocks available on TR."""
    return [{**s, "added_date": datetime.now().isoformat()} for s in SEED_EUROPEAN_STOCKS]


def validate_ticker(ticker: str) -> bool:
    """Return True if ticker has a valid live price in yfinance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None)
            return bool(price and price > 0)
        except Exception:
            return False


def build_universe(stocks: list[dict], etfs: list[dict]) -> pd.DataFrame:
    """Merge stocks and ETFs, deduplicate on yf_ticker."""
    all_assets = stocks + etfs
    df = pd.DataFrame(all_assets)
    df = df.drop_duplicates(subset=["yf_ticker"], keep="first")
    return df.reset_index(drop=True)


def refresh_universe(universe_path: str = UNIVERSE_PATH, validate: bool = True) -> pd.DataFrame:
    """Weekly refresh: validate existing tickers, add new candidates."""
    Path("data").mkdir(exist_ok=True)

    try:
        existing = pd.read_csv(universe_path)
        print(f"[universe] Loaded {len(existing)} existing assets")
    except FileNotFoundError:
        existing = pd.DataFrame(columns=["isin", "name", "yf_ticker", "type", "sector", "region", "added_date"])
        print("[universe] No existing universe found, building from scratch")

    if validate and not existing.empty:
        print("[universe] Validating existing tickers...")
        valid_mask = existing["yf_ticker"].apply(validate_ticker)
        removed_count = (~valid_mask).sum()
        existing = existing[valid_mask].copy()
        print(f"[universe] Removed {removed_count} invalid tickers")

    candidates = (
        download_sp500_tickers() +
        fetch_seed_european_stocks() +
        fetch_seed_etfs()
    )
    candidate_df = pd.DataFrame(candidates)
    existing_tickers = set(existing["yf_ticker"].tolist())
    new_candidates = candidate_df[~candidate_df["yf_ticker"].isin(existing_tickers)]

    if validate and not new_candidates.empty:
        print(f"[universe] Validating {len(new_candidates)} new candidates...")
        valid_new_mask = new_candidates["yf_ticker"].apply(validate_ticker)
        new_valid = new_candidates[valid_new_mask].copy()
    else:
        new_valid = new_candidates.copy()

    universe = pd.concat([existing, new_valid], ignore_index=True)
    universe.to_csv(universe_path, index=False)
    print(f"[universe] Universe: {len(universe)} total assets (+{len(new_valid)} added)")
    return universe


def load_universe(universe_path: str = UNIVERSE_PATH) -> list[dict]:
    """Load universe CSV and return list of dicts."""
    df = pd.read_csv(universe_path)
    return df.to_dict(orient="records")
