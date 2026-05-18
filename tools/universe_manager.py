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
    {"isin": "NL0010273215", "name": "ASML Holding", "yf_ticker": "ASML.AS", "type": "stock", "sector": "Information Technology", "region": "NL"},
    {"isin": "FR0000131104", "name": "BNP Paribas", "yf_ticker": "BNP.PA", "type": "stock", "sector": "Financials", "region": "FR"},
    {"isin": "FR0000120321", "name": "L'Oreal", "yf_ticker": "OR.PA", "type": "stock", "sector": "Consumer Staples", "region": "FR"},
    {"isin": "DE0005140008", "name": "Deutsche Bank", "yf_ticker": "DBK.DE", "type": "stock", "sector": "Financials", "region": "DE"},
    {"isin": "DE0007100000", "name": "Mercedes-Benz", "yf_ticker": "MBG.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0005552004", "name": "Deutsche Telekom", "yf_ticker": "DTE.DE", "type": "stock", "sector": "Communication Services", "region": "DE"},
    {"isin": "FR0000121014", "name": "LVMH", "yf_ticker": "MC.PA", "type": "stock", "sector": "Consumer Discretionary", "region": "FR"},
    {"isin": "NL0000009165", "name": "Heineken", "yf_ticker": "HEIA.AS", "type": "stock", "sector": "Consumer Staples", "region": "NL"},
    {"isin": "GB0008847096", "name": "Tesco PLC", "yf_ticker": "TSCO.L", "type": "stock", "sector": "Consumer Staples", "region": "GB"},
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
    {"isin": "IT0003128367", "name": "Enel SpA", "yf_ticker": "ENEL.MI", "type": "stock", "sector": "Utilities", "region": "IT"},
    {"isin": "DE0006231004", "name": "Infineon Technologies", "yf_ticker": "IFX.DE", "type": "stock", "sector": "Technology", "region": "DE"},
    {"isin": "NL0009434992", "name": "Airbus SE", "yf_ticker": "AIR.PA", "type": "stock", "sector": "Industrials", "region": "FR"},
    {"isin": "GB00BH4HKS39", "name": "Ryanair", "yf_ticker": "RYA.L", "type": "stock", "sector": "Industrials", "region": "IE"},
    {"isin": "DE0008232125", "name": "Deutsche Lufthansa", "yf_ticker": "LHA.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
    # Portfolio holdings — US large caps (Frankfurt cross-listed, rate via primary)
    {"isin": "US67066G1040", "name": "NVIDIA Corporation", "yf_ticker": "NVDA", "type": "stock", "sector": "Information Technology", "region": "US"},
    {"isin": "US0231351067", "name": "Amazon.com Inc.", "yf_ticker": "AMZN", "type": "stock", "sector": "Consumer Discretionary", "region": "US"},
    {"isin": "US02079K3059", "name": "Alphabet Inc. (Google)", "yf_ticker": "GOOGL", "type": "stock", "sector": "Communication Services", "region": "US"},
    # Portfolio holdings — Italian & Asian blue chips
    {"isin": "IT0000072618", "name": "Intesa Sanpaolo", "yf_ticker": "ISP.MI", "type": "stock", "sector": "Financials", "region": "IT"},
    {"isin": "IT0005239360", "name": "UniCredit", "yf_ticker": "UCG.MI", "type": "stock", "sector": "Financials", "region": "IT"},
    {"isin": "IT0003763665", "name": "Webuild S.p.A.", "yf_ticker": "WBD.MI", "type": "stock", "sector": "Industrials", "region": "IT"},
    {"isin": "IT0003128367", "name": "Enel SpA", "yf_ticker": "ENEL.MI", "type": "stock", "sector": "Utilities", "region": "IT"},
    {"isin": "IT0000066123", "name": "Eni SpA", "yf_ticker": "ENI.MI", "type": "stock", "sector": "Energy", "region": "IT"},
    {"isin": "IT0004176001", "name": "Ferrari N.V.", "yf_ticker": "RACE.MI", "type": "stock", "sector": "Consumer Discretionary", "region": "IT"},
    {"isin": "IT0003153415", "name": "Prysmian", "yf_ticker": "PRY.MI", "type": "stock", "sector": "Industrials", "region": "IT"},
    {"isin": "TW0002330008", "name": "Taiwan Semiconductor (TSMC)", "yf_ticker": "TSM", "type": "stock", "sector": "Information Technology", "region": "TW"},
    {"isin": "KR7005930003", "name": "Samsung Electronics", "yf_ticker": "005930.KS", "type": "stock", "sector": "Information Technology", "region": "KR"},
    # Japanese blue chips (Nikkei) — available on TR as ADRs or Tokyo listings
    {"isin": "JP3633400001", "name": "Toyota Motor", "yf_ticker": "TM", "type": "stock", "sector": "Consumer Discretionary", "region": "JP"},
    {"isin": "JP3633400002", "name": "Sony Group", "yf_ticker": "SONY", "type": "stock", "sector": "Consumer Discretionary", "region": "JP"},
    {"isin": "JP3633400003", "name": "Honda Motor", "yf_ticker": "HMC", "type": "stock", "sector": "Consumer Discretionary", "region": "JP"},
    {"isin": "JP3633400004", "name": "Softbank Group", "yf_ticker": "9984.T", "type": "stock", "sector": "Communication Services", "region": "JP"},
    {"isin": "JP3633400005", "name": "Keyence", "yf_ticker": "6861.T", "type": "stock", "sector": "Industrials", "region": "JP"},
    {"isin": "JP3633400006", "name": "Fast Retailing (Uniqlo)", "yf_ticker": "9983.T", "type": "stock", "sector": "Consumer Discretionary", "region": "JP"},
    {"isin": "JP3633400007", "name": "Tokyo Electron", "yf_ticker": "8035.T", "type": "stock", "sector": "Information Technology", "region": "JP"},
    {"isin": "JP3633400008", "name": "Mitsubishi UFJ Financial", "yf_ticker": "MUFG", "type": "stock", "sector": "Financials", "region": "JP"},
    {"isin": "JP3633400009", "name": "Nintendo", "yf_ticker": "NTDOY", "type": "stock", "sector": "Communication Services", "region": "JP"},
    {"isin": "JP3633400010", "name": "Hitachi", "yf_ticker": "HTHIY", "type": "stock", "sector": "Industrials", "region": "JP"},
    {"isin": "JP3633400011", "name": "Panasonic", "yf_ticker": "PCRFY", "type": "stock", "sector": "Consumer Discretionary", "region": "JP"},
    {"isin": "JP3633400012", "name": "Mitsubishi Corp", "yf_ticker": "MSBHF", "type": "stock", "sector": "Industrials", "region": "JP"},
    {"isin": "CNE1000002H1", "name": "Tencent Holdings", "yf_ticker": "TCEHY", "type": "stock", "sector": "Communication Services", "region": "CN"},
    {"isin": "CNE000000002", "name": "Alibaba Group", "yf_ticker": "BABA", "type": "stock", "sector": "Consumer Discretionary", "region": "CN"},
    {"isin": "IE00BZ12WP82", "name": "Medtronic", "yf_ticker": "MDT", "type": "stock", "sector": "Healthcare", "region": "IE"},
    {"isin": "NL0012866412", "name": "BE Semiconductor", "yf_ticker": "BESI.AS", "type": "stock", "sector": "Information Technology", "region": "NL"},
    {"isin": "FR0000131906", "name": "Capgemini", "yf_ticker": "CAP.PA", "type": "stock", "sector": "Information Technology", "region": "FR"},
    {"isin": "FR0000045072", "name": "Credit Agricole", "yf_ticker": "ACA.PA", "type": "stock", "sector": "Financials", "region": "FR"},
    {"isin": "ES0178430E18", "name": "Iberdrola", "yf_ticker": "IBE.MC", "type": "stock", "sector": "Utilities", "region": "ES"},
    {"isin": "DE000A1EWWW0", "name": "Adidas AG", "yf_ticker": "ADS.DE", "type": "stock", "sector": "Consumer Discretionary", "region": "DE"},
    {"isin": "DE0007236101", "name": "Siemens AG", "yf_ticker": "SIE.DE", "type": "stock", "sector": "Industrials", "region": "DE"},
]

SEED_ETFS = [
    # Broad market — world / regional
    {"isin": "IE00B4L5Y983", "name": "iShares Core MSCI World", "yf_ticker": "IWDA.AS", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B3RBWM25", "name": "Vanguard FTSE All-World", "yf_ticker": "VWRL.L", "type": "etf", "sector": "Blend", "region": "Global"},
    {"isin": "IE00B3XXRP09", "name": "Vanguard S&P 500", "yf_ticker": "VUSA.AS", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00B5BMR087", "name": "iShares Core S&P 500", "yf_ticker": "CSPX.L", "type": "etf", "sector": "Blend", "region": "US"},
    {"isin": "IE00BKM4GZ66", "name": "iShares Core MSCI Europe", "yf_ticker": "IEUA.AS", "type": "etf", "sector": "Blend", "region": "EU"},
    {"isin": "US9229087690", "name": "Vanguard FTSE Europe ETF", "yf_ticker": "VGK", "type": "etf", "sector": "Blend", "region": "EU"},
    # Regional emerging / single-country
    {"isin": "IE00B52MJY50", "name": "iShares Core MSCI EM", "yf_ticker": "IEMG", "type": "etf", "sector": "Blend", "region": "EM"},
    {"isin": "US46434G8473", "name": "iShares MSCI India ETF", "yf_ticker": "INDA", "type": "etf", "sector": "Blend", "region": "IN"},
    {"isin": "US4642872349", "name": "iShares MSCI China", "yf_ticker": "MCHI", "type": "etf", "sector": "Blend", "region": "CN"},
    {"isin": "US4642873859", "name": "iShares MSCI Japan", "yf_ticker": "EWJ", "type": "etf", "sector": "Blend", "region": "JP"},
    {"isin": "US4642874659", "name": "iShares MSCI South Korea", "yf_ticker": "EWY", "type": "etf", "sector": "Blend", "region": "KR"},
    {"isin": "US46434G1031", "name": "iShares MSCI Brazil", "yf_ticker": "EWZ", "type": "etf", "sector": "Blend", "region": "BR"},
    {"isin": "IE00B4L5Y983", "name": "iShares MSCI Germany", "yf_ticker": "EWG", "type": "etf", "sector": "Blend", "region": "DE"},
    # Macro hedge
    {"isin": "US78462F1030", "name": "SPDR Gold Shares", "yf_ticker": "GLD", "type": "etf", "sector": "Commodities", "region": "Global"},
]


def _wiki_fetch(url: str) -> "pd.DataFrame | None":
    """Fetch first table from a Wikipedia page, return DataFrame or None."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; universe-manager/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return pd.read_html(io.StringIO(resp.text), header=0)[0]
    except Exception as e:
        print(f"[universe] Wikipedia fetch failed ({url}): {e}")
        return None


def download_sp500_tickers() -> list[dict]:
    """Download S&P 500 components from Wikipedia."""
    df = _wiki_fetch("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if df is None:
        return []
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


def _wiki_fetch_table(url: str, table_index: int) -> "pd.DataFrame | None":
    """Fetch specific table from a Wikipedia page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; universe-manager/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), header=0)
        return tables[table_index] if table_index < len(tables) else None
    except Exception as e:
        print(f"[universe] Wikipedia fetch failed ({url}): {e}")
        return None


def download_dax_tickers() -> list[dict]:
    """Download DAX 40 from Wikipedia. Tickers already carry exchange suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/DAX", 4)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        # DAX tickers: some .DE, some .PA (Airbus) — keep as-is
        sector = str(row.get("Prime Standard Sector", "Unknown"))
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": sector, "region": "DE" if ticker.endswith(".DE") else "EU",
            "added_date": datetime.now().isoformat(),
        })
    return result


def download_ftse100_tickers() -> list[dict]:
    """Download FTSE 100 from Wikipedia. Tickers are bare — append .L."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/FTSE_100_Index", 6)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    sector_col = next((c for c in df.columns if "sector" in c.lower() or "classification" in c.lower()), None)
    for _, row in df.iterrows():
        raw = str(row.get("Ticker", "")).strip()
        if not raw or raw == "nan":
            continue
        ticker = f"{raw}.L" if not raw.endswith(".L") else raw
        sector = str(row.get(sector_col, "Unknown")) if sector_col else "Unknown"
        result.append({
            "isin": "", "name": str(row.get("Company", raw)),
            "yf_ticker": ticker, "type": "stock",
            "sector": sector, "region": "GB",
            "added_date": datetime.now().isoformat(),
        })
    return result


def download_cac40_tickers() -> list[dict]:
    """Download CAC 40 from Wikipedia. Tickers already carry .PA suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/CAC_40", 4)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        sector = str(row.get("Sector", "Unknown"))
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": sector, "region": "FR",
            "added_date": datetime.now().isoformat(),
        })
    return result


def download_eurostoxx50_tickers() -> list[dict]:
    """Download Euro Stoxx 50 from Wikipedia. Tickers have full exchange suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/Euro_Stoxx_50", 3)
    if df is None or "Ticker" not in df.columns:
        return []

    SUFFIX_REGION = {
        ".PA": "FR", ".DE": "DE", ".AS": "NL", ".MC": "ES",
        ".MI": "IT", ".BR": "BE", ".HE": "FI", ".IR": "IE",
    }
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        suffix = next((s for s in SUFFIX_REGION if ticker.endswith(s)), None)
        region = SUFFIX_REGION.get(suffix, "EU")
        result.append({
            "isin": "", "name": str(row.get("Name", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("Sector", "Unknown")),
            "region": region,
            "added_date": datetime.now().isoformat(),
        })
    return result


def download_nasdaq100_tickers() -> list[dict]:
    """Download NASDAQ 100 — overlaps with S&P 500 but adds pure-NASDAQ names."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/Nasdaq-100", 5)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("ICB Industry[14]", "Unknown")),
            "region": "US", "added_date": datetime.now().isoformat(),
        })
    return result


def download_omxs30_tickers() -> list[dict]:
    """Download OMX Stockholm 30 — .ST suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/OMX_Stockholm_30", 1)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("GICS sector", "Unknown")),
            "region": "SE", "added_date": datetime.now().isoformat(),
        })
    return result


def download_aex_tickers() -> list[dict]:
    """Download AEX 25 (Amsterdam) — .AS suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/AEX_index", 3)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("ICB Sector", "Unknown")),
            "region": "NL", "added_date": datetime.now().isoformat(),
        })
    return result


def download_ibex35_tickers() -> list[dict]:
    """Download IBEX 35 (Spain) — .MC suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/IBEX_35", 2)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": "", "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("Sector", "Unknown")),
            "region": "ES", "added_date": datetime.now().isoformat(),
        })
    return result


def download_ftse_mib_tickers() -> list[dict]:
    """Download FTSE MIB (Italy) — .MI suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/FTSE_MIB", 1)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": str(row.get("ISIN", "")),
            "name": str(row.get("Company", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("ICB Sector", "Unknown")),
            "region": "IT", "added_date": datetime.now().isoformat(),
        })
    return result


def download_smi_tickers() -> list[dict]:
    """Download SMI 20 (Switzerland) — .SW suffix."""
    df = _wiki_fetch_table("https://en.wikipedia.org/wiki/Swiss_Market_Index", 2)
    if df is None or "Ticker" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker == "nan":
            continue
        result.append({
            "isin": "", "name": str(row.get("Name", ticker)),
            "yf_ticker": ticker, "type": "stock",
            "sector": str(row.get("Industry", "Unknown")),
            "region": "CH", "added_date": datetime.now().isoformat(),
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

    print("[universe] Downloading index components (S&P500, DAX, FTSE100, CAC40, EuroStoxx50)...")
    # Seeds first — curated ISINs/names take priority over Wikipedia scrapes
    candidates = (
        fetch_seed_european_stocks() +
        fetch_seed_etfs() +
        download_sp500_tickers() +
        download_nasdaq100_tickers() +
        download_dax_tickers() +
        download_ftse100_tickers() +
        download_cac40_tickers() +
        download_eurostoxx50_tickers() +
        download_aex_tickers() +
        download_ibex35_tickers() +
        download_ftse_mib_tickers() +
        download_smi_tickers() +
        download_omxs30_tickers()
    )
    candidate_df = pd.DataFrame(candidates)
    # Deduplicate within candidates (keep first = higher-priority source)
    candidate_df = candidate_df.drop_duplicates(subset=["yf_ticker"], keep="first")
    existing_tickers = set(existing["yf_ticker"].tolist())
    new_candidates = candidate_df[~candidate_df["yf_ticker"].isin(existing_tickers)]

    if validate and not new_candidates.empty:
        print(f"[universe] Validating {len(new_candidates)} new candidates...")
        valid_new_mask = new_candidates["yf_ticker"].apply(validate_ticker)
        new_valid = new_candidates[valid_new_mask].copy()
    else:
        new_valid = new_candidates.copy()

    universe = pd.concat([existing, new_valid], ignore_index=True)
    universe = universe.drop_duplicates(subset=["yf_ticker"], keep="first")
    universe = universe.reset_index(drop=True)
    universe.to_csv(universe_path, index=False)
    print(f"[universe] Universe: {len(universe)} total assets (+{len(new_valid)} added)")
    return universe


def load_universe(universe_path: str = UNIVERSE_PATH) -> list[dict]:
    """Load universe CSV and return list of dicts."""
    df = pd.read_csv(universe_path)
    return df.to_dict(orient="records")
