"""Trade Republic universe: ISIN → yfinance ticker mapping and validation."""

import pandas as pd
import yfinance as yf
import warnings
from pathlib import Path

ISIN_SUFFIX_MAP = {
    "US": "",
    "DE": ".DE",
    "NL": ".AS",
    "FR": ".PA",
    "GB": ".L",
    "CH": ".SW",
    "SE": ".ST",
    "FI": ".HE",
    "ES": ".MC",
    "IT": ".MI",
    "BE": ".BR",
    "PT": ".LS",
    "AT": ".VI",
    "DK": ".CO",
    "NO": ".OL",
}


def load_universe(csv_path: str) -> pd.DataFrame:
    """Load TR universe from CSV. Accepts the seed file or a custom TR export.

    Expected columns: isin, name, yf_ticker, sector, region
    If yf_ticker is missing, derives it from ISIN prefix + region suffix.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "yf_ticker" not in df.columns:
        df["yf_ticker"] = df.apply(_derive_ticker, axis=1)

    df = df.dropna(subset=["yf_ticker"])
    df["yf_ticker"] = df["yf_ticker"].str.strip()
    return df


def _derive_ticker(row) -> str:
    """Derive yfinance ticker from ISIN country prefix and region."""
    isin = str(row.get("isin", "")).upper()
    country_code = isin[:2] if len(isin) >= 2 else ""
    suffix = ISIN_SUFFIX_MAP.get(country_code, "")

    name = str(row.get("name", ""))
    ticker_guess = name.split()[0].upper()
    return f"{ticker_guess}{suffix}"


def validate_tickers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Validate each ticker resolves in yfinance; drop invalid ones with warning."""
    valid_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df.iterrows():
            ticker = row["yf_ticker"]
            try:
                info = yf.Ticker(ticker).fast_info
                price = getattr(info, "last_price", None)
                if price and price > 0:
                    valid_rows.append(row)
                else:
                    if verbose:
                        print(f"  [universe] Dropping {ticker} — no market price")
            except Exception:
                if verbose:
                    print(f"  [universe] Dropping {ticker} — yfinance error")

    result = pd.DataFrame(valid_rows).reset_index(drop=True)
    if verbose:
        print(f"  [universe] {len(result)}/{len(df)} tickers validated")
    return result


def get_universe(csv_path: str, validate: bool = False) -> list[dict]:
    """Load and optionally validate the TR universe. Returns list of dicts."""
    df = load_universe(csv_path)
    if validate:
        df = validate_tickers(df)
    return df.to_dict(orient="records")
