"""
Data loading, cleaning, alignment, and processing utilities for
cross-venue crypto trade data.

Supports Binance, Bybit, and OKX perpetual futures trade data.
"""

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column name mappings per venue
# ---------------------------------------------------------------------------

BINANCE_COLUMNS = {
    "id": "trade_id",
    "price": "price",
    "qty": "quantity",
    "quoteQty": "quote_quantity",
    "time": "timestamp",
    "isBuyerMaker": "is_buyer_maker",
}

BYBIT_COLUMNS = {
    "timestamp": "timestamp",
    "symbol": "symbol",
    "side": "side",
    "size": "quantity",
    "price": "price",
}

OKX_COLUMNS = {
    "ts": "timestamp",
    "px": "price",
    "sz": "quantity",
    "side": "side",
}


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_binance_trades(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
) -> list[Path]:
    """Download daily trade data from Binance public data repository.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. "BTCUSDT".
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format (inclusive).
    output_dir : str or Path
        Directory to save downloaded CSV files.

    Returns
    -------
    list[Path]
        Paths to the downloaded CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://data.binance.vision/data/futures/um/daily/trades"
        f"/{symbol}/{symbol}-trades"
    )

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    downloaded = []
    for date in tqdm(dates, desc=f"Downloading {symbol} trades"):
        date_str = date.strftime("%Y-%m-%d")
        url = f"{base_url}-{date_str}.zip"
        csv_path = output_dir / f"{symbol}-trades-{date_str}.csv"

        if csv_path.exists():
            downloaded.append(csv_path)
            continue

        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            print(f"  Warning: failed to download {date_str} (HTTP {response.status_code})")
            continue

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            zf.extract(csv_name, output_dir)
            extracted = output_dir / csv_name
            if extracted != csv_path:
                extracted.rename(csv_path)

        downloaded.append(csv_path)

    print(f"Downloaded {len(downloaded)} / {len(dates)} daily files.")
    return downloaded


# ---------------------------------------------------------------------------
# Data loading and standardisation
# ---------------------------------------------------------------------------

def load_raw_trades(filepath: str | Path, venue: str) -> pd.DataFrame:
    """Load raw trade CSV and apply venue-specific column mapping.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    venue : str
        One of "binance", "bybit", "okx".

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised column names.
    """
    column_maps = {
        "binance": BINANCE_COLUMNS,
        "bybit": BYBIT_COLUMNS,
        "okx": OKX_COLUMNS,
    }

    if venue not in column_maps:
        raise ValueError(f"Unknown venue: {venue}. Expected one of {list(column_maps)}")

    df = pd.read_csv(filepath)

    # Binance CSVs sometimes have no header row — columns are positional
    if venue == "binance" and df.columns[0] != "id":
        df.columns = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]

    df = df.rename(columns=column_maps[venue])
    df["venue"] = venue
    return df


def standardise_columns(df: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Ensure consistent column types and formats across venues.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trade DataFrame (already column-renamed).
    venue : str
        Venue name for venue-specific logic.

    Returns
    -------
    pd.DataFrame
        Standardised DataFrame.
    """
    df = df.copy()

    # Ensure numeric types
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["price", "quantity", "timestamp"])

    return df


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to UTC datetime with millisecond precision.

    Handles both Unix millisecond timestamps (Binance, OKX) and
    ISO-format strings (Bybit).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'timestamp' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'timestamp' as a UTC-aware datetime column.
    """
    df = df.copy()

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        # Unix millisecond timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Trade side classification
# ---------------------------------------------------------------------------

def classify_trade_side(df: pd.DataFrame) -> pd.DataFrame:
    """Classify trade side using venue-native field or tick rule.

    For Binance, uses the `is_buyer_maker` field directly.
    For other venues with a `side` field, maps to +1 (buy) / -1 (sell).
    Falls back to the tick rule when neither is available.

    Parameters
    ----------
    df : pd.DataFrame
        Trade DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with a 'trade_sign' column (+1 = buy, -1 = sell).
    """
    df = df.copy()

    if "is_buyer_maker" in df.columns:
        # Binance: is_buyer_maker=True means the buyer was the maker,
        # so the trade was initiated by a seller (taker sells).
        df["trade_sign"] = np.where(df["is_buyer_maker"], -1, 1)
    elif "side" in df.columns:
        side_lower = df["side"].astype(str).str.lower()
        df["trade_sign"] = np.where(side_lower == "buy", 1, -1)
    else:
        # Tick rule: compare price to previous price
        df["trade_sign"] = _apply_tick_rule(df["price"].values)

    return df


def _apply_tick_rule(prices: np.ndarray) -> np.ndarray:
    """Apply the tick rule to classify trade direction.

    If price > previous price → buy (+1).
    If price < previous price → sell (-1).
    If price == previous price → carry forward previous classification.

    Parameters
    ----------
    prices : np.ndarray
        Array of trade prices.

    Returns
    -------
    np.ndarray
        Array of trade signs (+1 or -1).
    """
    signs = np.zeros(len(prices), dtype=np.int8)
    signs[0] = 1  # Default first trade to buy

    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            signs[i] = 1
        elif prices[i] < prices[i - 1]:
            signs[i] = -1
        else:
            signs[i] = signs[i - 1]

    return signs


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived fields: log returns and inter-trade durations.

    Parameters
    ----------
    df : pd.DataFrame
        Trade DataFrame with 'price' and 'timestamp' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - 'log_return': log(price_t / price_{t-1})
        - 'inter_trade_duration_ms': milliseconds between consecutive trades
    """
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    # Inter-trade durations in milliseconds
    df["inter_trade_duration_ms"] = (
        df["timestamp"].diff().dt.total_seconds() * 1000
    )

    return df


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame, venue: str = "") -> dict:
    """Run data quality checks and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Processed trade DataFrame.
    venue : str, optional
        Venue name for reporting.

    Returns
    -------
    dict
        Quality report with counts, date range, gap info, etc.
    """
    report = {
        "venue": venue,
        "total_trades": len(df),
        "date_range_start": df["timestamp"].min(),
        "date_range_end": df["timestamp"].max(),
        "null_prices": int(df["price"].isna().sum()),
        "null_quantities": int(df["quantity"].isna().sum()),
        "null_trade_signs": int(df["trade_sign"].isna().sum()) if "trade_sign" in df.columns else "N/A",
        "unique_dates": df["timestamp"].dt.date.nunique(),
    }

    # Check for timestamp gaps > 1 minute
    if "inter_trade_duration_ms" in df.columns:
        gaps = df["inter_trade_duration_ms"] > 60_000
        report["gaps_over_1min"] = int(gaps.sum())
        report["max_gap_seconds"] = float(df["inter_trade_duration_ms"].max() / 1000)

    # Price statistics
    report["price_min"] = float(df["price"].min())
    report["price_max"] = float(df["price"].max())
    report["price_mean"] = float(df["price"].mean())

    # Trade sign balance
    if "trade_sign" in df.columns:
        buys = (df["trade_sign"] == 1).sum()
        sells = (df["trade_sign"] == -1).sum()
        report["buy_fraction"] = float(buys / len(df))
        report["sell_fraction"] = float(sells / len(df))

    return report


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save processed DataFrame to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        Processed trade data.
    filepath : str or Path
        Output path (should end with .parquet).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False, engine="pyarrow")
    print(f"Saved {len(df):,} trades to {filepath}")


def load_processed(filepath: str | Path) -> pd.DataFrame:
    """Load processed trade data from Parquet.

    Parameters
    ----------
    filepath : str or Path
        Path to the Parquet file.

    Returns
    -------
    pd.DataFrame
        Trade data.
    """
    return pd.read_parquet(filepath, engine="pyarrow")
