"""
Data loading, cleaning, alignment, and processing utilities for
cross-venue crypto trade data.

Supports Binance, Bybit, and OKX perpetual futures plus Coinbase spot trade data.
"""

import gzip
import io
import json
import time
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

# Coinbase BTC-USD spot — note: real USD, not USDT.
# Price difference between USD and USDT is negligible for microstructure analysis.
COINBASE_COLUMNS = {
    "trade_id": "trade_id",
    "price": "price",
    "size": "quantity",
    "time": "timestamp",
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


def download_okx_trades(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
) -> list[Path]:
    """Download daily trade data from OKX public data repository.

    Tries the bulk download endpoint first (monthly ZIP files containing
    all futures trades). Falls back to the REST API if bulk download fails.

    Parameters
    ----------
    symbol : str
        Instrument ID, e.g. "BTC-USDT-SWAP".
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format (inclusive).
    output_dir : str or Path
        Directory to save filtered CSV files.

    Returns
    -------
    list[Path]
        Paths to the downloaded CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    downloaded = []
    bulk_failed = False

    for date in tqdm(dates, desc=f"Downloading OKX {symbol} trades"):
        date_str = date.strftime("%Y-%m-%d")
        csv_path = output_dir / f"{symbol}-trades-{date_str}.csv"

        if csv_path.exists():
            downloaded.append(csv_path)
            continue

        if not bulk_failed:
            yyyymm = date.strftime("%Y%m")
            url = (
                f"https://www.okx.com/cdn/okex/traderecords/trades/monthly/"
                f"{yyyymm}/allfuture-trades-{date_str}.zip"
            )

            try:
                response = requests.get(url, timeout=120)
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        # Find CSV(s) inside the ZIP and filter for our instrument
                        for name in zf.namelist():
                            if name.endswith(".csv"):
                                with zf.open(name) as f:
                                    df = pd.read_csv(f)
                                # Filter for the target instrument
                                inst_col = _find_instrument_column(df)
                                if inst_col:
                                    df = df[df[inst_col].astype(str).str.upper() == symbol.upper()]
                                if len(df) > 0:
                                    df.to_csv(csv_path, index=False)
                                    downloaded.append(csv_path)
                                    break
                        else:
                            print(f"  Warning: no data for {symbol} in ZIP for {date_str}")
                        continue
                    # If we get here, the ZIP was valid — continue to next date
                else:
                    print(f"  Bulk download failed for {date_str} (HTTP {response.status_code}), trying REST API...")
                    bulk_failed = True
            except Exception as e:
                print(f"  Bulk download error for {date_str}: {e}, trying REST API...")
                bulk_failed = True

        # Fallback: REST API — only practical for very recent data
        # (the API returns 100 trades per request, so going back more than
        # ~1 day is impractical)
        if bulk_failed or not csv_path.exists():
            days_ago = (datetime.utcnow() - date).days
            if days_ago <= 1:
                df = _download_okx_rest(symbol, date_str)
                if df is not None and len(df) > 0:
                    df.to_csv(csv_path, index=False)
                    downloaded.append(csv_path)
                else:
                    print(f"  Warning: no OKX data for {date_str}")
            else:
                print(f"  Warning: OKX data for {date_str} unavailable "
                      f"(bulk CDN not yet published; REST API impractical for {days_ago}-day-old data)")

    print(f"Downloaded {len(downloaded)} / {len(dates)} daily files.")
    return downloaded


def _find_instrument_column(df: pd.DataFrame) -> str | None:
    """Find the instrument/symbol column in an OKX DataFrame."""
    for col in ["instrument_id", "instId", "instrument", "symbol"]:
        if col in df.columns:
            return col
    return None


def _download_okx_rest(inst_id: str, date_str: str) -> pd.DataFrame | None:
    """Download trades for a single day via OKX REST API.

    Parameters
    ----------
    inst_id : str
        Instrument ID, e.g. "BTC-USDT-SWAP".
    date_str : str
        Date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame or None
        Trade data, or None if no data retrieved.
    """
    base_url = "https://www.okx.com/api/v5/market/history-trades"
    all_trades = []
    after = ""
    day_start = datetime.strptime(date_str, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = int(day_end.timestamp() * 1000)

    for _ in range(5000):  # safety limit
        params = {"instId": inst_id, "limit": "100"}
        if after:
            params["after"] = after

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json().get("data", [])
            if not data:
                break

            for trade in data:
                ts = int(trade["ts"])
                if ts < start_ms:
                    return pd.DataFrame(all_trades) if all_trades else None
                if ts < end_ms:
                    all_trades.append(trade)

            after = data[-1]["tradeId"]
            time.sleep(0.1)  # respect rate limits
        except Exception:
            break

    return pd.DataFrame(all_trades) if all_trades else None


def download_bybit_trades(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
) -> list[Path]:
    """Download daily trade data from Bybit public data repository.

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

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    downloaded = []
    for date in tqdm(dates, desc=f"Downloading Bybit {symbol} trades"):
        date_str = date.strftime("%Y-%m-%d")
        csv_path = output_dir / f"{symbol}-trades-{date_str}.csv"

        if csv_path.exists():
            downloaded.append(csv_path)
            continue

        url = f"https://public.bybit.com/trading/{symbol}/{symbol}{date_str}.csv.gz"

        try:
            response = requests.get(url, timeout=120)
            if response.status_code != 200:
                print(f"  Warning: failed to download {date_str} (HTTP {response.status_code})")
                continue

            # Decompress gzipped CSV
            csv_bytes = gzip.decompress(response.content)
            csv_path.write_bytes(csv_bytes)
            downloaded.append(csv_path)
        except Exception as e:
            print(f"  Warning: error downloading {date_str}: {e}")
            continue

    print(f"Downloaded {len(downloaded)} / {len(dates)} daily files.")
    return downloaded


def download_coinbase_trades(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
) -> list[Path]:
    """Download trade data from Coinbase REST API.

    Coinbase provides BTC-USD spot trades (real USD, not USDT).
    The API returns trades in reverse chronological order, paginated
    via the `after` cursor parameter.

    Parameters
    ----------
    symbol : str
        Product ID, e.g. "BTC-USD".
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

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # exclusive end

    base_url = f"https://api.exchange.coinbase.com/products/{symbol}/trades"
    all_trades = []
    after = None
    page_count = 0

    pbar = tqdm(desc=f"Downloading Coinbase {symbol} trades", unit=" trades")

    while True:
        params = {"limit": 1000}
        if after is not None:
            params["after"] = after

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  Warning: Coinbase API returned HTTP {resp.status_code}")
                break

            trades = resp.json()
            if not trades:
                break

            for trade in trades:
                trade_time = datetime.fromisoformat(
                    trade["time"].replace("Z", "+00:00")
                )
                trade_dt = trade_time.replace(tzinfo=None)

                if trade_dt < start:
                    # We've gone past our date range (API returns reverse chronological)
                    # Save what we have and stop
                    pbar.close()
                    return _save_coinbase_daily(all_trades, start, end, output_dir, symbol)

                if trade_dt < end:
                    all_trades.append(trade)
                    pbar.update(1)

            # Pagination: 'after' cursor from response headers
            after = resp.headers.get("cb-after")
            if after is None:
                break

            page_count += 1
            time.sleep(0.1)  # respect rate limits

        except Exception as e:
            print(f"  Warning: Coinbase API error: {e}")
            break

    pbar.close()
    return _save_coinbase_daily(all_trades, start, end, output_dir, symbol)


def _save_coinbase_daily(
    trades: list[dict],
    start: datetime,
    end: datetime,
    output_dir: Path,
    symbol: str,
) -> list[Path]:
    """Split Coinbase trades into daily CSV files.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts from the Coinbase API.
    start : datetime
        Start of date range.
    end : datetime
        End of date range (exclusive).
    output_dir : Path
        Directory to save CSV files.
    symbol : str
        Product ID for file naming.

    Returns
    -------
    list[Path]
        Paths to saved CSV files.
    """
    if not trades:
        print("No Coinbase trades downloaded.")
        return []

    df = pd.DataFrame(trades)
    df["time_parsed"] = pd.to_datetime(df["time"])
    df["date"] = df["time_parsed"].dt.date

    downloaded = []
    n_days = (end - start).days
    for i in range(n_days):
        date = (start + timedelta(days=i)).date()
        csv_path = output_dir / f"{symbol}-trades-{date}.csv"

        if csv_path.exists():
            downloaded.append(csv_path)
            continue

        day_df = df[df["date"] == date].drop(columns=["time_parsed", "date"])
        if len(day_df) > 0:
            day_df.to_csv(csv_path, index=False)
            downloaded.append(csv_path)

    print(f"Downloaded {len(downloaded)} / {n_days} daily files ({len(df):,} total trades).")
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
        One of "binance", "bybit", "okx", "coinbase".

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised column names.
    """
    column_maps = {
        "binance": BINANCE_COLUMNS,
        "bybit": BYBIT_COLUMNS,
        "okx": OKX_COLUMNS,
        "coinbase": COINBASE_COLUMNS,
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

    # Compute quote_qty if not present (Bybit, OKX, Coinbase)
    if "quote_qty" not in df.columns and "quote_quantity" not in df.columns:
        df["quote_qty"] = df["price"] * df["quantity"]

    # Drop rows with missing critical fields
    df = df.dropna(subset=["price", "quantity", "timestamp"])

    return df


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to UTC datetime with millisecond precision.

    Handles Unix millisecond timestamps (Binance, OKX), Unix second
    timestamps with fractional part (Bybit), and ISO-format strings
    (Coinbase).

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
        sample = float(df["timestamp"].iloc[0])
        if sample > 1e12:
            # Unix millisecond timestamps (Binance, OKX)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            # Unix second timestamps with fractional part (Bybit)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
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
