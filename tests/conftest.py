"""Shared fixtures for the crypto-venue-entropy test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_trades_df(rng):
    """1000-row synthetic trade DataFrame with known properties.

    Properties:
    - 600ms-spaced timestamps over 10 minutes
    - Brownian walk prices starting at 85000
    - Lognormal quantities (~0.01 BTC)
    - 600 buys / 400 sells (known Shannon H = 0.971 bits)
    - venue = "binance"
    """
    n = 1000
    timestamps = pd.date_range(
        "2026-01-30 12:00:00", periods=n, freq="600ms", tz="UTC"
    )
    prices = 85000.0 + np.cumsum(rng.normal(0, 5, n))
    prices = np.maximum(prices, 1.0)  # ensure positive

    quantities = rng.lognormal(mean=-4.6, sigma=0.5, size=n)

    # 600 buys (+1), 400 sells (-1) -> known 60/40 split
    trade_signs = np.array([1] * 600 + [-1] * 400)
    rng.shuffle(trade_signs)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": quantities,
        "quote_qty": prices * quantities,
        "venue": "binance",
        "trade_sign": trade_signs,
    })

    # Derived fields
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["inter_trade_duration_ms"] = (
        df["timestamp"].diff().dt.total_seconds() * 1000
    )

    return df


@pytest.fixture
def sample_trades_binance_raw():
    """50-row DataFrame with Binance raw column names."""
    n = 50
    base_ts = 1706616000000  # 2024-01-30 12:00:00 UTC in ms
    return pd.DataFrame({
        "id": range(n),
        "price": [f"{85000 + i * 0.5:.2f}" for i in range(n)],
        "qty": [f"{0.01 + i * 0.001:.4f}" for i in range(n)],
        "quoteQty": [f"{850 + i:.2f}" for i in range(n)],
        "time": [base_ts + i * 500 for i in range(n)],
        "isBuyerMaker": [i % 2 == 0 for i in range(n)],
    })


@pytest.fixture
def sample_trades_bybit_raw():
    """50-row DataFrame with Bybit raw column names."""
    n = 50
    base_ts = 1706616000.0  # Unix seconds
    return pd.DataFrame({
        "timestamp": [base_ts + i * 0.5 for i in range(n)],
        "symbol": ["BTCUSDT"] * n,
        "side": ["Buy" if i % 2 == 0 else "Sell" for i in range(n)],
        "size": [f"{0.01 + i * 0.001:.4f}" for i in range(n)],
        "price": [f"{85000 + i * 0.5:.2f}" for i in range(n)],
    })


@pytest.fixture
def sample_entropy_series(rng):
    """100-element entropy series with a known sharp drop at index 50."""
    high = rng.uniform(0.90, 0.99, 50)
    low = rng.uniform(0.30, 0.50, 49)
    return np.concatenate([high, [0.30], low])


@pytest.fixture
def sample_bimodal_prices(rng):
    """2000 prices from a mixture of two Gaussians at 80000 and 85000."""
    mode1 = rng.normal(80000, 200, 1000)
    mode2 = rng.normal(85000, 200, 1000)
    return np.concatenate([mode1, mode2])


@pytest.fixture
def two_venue_dfs(rng):
    """Source and target DataFrames where target lags source by 1 step.

    Useful for testing transfer entropy directionality.
    """
    n = 500
    timestamps = pd.date_range(
        "2026-01-30 12:00:00", periods=n, freq="200ms", tz="UTC"
    )

    source_signs = rng.choice([-1, 0, 1], size=n)
    # Target copies source with 1-step lag
    target_signs = np.zeros(n, dtype=int)
    target_signs[1:] = source_signs[:-1]

    quantities = rng.lognormal(mean=-4.6, sigma=0.5, size=n)

    source_df = pd.DataFrame({
        "timestamp": timestamps,
        "trade_sign": source_signs,
        "quantity": quantities,
        "venue": "binance",
    })
    target_df = pd.DataFrame({
        "timestamp": timestamps,
        "trade_sign": target_signs,
        "quantity": quantities,
        "venue": "bybit",
    })

    return {"source": source_df, "target": target_df}


@pytest.fixture
def sample_net_te_series(rng):
    """50-element net TE series with two known sign flips.

    Structure: 20 positive (Binance leads), 20 negative (Bybit leads),
    10 positive (Binance leads again). Flips occur at indices 20 and 40.
    """
    return pd.Series(
        np.concatenate([
            rng.uniform(0.01, 0.05, 20),
            rng.uniform(-0.05, -0.01, 20),
            rng.uniform(0.01, 0.05, 10),
        ])
    )


@pytest.fixture
def sample_price_series():
    """200-bar price series at 5-min resolution with known trend/reversal.

    Bars 0-99: linear uptrend 100 → 110.
    Bars 100-199: linear downtrend 110 → 100.
    """
    n = 200
    timestamps = pd.date_range(
        "2026-01-30 12:00:00", periods=n, freq="5min", tz="UTC"
    )
    up = np.linspace(100, 110, 100)
    down = np.linspace(110, 100, 100)
    prices = np.concatenate([up, down])
    return pd.Series(prices, index=timestamps)


@pytest.fixture
def sample_signals_df():
    """3 signals at known timestamps with known directions.

    Signal 1: long at bar 10 (price ~105.05)
    Signal 2: short at bar 60 (price ~106.06)
    Signal 3: long at bar 150 (price ~105.05 on downtrend)
    """
    timestamps = pd.date_range(
        "2026-01-30 12:00:00", periods=200, freq="5min", tz="UTC"
    )
    return pd.DataFrame({
        "timestamp": [timestamps[10], timestamps[60], timestamps[150]],
        "signal_type": ["low_entropy", "te_flip", "low_entropy"],
        "direction": [1, -1, 1],
    })
