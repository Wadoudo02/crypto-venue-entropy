"""Tests for src/microstructure.py -- ACF, cross-correlation, arrival rate, size distribution."""

import numpy as np
import pandas as pd
import pytest

from src.microstructure import (
    compute_trade_arrival_rate,
    cross_venue_correlation,
    trade_sign_autocorrelation,
    trade_size_distribution,
)


# ── trade_sign_autocorrelation ───────────────────────────────────────────────


def test_trade_sign_acf_alternating():
    signs = np.array([1, -1] * 200, dtype=float)
    acf_vals = trade_sign_autocorrelation(signs, max_lag=10)

    assert acf_vals[0] == pytest.approx(1.0, abs=1e-6)
    assert acf_vals[1] == pytest.approx(-1.0, abs=0.05)


def test_trade_sign_acf_output_length():
    signs = np.array([1, -1, 1, 1, -1] * 100, dtype=float)
    max_lag = 20
    acf_vals = trade_sign_autocorrelation(signs, max_lag=max_lag)
    assert len(acf_vals) == max_lag + 1


def test_trade_sign_acf_random():
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=500)
    acf_vals = trade_sign_autocorrelation(signs, max_lag=20)

    assert acf_vals[0] == pytest.approx(1.0, abs=1e-6)
    # Random signs: ACF should decay quickly to near zero
    assert all(abs(acf_vals[k]) < 0.15 for k in range(5, 21))


# ── cross_venue_correlation ──────────────────────────────────────────────────


def test_cross_venue_correlation_identical():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 200))

    result = cross_venue_correlation({"A": series, "B": series}, max_lag=5)

    assert result.loc[0].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_cross_venue_correlation_independent():
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(0, 1, 1000))
    s2 = pd.Series(rng.normal(0, 1, 1000))

    result = cross_venue_correlation({"A": s1, "B": s2}, max_lag=5)

    assert all(abs(result["A → B"]) < 0.1)


def test_cross_venue_correlation_output_structure():
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(0, 1, 200))
    s2 = pd.Series(rng.normal(0, 1, 200))
    s3 = pd.Series(rng.normal(0, 1, 200))

    result = cross_venue_correlation({"A": s1, "B": s2, "C": s3}, max_lag=5)

    # 3 venues -> 3 pairs: A→B, A→C, B→C
    assert len(result.columns) == 3
    assert list(result.index) == list(range(-5, 6))


# ── compute_trade_arrival_rate ───────────────────────────────────────────────


def test_compute_trade_arrival_rate(sample_trades_df):
    rate = compute_trade_arrival_rate(sample_trades_df, freq="1min")

    assert isinstance(rate, pd.Series)
    assert isinstance(rate.index, pd.DatetimeIndex)
    assert rate.sum() == 1000  # total trades


# ── trade_size_distribution ──────────────────────────────────────────────────


def test_trade_size_distribution_keys(sample_trades_df):
    dist = trade_size_distribution(sample_trades_df)

    expected_keys = {"mean", "median", "std", "skew", "kurtosis", "p95", "p99", "max"}
    assert expected_keys == set(dist.keys())


def test_trade_size_distribution_values(sample_trades_df):
    dist = trade_size_distribution(sample_trades_df)

    assert dist["mean"] > 0
    assert dist["median"] > 0
    assert dist["max"] >= dist["p99"] >= dist["p95"] >= dist["median"]
