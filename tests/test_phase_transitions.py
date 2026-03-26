"""Tests for src/phase_transitions.py -- regime detection and classification."""

import numpy as np
import pandas as pd
import pytest

from src.phase_transitions import (
    classify_regime,
    correlation_length,
    detect_entropy_discontinuities,
    order_flow_imbalance,
    realised_volatility,
    susceptibility,
)


# ── realised_volatility ──────────────────────────────────────────────────────


def test_realised_volatility_constant_price():
    prices = pd.Series([100.0] * 500)
    vol = realised_volatility(prices, window=50)

    # Constant price -> zero log returns -> zero volatility
    non_nan = vol.dropna()
    assert (non_nan == 0.0).all()


def test_realised_volatility_output_length():
    prices = pd.Series(np.linspace(100, 110, 200))
    vol = realised_volatility(prices, window=50)
    assert len(vol) == 200


def test_realised_volatility_positive():
    rng = np.random.default_rng(42)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 500)))
    prices = prices.clip(lower=1.0)
    vol = realised_volatility(prices, window=50)

    non_nan = vol.dropna()
    assert (non_nan >= 0).all()


# ── order_flow_imbalance ─────────────────────────────────────────────────────


def test_order_flow_imbalance_all_buys():
    signs = np.ones(500)
    imb = order_flow_imbalance(signs, window=50)

    # After window fills, should converge to +1.0
    assert imb[-1] == pytest.approx(1.0, abs=1e-10)


def test_order_flow_imbalance_balanced():
    signs = np.array([1, -1] * 250, dtype=float)
    imb = order_flow_imbalance(signs, window=100)

    # Alternating: rolling mean should be ~0
    non_nan = imb[~np.isnan(imb)]
    assert all(abs(v) < 0.05 for v in non_nan)


# ── susceptibility ───────────────────────────────────────────────────────────


def test_susceptibility_constant_imbalance():
    imb = np.full(500, 0.5)
    susc = susceptibility(imb, window=50)

    non_nan = susc[~np.isnan(susc)]
    assert all(v == pytest.approx(0.0, abs=1e-10) for v in non_nan)


def test_susceptibility_oscillating():
    imb = np.array([0.5, -0.5] * 250, dtype=float)
    susc = susceptibility(imb, window=100)

    non_nan = susc[~np.isnan(susc)]
    assert all(v > 0 for v in non_nan)


# ── correlation_length ───────────────────────────────────────────────────────


def test_correlation_length_white_noise():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 1, 5000)

    corr_len = correlation_length(returns, window=1000, max_lag=50)

    # White noise: tau_int should be close to 0.5 (the base term)
    valid = corr_len[~np.isnan(corr_len)]
    mean_tau = np.mean(valid)
    assert mean_tau < 2.0  # white noise -> low correlation length


def test_correlation_length_persistent_process():
    rng = np.random.default_rng(42)
    n = 5000
    # AR(1) with phi=0.9 -> persistent
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)

    corr_len = correlation_length(x, window=1000, max_lag=50, use_absolute=False)

    valid = corr_len[~np.isnan(corr_len)]
    mean_tau_persistent = np.mean(valid)

    # Compare with white noise
    white = rng.normal(0, 1, n)
    corr_len_white = correlation_length(white, window=1000, max_lag=50, use_absolute=False)
    valid_white = corr_len_white[~np.isnan(corr_len_white)]
    mean_tau_white = np.mean(valid_white)

    assert mean_tau_persistent > mean_tau_white


def test_correlation_length_output_length():
    returns = np.random.default_rng(42).normal(0, 1, 2000)
    window = 500
    corr_len = correlation_length(returns, window=window)

    assert len(corr_len) == 2000
    # First `window` values should be NaN
    assert all(np.isnan(corr_len[:window]))


# ── detect_entropy_discontinuities ───────────────────────────────────────────


def test_detect_entropy_discontinuities_with_known_jump(sample_entropy_series):
    disc, deriv = detect_entropy_discontinuities(sample_entropy_series, threshold=2.0)

    assert disc.dtype == bool
    assert len(disc) == 100
    assert len(deriv) == 100
    # The sharp drop at index 50 should be detected
    assert disc[50]


def test_detect_entropy_discontinuities_smooth():
    smooth = np.full(100, 0.95)
    disc, deriv = detect_entropy_discontinuities(smooth, threshold=2.0)

    assert not disc.any()


# ── classify_regime ──────────────────────────────────────────────────────────


def test_classify_regime_valid_labels():
    rng = np.random.default_rng(42)
    n = 1000
    vol = rng.uniform(0, 1, n)
    ent = rng.uniform(0, 1, n)
    corr = rng.uniform(0, 10, n)

    regimes = classify_regime(vol, ent, corr)

    valid_labels = {"hot", "cold", "critical", "transitional", "unknown"}
    assert all(r in valid_labels for r in regimes)


def test_classify_regime_nan_handling():
    n = 100
    vol = np.full(n, np.nan)
    ent = np.full(n, np.nan)
    corr = np.full(n, np.nan)

    # Need at least some valid values for percentile computation
    vol[:50] = 0.5
    ent[:50] = 0.5
    corr[:50] = 5.0

    regimes = classify_regime(vol, ent, corr)

    # NaN inputs should be classified as "unknown"
    assert all(regimes[i] == "unknown" for i in range(50, 100))
