"""Smoke tests for src/visualisation.py -- verify all plot functions run without error."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualisation import (
    plot_correlation_length_timeseries,
    plot_cross_correlation,
    plot_dwell_time_distribution,
    plot_entropy_comparison,
    plot_entropy_discontinuities,
    plot_entropy_timeseries,
    plot_entropy_vs_volatility,
    plot_free_energy_landscape,
    plot_intraday_pattern,
    plot_kramers_test,
    plot_metastable_levels,
    plot_phase_observables,
    plot_price_overlay,
    plot_regime_timeline,
    plot_support_resistance_comparison,
    plot_trade_sign_acf,
    plot_transfer_entropy,
    plot_transfer_entropy_heatmap,
    set_style,
)


@pytest.fixture
def timestamps():
    return pd.date_range("2026-01-30 12:00", periods=50, freq="5min", tz="UTC")


@pytest.fixture
def prices(timestamps):
    rng = np.random.default_rng(42)
    return pd.Series(85000 + np.cumsum(rng.normal(0, 10, 50)), index=timestamps)


def _assert_fig(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── Basic ────────────────────────────────────────────────────────────────────


def test_set_style():
    set_style()  # Should not raise


# ── Price / Microstructure ───────────────────────────────────────────────────


def test_plot_price_overlay():
    ts = pd.date_range("2026-01-30", periods=20, freq="1min", tz="UTC")
    dfs = {
        "binance": pd.DataFrame({"timestamp": ts, "price": np.linspace(85000, 85100, 20)}),
        "bybit": pd.DataFrame({"timestamp": ts, "price": np.linspace(85010, 85110, 20)}),
    }
    _assert_fig(plot_price_overlay(dfs))


def test_plot_trade_sign_acf():
    acf_dict = {
        "binance": np.linspace(1, 0, 50),
        "bybit": np.linspace(1, 0.1, 50),
    }
    _assert_fig(plot_trade_sign_acf(acf_dict))


def test_plot_cross_correlation():
    lags = list(range(-5, 6))
    df = pd.DataFrame({"A → B": np.random.default_rng(42).normal(0, 0.01, 11)}, index=lags)
    df.index.name = "lag"
    _assert_fig(plot_cross_correlation(df))


def test_plot_intraday_pattern():
    hourly = {"binance": pd.Series(np.random.default_rng(42).uniform(100, 500, 24), index=range(24))}
    _assert_fig(plot_intraday_pattern(hourly))


# ── Entropy ──────────────────────────────────────────────────────────────────


def test_plot_entropy_timeseries(timestamps):
    entropy_dict = {"binance": pd.Series(np.random.default_rng(42).uniform(0.8, 1.0, 50), index=timestamps)}
    _assert_fig(plot_entropy_timeseries(entropy_dict))


def test_plot_entropy_timeseries_with_price(timestamps, prices):
    entropy_dict = {"binance": pd.Series(np.random.default_rng(42).uniform(0.8, 1.0, 50), index=timestamps)}
    _assert_fig(plot_entropy_timeseries(entropy_dict, prices=prices))


def test_plot_entropy_comparison():
    ts = pd.date_range("2026-01-30", periods=20, freq="5min", tz="UTC")
    entropy_data = {
        "5min": {
            "binance": pd.DataFrame({"timestamp": ts, "normalised_entropy": np.random.default_rng(42).uniform(0.8, 1.0, 20)}),
        },
    }
    _assert_fig(plot_entropy_comparison(entropy_data))


# ── Transfer Entropy ─────────────────────────────────────────────────────────


def test_plot_transfer_entropy():
    ts = pd.date_range("2026-01-30", periods=20, freq="5min", tz="UTC")
    te_fwd = pd.DataFrame({"timestamp": ts, "te": np.random.default_rng(42).uniform(0, 0.1, 20)})
    te_rev = pd.DataFrame({"timestamp": ts, "te": np.random.default_rng(43).uniform(0, 0.1, 20)})
    _assert_fig(plot_transfer_entropy(te_fwd, te_rev))


def test_plot_entropy_vs_volatility():
    ts = pd.date_range("2026-01-30", periods=30, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    entropy_df = pd.DataFrame({"timestamp": ts, "normalised_entropy": rng.uniform(0.8, 1.0, 30)})
    vol = pd.Series(rng.uniform(0.001, 0.01, 30), index=ts)
    _assert_fig(plot_entropy_vs_volatility(entropy_df, vol, "binance"))


def test_plot_transfer_entropy_heatmap():
    te_matrix = pd.DataFrame(
        [[0.0, 0.05], [0.03, 0.0]],
        index=["Binance", "Bybit"],
        columns=["Binance", "Bybit"],
    )
    _assert_fig(plot_transfer_entropy_heatmap(te_matrix))


# ── Phase Transitions ────────────────────────────────────────────────────────


def test_plot_free_energy_landscape():
    rng = np.random.default_rng(42)
    price_grid = np.linspace(80000, 86000, 50)
    fe_2d = rng.uniform(0, 5, (3, 50))
    times = pd.date_range("2026-01-30", periods=3, freq="1h", tz="UTC")
    _assert_fig(plot_free_energy_landscape(price_grid, fe_2d, times))


def test_plot_regime_timeline(timestamps, prices):
    labels = np.random.default_rng(42).choice(["hot", "cold", "critical", "transitional"], size=50)
    regimes = pd.Series(labels, index=timestamps)
    _assert_fig(plot_regime_timeline(regimes, prices, smooth_window=1))


def test_plot_phase_observables(timestamps):
    rng = np.random.default_rng(42)
    n = 50
    _assert_fig(plot_phase_observables(
        timestamps=timestamps,
        temperature=rng.uniform(0, 0.01, n),
        order_parameter=rng.uniform(-1, 1, n),
        susceptibility=rng.uniform(0, 0.5, n),
    ))


def test_plot_correlation_length_timeseries(timestamps):
    corr_len = np.random.default_rng(42).uniform(0.5, 10, 50)
    _assert_fig(plot_correlation_length_timeseries(timestamps, corr_len))


def test_plot_entropy_discontinuities(timestamps):
    rng = np.random.default_rng(42)
    n = 50
    entropy = rng.uniform(0.8, 1.0, n)
    derivative = np.diff(entropy, prepend=entropy[0])
    disc = np.abs(derivative) > 0.05
    _assert_fig(plot_entropy_discontinuities(timestamps, entropy, derivative, disc))


# ── Metastability ────────────────────────────────────────────────────────────


def test_plot_metastable_levels(timestamps, prices):
    levels_df = pd.DataFrame({
        "timestamp": timestamps[:10],
        "price_level": np.linspace(84500, 85500, 10),
        "well_depth": np.random.default_rng(42).uniform(1, 5, 10),
    })
    _assert_fig(plot_metastable_levels(levels_df, prices))


def test_plot_dwell_time_distribution():
    dwell = {80000.0: list(np.random.default_rng(42).exponential(100, 50))}
    _assert_fig(plot_dwell_time_distribution(dwell))


def test_plot_dwell_time_distribution_empty():
    _assert_fig(plot_dwell_time_distribution({80000.0: []}))


def test_plot_kramers_test():
    result = {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "y": np.array([2.0, 3.5, 5.0, 6.5, 8.0]),
        "slope": 1.5,
        "intercept": 0.5,
        "correlation": 0.99,
    }
    _assert_fig(plot_kramers_test(result))


def test_plot_kramers_test_empty():
    result = {
        "x": np.array([]),
        "y": np.array([]),
        "slope": np.nan,
        "intercept": np.nan,
        "correlation": np.nan,
    }
    _assert_fig(plot_kramers_test(result))


def test_plot_support_resistance_comparison(timestamps, prices):
    physics_levels = pd.DataFrame({
        "timestamp": timestamps[:5],
        "price_level": [84500, 84800, 85000, 85200, 85500],
    })
    traditional = [84000.0, 85000.0, 86000.0]
    _assert_fig(plot_support_resistance_comparison(physics_levels, traditional, prices))
