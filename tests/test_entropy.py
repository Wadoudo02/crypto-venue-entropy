"""Tests for src/entropy.py -- Shannon entropy, transfer entropy, mutual information."""

import numpy as np
import pandas as pd
import pytest

from src.entropy import (
    mutual_information,
    rolling_shannon_entropy,
    rolling_shannon_entropy_volume_weighted,
    rolling_transfer_entropy,
    shannon_entropy,
    transfer_entropy,
)


@pytest.fixture(scope="module", autouse=True)
def _warm_up_numba():
    """Pre-compile numba JIT functions so individual tests stay fast."""
    transfer_entropy(np.array([1, -1, 0, 1, -1]), np.array([0, 1, -1, 1, 0]), 1)


# ── Shannon Entropy ──────────────────────────────────────────────────────────


def test_shannon_entropy_all_buys():
    assert shannon_entropy(np.ones(100)) == 0.0


def test_shannon_entropy_all_sells():
    assert shannon_entropy(-np.ones(100)) == 0.0


def test_shannon_entropy_balanced():
    signs = np.array([1, -1] * 50)
    assert shannon_entropy(signs) == pytest.approx(1.0, abs=1e-10)


def test_shannon_entropy_known_ratio():
    signs = np.array([1] * 75 + [-1] * 25)
    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert shannon_entropy(signs) == pytest.approx(expected, abs=1e-6)


def test_shannon_entropy_empty():
    assert shannon_entropy(np.array([])) == 0.0


# ── Rolling Shannon Entropy ──────────────────────────────────────────────────


def test_rolling_shannon_entropy_output_columns(sample_trades_df):
    result = rolling_shannon_entropy(sample_trades_df, window="1min")

    expected_cols = {"timestamp", "entropy", "normalised_entropy", "buy_fraction", "trade_count"}
    assert expected_cols == set(result.columns)
    assert len(result) > 0


def test_rolling_shannon_entropy_values_in_range(sample_trades_df):
    result = rolling_shannon_entropy(sample_trades_df, window="1min")

    assert (result["entropy"] >= 0).all()
    assert (result["entropy"] <= 1).all()
    assert (result["buy_fraction"] >= 0).all()
    assert (result["buy_fraction"] <= 1).all()


# ── Volume-Weighted Shannon Entropy ──────────────────────────────────────────


def test_rolling_shannon_entropy_volume_weighted_differs():
    rng = np.random.default_rng(99)
    n = 500
    timestamps = pd.date_range("2026-01-30 12:00:00", periods=n, freq="200ms", tz="UTC")
    # Interleave buys and sells so each window has a mix
    signs = np.array([1, 1, 1, -1, -1] * 100)
    # Buys have 10x larger size than sells -> volume-weighted buy_fraction >> count-based
    quantities = np.where(signs == 1, rng.uniform(0.1, 0.2, n), rng.uniform(0.01, 0.02, n))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "trade_sign": signs,
        "quantity": quantities,
    })

    unweighted = rolling_shannon_entropy(df, window="10s")
    weighted = rolling_shannon_entropy_volume_weighted(df, window="10s")

    assert len(weighted) > 0 and len(unweighted) > 0
    # Volume weighting should produce different entropy values
    min_len = min(len(weighted), len(unweighted))
    assert not np.allclose(
        unweighted["entropy"].values[:min_len],
        weighted["entropy"].values[:min_len],
    )


# ── Transfer Entropy ─────────────────────────────────────────────────────────


def test_transfer_entropy_independent_sequences():
    rng = np.random.default_rng(42)
    source = rng.choice([-1, 0, 1], size=2000)
    target = rng.choice([-1, 0, 1], size=2000)

    te = transfer_entropy(source, target, history_length=1)
    assert te < 0.05


def test_transfer_entropy_coupled_sequences():
    rng = np.random.default_rng(42)
    n = 2000
    source = rng.choice([-1, 0, 1], size=n)
    target = np.zeros(n, dtype=int)
    target[1:] = source[:-1]  # target copies source with 1-step lag

    te_forward = transfer_entropy(source, target, history_length=1)
    te_reverse = transfer_entropy(target, source, history_length=1)

    assert te_forward > 0.1  # significant forward TE
    assert te_forward > te_reverse * 2  # clear asymmetry


def test_transfer_entropy_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        transfer_entropy(np.array([1, -1, 0]), np.array([1, -1]), 1)


def test_transfer_entropy_short_sequence():
    result = transfer_entropy(np.array([1]), np.array([1]), history_length=2)
    assert result == 0.0


# ── Mutual Information ───────────────────────────────────────────────────────


def test_mutual_information_identical():
    x = np.array([1, -1, 0, 1, -1, 0, 1, 0, -1, 1] * 100)
    mi = mutual_information(x, x)
    assert mi > 0


def test_mutual_information_independent():
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 0, 1], size=10000)
    y = rng.choice([-1, 0, 1], size=10000)

    mi = mutual_information(x, y)
    assert mi < 0.02


def test_mutual_information_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        mutual_information(np.array([1, -1]), np.array([1]))


# ── Rolling Transfer Entropy ─────────────────────────────────────────────────


def test_rolling_transfer_entropy_output(two_venue_dfs):
    result = rolling_transfer_entropy(
        source_df=two_venue_dfs["source"],
        target_df=two_venue_dfs["target"],
        bin_freq="1s",
        window="30s",
        step="10s",
        history_length=1,
    )

    assert "timestamp" in result.columns
    assert "te" in result.columns
    assert "n_bins" in result.columns
    if len(result) > 0:
        assert (result["te"] >= 0).all()
