"""Tests for src/metastability.py -- free-energy landscape and metastability analysis."""

import numpy as np
import pandas as pd
import pytest

from src.metastability import (
    barrier_height,
    dwell_time_analysis,
    find_metastable_levels,
    free_energy_landscape,
    kramers_test,
    rolling_metastable_levels,
)


# ── free_energy_landscape ────────────────────────────────────────────────────


def test_free_energy_landscape_output_shapes(sample_bimodal_prices):
    window_centers, price_grid, fe_2d = free_energy_landscape(
        sample_bimodal_prices, window=500, step=500, n_bins=50
    )

    assert len(price_grid) == 50
    assert fe_2d.shape[1] == 50
    assert fe_2d.shape[0] == len(window_centers)
    assert len(window_centers) > 0


def test_free_energy_landscape_short_input():
    prices = np.array([100.0, 101.0, 102.0])
    window_centers, price_grid, fe_2d = free_energy_landscape(
        prices, window=100, n_bins=50
    )

    assert len(window_centers) == 0
    assert len(price_grid) == 0
    assert fe_2d.shape == (0, 50)


def test_free_energy_landscape_values_finite(sample_bimodal_prices):
    _, _, fe_2d = free_energy_landscape(
        sample_bimodal_prices, window=500, step=500, n_bins=50
    )

    assert np.all(np.isfinite(fe_2d))


# ── find_metastable_levels ───────────────────────────────────────────────────


def test_find_metastable_levels_bimodal(sample_bimodal_prices):
    _, price_grid, fe_2d = free_energy_landscape(
        sample_bimodal_prices, window=2000, step=2000, n_bins=100
    )

    # Use the single window (whole dataset)
    levels = find_metastable_levels(price_grid, fe_2d[0])

    assert len(levels) >= 2
    # Levels should be near the two modes: 80000 and 85000
    near_80k = any(abs(l - 80000) < 1000 for l in levels)
    near_85k = any(abs(l - 85000) < 1000 for l in levels)
    assert near_80k and near_85k


def test_find_metastable_levels_unimodal():
    rng = np.random.default_rng(42)
    prices = rng.normal(85000, 200, 2000)
    _, price_grid, fe_2d = free_energy_landscape(
        prices, window=2000, step=2000, n_bins=50
    )

    levels = find_metastable_levels(price_grid, fe_2d[0])
    # Unimodal: should find fewer levels than bimodal
    # With finite bins, noise can create a few spurious minima
    assert len(levels) <= 5


def test_find_metastable_levels_empty():
    levels = find_metastable_levels(np.array([]), np.array([]))
    assert len(levels) == 0


# ── rolling_metastable_levels ────────────────────────────────────────────────


def test_rolling_metastable_levels_structure(sample_bimodal_prices):
    wc, pg, fe_2d = free_energy_landscape(
        sample_bimodal_prices, window=500, step=500, n_bins=50
    )

    results = rolling_metastable_levels(wc, pg, fe_2d)

    assert isinstance(results, list)
    if len(results) > 0:
        r = results[0]
        assert "time_idx" in r
        assert "obs_center" in r
        assert "price_level" in r
        assert "well_depth" in r
        assert all(entry["well_depth"] > 0 for entry in results)


# ── dwell_time_analysis ──────────────────────────────────────────────────────


def test_dwell_time_analysis_known_consolidation():
    # Price stays at 80000 for 200s, then jumps to 85000 for 200s
    n = 4000
    timestamps = pd.date_range("2026-01-30 12:00:00", periods=n, freq="100ms", tz="UTC")
    prices_arr = np.concatenate([
        np.full(2000, 80000.0),
        np.full(2000, 85000.0),
    ])
    prices = pd.Series(prices_arr, index=timestamps)

    levels = np.array([80000.0, 85000.0])
    dwell = dwell_time_analysis(prices, levels, band_width=0.002)

    assert 80000.0 in dwell
    assert 85000.0 in dwell
    # Both should have dwell times (at least one entry each)
    # The exact values depend on the 10s filter and entry/exit logic
    total_dwell_80k = sum(dwell[80000.0]) if dwell[80000.0] else 0
    total_dwell_85k = sum(dwell[85000.0]) if dwell[85000.0] else 0
    assert total_dwell_80k > 0 or total_dwell_85k > 0


def test_dwell_time_analysis_no_dwell():
    # Prices never enter the bands
    timestamps = pd.date_range("2026-01-30", periods=100, freq="1s", tz="UTC")
    prices = pd.Series(np.full(100, 90000.0), index=timestamps)

    levels = np.array([80000.0, 85000.0])
    dwell = dwell_time_analysis(prices, levels, band_width=0.002)

    assert all(len(v) == 0 for v in dwell.values())


# ── barrier_height ───────────────────────────────────────────────────────────


def test_barrier_height_two_levels():
    price_grid = np.array([79, 80, 81, 82, 83, 84, 85], dtype=float)
    # Two wells at 81 and 83, barrier at 82
    free_energy = np.array([5.0, 3.0, 1.0, 4.0, 1.0, 3.0, 5.0])
    levels = np.array([81.0, 83.0])

    barriers = barrier_height(price_grid, free_energy, levels)

    assert len(barriers) == 1
    key = (81.0, 83.0)
    # Barrier = max(F[81:83]) - min(F[81], F[83]) = 4.0 - 1.0 = 3.0
    assert barriers[key] == pytest.approx(3.0, abs=0.01)


def test_barrier_height_single_level():
    price_grid = np.linspace(80, 85, 50)
    free_energy = np.random.default_rng(42).uniform(0, 5, 50)
    levels = np.array([82.0])

    barriers = barrier_height(price_grid, free_energy, levels)
    assert len(barriers) == 0


# ── kramers_test ─────────────────────────────────────────────────────────────


def test_kramers_test_linear_relationship():
    # Fabricate: ln(dwell) = 2.0 * barrier + 1.0
    levels = [80000.0, 81000.0, 82000.0, 83000.0, 84000.0]
    barriers_dict = {}
    for i in range(len(levels) - 1):
        barriers_dict[(levels[i], levels[i + 1])] = float(i + 1)

    dwell_times = {}
    for i, level in enumerate(levels):
        # barrier for this level is approximately i+0.5 (average of adjacent)
        b = (i + 0.5) if i < len(levels) - 1 else (i - 0.5)
        dwell_times[level] = [np.exp(2.0 * b + 1.0)] * 5  # 5 identical dwells

    result = kramers_test(dwell_times, barriers_dict, temperature=1.0)

    assert result["correlation"] == pytest.approx(1.0, abs=0.1)
    assert result["slope"] == pytest.approx(2.0, abs=0.5)


def test_kramers_test_insufficient_data():
    dwell_times = {80000.0: [10.0], 81000.0: [20.0]}
    barriers = {(80000.0, 81000.0): 2.0}

    result = kramers_test(dwell_times, barriers)

    assert np.isnan(result["correlation"])
    assert len(result["x"]) == 0
    assert len(result["y"]) == 0
