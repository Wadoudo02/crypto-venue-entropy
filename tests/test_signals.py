"""Tests for src/signals.py — trading signal generation functions."""

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    low_entropy_signal,
    te_leadership_flip,
    acf_risk_signal,
    crash_type_classifier,
    metastable_order_signal,
)


# ---------------------------------------------------------------------------
# Signal 1: low_entropy_signal
# ---------------------------------------------------------------------------


class TestLowEntropySignal:
    def test_known_percentile(self):
        """5th percentile of 0..99 is ~4.95; values 0-4 should fire."""
        s = pd.Series(np.arange(100, dtype=float))
        result = low_entropy_signal(s, threshold_pctile=5.0)
        assert result.sum() == 5  # values 0, 1, 2, 3, 4

    def test_all_equal(self):
        """When all values are identical, no value is strictly below threshold."""
        s = pd.Series([0.95] * 50)
        result = low_entropy_signal(s, threshold_pctile=5.0)
        assert result.sum() == 0

    def test_empty_series(self):
        """Empty input returns empty boolean Series."""
        result = low_entropy_signal(pd.Series(dtype=float))
        assert len(result) == 0
        assert result.dtype == bool

    def test_all_nan(self):
        """All-NaN series: quantile is NaN so no comparison is True."""
        s = pd.Series([np.nan] * 20)
        result = low_entropy_signal(s)
        assert result.sum() == 0

    def test_single_element(self):
        """Single element: threshold equals that value, strict < gives False."""
        s = pd.Series([0.5])
        result = low_entropy_signal(s)
        assert result.sum() == 0

    def test_custom_percentile(self):
        """10th percentile of 0..99 should fire for values 0-9."""
        s = pd.Series(np.arange(100, dtype=float))
        result = low_entropy_signal(s, threshold_pctile=10.0)
        assert result.sum() == 10


# ---------------------------------------------------------------------------
# Signal 2: te_leadership_flip
# ---------------------------------------------------------------------------


class TestTELeadershipFlip:
    def test_clear_reversal(self, sample_net_te_series):
        """Flip at idx 21 (second consecutive negative after 20 positives)."""
        result = te_leadership_flip(sample_net_te_series, consecutive_windows=2)
        assert result.iloc[21] is np.True_
        # Should also detect flip back to positive at idx 41
        assert result.iloc[41] is np.True_

    def test_single_window_no_fire(self):
        """A single-window sign change should NOT trigger with consecutive=2."""
        vals = [0.1] * 10 + [-0.05] + [0.1] * 10
        s = pd.Series(vals)
        result = te_leadership_flip(s, consecutive_windows=2)
        # The lone negative at idx 10 has run_length=1, should not fire
        assert result.iloc[10] is np.False_

    def test_consecutive_3(self, sample_net_te_series):
        """With consecutive_windows=3, flip detected one window later."""
        result = te_leadership_flip(sample_net_te_series, consecutive_windows=3)
        assert result.iloc[21] is np.False_
        assert result.iloc[22] is np.True_

    def test_empty_series(self):
        """Empty series returns all-False."""
        result = te_leadership_flip(pd.Series(dtype=float))
        assert len(result) == 0

    def test_all_same_sign(self):
        """No flips when all values have the same sign."""
        s = pd.Series([0.1] * 50)
        result = te_leadership_flip(s, consecutive_windows=2)
        assert result.sum() == 0

    def test_short_series(self):
        """Series shorter than consecutive_windows returns all False."""
        s = pd.Series([0.1])
        result = te_leadership_flip(s, consecutive_windows=2)
        assert len(result) == 1
        assert result.iloc[0] is np.False_


# ---------------------------------------------------------------------------
# Signal 3: acf_risk_signal
# ---------------------------------------------------------------------------


class TestACFRiskSignal:
    def test_spike_detected(self):
        """A value at 5x the baseline should trigger the signal."""
        vals = [1.0] * 50
        vals[40] = 5.0  # 5x spike
        s = pd.Series(vals)
        result = acf_risk_signal(s, multiplier=2.0, trailing_window=20)
        assert result.iloc[40] is np.True_

    def test_no_spike(self):
        """Constant series: no value exceeds 2x median (= itself)."""
        s = pd.Series([1.0] * 50)
        result = acf_risk_signal(s, multiplier=2.0, trailing_window=20)
        assert result.sum() == 0

    def test_custom_multiplier(self):
        """Value at 2.5x should NOT fire with multiplier=3.0."""
        vals = [1.0] * 50
        vals[40] = 2.5
        s = pd.Series(vals)
        result = acf_risk_signal(s, multiplier=3.0, trailing_window=20)
        assert result.iloc[40] is np.False_

    def test_nan_handling(self):
        """NaN values should produce False, not errors."""
        vals = [1.0] * 20 + [np.nan] * 5 + [1.0] * 25
        s = pd.Series(vals)
        result = acf_risk_signal(s, trailing_window=10)
        assert not result.iloc[22]  # NaN position

    def test_insufficient_history(self):
        """First trailing_window values should be False (no median yet)."""
        s = pd.Series([1.0] * 50)
        result = acf_risk_signal(s, trailing_window=20)
        assert result.iloc[:20].sum() == 0

    def test_empty_series(self):
        """Empty input returns empty boolean Series."""
        result = acf_risk_signal(pd.Series(dtype=float))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Signal 4: crash_type_classifier
# ---------------------------------------------------------------------------


class TestCrashTypeClassifier:
    def _make_series(self, n=200):
        """Helper: create random-ish feature series for testing."""
        rng = np.random.default_rng(42)
        return (
            pd.Series(rng.uniform(0.3, 1.0, n)),   # entropy
            pd.Series(rng.uniform(0.001, 0.05, n)), # volatility
            pd.Series(rng.uniform(0.5, 5.0, n)),    # tau_int
        )

    def test_info_driven_signature(self):
        """Low entropy + high vol + high tau → info_driven."""
        n = 100
        ent = pd.Series([0.95] * n)
        vol = pd.Series([0.01] * n)
        tau = pd.Series([1.0] * n)
        # Make last 5 points extreme
        ent.iloc[-5:] = 0.1
        vol.iloc[-5:] = 0.5
        tau.iloc[-5:] = 10.0
        result = crash_type_classifier(ent, vol, tau)
        assert (result.iloc[-5:] == "info_driven").all()

    def test_mechanical_signature(self):
        """High vol + normal entropy → mechanical."""
        n = 100
        ent = pd.Series([0.95] * n)
        vol = pd.Series([0.01] * n)
        tau = pd.Series([1.0] * n)
        # High vol but keep entropy high (normal) and tau low
        vol.iloc[-5:] = 0.5
        result = crash_type_classifier(ent, vol, tau)
        assert (result.iloc[-5:] == "mechanical").all()

    def test_normal(self):
        """Normal everything → all normal."""
        n = 100
        ent = pd.Series(np.linspace(0.8, 1.0, n))
        vol = pd.Series(np.linspace(0.01, 0.03, n))
        tau = pd.Series(np.linspace(0.5, 2.0, n))
        result = crash_type_classifier(ent, vol, tau)
        # Middle values should be "normal"
        assert result.iloc[50] == "normal"

    def test_valid_labels_only(self):
        """All outputs must be one of the three valid labels."""
        ent, vol, tau = self._make_series()
        result = crash_type_classifier(ent, vol, tau)
        valid = {"info_driven", "mechanical", "normal"}
        assert set(result.unique()).issubset(valid)

    def test_empty_series(self):
        """Empty input returns empty string Series."""
        empty = pd.Series(dtype=float)
        result = crash_type_classifier(empty, empty, empty)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Signal 5: metastable_order_signal
# ---------------------------------------------------------------------------


class TestMetastableOrderSignal:
    def test_deep_well_place(self):
        """Well depth > threshold → 'place'."""
        depths = pd.Series([7.0, 6.0, 8.0])
        result = metastable_order_signal(depths, threshold=5.0)
        assert (result["signal"] == "place").all()

    def test_shallow_well_avoid(self):
        """Well depth < threshold → 'avoid'."""
        depths = pd.Series([2.0, 3.0, 1.5])
        result = metastable_order_signal(depths, threshold=5.0)
        assert (result["signal"] == "avoid").all()

    def test_custom_threshold(self):
        """With threshold=3.0, depth=4.0 should fire 'place'."""
        depths = pd.Series([4.0, 2.0])
        result = metastable_order_signal(depths, threshold=3.0)
        assert result["signal"].iloc[0] == "place"
        assert result["signal"].iloc[1] == "avoid"

    def test_timeout_value(self):
        """stale_timeout_s column should contain the configured value."""
        depths = pd.Series([7.0])
        result = metastable_order_signal(depths, stale_timeout_s=60.0)
        assert result["stale_timeout_s"].iloc[0] == 60.0

    def test_empty_series(self):
        """Empty input returns empty DataFrame with correct columns."""
        result = metastable_order_signal(pd.Series(dtype=float))
        assert len(result) == 0
        assert list(result.columns) == ["signal", "stale_timeout_s"]

    def test_mixed_depths(self):
        """Mix of deep and shallow wells."""
        depths = pd.Series([7.0, 2.0, 5.1, 4.9, 10.0])
        result = metastable_order_signal(depths, threshold=5.0)
        expected = ["place", "avoid", "place", "avoid", "place"]
        assert list(result["signal"]) == expected
