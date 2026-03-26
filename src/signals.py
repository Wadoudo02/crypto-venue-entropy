"""
Signal generation from entropy and microstructure features.

Extracts the five trading signals identified in the synthesis analysis
into reusable functions. Each function takes a feature Series or DataFrame
and returns a signal Series (boolean or categorical).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default thresholds (from original analysis, Jan 30 – Feb 6 2026)
# ---------------------------------------------------------------------------

DEFAULT_ENTROPY_PCTILE = 5.0
DEFAULT_TE_CONSECUTIVE = 2
DEFAULT_ACF_MULTIPLIER = 2.0
DEFAULT_ACF_TRAILING_WINDOW = 20
DEFAULT_WELL_DEPTH_THRESHOLD = 5.0
DEFAULT_STALE_TIMEOUT_S = 41.0


# ---------------------------------------------------------------------------
# Signal 1: Low-entropy directional flow
# ---------------------------------------------------------------------------


def low_entropy_signal(
    entropy_series: pd.Series,
    threshold_pctile: float = DEFAULT_ENTROPY_PCTILE,
) -> pd.Series:
    """Flag windows where Shannon entropy drops below Nth percentile.

    When Binance entropy falls below the 5th percentile, 88.1% of signals
    precede |return| > 0.05% within 5 minutes in the original dataset.

    Parameters
    ----------
    entropy_series : pd.Series
        Rolling Shannon entropy values (typically 5-min windows).
    threshold_pctile : float
        Percentile below which the signal fires (default 5.0).

    Returns
    -------
    pd.Series
        Boolean series; True where entropy is below threshold.
    """
    if entropy_series.empty:
        return pd.Series(dtype=bool)

    threshold = entropy_series.quantile(threshold_pctile / 100.0)
    return entropy_series < threshold


# ---------------------------------------------------------------------------
# Signal 2: Transfer entropy leadership reversal
# ---------------------------------------------------------------------------


def te_leadership_flip(
    net_te_series: pd.Series,
    consecutive_windows: int = DEFAULT_TE_CONSECUTIVE,
) -> pd.Series:
    """Detect sustained reversals in information leadership between venues.

    A flip is confirmed when the sign of net transfer entropy
    (TE_forward - TE_reverse) changes and persists for at least
    ``consecutive_windows`` consecutive periods.

    Parameters
    ----------
    net_te_series : pd.Series
        Net transfer entropy (positive = venue A leads).
    consecutive_windows : int
        Minimum run length to confirm a flip (default 2).

    Returns
    -------
    pd.Series
        Boolean series; True at the window where a flip is confirmed.
    """
    if net_te_series.empty or len(net_te_series) < consecutive_windows:
        return pd.Series(False, index=net_te_series.index, dtype=bool)

    signs = np.sign(net_te_series.values)
    result = np.zeros(len(signs), dtype=bool)

    run_length = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] or signs[i] == 0:
            run_length += 1
        else:
            run_length = 1

        if run_length == consecutive_windows and signs[i] != 0:
            # Check that the sign actually differs from the run before this one
            pre_run_start = i - consecutive_windows
            if pre_run_start >= 0 and signs[pre_run_start] != signs[i]:
                result[i] = True

    return pd.Series(result, index=net_te_series.index, dtype=bool)


# ---------------------------------------------------------------------------
# Signal 3: ACF risk (critical slowing down)
# ---------------------------------------------------------------------------


def acf_risk_signal(
    tau_int_series: pd.Series,
    multiplier: float = DEFAULT_ACF_MULTIPLIER,
    trailing_window: int = DEFAULT_ACF_TRAILING_WINDOW,
) -> pd.Series:
    """Flag windows where integrated autocorrelation time spikes.

    Elevated tau_int signals critical slowing down; in the original
    dataset, tau_int > 90th percentile preceded 1.65x baseline volatility.

    Parameters
    ----------
    tau_int_series : pd.Series
        Integrated autocorrelation time series.
    multiplier : float
        Multiple of trailing median that triggers the signal (default 2.0).
    trailing_window : int
        Lookback window for computing the trailing median (default 20).

    Returns
    -------
    pd.Series
        Boolean series; True where tau_int exceeds multiplier * trailing median.
    """
    if tau_int_series.empty:
        return pd.Series(dtype=bool)

    trailing_median = tau_int_series.rolling(
        window=trailing_window, min_periods=trailing_window
    ).median()

    signal = tau_int_series > (multiplier * trailing_median)
    return signal.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Signal 4: Crash type classification
# ---------------------------------------------------------------------------


def crash_type_classifier(
    entropy: pd.Series,
    volatility: pd.Series,
    tau_int: pd.Series,
    entropy_low_pctile: float = 10.0,
    vol_high_pctile: float = 90.0,
    tau_high_pctile: float = 90.0,
) -> pd.Series:
    """Classify each window as information-driven crash, mechanical crash, or normal.

    Information-driven (Jan 31 pattern): entropy collapse + high volatility
    + elevated tau_int (informed directional trading).

    Mechanical (Feb 5-6 pattern): high volatility with normal entropy
    (forced liquidations, balanced but large order flow).

    Parameters
    ----------
    entropy : pd.Series
        Rolling Shannon entropy.
    volatility : pd.Series
        Rolling realised volatility (temperature).
    tau_int : pd.Series
        Integrated autocorrelation time.
    entropy_low_pctile : float
        Percentile below which entropy is considered "low" (default 10).
    vol_high_pctile : float
        Percentile above which volatility is considered "high" (default 90).
    tau_high_pctile : float
        Percentile above which tau_int is considered "high" (default 90).

    Returns
    -------
    pd.Series
        Categorical string series: "info_driven", "mechanical", or "normal".
    """
    if entropy.empty:
        return pd.Series(dtype=str)

    ent_low = entropy < entropy.quantile(entropy_low_pctile / 100.0)
    vol_high = volatility > volatility.quantile(vol_high_pctile / 100.0)
    tau_high = tau_int > tau_int.quantile(tau_high_pctile / 100.0)

    labels = pd.Series("normal", index=entropy.index)
    labels[vol_high & ~ent_low] = "mechanical"
    labels[ent_low & vol_high & tau_high] = "info_driven"

    return labels


# ---------------------------------------------------------------------------
# Signal 5: Metastable level order placement
# ---------------------------------------------------------------------------


def metastable_order_signal(
    well_depths: pd.Series,
    threshold: float = DEFAULT_WELL_DEPTH_THRESHOLD,
    stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
) -> pd.DataFrame:
    """Generate order placement signals based on free-energy well depth.

    Levels with deep wells (strong metastable states) are suitable for
    passive limit orders. The stale timeout reflects the median dwell
    time from the original analysis (41s) — orders not filled within
    this window should be cancelled.

    Parameters
    ----------
    well_depths : pd.Series
        Well depth at each metastable level.
    threshold : float
        Minimum well depth to emit a "place" signal (default 5.0).
    stale_timeout_s : float
        Recommended cancel-after duration in seconds (default 41.0).

    Returns
    -------
    pd.DataFrame
        Columns: "signal" ("place" or "avoid"), "stale_timeout_s".
    """
    if well_depths.empty:
        return pd.DataFrame(
            columns=["signal", "stale_timeout_s"],
            dtype=object,
        )

    signal = pd.Series("avoid", index=well_depths.index)
    signal[well_depths > threshold] = "place"

    return pd.DataFrame({
        "signal": signal,
        "stale_timeout_s": stale_timeout_s,
    })
