"""
Phase transition detection using statistical mechanics analogues.

Maps market observables to thermodynamic quantities:
- Temperature ~ realised volatility
- Order parameter ~ net order flow imbalance (magnetisation)
- Susceptibility ~ variance of imbalance
- Correlation length ~ ACF decay scale

Detects regime shifts via entropy discontinuities and correlation
length divergence (critical slowing down).
"""

import numpy as np
import pandas as pd


def realised_volatility(
    prices: pd.Series,
    window: int = 300,
) -> pd.Series:
    """Compute rolling realised volatility (temperature analogue).

    Parameters
    ----------
    prices : pd.Series
        Price series.
    window : int
        Rolling window size (number of observations).

    Returns
    -------
    pd.Series
        Rolling realised volatility.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std()


def order_flow_imbalance(
    signs: np.ndarray,
    window: int = 300,
) -> np.ndarray:
    """Compute rolling order flow imbalance (order parameter / magnetisation).

    Imbalance = mean(signs) in window.
    +1 = all buys (fully magnetised), 0 = balanced (disordered).

    Parameters
    ----------
    signs : np.ndarray
        Trade sign array (+1 / -1).
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling imbalance values in [-1, 1].
    """
    s = pd.Series(signs)
    return s.rolling(window).mean().values


def susceptibility(
    imbalance: np.ndarray,
    window: int = 300,
) -> np.ndarray:
    """Compute rolling susceptibility (variance of order flow imbalance).

    Peaks near critical points where the market is most sensitive
    to perturbations.

    Parameters
    ----------
    imbalance : np.ndarray
        Order flow imbalance series.
    window : int
        Rolling window for variance computation.

    Returns
    -------
    np.ndarray
        Rolling susceptibility (variance).
    """
    s = pd.Series(imbalance)
    return s.rolling(window).var().values


def correlation_length(
    returns: np.ndarray,
    window: int = 1000,
    threshold: float = 1 / np.e,
    max_lag: int = 100,
    use_absolute: bool = True,
) -> np.ndarray:
    """Compute rolling correlation length from autocorrelation decay.

    Correlation length = lag at which ACF drops below threshold.
    Diverging correlation length signals critical slowing down.

    By default uses absolute returns |r| rather than raw returns,
    because raw returns are nearly uncorrelated (EMH), but absolute
    returns exhibit volatility clustering with significant, slowly
    decaying autocorrelation — the relevant signal for detecting
    regime persistence and critical slowing down.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    window : int
        Rolling window size.
    threshold : float
        ACF threshold for defining correlation length (default 1/e).
    max_lag : int
        Maximum lag to search for threshold crossing.
    use_absolute : bool
        If True, compute ACF of |returns| (volatility clustering).
        If False, compute ACF of raw returns.

    Returns
    -------
    np.ndarray
        Rolling correlation length values. Returns max_lag if threshold not crossed.
    """
    series = np.abs(returns) if use_absolute else returns
    n = len(series)
    corr_lengths = np.full(n, np.nan)

    for i in range(window, n):
        window_data = series[i - window:i]
        # Remove mean for autocorrelation
        window_data = window_data - np.nanmean(window_data)

        # Compute ACF up to max_lag
        acf_vals = np.zeros(max_lag + 1)
        acf_vals[0] = 1.0  # By definition

        variance = np.nanvar(window_data)
        if variance == 0:
            corr_lengths[i] = 0
            continue

        for lag in range(1, max_lag + 1):
            if lag >= len(window_data):
                break
            # Autocorrelation at lag k
            covariance = np.nanmean(window_data[:-lag] * window_data[lag:])
            acf_vals[lag] = covariance / variance

        # Find first lag where ACF drops below threshold
        below_threshold = np.where(acf_vals[1:] < threshold)[0]
        if len(below_threshold) > 0:
            corr_lengths[i] = below_threshold[0] + 1  # +1 because we skipped lag 0
        else:
            corr_lengths[i] = max_lag  # Didn't cross threshold

    return corr_lengths


def detect_entropy_discontinuities(
    entropy_series: np.ndarray,
    threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect sharp jumps in entropy time series.

    Identifies both first-order-like (sudden jumps) and second-order-like
    (continuous with diverging derivative) transitions.

    Parameters
    ----------
    entropy_series : np.ndarray
        Shannon entropy time series.
    threshold : float
        Number of standard deviations for jump detection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Boolean array marking detected discontinuity locations
        - First derivative (rate of change) of entropy series
    """
    # Compute first derivative (rate of change)
    derivative = np.zeros_like(entropy_series)
    derivative[1:] = np.diff(entropy_series)

    # Compute statistics of derivative
    mean_deriv = np.nanmean(derivative)
    std_deriv = np.nanstd(derivative)

    # Flag discontinuities: |derivative| > threshold * std
    discontinuities = np.abs(derivative - mean_deriv) > (threshold * std_deriv)

    return discontinuities, derivative


def classify_regime(
    volatility: np.ndarray,
    entropy: np.ndarray,
    corr_length: np.ndarray,
    susceptibility: np.ndarray = None,
) -> np.ndarray:
    """Classify market regime using phase transition framework.

    Regimes:
    - "hot": high entropy, high volatility, low correlation length (disordered)
    - "cold": low entropy, low volatility, high correlation length (ordered/trending)
    - "critical": intermediate entropy, diverging corr length, high susceptibility

    Parameters
    ----------
    volatility : np.ndarray
        Realised volatility (temperature).
    entropy : np.ndarray
        Shannon entropy of trade signs.
    corr_length : np.ndarray
        Autocorrelation decay length.
    susceptibility : np.ndarray, optional
        Variance of order flow imbalance. If provided, used for critical regime detection.

    Returns
    -------
    np.ndarray
        String array of regime labels.
    """
    n = len(entropy)
    regimes = np.full(n, "", dtype=object)

    # Compute percentile thresholds for each observable
    # Filter out NaN values for threshold calculation
    valid_vol = volatility[~np.isnan(volatility)]
    valid_ent = entropy[~np.isnan(entropy)]
    valid_corr = corr_length[~np.isnan(corr_length)]

    vol_25, vol_75 = np.percentile(valid_vol, [25, 75])
    ent_25, ent_75 = np.percentile(valid_ent, [25, 75])
    corr_25, corr_75 = np.percentile(valid_corr, [25, 75])

    # Compute correlation length rate of change for "diverging" detection
    corr_deriv = np.zeros_like(corr_length)
    corr_deriv[1:] = np.diff(corr_length)
    valid_deriv = corr_deriv[~np.isnan(corr_deriv)]
    corr_deriv_90 = np.percentile(valid_deriv, 90)  # Rapidly increasing

    if susceptibility is not None:
        valid_susc = susceptibility[~np.isnan(susceptibility)]
        susc_90 = np.percentile(valid_susc, 90)

    for i in range(n):
        # Skip if any observable is NaN
        if np.isnan(volatility[i]) or np.isnan(entropy[i]) or np.isnan(corr_length[i]):
            regimes[i] = "unknown"
            continue

        # Critical regime: diverging correlation length or high susceptibility
        is_critical = False
        if corr_deriv[i] > corr_deriv_90:  # Correlation length rapidly increasing
            is_critical = True
        elif susceptibility is not None and not np.isnan(susceptibility[i]) and susceptibility[i] > susc_90:
            is_critical = True

        if is_critical:
            regimes[i] = "critical"
            continue

        # Score-based classification: count how many indicators point hot vs cold
        # Hot indicators: high entropy, high volatility, low correlation length
        hot_score = 0
        cold_score = 0

        if entropy[i] > ent_75:
            hot_score += 1
        elif entropy[i] < ent_25:
            cold_score += 1

        if volatility[i] > vol_75:
            hot_score += 1
        elif volatility[i] < vol_25:
            cold_score += 1

        if corr_length[i] < corr_25:
            hot_score += 1
        elif corr_length[i] > corr_75:
            cold_score += 1

        # Classify: need at least 2 out of 3 indicators to agree
        if hot_score >= 2:
            regimes[i] = "hot"
        elif cold_score >= 2:
            regimes[i] = "cold"
        else:
            regimes[i] = "transitional"

    return regimes
