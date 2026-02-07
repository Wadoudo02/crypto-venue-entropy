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
) -> np.ndarray:
    """Compute rolling correlation length from return autocorrelation decay.

    Correlation length = lag at which ACF drops below threshold.
    Diverging correlation length signals critical slowing down.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    window : int
        Rolling window size.
    threshold : float
        ACF threshold for defining correlation length (default 1/e).

    Returns
    -------
    np.ndarray
        Rolling correlation length values.
    """
    raise NotImplementedError("Correlation length will be implemented in Phase 4.")


def detect_entropy_discontinuities(
    entropy_series: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
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
    np.ndarray
        Boolean array marking detected discontinuity locations.
    """
    raise NotImplementedError("Entropy discontinuity detection will be implemented in Phase 4.")


def classify_regime(
    volatility: np.ndarray,
    entropy: np.ndarray,
    corr_length: np.ndarray,
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

    Returns
    -------
    np.ndarray
        String array of regime labels.
    """
    raise NotImplementedError("Regime classification will be implemented in Phase 4.")
