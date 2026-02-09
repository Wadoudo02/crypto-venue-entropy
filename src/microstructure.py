"""
Microstructure analysis utilities for cross-venue crypto trade data.

Provides tools for analysing trade sign persistence, cross-venue
correlations, trade arrival rates, and size distributions.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def trade_sign_autocorrelation(
    signs: np.ndarray,
    max_lag: int = 100,
) -> np.ndarray:
    """Compute autocorrelation function of trade signs.

    High persistence (slow ACF decay) suggests informed/directional flow.
    Rapid decay suggests noise/market-making activity.

    Parameters
    ----------
    signs : np.ndarray
        Array of trade signs (+1 or -1).
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray
        ACF values from lag 0 to max_lag.
    """
    return acf(signs, nlags=max_lag, fft=True)


def cross_venue_correlation(
    returns_dict: dict[str, pd.Series],
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compute cross-correlation between venue return series at various lags.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of venue name to return series (should be time-aligned).
    max_lag : int
        Maximum lag (in number of periods) for cross-correlation.

    Returns
    -------
    pd.DataFrame
        Cross-correlation matrix indexed by lag, with venue-pair columns.
    """
    venues = list(returns_dict.keys())
    lags = range(-max_lag, max_lag + 1)
    results = {}

    for i, v1 in enumerate(venues):
        for v2 in venues[i + 1:]:
            s1 = returns_dict[v1].fillna(0.0)
            s2 = returns_dict[v2].fillna(0.0)
            corrs = []
            for lag in lags:
                if lag > 0:
                    corrs.append(s1.iloc[:-lag].corr(s2.iloc[lag:]))
                elif lag < 0:
                    corrs.append(s1.iloc[-lag:].corr(s2.iloc[:lag]))
                else:
                    corrs.append(s1.corr(s2))
            results[f"{v1} → {v2}"] = corrs

    df_out = pd.DataFrame(results, index=list(lags))
    df_out.index.name = "lag"
    return df_out


def compute_trade_arrival_rate(
    df: pd.DataFrame,
    freq: str = "1s",
) -> pd.Series:
    """Compute trade arrival rate at a given frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Trade DataFrame with 'timestamp' column.
    freq : str
        Resampling frequency (e.g. '1s', '1min', '1h').

    Returns
    -------
    pd.Series
        Trade count per period.
    """
    return df.set_index("timestamp").resample(freq).size()


def trade_size_distribution(df: pd.DataFrame) -> dict:
    """Compute summary statistics of trade size distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Trade DataFrame with 'quantity' column.

    Returns
    -------
    dict
        Distribution statistics including mean, median, std, skew, kurtosis.
    """
    q = df["quantity"]
    return {
        "mean": float(q.mean()),
        "median": float(q.median()),
        "std": float(q.std()),
        "skew": float(q.skew()),
        "kurtosis": float(q.kurtosis()),
        "p95": float(q.quantile(0.95)),
        "p99": float(q.quantile(0.99)),
        "max": float(q.max()),
    }
