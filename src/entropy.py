"""
Information-theoretic measures for cross-venue trade flow analysis.

Implements Shannon entropy, transfer entropy, and mutual information
for quantifying randomness, information content, and directional
information flow between crypto trading venues.
"""

import numpy as np
import pandas as pd


def shannon_entropy(signs: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling Shannon entropy of a trade sign sequence.

    H = -sum(p(x) * log2(p(x))) for x in {buy, sell}.
    Maximum entropy = 1 bit (50/50 split), minimum = 0 (all one side).

    Parameters
    ----------
    signs : np.ndarray
        Array of trade signs (+1 or -1).
    window : int
        Rolling window size (number of trades).

    Returns
    -------
    np.ndarray
        Rolling Shannon entropy values (NaN for initial window).
    """
    n = len(signs)
    entropy = np.full(n, np.nan)

    for i in range(window - 1, n):
        w = signs[i - window + 1 : i + 1]
        p_buy = np.sum(w == 1) / window
        p_sell = 1 - p_buy

        h = 0.0
        if p_buy > 0:
            h -= p_buy * np.log2(p_buy)
        if p_sell > 0:
            h -= p_sell * np.log2(p_sell)

        entropy[i] = h

    return entropy


def normalised_entropy(signs: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling normalised Shannon entropy (H / H_max).

    Scaled to [0, 1] where 1 = maximum randomness, 0 = fully directional.

    Parameters
    ----------
    signs : np.ndarray
        Array of trade signs (+1 or -1).
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Normalised entropy values.
    """
    h = shannon_entropy(signs, window)
    h_max = 1.0  # log2(2) = 1 for binary alphabet
    return h / h_max


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    window: int | None = None,
) -> float | np.ndarray:
    """Compute transfer entropy TE(source -> target).

    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Measures how much knowing the source's past reduces uncertainty
    about the target's future, beyond what the target's own past provides.

    Parameters
    ----------
    source : np.ndarray
        Source venue trade sign sequence (discretised).
    target : np.ndarray
        Target venue trade sign sequence (discretised).
    lag : int
        History length (number of past steps to condition on).
    window : int or None
        If provided, compute rolling TE over this window size.
        If None, compute a single TE value over the entire sequence.

    Returns
    -------
    float or np.ndarray
        Transfer entropy value(s).
    """
    raise NotImplementedError("Transfer entropy will be implemented in Phase 3.")


def net_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    window: int | None = None,
) -> float | np.ndarray:
    """Compute net transfer entropy: TE(source->target) - TE(target->source).

    Positive values indicate the source leads the target.

    Parameters
    ----------
    source : np.ndarray
        First venue trade sign sequence.
    target : np.ndarray
        Second venue trade sign sequence.
    lag : int
        History length.
    window : int or None
        Rolling window size, or None for single value.

    Returns
    -------
    float or np.ndarray
        Net transfer entropy value(s).
    """
    raise NotImplementedError("Net transfer entropy will be implemented in Phase 3.")


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 0,
) -> float:
    """Compute mutual information MI(X; Y) between two discrete sequences.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)

    Measures total shared information (non-directional).

    Parameters
    ----------
    x : np.ndarray
        First discrete sequence.
    y : np.ndarray
        Second discrete sequence.
    lag : int
        Lag to apply to y before computing MI.

    Returns
    -------
    float
        Mutual information in bits.
    """
    raise NotImplementedError("Mutual information will be implemented in Phase 3.")
