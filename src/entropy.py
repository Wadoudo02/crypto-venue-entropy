"""
Information-theoretic measures for cross-venue trade flow analysis.

Implements Shannon entropy, transfer entropy, and mutual information
for quantifying randomness, information content, and directional
information flow between crypto trading venues.
"""

import numpy as np
import pandas as pd
from numba import njit


# ---------------------------------------------------------------------------
# Shannon Entropy
# ---------------------------------------------------------------------------


def shannon_entropy(signs: np.ndarray) -> float:
    """Compute Shannon entropy of a binary trade sign sequence.

    H = -sum(p(x) * log2(p(x))) for x in {+1, -1}

    Parameters
    ----------
    signs : np.ndarray
        Array of trade signs (+1 or -1).

    Returns
    -------
    float
        Entropy in bits. Max = 1.0 (50/50), Min = 0.0 (all one sign).
    """
    if len(signs) == 0:
        return 0.0
    n = len(signs)
    p_buy = np.sum(signs == 1) / n
    p_sell = 1.0 - p_buy

    h = 0.0
    if p_buy > 0:
        h -= p_buy * np.log2(p_buy)
    if p_sell > 0:
        h -= p_sell * np.log2(p_sell)
    return float(h)


def rolling_shannon_entropy(
    df: pd.DataFrame,
    window: str = "5min",
    sign_col: str = "trade_sign",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute Shannon entropy of trade signs in rolling time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame with sign and timestamp columns.
    window : str
        Pandas frequency string for window size (e.g. '1min', '5min', '15min').
    sign_col : str
        Column containing trade signs (+1 / -1).
    timestamp_col : str
        Column containing timestamps.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, entropy, normalised_entropy, buy_fraction, trade_count.
    """
    grouped = df.set_index(timestamp_col)[sign_col].resample(window)

    records = []
    for ts, group in grouped:
        signs = group.values
        n = len(signs)
        if n == 0:
            continue
        p_buy = np.sum(signs == 1) / n
        p_sell = 1.0 - p_buy

        h = 0.0
        if p_buy > 0:
            h -= p_buy * np.log2(p_buy)
        if p_sell > 0:
            h -= p_sell * np.log2(p_sell)

        records.append({
            "timestamp": ts,
            "entropy": h,
            "normalised_entropy": h / 1.0,  # H_max = 1 bit for binary
            "buy_fraction": p_buy,
            "trade_count": n,
        })

    return pd.DataFrame(records)


def rolling_shannon_entropy_volume_weighted(
    df: pd.DataFrame,
    window: str = "5min",
    sign_col: str = "trade_sign",
    quantity_col: str = "quantity",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute Shannon entropy where each trade's contribution is weighted by size.

    Instead of p(buy) = count(buys) / count(all), uses
    p(buy) = sum(buy_volumes) / sum(all_volumes).

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame.
    window : str
        Pandas frequency string for window size.
    sign_col : str
        Column containing trade signs (+1 / -1).
    quantity_col : str
        Column containing trade sizes.
    timestamp_col : str
        Column containing timestamps.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, entropy, normalised_entropy, buy_fraction, trade_count.
    """
    indexed = df.set_index(timestamp_col)[[sign_col, quantity_col]]
    grouped = indexed.resample(window)

    records = []
    for ts, group in grouped:
        n = len(group)
        if n == 0:
            continue

        buy_mask = group[sign_col].values == 1
        buy_vol = group[quantity_col].values[buy_mask].sum()
        total_vol = group[quantity_col].values.sum()

        if total_vol == 0:
            continue

        p_buy = buy_vol / total_vol
        p_sell = 1.0 - p_buy

        h = 0.0
        if p_buy > 0:
            h -= p_buy * np.log2(p_buy)
        if p_sell > 0:
            h -= p_sell * np.log2(p_sell)

        records.append({
            "timestamp": ts,
            "entropy": h,
            "normalised_entropy": h / 1.0,
            "buy_fraction": p_buy,
            "trade_count": n,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Transfer Entropy
# ---------------------------------------------------------------------------


@njit
def _te_core(source: np.ndarray, target: np.ndarray, history_length: int) -> float:
    """Numba-accelerated transfer entropy computation.

    Computes TE(source -> target) using histogram-based probability estimation
    on a ternary alphabet {-1, 0, +1} mapped to {0, 1, 2}.

    Parameters
    ----------
    source : np.ndarray
        Source sequence mapped to {0, 1, 2}.
    target : np.ndarray
        Target sequence mapped to {0, 1, 2}.
    history_length : int
        Number of past steps to condition on (k).

    Returns
    -------
    float
        Transfer entropy in bits.
    """
    n = len(source)
    if n <= history_length:
        return 0.0

    alphabet_size = 3
    # Number of possible history patterns: alphabet_size^history_length
    n_patterns = 1
    for _ in range(history_length):
        n_patterns *= alphabet_size

    # Count joint occurrences:
    # joint[y_fut, y_past_pattern, x_past_pattern]
    joint = np.zeros((alphabet_size, n_patterns, n_patterns), dtype=np.float64)

    for t in range(history_length, n):
        y_fut = target[t]

        # Encode history patterns as base-3 numbers
        y_past = 0
        x_past = 0
        base = 1
        for k in range(history_length):
            y_past += target[t - 1 - k] * base
            x_past += source[t - 1 - k] * base
            base *= alphabet_size

        joint[y_fut, y_past, x_past] += 1.0

    total = n - history_length
    if total == 0:
        return 0.0

    # Normalise to joint probability
    for i in range(alphabet_size):
        for j in range(n_patterns):
            for k in range(n_patterns):
                joint[i, j, k] /= total

    # Marginals: p(y_past, x_past) and p(y_fut, y_past)
    p_yx = np.zeros((n_patterns, n_patterns), dtype=np.float64)  # p(y_past, x_past)
    p_yfy = np.zeros((alphabet_size, n_patterns), dtype=np.float64)  # p(y_fut, y_past)
    p_y = np.zeros(n_patterns, dtype=np.float64)  # p(y_past)

    for i in range(alphabet_size):
        for j in range(n_patterns):
            for k in range(n_patterns):
                p_yx[j, k] += joint[i, j, k]
                p_yfy[i, j] += joint[i, j, k]

    for j in range(n_patterns):
        for k in range(n_patterns):
            p_y[j] += p_yx[j, k]

    # TE = sum p(y_fut, y_past, x_past) * log2( p(y_fut|y_past,x_past) / p(y_fut|y_past) )
    te = 0.0
    log2 = np.log(2.0)
    for i in range(alphabet_size):
        for j in range(n_patterns):
            for k in range(n_patterns):
                p_joint = joint[i, j, k]
                if p_joint < 1e-15:
                    continue
                p_cond_full = p_joint / p_yx[j, k] if p_yx[j, k] > 1e-15 else 0.0
                p_cond_target = p_yfy[i, j] / p_y[j] if p_y[j] > 1e-15 else 0.0
                if p_cond_full > 1e-15 and p_cond_target > 1e-15:
                    te += p_joint * (np.log(p_cond_full / p_cond_target) / log2)

    return te


def _map_ternary(arr: np.ndarray) -> np.ndarray:
    """Map ternary values {-1, 0, +1} to {0, 1, 2} for numba indexing."""
    return (arr + 1).astype(np.int64)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    history_length: int = 1,
) -> float:
    """Compute transfer entropy TE(source -> target).

    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Uses histogram-based probability estimation on ternary alphabet {-1, 0, +1}.

    Parameters
    ----------
    source : np.ndarray
        Discretised source time series (values in {-1, 0, +1}).
    target : np.ndarray
        Discretised target time series (same length as source).
    history_length : int
        Number of past steps to condition on (k in the literature).

    Returns
    -------
    float
        Transfer entropy in bits (non-negative).
    """
    if len(source) != len(target):
        raise ValueError("source and target must have the same length")
    if len(source) <= history_length:
        return 0.0

    src_mapped = _map_ternary(source)
    tgt_mapped = _map_ternary(target)
    return _te_core(src_mapped, tgt_mapped, history_length)


def _resample_to_bins(
    df: pd.DataFrame,
    bin_freq: str = "1s",
    sign_col: str = "trade_sign",
    quantity_col: str = "quantity",
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Resample trade data to fixed-frequency bins with ternary net sign.

    For each bin: net_sign = sign(sum(trade_sign * quantity)).
    Bins with no trades get 0 (balanced/no information).

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level DataFrame.
    bin_freq : str
        Bin frequency (e.g. '1s').

    Returns
    -------
    pd.Series
        Ternary series (+1, -1, 0) indexed by bin timestamp.
    """
    indexed = df.set_index(timestamp_col)
    weighted = (indexed[sign_col] * indexed[quantity_col]).resample(bin_freq).sum()
    return np.sign(weighted).fillna(0).astype(int)


def rolling_transfer_entropy(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_freq: str = "1s",
    window: str = "30min",
    step: str = "5min",
    history_length: int = 1,
) -> pd.DataFrame:
    """Compute transfer entropy in rolling windows.

    Pipeline:
    1. Resample both trade DataFrames to `bin_freq` resolution with ternary net sign.
    2. Roll a window of size `window`, stepping by `step`.
    3. In each window, compute TE(source -> target).

    Parameters
    ----------
    source_df : pd.DataFrame
        Source venue trade DataFrame.
    target_df : pd.DataFrame
        Target venue trade DataFrame.
    bin_freq : str
        Bin frequency for discretisation (e.g. '1s').
    window : str
        Rolling window size (e.g. '30min').
    step : str
        Step size between windows (e.g. '5min').
    history_length : int
        Number of past bins to condition on.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, te, n_bins.
    """
    # Step 1: resample to bins
    src_bins = _resample_to_bins(source_df, bin_freq)
    tgt_bins = _resample_to_bins(target_df, bin_freq)

    # Align to common index
    common_idx = src_bins.index.union(tgt_bins.index)
    src_bins = src_bins.reindex(common_idx, fill_value=0)
    tgt_bins = tgt_bins.reindex(common_idx, fill_value=0)

    # Step 2: compute rolling TE
    window_td = pd.Timedelta(window)
    step_td = pd.Timedelta(step)

    start = common_idx.min()
    end = common_idx.max()

    records = []
    current = start
    while current + window_td <= end:
        win_start = current
        win_end = current + window_td
        mask = (common_idx >= win_start) & (common_idx < win_end)

        src_win = src_bins.values[mask]
        tgt_win = tgt_bins.values[mask]
        n_bins = int(np.sum((src_win != 0) | (tgt_win != 0)))

        te_val = transfer_entropy(src_win, tgt_win, history_length)

        records.append({
            "timestamp": win_start + window_td / 2,  # window centre
            "te": te_val,
            "n_bins": n_bins,
        })
        current += step_td

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information MI(X; Y) = H(X) + H(Y) - H(X, Y).

    Non-directional measure of shared information between two sequences.
    Uses ternary alphabet {-1, 0, +1}.

    Parameters
    ----------
    x : np.ndarray
        First discrete sequence (values in {-1, 0, +1}).
    y : np.ndarray
        Second discrete sequence (same length).

    Returns
    -------
    float
        Mutual information in bits.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n == 0:
        return 0.0

    alphabet = np.array([-1, 0, 1])

    # Marginal entropies
    def _entropy(arr: np.ndarray) -> float:
        h = 0.0
        for val in alphabet:
            p = np.sum(arr == val) / n
            if p > 0:
                h -= p * np.log2(p)
        return h

    h_x = _entropy(x)
    h_y = _entropy(y)

    # Joint entropy
    h_xy = 0.0
    for vx in alphabet:
        for vy in alphabet:
            p = np.sum((x == vx) & (y == vy)) / n
            if p > 0:
                h_xy -= p * np.log2(p)

    return float(h_x + h_y - h_xy)
