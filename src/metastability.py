"""
Metastability analysis using free-energy landscape analogues.

Identifies quasi-stable price levels where the market lingers before
transitioning, analogous to metastable states in statistical mechanics.

Key concepts:
- Free energy F(x) = -kT * ln(P(x)), where P(x) is empirical price density
- Local minima in F(x) = metastable states (support/resistance levels)
- Barrier heights govern escape times (Kramers theory)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import gaussian_kde


def free_energy_landscape(
    prices: np.ndarray,
    window: int = 3600,
    step: int = 3600,
    n_bins: int = 50,
    kT: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct rolling free-energy landscape F(x) = -kT * ln(P(x)).

    For each rolling window, compute a histogram of prices to get P(x),
    then compute F(x). Returns a 2D array: rows = time windows, cols = price bins.

    Parameters
    ----------
    prices : np.ndarray
        Price series (full dataset).
    window : int
        Rolling window size (number of observations).
    step : int
        Step size between windows (stride).
    n_bins : int
        Number of price bins for the histogram.
    kT : float
        Temperature parameter (can use realised volatility or just 1.0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - window_centers: array of observation indices for window centers
        - price_grid: array of price bin centers (length n_bins)
        - free_energy_2d: array of shape (n_windows, n_bins)
    """
    if len(prices) < window:
        return np.array([]), np.array([]), np.array([]).reshape(0, n_bins)

    # Global price grid
    price_min, price_max = prices.min(), prices.max()
    price_grid = np.linspace(price_min, price_max, n_bins)

    # Rolling window computation
    n_windows = (len(prices) - window) // step + 1
    free_energy_2d = np.zeros((n_windows, n_bins))
    window_centers = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        start = i * step
        end = start + window
        window_prices = prices[start:end]
        window_centers[i] = (start + end) // 2

        # Histogram with global bins
        counts, _ = np.histogram(window_prices, bins=n_bins,
                                  range=(price_min, price_max))

        # Add epsilon and normalize
        epsilon = 1e-10
        prob_density = (counts + epsilon) / (counts.sum() + n_bins * epsilon)

        # Free energy
        free_energy_2d[i, :] = -kT * np.log(prob_density)

    return window_centers, price_grid, free_energy_2d


def find_metastable_levels(
    price_grid: np.ndarray,
    free_energy: np.ndarray,
    prominence: float = None,
) -> np.ndarray:
    """Find metastable price levels (local minima of free energy).

    Parameters
    ----------
    price_grid : np.ndarray
        Price values corresponding to free energy evaluations.
    free_energy : np.ndarray
        Free energy values (1D array for single window).
    prominence : float, optional
        Minimum prominence for peak filtering. If None, use 0.5*std(free_energy).

    Returns
    -------
    np.ndarray
        Price levels corresponding to metastable states.
    """
    if len(free_energy) == 0:
        return np.array([])

    if prominence is None:
        # Only use populated bins (below the epsilon ceiling) to set prominence.
        # Empty bins inflate std(F) and suppress real structure detection.
        F_ceiling = free_energy.max()
        populated_mask = free_energy < (F_ceiling - 1.0)
        if populated_mask.sum() > 3:
            prominence = 0.5 * np.std(free_energy[populated_mask])
        else:
            prominence = 0.5 * np.std(free_energy)

    # Find minima (peaks of -F)
    minima_indices, properties = find_peaks(-free_energy,
                                             distance=3,
                                             prominence=prominence)

    # Convert indices to price levels
    metastable_levels = price_grid[minima_indices]

    return metastable_levels


def rolling_metastable_levels(
    window_centers: np.ndarray,
    price_grid: np.ndarray,
    free_energy_2d: np.ndarray,
    prominence: float = None,
) -> list[dict]:
    """Find metastable levels for each time window.

    Parameters
    ----------
    window_centers : np.ndarray
        Observation indices for window centers.
    price_grid : np.ndarray
        Price grid for all windows.
    free_energy_2d : np.ndarray
        2D array (n_windows × n_bins) of free energy values.
    prominence : float, optional
        Minimum prominence for peak filtering.

    Returns
    -------
    list[dict]
        List of dicts with keys: time_idx, obs_center, price_level, well_depth
    """
    results = []
    for i in range(len(window_centers)):
        F = free_energy_2d[i, :]
        levels = find_metastable_levels(price_grid, F, prominence=prominence)

        # Compute well depths
        F_max = F.max()
        for level_price in levels:
            level_idx = np.argmin(np.abs(price_grid - level_price))
            F_min = F[level_idx]

            # Find barrier height: max F in neighborhood
            neighborhood = slice(max(0, level_idx - 5), min(len(F), level_idx + 6))
            F_barrier = F[neighborhood].max()
            well_depth = F_barrier - F_min

            # Skip levels where the "barrier" is just the cliff into empty bins
            if well_depth > (F_max * 0.8):
                continue

            results.append({
                'time_idx': i,
                'obs_center': window_centers[i],
                'price_level': level_price,
                'well_depth': well_depth,
            })

    return results


def dwell_time_analysis(
    prices: pd.Series,
    levels: np.ndarray,
    band_width: float = 0.002,
) -> dict[float, list[float]]:
    """Compute dwell times at each metastable level.

    Measures how long price remains within a band around each level,
    analogous to lifetime of a metastable state.

    Parameters
    ----------
    prices : pd.Series
        Price time series with datetime index.
    levels : np.ndarray
        Metastable price levels.
    band_width : float
        Fractional band width around each level (e.g. 0.002 = 0.2%).

    Returns
    -------
    dict[float, list[float]]
        Mapping of level -> list of dwell times in seconds.
    """
    dwell_times = {float(level): [] for level in levels}

    for level in levels:
        lower = level * (1 - band_width)
        upper = level * (1 + band_width)

        # Find all excursions into band
        in_band = (prices >= lower) & (prices <= upper)

        # Detect entry/exit transitions
        entries = (in_band) & (~in_band.shift(1, fill_value=False))
        exits = (~in_band) & (in_band.shift(1, fill_value=False))

        entry_times = prices.index[entries]
        exit_times = prices.index[exits]

        # Match entries to exits
        for entry_time in entry_times:
            # Find next exit after this entry
            later_exits = exit_times[exit_times > entry_time]
            if len(later_exits) > 0:
                exit_time = later_exits[0]
                dwell_seconds = (exit_time - entry_time).total_seconds()
                if dwell_seconds >= 10:  # Filter noise
                    dwell_times[float(level)].append(dwell_seconds)

    return dwell_times


def barrier_height(
    price_grid: np.ndarray,
    free_energy: np.ndarray,
    levels: np.ndarray,
) -> dict[tuple[float, float], float]:
    """Compute energy barrier heights between adjacent metastable levels.

    Higher barriers = more stable levels, longer escape times.

    Parameters
    ----------
    price_grid : np.ndarray
        Price values for the free energy landscape.
    free_energy : np.ndarray
        Free energy values (1D for single window).
    levels : np.ndarray
        Metastable price levels (from find_metastable_levels).

    Returns
    -------
    dict[tuple[float, float], float]
        Mapping of (level_i, level_j) -> barrier height.
    """
    barriers = {}
    if len(levels) < 2:
        return barriers

    levels_sorted = np.sort(levels)

    for i in range(len(levels_sorted) - 1):
        level_i = levels_sorted[i]
        level_j = levels_sorted[i + 1]

        # Find indices
        idx_i = np.argmin(np.abs(price_grid - level_i))
        idx_j = np.argmin(np.abs(price_grid - level_j))

        # Find max F between them (barrier)
        between = free_energy[idx_i:idx_j + 1]
        if len(between) > 0:
            F_barrier = between.max()

            # Barrier height relative to lower well
            F_i = free_energy[idx_i]
            F_j = free_energy[idx_j]
            barrier = F_barrier - min(F_i, F_j)

            barriers[(float(level_i), float(level_j))] = float(barrier)

    return barriers


def kramers_test(
    dwell_times: dict[float, list[float]],
    barriers: dict[tuple[float, float], float],
    temperature: float = 1.0,
) -> dict:
    """Test Kramers escape theory: ln(dwell) ~ barrier/T.

    Parameters
    ----------
    dwell_times : dict
        Mapping level -> list of dwell times.
    barriers : dict
        Mapping (level_i, level_j) -> barrier height.
    temperature : float
        Temperature parameter kT.

    Returns
    -------
    dict
        Keys: correlation, slope, intercept, x, y (for plotting)
    """
    x = []  # barrier / kT
    y = []  # ln(dwell time)

    # For each level with dwell times, find associated barriers
    for level, dwells in dwell_times.items():
        if len(dwells) == 0:
            continue

        mean_dwell = np.mean(dwells)

        # Find barrier for this level (use average of adjacent barriers)
        # Tolerance-based matching (barriers may be keyed by rounded price bins)
        relevant_barriers = []
        for (l1, l2), b in barriers.items():
            if abs(level - l1) <= 500 or abs(level - l2) <= 500:
                relevant_barriers.append(b)
        if len(relevant_barriers) == 0:
            continue

        barrier = np.mean(relevant_barriers)

        x.append(barrier / temperature)
        y.append(np.log(mean_dwell))

    if len(x) < 3:
        return {
            'correlation': np.nan,
            'slope': np.nan,
            'intercept': np.nan,
            'x': np.array([]),
            'y': np.array([])
        }

    x = np.array(x)
    y = np.array(y)

    # Linear fit
    correlation = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    return {
        'correlation': correlation,
        'slope': slope,
        'intercept': intercept,
        'x': x,
        'y': y,
    }
