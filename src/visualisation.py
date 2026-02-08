"""
Plotting utilities with consistent styling for the crypto-venue-entropy project.

Provides a unified visual style and reusable plotting functions for
microstructure, entropy, phase transition, and metastability analyses.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colour palette: one colour per venue
VENUE_COLOURS = {
    "binance": "#F0B90B",   # Binance yellow
    "bybit": "#F7A600",     # Bybit orange
    "okx": "#121212",       # OKX dark
    "coinbase": "#0052FF",  # Coinbase blue
}

REGIME_COLOURS = {
    "hot": "#e74c3c",
    "cold": "#3498db",
    "critical": "#f39c12",
}


def set_style() -> None:
    """Set consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
    })
    sns.set_palette("deep")


def plot_price_overlay(
    dfs_dict: dict[str, pd.DataFrame],
    title: str = "BTC-USDT Price Across Venues",
    freq: str = "1min",
) -> plt.Figure:
    """Plot price series from multiple venues overlaid.

    Parameters
    ----------
    dfs_dict : dict[str, pd.DataFrame]
        Mapping of venue name to trade DataFrame (with 'timestamp', 'price').
    title : str
        Plot title.
    freq : str
        Resampling frequency for cleaner visualisation.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    for venue, df in dfs_dict.items():
        resampled = (
            df.set_index("timestamp")["price"]
            .resample(freq)
            .last()
            .dropna()
        )
        colour = VENUE_COLOURS.get(venue, None)
        ax.plot(resampled.index, resampled.values, label=venue.capitalize(),
                color=colour, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_entropy_timeseries(
    entropy_dict: dict[str, pd.Series],
    prices: pd.Series | None = None,
    title: str = "Shannon Entropy of Trade Signs",
) -> plt.Figure:
    """Plot entropy time series, optionally with price overlay.

    Parameters
    ----------
    entropy_dict : dict[str, pd.Series]
        Mapping of venue name to entropy Series (datetime-indexed).
    prices : pd.Series or None
        Optional price series to overlay on secondary axis.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax1 = plt.subplots()

    for venue, entropy in entropy_dict.items():
        colour = VENUE_COLOURS.get(venue, None)
        ax1.plot(entropy.index, entropy.values, label=f"{venue.capitalize()} entropy",
                 color=colour, alpha=0.8)

    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper left")

    if prices is not None:
        ax2 = ax1.twinx()
        ax2.plot(prices.index, prices.values, color="grey", alpha=0.4,
                 linewidth=1, label="Price")
        ax2.set_ylabel("Price (USDT)")
        ax2.legend(loc="upper right")

    ax1.set_title(title)
    ax1.set_xlabel("Time (UTC)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_transfer_entropy_heatmap(
    te_matrix: pd.DataFrame,
    title: str = "Transfer Entropy Between Venues",
) -> plt.Figure:
    """Plot transfer entropy as a heatmap.

    Parameters
    ----------
    te_matrix : pd.DataFrame
        Square DataFrame where te_matrix[i][j] = TE(i -> j).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        te_matrix,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        square=True,
        cbar_kws={"label": "Transfer Entropy (bits)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Venue")
    ax.set_ylabel("Source Venue")
    fig.tight_layout()
    return fig


def plot_free_energy_landscape(
    price_grid: np.ndarray,
    free_energy_2d: np.ndarray,
    times: pd.DatetimeIndex,
    title: str = "Free-Energy Landscape Evolution",
) -> plt.Figure:
    """Plot 2D heatmap of evolving free-energy landscape.

    Parameters
    ----------
    price_grid : np.ndarray
        Price values (y-axis).
    free_energy_2d : np.ndarray
        2D array of shape (n_times, n_prices).
    times : pd.DatetimeIndex
        Time values (x-axis).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.pcolormesh(
        times, price_grid, free_energy_2d.T,
        cmap="viridis_r", shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Free Energy (a.u.)")
    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_regime_timeline(
    regimes: pd.Series,
    prices: pd.Series,
    title: str = "Market Regime Classification",
) -> plt.Figure:
    """Plot regime classification as coloured background with price overlay.

    Parameters
    ----------
    regimes : pd.Series
        Regime labels (datetime-indexed): "hot", "cold", or "critical".
    prices : pd.Series
        Price series (datetime-indexed).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    ax.plot(prices.index, prices.values, color="black", linewidth=1, alpha=0.8)

    for regime, colour in REGIME_COLOURS.items():
        mask = regimes == regime
        if mask.any():
            ax.fill_between(
                regimes.index, prices.min(), prices.max(),
                where=mask, alpha=0.15, color=colour, label=regime.capitalize(),
            )

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
