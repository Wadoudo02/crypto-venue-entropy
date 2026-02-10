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
    "bybit": "#7B2FBE",     # Bybit purple
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


# ---------------------------------------------------------------------------
# Phase 2: Microstructure exploration plots
# ---------------------------------------------------------------------------


def plot_trade_sign_acf(
    acf_dict: dict[str, np.ndarray],
    title: str = "Trade Sign Autocorrelation by Venue",
) -> plt.Figure:
    """Overlay ACF curves of trade signs for multiple venues.

    Parameters
    ----------
    acf_dict : dict[str, np.ndarray]
        Mapping of venue name to ACF array (lag 0 … max_lag).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    for venue, acf_vals in acf_dict.items():
        colour = VENUE_COLOURS.get(venue, None)
        ax.plot(range(len(acf_vals)), acf_vals, label=venue.capitalize(),
                color=colour, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axhline(1 / np.e, color="grey", linewidth=0.8, linestyle=":",
               label="1/e threshold")
    ax.set_title(title)
    ax.set_xlabel("Lag (trades)")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    fig.tight_layout()
    return fig

from matplotlib.ticker import FormatStrFormatter

def plot_cross_correlation(
    xcorr_df: pd.DataFrame,
    title: str = "Cross-Venue Return Correlation by Lag",
    zero_the_axis = False,
) -> plt.Figure:
    """Plot cross-correlation between venue pairs at various lags.

    Parameters
    ----------
    xcorr_df : pd.DataFrame
        DataFrame indexed by lag with a column per venue pair.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots()

    for col in xcorr_df.columns:
        ax.plot(xcorr_df.index, xcorr_df[col], label=col, alpha=0.8)

    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    if zero_the_axis:
        ymin, ymax = xcorr_df.min().min(), xcorr_df.max().max()
        ax.set_ylim(0, ymax + 0.1)
    else:
        ymin, ymax = xcorr_df.min().min(), xcorr_df.max().max()
        pad = 0.1 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(title)
    ax.set_xlabel("Lag (periods)")
    ax.set_ylabel("Pearson Correlation")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_intraday_pattern(
    hourly_dict: dict[str, pd.Series],
    ylabel: str = "Value",
    title: str = "Intraday Pattern by Venue",
) -> plt.Figure:
    """Side-by-side bar chart of intraday patterns for multiple venues.

    Parameters
    ----------
    hourly_dict : dict[str, pd.Series]
        Mapping of venue name to Series indexed by hour (0–23).
    ylabel : str
        Y-axis label.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots()
    venues = list(hourly_dict.keys())
    n = len(venues)
    width = 0.8 / n
    hours = np.arange(24)

    for i, venue in enumerate(venues):
        colour = VENUE_COLOURS.get(venue, None)
        values = hourly_dict[venue].reindex(hours, fill_value=0)
        ax.bar(hours + i * width - 0.4 + width / 2, values,
               width=width, label=venue.capitalize(), color=colour, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(hours)
    ax.legend()
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


def plot_entropy_comparison(
    entropy_data: dict[str, dict[str, pd.DataFrame]],
    title: str = "Shannon Entropy at Multiple Window Sizes",
) -> plt.Figure:
    """Multi-panel figure: one row per window size, both venues overlaid.

    Parameters
    ----------
    entropy_data : dict[str, dict[str, pd.DataFrame]]
        Nested dict: {window_label: {venue_name: entropy_df}}.
        Each entropy_df has columns 'timestamp' and 'normalised_entropy'.
    title : str
        Overall figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    windows = list(entropy_data.keys())
    n_panels = len(windows)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, win_label in zip(axes, windows):
        for venue, edf in entropy_data[win_label].items():
            colour = VENUE_COLOURS.get(venue, None)
            ax.plot(edf["timestamp"], edf["normalised_entropy"],
                    label=venue.capitalize(), color=colour, alpha=0.7, linewidth=0.8)
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"Entropy ({win_label})")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower left")

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_transfer_entropy(
    te_forward: pd.DataFrame,
    te_reverse: pd.DataFrame,
    source_name: str = "Binance",
    target_name: str = "Bybit",
    price_data: pd.Series | None = None,
    title: str = None,
) -> plt.Figure:
    """Plot rolling transfer entropy in both directions with net TE panel.

    Parameters
    ----------
    te_forward : pd.DataFrame
        Rolling TE(source -> target) with 'timestamp' and 'te' columns.
    te_reverse : pd.DataFrame
        Rolling TE(target -> source) with 'timestamp' and 'te' columns.
    source_name : str
        Name of source venue.
    target_name : str
        Name of target venue.
    price_data : pd.Series or None
        Optional price series for overlay.
    title : str or None
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if title is None:
        title = f"Rolling Transfer Entropy: {source_name} ↔ {target_name}"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: TE in both directions
    ax1.plot(te_forward["timestamp"], te_forward["te"],
             color=VENUE_COLOURS.get(source_name.lower(), "#F0B90B"),
             label=f"TE({source_name}→{target_name})", alpha=0.8)
    ax1.plot(te_reverse["timestamp"], te_reverse["te"],
             color=VENUE_COLOURS.get(target_name.lower(), "#7B2FBE"),
             label=f"TE({target_name}→{source_name})", alpha=0.8)
    ax1.set_ylabel("Transfer Entropy (bits)")
    ax1.legend(loc="upper left")
    ax1.set_title(title)

    if price_data is not None:
        ax1b = ax1.twinx()
        ax1b.plot(price_data.index, price_data.values, color="grey",
                  alpha=0.3, linewidth=1, label="Price")
        ax1b.set_ylabel("Price (USDT)")
        ax1b.legend(loc="upper right")

    # Bottom panel: Net TE
    net_te = te_forward["te"].values - te_reverse["te"].values
    timestamps = te_forward["timestamp"]
    ax2.fill_between(timestamps, net_te, 0,
                     where=net_te >= 0, color="#2ecc71", alpha=0.4,
                     label=f"{source_name} leads")
    ax2.fill_between(timestamps, net_te, 0,
                     where=net_te < 0, color="#e74c3c", alpha=0.4,
                     label=f"{target_name} leads")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Net TE (bits)")
    ax2.set_xlabel("Time (UTC)")
    ax2.legend(loc="lower left")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_entropy_vs_volatility(
    entropy_data: pd.DataFrame,
    volatility_data: pd.Series,
    venue_name: str,
    title: str = None,
) -> plt.Figure:
    """Scatter plot of normalised entropy vs realised volatility.

    Tests the statistical mechanics hypothesis: high volatility -> high entropy.

    Parameters
    ----------
    entropy_data : pd.DataFrame
        DataFrame with 'timestamp' and 'normalised_entropy' columns.
    volatility_data : pd.Series
        Realised volatility series (datetime-indexed).
    venue_name : str
        Venue name for labelling.
    title : str or None
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if title is None:
        title = f"Entropy vs Realised Volatility — {venue_name.capitalize()}"

    # Align on timestamps
    ent = entropy_data.set_index("timestamp")["normalised_entropy"]
    common = ent.index.intersection(volatility_data.index)
    ent_aligned = ent.loc[common]
    vol_aligned = volatility_data.loc[common]

    fig, ax = plt.subplots(figsize=(8, 6))
    colour = VENUE_COLOURS.get(venue_name.lower(), "#333333")
    ax.scatter(vol_aligned.values, ent_aligned.values, alpha=0.15, s=10,
               color=colour, edgecolors="none")

    # Correlation annotation
    corr = np.corrcoef(vol_aligned.values, ent_aligned.values)[0, 1]
    ax.annotate(f"ρ = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Realised Volatility")
    ax.set_ylabel("Normalised Entropy")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
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
    smooth_window: int = 6,
) -> plt.Figure:
    """Plot regime classification as coloured background with price overlay.

    Applies a rolling majority-vote smoothing to produce visually coherent
    regime blocks, consistent with the macroscopic perspective of the
    phase transition framework.

    Parameters
    ----------
    regimes : pd.Series
        Regime labels (datetime-indexed): "hot", "cold", "critical", "transitional".
    prices : pd.Series
        Price series (datetime-indexed).
    title : str
        Plot title.
    smooth_window : int
        Rolling window for majority-vote smoothing (default 6 = 30 min at 5-min resolution).

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # --- Majority-vote smoothing ---
    if smooth_window > 1:
        smoothed = regimes.copy()
        half = smooth_window // 2
        for i in range(half, len(regimes) - half):
            window = regimes.iloc[i - half:i + half + 1]
            counts = window.value_counts()
            smoothed.iloc[i] = counts.index[0]  # Most frequent label
        regimes = smoothed

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(prices.index, prices.values, color="black", linewidth=1, alpha=0.8,
            zorder=3)

    # --- Plot contiguous regime blocks as axvspan ---
    for regime, colour in REGIME_COLOURS.items():
        mask = regimes == regime
        if not mask.any():
            continue

        # Find contiguous blocks
        indices = mask.index[mask]
        if len(indices) == 0:
            continue

        # Group consecutive timestamps
        blocks = []
        block_start = indices[0]
        block_end = indices[0]

        for j in range(1, len(indices)):
            # Check if this index is consecutive (within 2x the step size)
            gap = (indices[j] - indices[j - 1]).total_seconds()
            step = pd.Timedelta(regimes.index.freq or "5min").total_seconds()
            if gap <= step * 1.5:
                block_end = indices[j]
            else:
                blocks.append((block_start, block_end))
                block_start = indices[j]
                block_end = indices[j]
        blocks.append((block_start, block_end))

        # Plot each block
        for k, (start, end) in enumerate(blocks):
            # Extend end by one step so the block has visible width
            end_ext = end + pd.Timedelta(regimes.index.freq or "5min")
            ax.axvspan(start, end_ext, alpha=0.25, color=colour,
                       label=regime.capitalize() if k == 0 else None,
                       zorder=1)

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_phase_observables(
    timestamps: pd.Index,
    temperature: np.ndarray,
    order_parameter: np.ndarray,
    susceptibility: np.ndarray,
    prices: pd.Series = None,
    title: str = "Phase Transition Observables",
) -> plt.Figure:
    """Plot temperature, order parameter, and susceptibility with price overlay.

    Parameters
    ----------
    timestamps : pd.Index
        Datetime index for x-axis.
    temperature : np.ndarray
        Temperature analogue (realized volatility).
    order_parameter : np.ndarray
        Order parameter (order flow imbalance).
    susceptibility : np.ndarray
        Susceptibility (variance of imbalance).
    prices : pd.Series, optional
        Price series for overlay.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure with 4 subplots.
    """
    if prices is not None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    plot_idx = 0

    # Price panel (if provided)
    if prices is not None:
        axes[plot_idx].plot(prices.index, prices.values, color="black", linewidth=1, alpha=0.8)
        axes[plot_idx].set_ylabel("Price (USDT)")
        axes[plot_idx].set_title("BTC Price")
        plot_idx += 1

    # Temperature (volatility)
    axes[plot_idx].plot(timestamps, temperature, color="#e74c3c", linewidth=1, alpha=0.8)
    axes[plot_idx].set_ylabel("Temperature\n(Volatility)")
    axes[plot_idx].set_title("Temperature Analogue (Realized Volatility)")
    plot_idx += 1

    # Order parameter (imbalance)
    axes[plot_idx].plot(timestamps, order_parameter, color="#3498db", linewidth=1, alpha=0.8)
    axes[plot_idx].axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[plot_idx].set_ylabel("Order Parameter\n(Imbalance)")
    axes[plot_idx].set_title("Order Parameter (Net Order Flow Imbalance)")
    axes[plot_idx].set_ylim(-1.1, 1.1)
    plot_idx += 1

    # Susceptibility
    axes[plot_idx].plot(timestamps, susceptibility, color="#f39c12", linewidth=1, alpha=0.8)
    axes[plot_idx].set_ylabel("Susceptibility\n(Variance)")
    axes[plot_idx].set_title("Susceptibility (Variance of Imbalance)")
    axes[plot_idx].set_xlabel("Time (UTC)")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_correlation_length_timeseries(
    timestamps: pd.Index,
    correlation_length: np.ndarray,
    prices: pd.Series = None,
    title: str = "Correlation Length Evolution",
) -> plt.Figure:
    """Plot correlation length time series with price overlay.

    Parameters
    ----------
    timestamps : pd.Index
        Datetime index for x-axis.
    correlation_length : np.ndarray
        Correlation length values.
    prices : pd.Series, optional
        Price series for overlay.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if prices is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        # Price panel
        ax1.plot(prices.index, prices.values, color="black", linewidth=1, alpha=0.8)
        ax1.set_ylabel("Price (USDT)")
        ax1.set_title("BTC Price")
        # Correlation length panel
        ax2.plot(timestamps, correlation_length, color="#9b59b6", linewidth=1, alpha=0.8)
        ax2.fill_between(timestamps, 0, correlation_length, alpha=0.2, color="#9b59b6")
        ax2.set_ylabel("Correlation Length\n(lags)")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_title("Correlation Length (Critical Slowing Down)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.suptitle(title, fontsize=14, y=0.995)
        fig.autofmt_xdate()
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(timestamps, correlation_length, color="#9b59b6", linewidth=1, alpha=0.8)
        ax.fill_between(timestamps, 0, correlation_length, alpha=0.2, color="#9b59b6")
        ax.set_ylabel("Correlation Length (lags)")
        ax.set_xlabel("Time (UTC)")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()

    return fig


def plot_entropy_discontinuities(
    timestamps: pd.Index,
    entropy: np.ndarray,
    derivative: np.ndarray,
    discontinuities: np.ndarray,
    prices: pd.Series = None,
    title: str = "Entropy Discontinuity Detection",
) -> plt.Figure:
    """Plot entropy, its derivative, and detected discontinuities.

    Parameters
    ----------
    timestamps : pd.Index
        Datetime index for x-axis.
    entropy : np.ndarray
        Shannon entropy time series.
    derivative : np.ndarray
        First derivative of entropy.
    discontinuities : np.ndarray
        Boolean array marking discontinuity locations.
    prices : pd.Series, optional
        Price series for overlay.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib figure with 3 subplots.
    """
    if prices is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        # Price panel
        axes[0].plot(prices.index, prices.values, color="black", linewidth=1, alpha=0.8)
        axes[0].set_ylabel("Price (USDT)")
        axes[0].set_title("BTC Price")
        plot_idx = 1
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plot_idx = 0

    # Entropy panel
    axes[plot_idx].plot(timestamps, entropy, color="#2ecc71", linewidth=1, alpha=0.8)
    # Mark discontinuities
    if discontinuities.any():
        disc_times = timestamps[discontinuities]
        disc_values = entropy[discontinuities]
        axes[plot_idx].scatter(disc_times, disc_values, color="#e74c3c", s=50,
                              marker='x', zorder=5, label="Discontinuities")
    axes[plot_idx].set_ylabel("Entropy (bits)")
    axes[plot_idx].set_title("Shannon Entropy Time Series")
    axes[plot_idx].legend()
    plot_idx += 1

    # Derivative panel
    axes[plot_idx].plot(timestamps, derivative, color="#3498db", linewidth=0.8, alpha=0.8)
    axes[plot_idx].axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    # Mark discontinuities
    if discontinuities.any():
        disc_times = timestamps[discontinuities]
        disc_deriv = derivative[discontinuities]
        axes[plot_idx].scatter(disc_times, disc_deriv, color="#e74c3c", s=50,
                               marker='x', zorder=5)
    axes[plot_idx].set_ylabel("dH/dt")
    axes[plot_idx].set_xlabel("Time (UTC)")
    axes[plot_idx].set_title("Entropy Rate of Change (First Derivative)")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
