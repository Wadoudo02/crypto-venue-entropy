"""
Lightweight event-driven backtester for entropy-based signals.

Not a full production framework, but sufficient for signal validation.
Supports individual signal backtesting, combined strategies with priority
hierarchies, random baseline comparison, and walk-forward validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Record of a single executed trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    pnl: float  # net of transaction costs
    signal_type: str


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""

    trades: List[Trade]
    equity_curve: pd.Series
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    num_trades: int

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            "Backtest Summary",
            "=" * 40,
            f"Total trades:    {self.num_trades}",
            f"Total PnL:       {self.total_pnl:,.2f}",
            f"Sharpe ratio:    {self.sharpe_ratio:.3f}",
            f"Max drawdown:    {self.max_drawdown:.2%}",
            f"Win rate:        {self.win_rate:.2%}",
            f"Avg win:         {self.avg_win:,.2f}",
            f"Avg loss:        {self.avg_loss:,.2f}",
            f"Profit factor:   {self.profit_factor:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------------


def compute_sharpe_ratio(
    equity_curve: pd.Series,
    periods_per_year: Optional[float] = None,
) -> float:
    """Annualised Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity values indexed by timestamp (or integer).
    periods_per_year : float, optional
        If *None*, the frequency is inferred from the index timestamps.
        For 5-min bars: 365.25 * 24 * 12 = 105 192.

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns 0.0 when there are fewer than
        2 observations or returns have zero standard deviation.
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) < 1 or returns.std() == 0:
        return 0.0

    if periods_per_year is None:
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            dt = equity_curve.index.to_series().diff().median()
            periods_per_year = pd.Timedelta(days=365.25) / dt
        else:
            periods_per_year = 252  # default to daily

    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction (0 to 1).

    Parameters
    ----------
    equity_curve : pd.Series
        Equity values.

    Returns
    -------
    float
        Drawdown fraction.  0.0 if the curve is monotonically increasing
        or has fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0

    cummax = equity_curve.cummax()
    drawdown = (cummax - equity_curve) / cummax
    return float(drawdown.max())


def compute_trade_metrics(trades: List[Trade]) -> dict:
    """Derive aggregate metrics from a list of trades.

    Returns
    -------
    dict
        Keys: win_rate, avg_win, avg_loss, profit_factor, num_trades.
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    total_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    total_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0

    if total_losses > 0:
        profit_factor = total_wins / total_losses
    elif total_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "num_trades": len(pnls),
    }


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------


class SignalBacktester:
    """Event-driven backtester for pre-computed signal DataFrames.

    Parameters
    ----------
    prices : pd.Series
        Mid-prices indexed by timestamp at signal resolution.
    signals : pd.DataFrame
        Columns: ``timestamp``, ``signal_type``, ``direction`` (+1 long / -1 short).
    transaction_cost_bps : float
        Round-trip transaction cost in basis points (default 2.0).
    position_size_frac : float
        Fraction of equity risked per trade (default 0.01 = 1 %).
    initial_equity : float
        Starting equity in notional terms (default 100 000).
    holding_period : int or None
        If set, exit after this many bars.  Otherwise the position is held
        until the end of the price series.
    """

    def __init__(
        self,
        prices: pd.Series,
        signals: pd.DataFrame,
        transaction_cost_bps: float = 2.0,
        position_size_frac: float = 0.01,
        initial_equity: float = 100_000.0,
        holding_period: Optional[int] = None,
    ):
        self.prices = prices.sort_index()
        self.signals = signals.sort_values("timestamp").reset_index(drop=True)
        self.transaction_cost_bps = transaction_cost_bps
        self.position_size_frac = position_size_frac
        self.initial_equity = initial_equity
        self.holding_period = holding_period

    # ----- internal helpers ------------------------------------------------

    def _next_price_after(self, t: pd.Timestamp):
        """Return (timestamp, price) for the first bar strictly after *t*.

        Returns (None, None) if no such bar exists.
        """
        mask = self.prices.index > t
        if not mask.any():
            return None, None
        idx = self.prices.index[mask][0]
        return idx, float(self.prices.loc[idx])

    def _exit_price(self, entry_idx):
        """Return (timestamp, price) for the exit bar.

        Uses *holding_period* bars after entry if set, otherwise falls
        back to the last bar in the series.
        """
        loc = self.prices.index.get_loc(entry_idx)

        if self.holding_period is not None:
            exit_loc = min(loc + self.holding_period, len(self.prices) - 1)
        else:
            exit_loc = len(self.prices) - 1

        exit_idx = self.prices.index[exit_loc]
        return exit_idx, float(self.prices.iloc[exit_loc])

    # ----- public API ------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return aggregated results.

        The loop processes signals chronologically.  Only one position is
        open at a time (no overlapping trades).
        """
        trades: List[Trade] = []
        equity = self.initial_equity
        equity_points = [(self.prices.index[0], equity)]

        current_exit_time = None  # tracks when the active position closes

        for _, sig in self.signals.iterrows():
            sig_time = sig["timestamp"]

            # Skip if we still have an open position
            if current_exit_time is not None and sig_time <= current_exit_time:
                continue

            # Entry: first price strictly AFTER signal time (no look-ahead)
            entry_time, entry_price = self._next_price_after(sig_time)
            if entry_time is None:
                continue

            # Exit
            exit_time, exit_price = self._exit_price(entry_time)

            # Position size
            position_value = equity * self.position_size_frac

            # PnL
            direction = sig["direction"]
            if direction == 1:  # long
                gross_return = (exit_price - entry_price) / entry_price
            else:  # short
                gross_return = (entry_price - exit_price) / entry_price

            cost = 2 * self.transaction_cost_bps / 10_000  # round-trip
            net_return = gross_return - cost
            pnl = position_value * net_return

            equity += pnl

            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                side="long" if direction == 1 else "short",
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                signal_type=sig["signal_type"],
            )
            trades.append(trade)
            equity_points.append((exit_time, equity))
            current_exit_time = exit_time

        # Build equity curve
        if equity_points:
            eq_idx, eq_vals = zip(*equity_points)
            equity_curve = pd.Series(list(eq_vals), index=list(eq_idx))
        else:
            equity_curve = pd.Series([self.initial_equity], index=[self.prices.index[0]])

        # Compute metrics
        metrics = compute_trade_metrics(trades)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            total_pnl=equity - self.initial_equity,
            sharpe_ratio=compute_sharpe_ratio(equity_curve),
            max_drawdown=compute_max_drawdown(equity_curve),
            **metrics,
        )


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


def random_baseline(
    prices: pd.Series,
    num_trades: int,
    holding_period: int,
    transaction_cost_bps: float = 2.0,
    position_size_frac: float = 0.01,
    initial_equity: float = 100_000.0,
    seed: int = 42,
) -> BacktestResult:
    """Generate a random-entry baseline for comparison.

    Randomly selects *num_trades* entry points, alternating long/short,
    with the same holding period and transaction costs as the real backtest.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by timestamp.
    num_trades : int
        Number of random trades to place.
    holding_period : int
        Bars to hold each trade.
    transaction_cost_bps : float
        Round-trip cost in basis points.
    position_size_frac : float
        Fraction of equity per trade.
    initial_equity : float
        Starting equity.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BacktestResult
    """
    rng = np.random.default_rng(seed)

    max_entry = len(prices) - holding_period - 1
    if max_entry < 1 or num_trades < 1:
        equity_curve = pd.Series(
            [initial_equity], index=[prices.index[0]]
        )
        return BacktestResult(
            trades=[],
            equity_curve=equity_curve,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            num_trades=0,
        )

    entry_locs = np.sort(
        rng.choice(range(1, max_entry + 1), size=min(num_trades, max_entry), replace=False)
    )
    directions = rng.choice([-1, 1], size=len(entry_locs))

    signals = pd.DataFrame({
        "timestamp": [prices.index[i] for i in entry_locs],
        "signal_type": "random",
        "direction": directions,
    })

    bt = SignalBacktester(
        prices=prices,
        signals=signals,
        transaction_cost_bps=transaction_cost_bps,
        position_size_frac=position_size_frac,
        initial_equity=initial_equity,
        holding_period=holding_period,
    )
    return bt.run()
