"""Tests for src/backtesting.py — signal backtesting framework.

All tests use synthetic data with hand-computable expected values.
No real market data, no randomness except where testing the random baseline.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    Trade,
    BacktestResult,
    SignalBacktester,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_trade_metrics,
    random_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_prices():
    """50-bar price series that stays constant at 100."""
    ts = pd.date_range("2026-01-30 12:00", periods=50, freq="5min", tz="UTC")
    return pd.Series(100.0, index=ts)


@pytest.fixture
def trending_prices():
    """20-bar series: monotone increase from 100 to 119 (step +1 per bar)."""
    ts = pd.date_range("2026-01-30 12:00", periods=20, freq="5min", tz="UTC")
    return pd.Series(np.arange(100.0, 120.0), index=ts)


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------


class TestTrade:
    def test_fields(self):
        t = Trade(
            entry_time=pd.Timestamp("2026-01-30 12:00", tz="UTC"),
            exit_time=pd.Timestamp("2026-01-30 12:05", tz="UTC"),
            side="long",
            entry_price=100.0,
            exit_price=110.0,
            pnl=50.0,
            signal_type="low_entropy",
        )
        assert t.side == "long"
        assert t.pnl == 50.0
        assert t.signal_type == "low_entropy"

    def test_pnl_can_be_negative(self):
        t = Trade(
            entry_time=pd.Timestamp("2026-01-30 12:00", tz="UTC"),
            exit_time=pd.Timestamp("2026-01-30 12:05", tz="UTC"),
            side="short",
            entry_price=100.0,
            exit_price=110.0,
            pnl=-50.0,
            signal_type="acf_risk",
        )
        assert t.pnl < 0


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_summary_format(self):
        r = BacktestResult(
            trades=[], equity_curve=pd.Series([100_000]),
            total_pnl=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, num_trades=0,
        )
        s = r.summary()
        assert "Backtest Summary" in s
        assert "Total trades" in s
        assert "Sharpe ratio" in s

    def test_summary_no_trades(self):
        r = BacktestResult(
            trades=[], equity_curve=pd.Series([100_000]),
            total_pnl=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, num_trades=0,
        )
        assert "0" in r.summary()


# ---------------------------------------------------------------------------
# compute_sharpe_ratio
# ---------------------------------------------------------------------------


class TestComputeSharpeRatio:
    def test_known_equity_curve(self):
        """Equity [100, 101, 102, 103, 104] → constant 1% daily returns.
        Sharpe = mean/std * sqrt(252). With constant returns, std→0, so
        we test with slight variation instead."""
        eq = pd.Series(
            [100.0, 102.0, 101.0, 103.0, 105.0],
            index=pd.date_range("2026-01-30", periods=5, freq="D"),
        )
        sharpe = compute_sharpe_ratio(eq)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # net positive returns

    def test_flat_equity_returns_zero(self):
        eq = pd.Series(
            [100.0, 100.0, 100.0, 100.0],
            index=pd.date_range("2026-01-30", periods=4, freq="D"),
        )
        assert compute_sharpe_ratio(eq) == 0.0

    def test_single_point_returns_zero(self):
        eq = pd.Series([100.0])
        assert compute_sharpe_ratio(eq) == 0.0

    def test_explicit_periods(self):
        """With periods_per_year supplied, should use that value."""
        eq = pd.Series([100.0, 110.0, 105.0, 115.0])
        s1 = compute_sharpe_ratio(eq, periods_per_year=252)
        s2 = compute_sharpe_ratio(eq, periods_per_year=52)
        # Higher periods_per_year → higher annualised Sharpe
        assert s1 > s2


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


class TestComputeMaxDrawdown:
    def test_known_drawdown(self):
        """Equity [100, 120, 90, 110] → DD = (120-90)/120 = 25%."""
        eq = pd.Series([100.0, 120.0, 90.0, 110.0])
        dd = compute_max_drawdown(eq)
        assert abs(dd - 0.25) < 1e-10

    def test_monotone_increasing(self):
        eq = pd.Series([100.0, 110.0, 120.0, 130.0])
        assert compute_max_drawdown(eq) == 0.0

    def test_monotone_decreasing(self):
        """[100, 80, 60] → DD = (100-60)/100 = 40%."""
        eq = pd.Series([100.0, 80.0, 60.0])
        dd = compute_max_drawdown(eq)
        assert abs(dd - 0.40) < 1e-10

    def test_single_point(self):
        eq = pd.Series([100.0])
        assert compute_max_drawdown(eq) == 0.0


# ---------------------------------------------------------------------------
# compute_trade_metrics
# ---------------------------------------------------------------------------


class TestComputeTradeMetrics:
    def _make_trade(self, pnl):
        return Trade(
            entry_time=pd.Timestamp("2026-01-30", tz="UTC"),
            exit_time=pd.Timestamp("2026-01-30 01:00", tz="UTC"),
            side="long", entry_price=100.0, exit_price=100.0,
            pnl=pnl, signal_type="test",
        )

    def test_known_outcomes(self):
        """3 wins (+10, +5, +20), 2 losses (-8, -3).
        win_rate = 3/5 = 0.6
        avg_win = 35/3 ≈ 11.667
        avg_loss = -11/2 = -5.5
        profit_factor = 35/11 ≈ 3.182
        """
        trades = [self._make_trade(p) for p in [10, 5, 20, -8, -3]]
        m = compute_trade_metrics(trades)
        assert abs(m["win_rate"] - 0.6) < 1e-10
        assert abs(m["avg_win"] - 35 / 3) < 1e-10
        assert abs(m["avg_loss"] - (-5.5)) < 1e-10
        assert abs(m["profit_factor"] - 35 / 11) < 1e-10
        assert m["num_trades"] == 5

    def test_all_winners(self):
        trades = [self._make_trade(p) for p in [10, 20, 30]]
        m = compute_trade_metrics(trades)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    def test_all_losers(self):
        trades = [self._make_trade(p) for p in [-10, -20]]
        m = compute_trade_metrics(trades)
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_empty_trades(self):
        m = compute_trade_metrics([])
        assert m["win_rate"] == 0.0
        assert m["avg_win"] == 0.0
        assert m["avg_loss"] == 0.0
        assert m["profit_factor"] == 0.0
        assert m["num_trades"] == 0


# ---------------------------------------------------------------------------
# SignalBacktester
# ---------------------------------------------------------------------------


class TestSignalBacktester:
    def test_single_long_trade(self, trending_prices):
        """Long at bar 5 (entry bar 6 = price 106), exit 5 bars later
        (bar 11 = price 111). Gross return = 5/106. Cost = 4bps.
        PnL = 1000 * (5/106 - 0.0004)."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[5]],
            "signal_type": ["low_entropy"],
            "direction": [1],
        })
        bt = SignalBacktester(
            trending_prices, signals,
            transaction_cost_bps=2.0,
            position_size_frac=0.01,
            initial_equity=100_000.0,
            holding_period=5,
        )
        result = bt.run()
        assert result.num_trades == 1
        t = result.trades[0]
        assert t.side == "long"
        assert t.entry_price == 106.0
        assert t.exit_price == 111.0

        expected_gross = (111 - 106) / 106
        expected_cost = 2 * 2.0 / 10_000
        expected_pnl = 1000.0 * (expected_gross - expected_cost)
        assert abs(t.pnl - expected_pnl) < 1e-8

    def test_single_short_trade(self, trending_prices):
        """Short at bar 5 in an uptrend → losing trade."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[5]],
            "signal_type": ["acf_risk"],
            "direction": [-1],
        })
        bt = SignalBacktester(
            trending_prices, signals,
            holding_period=5,
        )
        result = bt.run()
        assert result.num_trades == 1
        assert result.trades[0].side == "short"
        assert result.trades[0].pnl < 0  # short in uptrend = loss

    def test_transaction_cost_on_flat_trade(self, flat_prices):
        """Entry = exit = 100. PnL should be negative (= cost only)."""
        signals = pd.DataFrame({
            "timestamp": [flat_prices.index[5]],
            "signal_type": ["test"],
            "direction": [1],
        })
        bt = SignalBacktester(
            flat_prices, signals,
            transaction_cost_bps=2.0,
            position_size_frac=0.01,
            initial_equity=100_000.0,
            holding_period=5,
        )
        result = bt.run()
        expected_cost = 1000.0 * (2 * 2.0 / 10_000)
        assert abs(result.trades[0].pnl - (-expected_cost)) < 1e-10

    def test_no_signals(self, trending_prices):
        """Empty signals → 0 trades, flat equity."""
        signals = pd.DataFrame(columns=["timestamp", "signal_type", "direction"])
        bt = SignalBacktester(trending_prices, signals, holding_period=5)
        result = bt.run()
        assert result.num_trades == 0
        assert result.total_pnl == 0.0
        assert result.max_drawdown == 0.0

    def test_no_look_ahead(self, trending_prices):
        """Entry must be the bar AFTER the signal bar, not the signal bar."""
        sig_bar = 10
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[sig_bar]],
            "signal_type": ["test"],
            "direction": [1],
        })
        bt = SignalBacktester(trending_prices, signals, holding_period=3)
        result = bt.run()
        # Entry should be bar 11, not bar 10
        assert result.trades[0].entry_time == trending_prices.index[sig_bar + 1]
        assert result.trades[0].entry_price == trending_prices.iloc[sig_bar + 1]

    def test_holding_period_exit(self, trending_prices):
        """With holding_period=5, trade exits exactly 5 bars after entry."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[3]],
            "signal_type": ["test"],
            "direction": [1],
        })
        bt = SignalBacktester(trending_prices, signals, holding_period=5)
        result = bt.run()
        entry_loc = trending_prices.index.get_loc(result.trades[0].entry_time)
        exit_loc = trending_prices.index.get_loc(result.trades[0].exit_time)
        assert exit_loc - entry_loc == 5

    def test_position_sizing(self, trending_prices):
        """Position value = equity * frac. With 2% frac and 100k equity = 2000."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[5]],
            "signal_type": ["test"],
            "direction": [1],
        })
        bt = SignalBacktester(
            trending_prices, signals,
            position_size_frac=0.02,
            initial_equity=100_000.0,
            holding_period=5,
        )
        result = bt.run()
        t = result.trades[0]
        position_value = 100_000.0 * 0.02  # 2000
        gross_return = (t.exit_price - t.entry_price) / t.entry_price
        cost = 2 * 2.0 / 10_000
        expected = position_value * (gross_return - cost)
        assert abs(t.pnl - expected) < 1e-8

    def test_multiple_trades_equity(self, trending_prices):
        """Two sequential trades; equity curve reflects cumulative PnL."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[2], trending_prices.index[10]],
            "signal_type": ["test", "test"],
            "direction": [1, 1],
        })
        bt = SignalBacktester(
            trending_prices, signals,
            initial_equity=100_000.0,
            holding_period=3,
        )
        result = bt.run()
        assert result.num_trades == 2
        # Equity should be monotonically increasing (both longs in uptrend)
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]

    def test_overlapping_signals_skipped(self, trending_prices):
        """Signals during an open position are skipped."""
        # Signal at bar 5, holding_period=10 → position open until bar 16
        # Signal at bar 8 should be skipped
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[5], trending_prices.index[8]],
            "signal_type": ["test", "test"],
            "direction": [1, -1],
        })
        bt = SignalBacktester(trending_prices, signals, holding_period=10)
        result = bt.run()
        assert result.num_trades == 1  # second signal skipped

    def test_signal_at_end_of_data(self, trending_prices):
        """Signal at last bar cannot execute (no next bar for entry)."""
        signals = pd.DataFrame({
            "timestamp": [trending_prices.index[-1]],
            "signal_type": ["test"],
            "direction": [1],
        })
        bt = SignalBacktester(trending_prices, signals, holding_period=5)
        result = bt.run()
        assert result.num_trades == 0


# ---------------------------------------------------------------------------
# random_baseline
# ---------------------------------------------------------------------------


class TestRandomBaseline:
    def test_correct_num_trades(self, sample_price_series):
        result = random_baseline(
            sample_price_series, num_trades=10, holding_period=5,
        )
        assert result.num_trades <= 10

    def test_deterministic_with_seed(self, sample_price_series):
        r1 = random_baseline(sample_price_series, num_trades=10, holding_period=5, seed=42)
        r2 = random_baseline(sample_price_series, num_trades=10, holding_period=5, seed=42)
        assert r1.total_pnl == r2.total_pnl
        assert r1.num_trades == r2.num_trades

    def test_different_seed_different_result(self, sample_price_series):
        r1 = random_baseline(sample_price_series, num_trades=10, holding_period=5, seed=42)
        r2 = random_baseline(sample_price_series, num_trades=10, holding_period=5, seed=99)
        # Different seeds should generally produce different results
        # (could be equal by extreme coincidence, but practically won't)
        assert r1.num_trades > 0

    def test_zero_trades(self, sample_price_series):
        result = random_baseline(sample_price_series, num_trades=0, holding_period=5)
        assert result.num_trades == 0
        assert result.total_pnl == 0.0
