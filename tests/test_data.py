"""Tests for src/data.py -- loading, standardisation, and quality checks."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data import (
    align_timestamps,
    classify_trade_side,
    compute_derived_fields,
    download_binance_trades,
    load_processed,
    load_raw_trades,
    run_quality_checks,
    save_processed,
    standardise_columns,
)


# ── load_raw_trades ──────────────────────────────────────────────────────────


def test_load_raw_trades_binance(tmp_path, sample_trades_binance_raw):
    csv_path = tmp_path / "binance_trades.csv"
    sample_trades_binance_raw.to_csv(csv_path, index=False)

    df = load_raw_trades(csv_path, "binance")

    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "timestamp" in df.columns
    assert (df["venue"] == "binance").all()
    assert len(df) == 50


def test_load_raw_trades_bybit(tmp_path, sample_trades_bybit_raw):
    csv_path = tmp_path / "bybit_trades.csv"
    sample_trades_bybit_raw.to_csv(csv_path, index=False)

    df = load_raw_trades(csv_path, "bybit")

    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "side" in df.columns
    assert (df["venue"] == "bybit").all()
    assert len(df) == 50


def test_load_raw_trades_unknown_venue(tmp_path):
    csv_path = tmp_path / "dummy.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unknown venue"):
        load_raw_trades(csv_path, "kraken")


# ── standardise_columns ─────────────────────────────────────────────────────


def test_standardise_columns_numeric_coercion():
    df = pd.DataFrame({
        "price": ["85000.50", "85001.00", "85002.25"],
        "quantity": ["0.010", "0.020", "0.015"],
        "timestamp": [1, 2, 3],
        "venue": ["bybit"] * 3,
    })

    result = standardise_columns(df, "bybit")

    assert pd.api.types.is_numeric_dtype(result["price"])
    assert pd.api.types.is_numeric_dtype(result["quantity"])
    assert "quote_qty" in result.columns
    assert result["quote_qty"].iloc[0] == pytest.approx(85000.50 * 0.010, rel=1e-6)


def test_standardise_columns_drops_nulls():
    df = pd.DataFrame({
        "price": [85000.0, np.nan, 85002.0],
        "quantity": [0.01, 0.02, 0.03],
        "timestamp": [1, 2, 3],
        "venue": ["binance"] * 3,
    })

    result = standardise_columns(df, "binance")
    assert len(result) == 2
    assert result["price"].isna().sum() == 0


# ── align_timestamps ────────────────────────────────────────────────────────


def test_align_timestamps_unix_ms():
    df = pd.DataFrame({
        "timestamp": [1706616000000, 1706616001000, 1706616002000],
    })

    result = align_timestamps(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].dt.tz is not None  # UTC-aware
    assert result["timestamp"].is_monotonic_increasing


def test_align_timestamps_unix_seconds():
    df = pd.DataFrame({
        "timestamp": [1706616000.0, 1706616000.5, 1706616001.0],
    })

    result = align_timestamps(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].dt.tz is not None


def test_align_timestamps_iso_strings():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-30T12:00:00Z",
            "2026-01-30T12:00:01Z",
            "2026-01-30T12:00:02Z",
        ],
    })

    result = align_timestamps(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].is_monotonic_increasing


# ── classify_trade_side ──────────────────────────────────────────────────────


def test_classify_trade_side_binance():
    df = pd.DataFrame({
        "is_buyer_maker": [True, False, True],
        "price": [100, 101, 102],
    })

    result = classify_trade_side(df)

    # is_buyer_maker=True -> taker sells -> -1
    np.testing.assert_array_equal(result["trade_sign"].values, [-1, 1, -1])


def test_classify_trade_side_with_side_column():
    df = pd.DataFrame({
        "side": ["buy", "sell", "buy"],
        "price": [100, 101, 102],
    })

    result = classify_trade_side(df)
    np.testing.assert_array_equal(result["trade_sign"].values, [1, -1, 1])


def test_classify_trade_side_tick_rule():
    df = pd.DataFrame({
        "price": [100.0, 101.0, 101.0, 99.0],
    })

    result = classify_trade_side(df)

    # first=+1 (default), up=+1, same=carry(+1), down=-1
    np.testing.assert_array_equal(result["trade_sign"].values, [1, 1, 1, -1])


# ── compute_derived_fields ───────────────────────────────────────────────────


def test_compute_derived_fields_log_return():
    df = pd.DataFrame({
        "price": [100.0, 110.0, 100.0],
        "timestamp": pd.date_range("2026-01-30", periods=3, freq="1s", tz="UTC"),
    })

    result = compute_derived_fields(df)

    assert np.isnan(result["log_return"].iloc[0])
    assert result["log_return"].iloc[1] == pytest.approx(np.log(110 / 100), rel=1e-10)
    assert result["log_return"].iloc[2] == pytest.approx(np.log(100 / 110), rel=1e-10)


def test_compute_derived_fields_inter_trade_duration():
    df = pd.DataFrame({
        "price": [100.0, 101.0, 102.0],
        "timestamp": pd.date_range("2026-01-30", periods=3, freq="1s", tz="UTC"),
    })

    result = compute_derived_fields(df)

    assert np.isnan(result["inter_trade_duration_ms"].iloc[0])
    assert result["inter_trade_duration_ms"].iloc[1] == pytest.approx(1000.0)
    assert result["inter_trade_duration_ms"].iloc[2] == pytest.approx(1000.0)


# ── run_quality_checks ───────────────────────────────────────────────────────


def test_run_quality_checks_report_structure(sample_trades_df):
    report = run_quality_checks(sample_trades_df, venue="binance")

    expected_keys = {
        "venue", "total_trades", "date_range_start", "date_range_end",
        "null_prices", "null_quantities", "price_min", "price_max",
        "price_mean", "buy_fraction", "sell_fraction",
    }
    assert expected_keys.issubset(report.keys())
    assert report["venue"] == "binance"
    assert report["total_trades"] == 1000
    assert report["buy_fraction"] == pytest.approx(0.6, abs=0.01)


# ── save / load processed ───────────────────────────────────────────────────


def test_save_and_load_processed_roundtrip(tmp_path, sample_trades_df):
    path = tmp_path / "test.parquet"
    save_processed(sample_trades_df, path)

    assert path.exists()

    loaded = load_processed(path)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        sample_trades_df.reset_index(drop=True),
    )


# ── download (mocked) ───────────────────────────────────────────────────────


def test_download_binance_trades_mocked(tmp_path):
    csv_content = b"id,price,qty,quoteQty,time,isBuyerMaker\n1,85000,0.01,850,1706616000000,true\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-trades-2026-01-30.csv", csv_content)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.getvalue()

    with patch("src.data.requests.get", return_value=mock_resp):
        paths = download_binance_trades(
            symbol="BTCUSDT",
            start_date="2026-01-30",
            end_date="2026-01-30",
            output_dir=tmp_path,
        )

    assert len(paths) == 1
    assert paths[0].exists()
