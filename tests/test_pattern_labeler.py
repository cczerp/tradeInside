"""Tests for pattern_labeler.

Requires pandas. Skips cleanly if pandas isn't installed.
"""
from datetime import date

import pytest

pd = pytest.importorskip("pandas")

from pattern_labeler import (
    PATTERN_COLUMNS,
    apply_all_labels,
    label_large_trade,
    label_pre_event,
    label_same_company_cluster,
    label_sector_cluster,
)


def _trades(rows):
    """Build a trades DataFrame with the columns the labelers need."""
    df = pd.DataFrame(rows)
    df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.date
    if "trade_value" not in df.columns:
        df["trade_value"] = 1000.0
    if "trader" not in df.columns:
        df["trader"] = "Anon"
    if "ticker" not in df.columns:
        df["ticker"] = "AAA"
    return df


# --- label_large_trade -----------------------------------------------------

def test_large_trade_fires_when_value_exceeds_3x_trader_mean():
    # 6×$100 + 1×$10000 → mean = $1514, 3× = $4542, $10000 > $4542 fires
    rows = [{"trader": "Jane", "trade_value": 100.0, "txn_date": f"2025-01-0{i+1}"}
            for i in range(6)]
    rows.append({"trader": "Jane", "trade_value": 10_000.0, "txn_date": "2025-01-07"})
    df = _trades(rows)
    out = label_large_trade(df)
    assert out.tolist() == [False] * 6 + [True]


def test_large_trade_uses_per_trader_mean_not_global():
    df = _trades([
        # Big: two huge trades, neither >3× the other → no fire
        {"trader": "Big",   "trade_value": 1_000_000.0, "txn_date": "2025-01-01"},
        {"trader": "Big",   "trade_value": 1_000_000.0, "txn_date": "2025-01-02"},
        # Small: 5×$100 + 1×$5000 → mean=916.7, 3×=2750, $5000 fires
        {"trader": "Small", "trade_value": 100.0,  "txn_date": "2025-01-03"},
        {"trader": "Small", "trade_value": 100.0,  "txn_date": "2025-01-04"},
        {"trader": "Small", "trade_value": 100.0,  "txn_date": "2025-01-05"},
        {"trader": "Small", "trade_value": 100.0,  "txn_date": "2025-01-06"},
        {"trader": "Small", "trade_value": 100.0,  "txn_date": "2025-01-07"},
        {"trader": "Small", "trade_value": 5000.0, "txn_date": "2025-01-08"},
    ])
    out = label_large_trade(df)
    # Big's trades stay False because neither is large vs Big's own mean.
    # Small's last trade fires; the per-trader mean would not have fired
    # if we'd used the global mean (which is dragged up by Big).
    assert out.tolist() == [False, False, False, False, False, False, False, True]


def test_large_trade_singleton_trader_never_fires():
    df = _trades([
        {"trader": "Solo", "trade_value": 999_999.0, "txn_date": "2025-01-01"},
    ])
    assert label_large_trade(df).tolist() == [False]


# --- label_same_company_cluster --------------------------------------------

def test_same_company_cluster_fires_with_three_distinct_insiders():
    df = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"},
        {"trader": "B", "ticker": "FOO", "txn_date": "2025-01-03"},
        {"trader": "C", "ticker": "FOO", "txn_date": "2025-01-05"},
    ])
    out = label_same_company_cluster(df, window_days=7, min_insiders=3)
    # All three are within 7 days of each other and the ticker has 3 distinct traders
    assert out.tolist() == [True, True, True]


def test_same_company_cluster_does_not_fire_for_single_trader_repeating():
    df = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"},
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-02"},
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-03"},
    ])
    out = label_same_company_cluster(df, window_days=7, min_insiders=3)
    assert out.tolist() == [False, False, False]


def test_same_company_cluster_window_isolates_distant_trades():
    df = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"},
        {"trader": "B", "ticker": "FOO", "txn_date": "2025-01-02"},
        {"trader": "C", "ticker": "FOO", "txn_date": "2025-06-01"},  # months later
    ])
    out = label_same_company_cluster(df, window_days=7, min_insiders=3)
    # 3 distinct traders for ticker overall, but the june trade isn't in
    # window with the jan ones — so no row sees 3 distinct in its window
    assert out.tolist() == [False, False, False]


# --- label_sector_cluster --------------------------------------------------

def test_sector_cluster_fires_across_different_tickers_in_same_sector():
    df = _trades([
        {"trader": "A", "ticker": "MSFT", "txn_date": "2025-01-01"},
        {"trader": "B", "ticker": "AAPL", "txn_date": "2025-01-02"},
        {"trader": "C", "ticker": "GOOG", "txn_date": "2025-01-03"},
    ])
    sector_map = {"MSFT": "Tech", "AAPL": "Tech", "GOOG": "Tech"}
    out = label_sector_cluster(df, sector_map, window_days=3, min_traders=3)
    assert out.tolist() == [True, True, True]


def test_sector_cluster_does_not_fire_when_tickers_unmapped():
    df = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"},
        {"trader": "B", "ticker": "BAR", "txn_date": "2025-01-02"},
        {"trader": "C", "ticker": "BAZ", "txn_date": "2025-01-03"},
    ])
    out = label_sector_cluster(df, sector_map={}, window_days=3, min_traders=3)
    assert out.tolist() == [False, False, False]


# --- label_pre_event -------------------------------------------------------

def test_pre_event_fires_within_window_before_event():
    trades = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"},
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-08"},  # 7 days before -- inclusive
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-02-01"},  # 30 days before -- outside
    ])
    events = pd.DataFrame([{"ticker": "FOO", "event_date": pd.Timestamp("2025-01-15")}])
    out = label_pre_event(trades, events, window_days=7)
    assert out.tolist() == [False, True, False]


def test_pre_event_does_not_fire_when_trade_is_after_event():
    trades = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-20"},  # AFTER event
    ])
    events = pd.DataFrame([{"ticker": "FOO", "event_date": pd.Timestamp("2025-01-15")}])
    assert label_pre_event(trades, events, window_days=7).tolist() == [False]


def test_pre_event_handles_empty_events():
    trades = _trades([{"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"}])
    out = label_pre_event(trades, pd.DataFrame(columns=["ticker", "event_date"]), window_days=7)
    assert out.tolist() == [False]


def test_pre_event_handles_none_events():
    trades = _trades([{"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"}])
    assert label_pre_event(trades, None, window_days=7).tolist() == [False]


# --- apply_all_labels ------------------------------------------------------

def test_apply_all_labels_adds_expected_columns():
    df = _trades([
        {"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01", "trade_value": 100.0},
        {"trader": "B", "ticker": "FOO", "txn_date": "2025-01-02", "trade_value": 100.0},
        {"trader": "C", "ticker": "FOO", "txn_date": "2025-01-03", "trade_value": 100.0},
    ])
    apply_all_labels(df, sector_map={}, events_df=None)
    for col in PATTERN_COLUMNS:
        assert col in df.columns
    assert "pattern_count" in df.columns
    assert "pattern_set" in df.columns
    assert "pattern_count_bucket" in df.columns
    # Same-company cluster should fire on all three rows
    assert df["pat_same_company"].tolist() == [True, True, True]
    # Pattern set string includes the rule name
    assert all("same_company" in s for s in df["pattern_set"])


def test_pattern_count_bucket_thresholds():
    df = _trades([{"trader": "A", "ticker": "FOO", "txn_date": "2025-01-01"}])
    apply_all_labels(df, sector_map={}, events_df=None)
    # Singleton, no events, no sector → all rules off
    assert df["pattern_count"].iloc[0] == 0
    assert df["pattern_count_bucket"].iloc[0] == "0_none"
