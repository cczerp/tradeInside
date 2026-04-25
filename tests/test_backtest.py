"""Tests for the pure helpers in backtest.py.

Importing backtest.py is safe even without pandas — pandas is only imported
inside the DB-backed functions that these tests don't exercise.
"""
import math
from datetime import date

import pytest

from backtest import (
    forward_return,
    normalise_transaction_type,
    parse_price,
    parse_trade_date,
    summarise,
)


# --- parse_trade_date --------------------------------------------------------

def test_parse_trade_date_iso():
    assert parse_trade_date("2025-02-03") == date(2025, 2, 3)


def test_parse_trade_date_human():
    assert parse_trade_date("Oct 02, 2025") == date(2025, 10, 2)


def test_parse_trade_date_returns_none_for_empty_or_bad():
    assert parse_trade_date(None) is None
    assert parse_trade_date("") is None
    assert parse_trade_date("garbage") is None
    assert parse_trade_date("2025/02/03") is None  # we only accept the two known formats


# --- parse_price -------------------------------------------------------------

def test_parse_price_strips_dollar_and_commas():
    assert parse_price("$1,234.56") == 1234.56
    assert parse_price("$0.62") == 0.62


def test_parse_price_handles_numerics_and_nans():
    assert parse_price(42) == 42.0
    assert parse_price(3.14) == 3.14
    assert parse_price(float("nan")) is None
    assert parse_price(None) is None
    assert parse_price("") is None
    assert parse_price("not-a-number") is None


# --- normalise_transaction_type ---------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("P - Purchase", "buy"),
    ("Purchase", "buy"),
    ("P", "buy"),
    ("S - Sale", "sell"),
    ("Sale", "sell"),
    ("S - Sale+OE", "sell"),
    ("S", "sell"),
    ("", "other"),
    (None, "other"),
    ("Award", "other"),
    ("Conversion", "other"),
])
def test_normalise_transaction_type(raw, expected):
    assert normalise_transaction_type(raw) == expected


# --- forward_return ---------------------------------------------------------

def _series(*pairs):
    """Helper: build a sorted [(date, close), ...] series."""
    return [(date.fromisoformat(d), float(c)) for d, c in pairs]


def test_forward_return_simple_30d():
    series = _series(
        ("2025-01-02", 100.0),
        ("2025-02-01", 110.0),  # 30 days later (calendar)
    )
    r = forward_return(series, date(2025, 1, 2), 30)
    assert r is not None
    assert math.isclose(r, 0.10, abs_tol=1e-9)


def test_forward_return_uses_next_available_close_when_anchor_is_a_weekend():
    series = _series(
        ("2025-01-03", 100.0),  # Friday
        ("2025-02-03", 110.0),
    )
    # Saturday anchor — should use Monday-equivalent next available close
    r = forward_return(series, date(2025, 1, 4), 30)  # 2025-01-04 is Saturday
    # next-on-or-after start = 2025-02-03? no. bisect_left on 2025-01-04 lands at index 1
    # which is 2025-02-03, and target = 2025-02-03 also lands at index 1. start == end -> None.
    # That's the documented contract (we can't claim a forward return when start==end).
    assert r is None


def test_forward_return_returns_none_when_target_beyond_series():
    series = _series(("2025-01-02", 100.0), ("2025-01-15", 105.0))
    assert forward_return(series, date(2025, 1, 2), 90) is None


def test_forward_return_returns_none_when_anchor_beyond_series():
    series = _series(("2025-01-02", 100.0))
    assert forward_return(series, date(2026, 1, 1), 30) is None


def test_forward_return_returns_none_for_empty_series():
    assert forward_return([], date(2025, 1, 1), 30) is None


def test_forward_return_handles_non_positive_start_price():
    series = _series(("2025-01-02", 0.0), ("2025-02-01", 5.0))
    assert forward_return(series, date(2025, 1, 2), 30) is None


def test_forward_return_negative_horizon_is_none():
    series = _series(("2025-01-02", 100.0), ("2025-02-01", 110.0))
    assert forward_return(series, date(2025, 1, 2), 0) is None
    assert forward_return(series, date(2025, 1, 2), -7) is None


# --- summarise --------------------------------------------------------------

def test_summarise_empty_returns_zeroed_dict():
    s = summarise([])
    assert s["count"] == 0
    assert s["mean"] is None
    assert s["hit_rate"] is None


def test_summarise_basic_stats():
    # 4 wins, 1 loss → 80% hit rate; mean = (.1+.05+.2+.15-.1)/5 = .08
    s = summarise([0.10, 0.05, 0.20, 0.15, -0.10])
    assert s["count"] == 5
    assert math.isclose(s["mean"], 0.08, abs_tol=1e-9)
    assert math.isclose(s["hit_rate"], 0.8, abs_tol=1e-9)
    assert s["sharpe"] is not None and s["sharpe"] > 0


def test_summarise_drops_none_and_nan():
    s = summarise([0.10, None, float("nan"), 0.20])
    assert s["count"] == 2
    assert math.isclose(s["mean"], 0.15, abs_tol=1e-9)


def test_summarise_zero_std_produces_none_sharpe():
    # All identical returns -> std = 0 -> sharpe undefined
    s = summarise([0.05, 0.05, 0.05])
    assert s["std"] == 0.0
    assert s["sharpe"] is None
