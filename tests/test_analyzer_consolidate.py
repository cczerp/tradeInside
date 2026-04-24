"""Tests for analyzer.consolidate_patterns.

analyzer.py imports pandas and yfinance at module scope; this test skips if
either is missing.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from analyzer import consolidate_patterns


def test_empty_list_returns_empty():
    assert consolidate_patterns([]) == []


def test_single_timing_pattern_passes_through():
    out = consolidate_patterns(["Perfect timing: AAPL (+15.0% in 14 days)"])
    assert len(out) == 1
    assert "AAPL" in out[0]
    assert "+15.0%" in out[0]


def test_multiple_same_ticker_timings_collapse_with_count():
    patterns = [
        "Perfect timing: AAPL (+15.0% in 14 days)",
        "Perfect timing: AAPL (+20.0% in 14 days)",
        "Perfect timing: AAPL (+18.0% in 14 days)",
    ]
    out = consolidate_patterns(patterns)
    merged = [p for p in out if "AAPL" in p]
    assert len(merged) == 1
    assert "x3" in merged[0]
    assert "+15.0-20.0%" in merged[0]


def test_non_timing_patterns_pass_through_unchanged():
    p = "Same-company cluster: 3 insiders at AAPL within 7 days"
    assert consolidate_patterns([p]) == [p]


def test_coordinated_pattern_dedupes():
    p = "Coordinated: 5 traders in Tech on 2024-01-15"
    out = consolidate_patterns([p, p, p])
    assert out.count(p) == 1
