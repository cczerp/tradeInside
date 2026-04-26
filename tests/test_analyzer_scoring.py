"""Tests for analyzer's data-driven scoring constants.

Importing analyzer.py requires pandas + yfinance. Skips cleanly if either
isn't installed.
"""
import os

import pytest

pytest.importorskip("pandas")
pytest.importorskip("yfinance")


def _reload_analyzer(monkeypatch, *, legacy: bool):
    """Reload analyzer with ANALYZER_USE_LEGACY_SCORING set or unset."""
    if legacy:
        monkeypatch.setenv("ANALYZER_USE_LEGACY_SCORING", "1")
    else:
        monkeypatch.delenv("ANALYZER_USE_LEGACY_SCORING", raising=False)
    import importlib
    import analyzer
    return importlib.reload(analyzer)


def test_default_large_trade_score_is_zero(monkeypatch):
    a = _reload_analyzer(monkeypatch, legacy=False)
    assert a.LARGE_TRADE_SCORE == 0


def test_legacy_large_trade_score_is_eight(monkeypatch):
    a = _reload_analyzer(monkeypatch, legacy=True)
    assert a.LARGE_TRADE_SCORE == 8


def test_default_pattern_multiplier_is_flat(monkeypatch):
    a = _reload_analyzer(monkeypatch, legacy=False)
    assert a.get_pattern_multiplier(0) == 0.0
    assert a.get_pattern_multiplier(1) == 1.0
    assert a.get_pattern_multiplier(2) == 1.0
    assert a.get_pattern_multiplier(3) == 1.0
    assert a.get_pattern_multiplier(7) == 1.0


def test_legacy_pattern_multiplier_keeps_old_ladder(monkeypatch):
    a = _reload_analyzer(monkeypatch, legacy=True)
    assert a.get_pattern_multiplier(0) == 0.0
    assert a.get_pattern_multiplier(1) == 1.0
    assert a.get_pattern_multiplier(2) == 1.5
    assert a.get_pattern_multiplier(3) == 2.5
    assert a.get_pattern_multiplier(10) == 2.5  # caps at the 3+ bucket


def test_pattern_multiplier_handles_negative_count(monkeypatch):
    a = _reload_analyzer(monkeypatch, legacy=False)
    assert a.get_pattern_multiplier(-1) == 0.0
