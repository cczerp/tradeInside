"""Tests for pure helpers in migrate_data.

Requires pandas (since migrate_data imports it at module scope). Skips cleanly
if pandas isn't installed.
"""
from datetime import date

import pytest

pytest.importorskip("pandas")

from migrate_data import parse_date, clean_numeric


def test_parse_date_handles_mm_dd_yyyy():
    assert parse_date("01/15/2024") == date(2024, 1, 15)


def test_parse_date_handles_iso():
    assert parse_date("2024-01-15") == date(2024, 1, 15)


def test_parse_date_returns_none_on_empty_or_garbage():
    assert parse_date("") is None
    assert parse_date("not-a-date") is None


def test_clean_numeric_strips_currency_and_commas():
    assert clean_numeric("$1,234.56") == 1234.56
    assert clean_numeric("  $42 ") == 42.0


def test_clean_numeric_returns_none_on_bad_input():
    assert clean_numeric("") is None
    assert clean_numeric("abc") is None
