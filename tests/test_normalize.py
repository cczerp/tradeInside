"""Tests for normalize.normalize_columns.

Requires pandas. Skips cleanly if it isn't installed.
"""
import pytest

pd = pytest.importorskip("pandas")

from normalize import normalize_columns, get_standard_columns


def test_basic_rename():
    df = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Insider Name": ["Jane Doe"],
        "Trade Date": ["2024-01-15"],
        "Transaction": ["Purchase"],
        "Qty": [100],
        "Share Price": [180.0],
    })
    out = normalize_columns(df)
    assert set(["ticker", "trader_name", "transaction_date",
                "transaction_type", "shares", "price"]).issubset(out.columns)


def test_nbsp_columns_are_handled():
    # OpenInsider / QuiverQuant pages sometimes render headers with NBSP.
    df = pd.DataFrame({
        "Trade\xa0Date": ["2024-01-15"],
        "Trade\xa0Type": ["P"],
        "Company\xa0Name": ["Apple"],
        "Filing\xa0Date": ["2024-01-17"],
    })
    out = normalize_columns(df)
    assert "transaction_date" in out.columns
    assert "transaction_type" in out.columns
    assert "company" in out.columns
    assert "filing_date" in out.columns


def test_unmapped_columns_pass_through_unchanged():
    df = pd.DataFrame({"ticker": ["AAPL"], "weirdcustom": [1]})
    out = normalize_columns(df)
    assert "weirdcustom" in out.columns
    assert "ticker" in out.columns


def test_case_insensitive_matching():
    df = pd.DataFrame({"TICKER": ["AAPL"], "SYMBOL": ["MSFT"]})
    out = normalize_columns(df)
    # Both map to 'ticker'; pandas will end up with duplicate names but they
    # are both normalised (we just verify neither original header survived).
    assert "TICKER" not in out.columns
    assert "SYMBOL" not in out.columns


def test_politician_and_fund_aliases_map_to_trader_name():
    df = pd.DataFrame({"Politician": ["Pelosi"]})
    out = normalize_columns(df)
    assert "trader_name" in out.columns

    df2 = pd.DataFrame({"Fund": ["Berkshire"]})
    out2 = normalize_columns(df2)
    assert "trader_name" in out2.columns


def test_standard_columns_list_contains_core_fields():
    cols = get_standard_columns()
    for expected in ("trader_name", "ticker", "transaction_type",
                     "transaction_date", "shares", "price"):
        assert expected in cols
