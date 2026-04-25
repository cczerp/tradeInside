"""Tests for benchmarks.py — sector → ETF mapping."""
from benchmarks import (
    FALLBACK_ETF,
    SECTOR_TO_ETF,
    all_benchmark_symbols,
    benchmark_for_ticker,
)


def test_known_sector_maps_to_expected_etf():
    sector_map = {"NVDA": "Tech", "PFE": "Pharma", "JPM": "Finance"}
    assert benchmark_for_ticker("NVDA", sector_map) == "XLK"
    assert benchmark_for_ticker("PFE", sector_map) == "XBI"
    assert benchmark_for_ticker("JPM", sector_map) == "XLF"


def test_unknown_ticker_falls_back_to_spy():
    assert benchmark_for_ticker("ZZZZZ", {}) == FALLBACK_ETF
    assert benchmark_for_ticker(None, {}) == FALLBACK_ETF
    assert benchmark_for_ticker("", {}) == FALLBACK_ETF


def test_unmapped_sector_falls_back_to_spy():
    sector_map = {"FOO": "Holdings", "BAR": "Other"}
    assert benchmark_for_ticker("FOO", sector_map) == FALLBACK_ETF
    assert benchmark_for_ticker("BAR", sector_map) == FALLBACK_ETF


def test_ticker_case_is_normalised():
    sector_map = {"NVDA": "Tech"}
    assert benchmark_for_ticker("nvda", sector_map) == "XLK"


def test_all_benchmark_symbols_includes_spy_and_is_sorted():
    syms = all_benchmark_symbols()
    assert FALLBACK_ETF in syms
    assert syms == sorted(set(syms))


def test_all_benchmark_symbols_with_sector_map_filters_to_needed():
    sector_map = {"NVDA": "Tech", "PFE": "Pharma"}
    syms = all_benchmark_symbols(sector_map)
    assert "XLK" in syms
    assert "XBI" in syms
    assert "SPY" in syms  # always included as fallback
    # We didn't ask for any Energy/Finance/etc. tickers
    assert "XLE" not in syms


def test_sector_to_etf_has_no_self_reference():
    """Sanity: every value in SECTOR_TO_ETF should be a real ETF ticker,
    not a sector label."""
    sector_labels = set(SECTOR_TO_ETF.keys())
    for sector, etf in SECTOR_TO_ETF.items():
        assert etf not in sector_labels, f"{sector} → {etf} looks like a sector label, not an ETF"
        assert len(etf) <= 5  # ETF tickers are short
        assert etf.isupper()
