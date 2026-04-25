"""Sector → ETF mapping for benchmark selection.

The analyzer already labels every ticker with a sector (analyzer.SECTOR_MAP).
This module maps those sector labels onto the sector-SPDR ETF that best
tracks them, so `backtest.py --sector-adjust` can subtract a sector-aware
benchmark instead of plain SPY.

A purer benchmark = a more honest alpha number. Selling "outperforms SPY"
is a weaker pitch to a quant buyer than "outperforms its own sector ETF."

Tickers whose sector isn't mapped (or not in SECTOR_MAP at all) fall back
to SPY.
"""
from __future__ import annotations

from typing import Optional

# Sector SPDRs cover the GICS top level cleanly. XBI is added because the
# analyzer separates Biotech/Pharma from broader Healthcare and the
# small/mid biotech ETF tracks insider-trading-active names better than
# XLV.
DEFAULT_BENCHMARKS = ("SPY", "XLK", "XLF", "XLV", "XBI", "XLE", "XLI",
                      "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB")

SECTOR_TO_ETF = {
    # Tech / Software / Telecom / Media / Gaming → tech & comms
    "Tech":            "XLK",
    "Software":        "XLK",
    "Telecom":         "XLC",
    "Media":           "XLC",
    "Gaming":          "XLC",
    "Crypto":          "XLK",   # mostly tech-correlated equity tickers

    # Healthcare branches
    "Biotech":         "XBI",
    "Pharma":          "XBI",
    "Healthcare":      "XLV",

    # Financials
    "Finance":          "XLF",
    "Insurance":        "XLF",
    "SpecialtyFinance": "XLF",

    # Energy / Materials / Industrials
    "Energy":          "XLE",
    "Mining":          "XLB",
    "Industrial":      "XLI",
    "Aerospace":       "XLI",
    "Shipping":        "XLI",

    # Consumer / Real Estate / Utilities
    "Retail":          "XLY",
    "Consumer":        "XLP",
    "Agriculture":     "XLP",
    "RealEstate":      "XLRE",
    "Cannabis":        "XLY",   # closest match — discretionary
    "Utilities":       "XLU",

    # Long tail of analyzer sectors
    "Services":        "XLY",
    "Construction":    "XLI",
    "Chemical":        "XLB",
    "Semiconductor":   "XLK",
    "Diagnostics":     "XLV",
    "Marketing":       "XLC",
    "Marine":          "XLI",
    "Research":        "XLV",
    "Environmental":   "XLB",
    # "Holdings" / "Other" intentionally absent — fall through to SPY.
}

FALLBACK_ETF = "SPY"


def benchmark_for_ticker(ticker: str, sector_map: dict) -> str:
    """Return the best benchmark ETF symbol for a given ticker.

    ``sector_map`` is a {ticker: sector_name} dict (typically
    analyzer.SECTOR_MAP). Falls back to SPY when the ticker is unknown
    or its sector has no ETF mapping.
    """
    if not ticker:
        return FALLBACK_ETF
    sector = sector_map.get(ticker.upper())
    if not sector:
        return FALLBACK_ETF
    return SECTOR_TO_ETF.get(sector, FALLBACK_ETF)


def all_benchmark_symbols(sector_map: Optional[dict] = None) -> list[str]:
    """All benchmark symbols we might need, deduplicated and sorted.

    Includes the fallback (SPY) plus every ETF referenced by the sector
    mapping. Pass a ``sector_map`` to filter to only the ETFs your data
    actually needs (saves a few yfinance calls).
    """
    if sector_map is None:
        return sorted(set(DEFAULT_BENCHMARKS))
    needed = {FALLBACK_ETF}
    for ticker, sector in sector_map.items():
        etf = SECTOR_TO_ETF.get(sector, FALLBACK_ETF)
        needed.add(etf)
    return sorted(needed)
