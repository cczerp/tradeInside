"""Standalone re-implementation of the analyzer's pattern detection rules.

Each labeler returns a boolean Series aligned to ``trades_df``: True when
the rule fires for that row. Together they let backtest.py answer "does
each individual rule actually predict abnormal returns, or is the
analyzer's score dominated by one rule pulling the rest along?"

Rules implemented:
  - large_trade:        trade $ value > multiplier × trader's mean trade $
  - same_company_cluster: ≥ min_insiders distinct traders trading the same
                          ticker inside a ±window_days window centered on
                          the row
  - sector_cluster:     ≥ min_traders distinct traders trading the same
                          sector inside a ±window_days window
  - pre_event:          a corporate event for this ticker falls within
                          window_days *after* the trade

Rules deliberately not implemented here:
  - perfect_timing: circular when the goal is to measure forward returns
  - late_filing: filing-date data is dirty across sources
  - coordinated (≥5 traders same sector same day): the analyzer's
    notorious 42-trader false positive — we'll surface it via
    sector_cluster + pattern_count instead, and use the per-rule numbers
    to decide whether/how to keep it
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

LARGE_TRADE_MULTIPLIER = 3.0
SAME_COMPANY_WINDOW_DAYS = 7
SAME_COMPANY_MIN_INSIDERS = 3
SECTOR_WINDOW_DAYS = 3
SECTOR_MIN_TRADERS = 3
PRE_EVENT_WINDOW_DAYS = 7

PATTERN_COLUMNS = (
    "pat_large_trade",
    "pat_same_company",
    "pat_sector_cluster",
    "pat_pre_event",
)


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def label_large_trade(trades_df, multiplier: float = LARGE_TRADE_MULTIPLIER):
    """Per-trader 'unusually large' trade: $ value > multiplier × trader mean.

    Requires ``trade_value`` and ``trader`` columns (provided by
    backtest.add_segmentation_columns). Traders with only one trade can't
    have an above-average trade by this definition, so they evaluate to
    False (mean == value, ratio == 1).
    """
    import pandas as pd

    trader_mean = trades_df.groupby("trader")["trade_value"].transform("mean")
    fired = trades_df["trade_value"] > (multiplier * trader_mean)
    return fired.fillna(False).astype(bool)


def _window_distinct_count(group, anchor_date, anchor_idx, window_days, key_col):
    """Helper: in ``group``, count distinct ``key_col`` values within
    ±window_days of ``anchor_date``, excluding nothing (the anchor itself
    is included since we want 'is this trade part of a cluster')."""
    start = anchor_date - timedelta(days=window_days)
    end = anchor_date + timedelta(days=window_days)
    mask = (group["txn_date"] >= start) & (group["txn_date"] <= end)
    return int(group.loc[mask, key_col].nunique())


def label_same_company_cluster(
    trades_df,
    window_days: int = SAME_COMPANY_WINDOW_DAYS,
    min_insiders: int = SAME_COMPANY_MIN_INSIDERS,
):
    """Insider cluster on a single ticker.

    True when ≥ min_insiders distinct traders traded ticker T within
    ±window_days of the row's txn_date.
    """
    import pandas as pd

    fired = pd.Series(False, index=trades_df.index)
    for ticker, group in trades_df.groupby("ticker"):
        if group["trader"].nunique() < min_insiders:
            continue
        for idx, row in group.iterrows():
            distinct = _window_distinct_count(
                group, row["txn_date"], idx, window_days, "trader"
            )
            if distinct >= min_insiders:
                fired.at[idx] = True
    return fired


def label_sector_cluster(
    trades_df,
    sector_map: dict,
    window_days: int = SECTOR_WINDOW_DAYS,
    min_traders: int = SECTOR_MIN_TRADERS,
):
    """Cluster across a whole sector.

    True when ≥ min_traders distinct traders traded any ticker mapped to
    sector S within ±window_days. Tickers without a sector mapping are
    skipped (treated as no signal). The notorious 'Coordinated' analyzer
    rule is a strict subset of this with a different threshold; this
    labeler lets us measure whether the *idea* has signal independent of
    the threshold.
    """
    import pandas as pd

    df = trades_df.assign(sector=trades_df["ticker"].map(lambda t: sector_map.get(t)))
    fired = pd.Series(False, index=trades_df.index)
    for sector, group in df.dropna(subset=["sector"]).groupby("sector"):
        if group["trader"].nunique() < min_traders:
            continue
        for idx, row in group.iterrows():
            distinct = _window_distinct_count(
                group, row["txn_date"], idx, window_days, "trader"
            )
            if distinct >= min_traders:
                fired.at[idx] = True
    return fired


def label_pre_event(
    trades_df,
    events_df,
    window_days: int = PRE_EVENT_WINDOW_DAYS,
):
    """Trade lands within ``window_days`` *before* a corporate event.

    ``events_df`` must have ``ticker`` and ``event_date`` columns
    (event_date as datetime.date or pandas Timestamp). Returns all-False
    when ``events_df`` is None or empty.
    """
    import pandas as pd

    if events_df is None or len(events_df) == 0:
        return pd.Series(False, index=trades_df.index)

    # {ticker: sorted list of event dates}
    by_ticker: dict[str, list] = {}
    for tkr, sub in events_df.groupby("ticker"):
        dates = sorted(d for d in sub["event_date"] if d is not None)
        if dates:
            by_ticker[tkr] = dates

    fired = pd.Series(False, index=trades_df.index)
    for idx, row in trades_df.iterrows():
        anchor = row["txn_date"]
        if anchor is None:
            continue
        events = by_ticker.get(row["ticker"], [])
        upper = anchor + timedelta(days=window_days)
        for ev in events:
            # Pandas Timestamp comparison with datetime.date works both ways.
            ev_d = ev.date() if hasattr(ev, "date") else ev
            if anchor < ev_d <= upper:
                fired.at[idx] = True
                break
    return fired


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def apply_all_labels(
    trades_df,
    sector_map: Optional[dict] = None,
    events_df=None,
):
    """Add the four boolean pattern columns plus pattern_count and pattern_set.

    pattern_count: 0..4 — the number of independent rules that fired
    pattern_set:   comma-joined sorted rule names (or 'none')

    Returns the modified DataFrame (mutates in place and also returns it).
    """
    sector_map = sector_map or {}

    trades_df["pat_large_trade"] = label_large_trade(trades_df)
    trades_df["pat_same_company"] = label_same_company_cluster(trades_df)
    trades_df["pat_sector_cluster"] = label_sector_cluster(trades_df, sector_map)
    trades_df["pat_pre_event"] = label_pre_event(trades_df, events_df)

    trades_df["pattern_count"] = sum(trades_df[c].astype(int) for c in PATTERN_COLUMNS)

    def _set(row):
        active = sorted(
            c.replace("pat_", "") for c in PATTERN_COLUMNS if row[c]
        )
        return ",".join(active) if active else "none"
    trades_df["pattern_set"] = trades_df.apply(_set, axis=1)

    # Bucketed pattern_count for cleaner segmentation.
    def _bucket(c):
        if c == 0:
            return "0_none"
        if c == 1:
            return "1_one"
        if c == 2:
            return "2_two"
        return "3+_stacked"
    trades_df["pattern_count_bucket"] = trades_df["pattern_count"].map(_bucket)

    return trades_df
