"""Forward-return backtest for insider trades.

Loads trades from tradeinsider.db, computes 1/7/30/90-day forward returns
from stock_prices, and reports per-segment statistics (count, mean return,
hit rate, Sharpe). Optional market-adjustment against SPY when SPY data is
present in the price table.

The pure helpers near the top (parse_trade_date, normalise_transaction_type,
parse_price, forward_return, summarise) take simple Python data structures
and are exercised by tests/test_backtest.py without requiring pandas.

Run:
    python3 backtest.py
    python3 backtest.py --since 2025-01-01 --horizons 7,30,90
    python3 backtest.py --market-adjust    # requires SPY rows in stock_prices
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from bisect import bisect_left
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

from logging_config import get_logger

log = get_logger(__name__)

DB_FILE = os.path.join(os.path.dirname(__file__), "tradeinsider.db")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
DEFAULT_HORIZONS = (1, 7, 30, 90)
SUCCESS_THRESHOLD = 0.0  # forward return strictly above this counts as a hit


# ---------------------------------------------------------------------------
# Pure helpers (covered by tests/test_backtest.py)
# ---------------------------------------------------------------------------

_HUMAN_DATE = re.compile(r"^[A-Za-z]{3} \d{1,2}, \d{4}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_trade_date(value: Optional[str]) -> Optional[date]:
    """Parse a trade transaction_date as stored in the DB.

    Handles ISO ('2025-02-03') from openinsider and human ('Oct 02, 2025')
    from quiver. Returns None for empty / unparseable input.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if _ISO_DATE.match(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return None
    if _HUMAN_DATE.match(s):
        try:
            return datetime.strptime(s, "%b %d, %Y").date()
        except ValueError:
            return None
    return None


def parse_price(value) -> Optional[float]:
    """Parse a price field that may be a float, int, or '$1,234.56' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    s = str(value).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalise_transaction_type(value: Optional[str]) -> str:
    """Collapse the messy transaction_type column into 'buy', 'sell', 'other'.

    OpenInsider prefixes with 'P -' / 'S -'; quiver uses bare 'Purchase' /
    'Sale'. Anything ambiguous is 'other'.
    """
    if not value:
        return "other"
    s = str(value).lower()
    if "purchase" in s or s.startswith("p -") or s == "p":
        return "buy"
    if "sale" in s or s.startswith("s -") or s == "s":
        return "sell"
    return "other"


def forward_return(
    sorted_prices: list[tuple[date, float]],
    anchor: date,
    horizon_days: int,
) -> Optional[float]:
    """Compute forward return between the first close on/after ``anchor`` and
    the first close on/after ``anchor + horizon_days``.

    ``sorted_prices`` is a list of (date, close) sorted ascending by date.
    Using the next-available-close on each side handles weekends / holidays
    without requiring a trading-day calendar. Returns None if either side
    can't be located, or if the start price is non-positive.
    """
    if not sorted_prices or horizon_days <= 0:
        return None

    dates = [d for d, _ in sorted_prices]

    i = bisect_left(dates, anchor)
    if i >= len(dates):
        return None
    start_date, start_price = sorted_prices[i]
    if start_price is None or start_price <= 0:
        return None

    target = anchor + timedelta(days=horizon_days)
    j = bisect_left(dates, target)
    if j >= len(dates):
        return None
    end_date, end_price = sorted_prices[j]
    if end_price is None or end_price <= 0:
        return None

    # Sanity: don't claim a return if we couldn't actually advance forward.
    if end_date <= start_date:
        return None

    return (end_price / start_price) - 1.0


def summarise(returns: Iterable[float], success_threshold: float = SUCCESS_THRESHOLD) -> dict:
    """Summary statistics for a list of forward returns.

    Returns dict with: count, mean, median, std, hit_rate, sharpe.
    'sharpe' is mean/std (no risk-free adjustment, no annualisation) — a
    coarse signal-to-noise indicator, not a tradable Sharpe.
    """
    values = [float(r) for r in returns if r is not None and not math.isnan(float(r))]
    n = len(values)
    if n == 0:
        return {
            "count": 0, "mean": None, "median": None, "std": None,
            "hit_rate": None, "sharpe": None,
        }
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    hit_rate = sum(1 for v in values if v > success_threshold) / n
    sharpe = (mean / std) if std > 0 else None
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std": std,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# DB-backed orchestration (uses pandas for ergonomic reporting)
# ---------------------------------------------------------------------------


def _connect(db_file: str) -> sqlite3.Connection:
    return sqlite3.connect(db_file)


def load_trades(conn: sqlite3.Connection, since: Optional[date] = None):
    """Return the trades table, with parsed date, normalised side, and the
    trader name coalesced across insider/politician/fund. Rows with an
    unparseable date are dropped."""
    import pandas as pd

    df = pd.read_sql_query(
        "SELECT id, source, ticker, transaction_date, transaction_type, "
        "       insider_name, politician_name, fund_name, shares, price "
        "FROM trades",
        conn,
    )
    df["txn_date"] = df["transaction_date"].map(parse_trade_date)
    df["side"] = df["transaction_type"].map(normalise_transaction_type)
    df["price_f"] = df["price"].map(parse_price)
    df["trader"] = (
        df["insider_name"].fillna("")
        .where(df["insider_name"].notna(), df["politician_name"].fillna(""))
    )
    df.loc[df["trader"] == "", "trader"] = df["fund_name"].fillna("")

    df = df[df["txn_date"].notna()].copy()
    if since is not None:
        df = df[df["txn_date"] >= since].copy()
    return df


def load_prices(conn: sqlite3.Connection, tickers: Iterable[str]) -> dict[str, list[tuple[date, float]]]:
    """Return {ticker: sorted [(date, close), ...]} for the given tickers."""
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return {}

    out: dict[str, list[tuple[date, float]]] = {t: [] for t in tickers}
    chunk = 500
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        cur = conn.execute(
            f"SELECT ticker, date, close FROM stock_prices "
            f"WHERE ticker IN ({placeholders}) AND close IS NOT NULL "
            f"ORDER BY ticker, date",
            batch,
        )
        for tkr, d, close in cur:
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            out[tkr].append((dt, float(close)))
    # Already ORDERed in SQL but be defensive.
    for tkr in out:
        out[tkr].sort()
    return out


def compute_returns_table(trades_df, prices_by_ticker: dict, horizons: Iterable[int]):
    """Add ret_<H>d columns to a trades dataframe."""
    import pandas as pd

    horizons = list(horizons)
    new_cols = {f"ret_{h}d": [] for h in horizons}

    for _, row in trades_df.iterrows():
        anchor = row["txn_date"]
        prices = prices_by_ticker.get(row["ticker"], [])
        for h in horizons:
            ret = forward_return(prices, anchor, h)
            new_cols[f"ret_{h}d"].append(ret)

    for col, values in new_cols.items():
        trades_df[col] = values
    return trades_df


def add_segmentation_columns(trades_df):
    """Add unbiased segmentation columns used in the per-segment report.

    - ``trader_freq``: how many trades this trader has in the dataset
      (singleton / light_2-5 / moderate_6-20 / heavy_21+). Tests whether
      analyzer's repeat-offender multipliers are justified.
    - ``trade_size_q``: $ value quartile (q1=smallest .. q4=largest). Tests
      whether the analyzer's 3x-large-trade rule has predictive power.
    """
    counts = trades_df.groupby("trader").size()

    def freq_bucket(c):
        if c <= 1:
            return "singleton"
        if c <= 5:
            return "light_2-5"
        if c <= 20:
            return "moderate_6-20"
        return "heavy_21+"

    trades_df["trader_freq"] = trades_df["trader"].map(counts).map(freq_bucket)

    # $ value = shares * price; many rows have NaN shares or price, so this
    # column will have NaNs that pandas .qcut handles by leaving them out.
    import pandas as pd
    shares = pd.to_numeric(trades_df["shares"], errors="coerce")
    value = shares * trades_df["price_f"]
    trades_df["trade_value"] = value
    try:
        trades_df["trade_size_q"] = pd.qcut(
            value, q=4, labels=["q1_smallest", "q2", "q3", "q4_largest"],
            duplicates="drop",
        ).astype(object)
    except ValueError:
        trades_df["trade_size_q"] = None
    trades_df["trade_size_q"] = trades_df["trade_size_q"].fillna("unknown")

    return trades_df


def market_adjust(returns_df, spy_prices: list[tuple[date, float]], horizons: Iterable[int]):
    """Subtract SPY's same-window return from each ret_<H>d, producing
    excess_<H>d columns. No-op if spy_prices is empty."""
    if not spy_prices:
        return returns_df
    for h in horizons:
        col = f"ret_{h}d"
        excess_col = f"excess_{h}d"
        spy_returns = [
            forward_return(spy_prices, d, h) if d is not None else None
            for d in returns_df["txn_date"]
        ]
        returns_df[excess_col] = [
            (r - s) if (r is not None and s is not None) else None
            for r, s in zip(returns_df[col], spy_returns)
        ]
    return returns_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _flip_sign_for_sells(returns_df, return_col: str):
    """For sales, an *insider win* is the price going down — so flip the sign
    of sell returns when computing 'hit rate' from the trader's perspective.
    Returns a list of perspective-adjusted returns aligned with the df."""
    sign = returns_df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
    return [
        (sign_v * r) if (r is not None) else None
        for sign_v, r in zip(sign, returns_df[return_col])
    ]


def segment_report(returns_df, segment_col: str, horizons: Iterable[int],
                   adjusted: bool = False) -> list[dict]:
    """Per-segment summary table for each horizon.

    Reports both raw returns and 'insider perspective' returns (flipped sign
    for sells) so the hit-rate column is interpretable: hit = the trade went
    the way the insider was positioned.
    """
    out = []
    prefix = "excess_" if adjusted else "ret_"
    for seg, group in returns_df.groupby(segment_col, dropna=False):
        for h in horizons:
            col = f"{prefix}{h}d"
            if col not in group.columns:
                continue
            persp = _flip_sign_for_sells(group, col)
            stats = summarise(persp)
            stats.update({
                "segment_field": segment_col,
                "segment_value": "" if seg is None else str(seg),
                "horizon_days": h,
                "adjusted": adjusted,
            })
            out.append(stats)
    return out


def render_text_report(rows: list[dict], title: str) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(title)
    lines.append("=" * 90)
    if not rows:
        lines.append("(no rows)")
        return "\n".join(lines)

    header = f"{'segment':<30}{'horizon':>9}{'n':>7}{'mean':>10}{'median':>10}{'hit%':>8}{'sharpe':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    rows_sorted = sorted(rows, key=lambda r: (r["segment_field"], r["segment_value"], r["horizon_days"]))
    last_seg = None
    for r in rows_sorted:
        seg_label = f"{r['segment_field']}={r['segment_value'] or '∅'}"
        if seg_label == last_seg:
            seg_label_disp = ""
        else:
            seg_label_disp = seg_label
            last_seg = seg_label
        mean_s = "—" if r["mean"] is None else f"{r['mean'] * 100:+.2f}%"
        med_s = "—" if r["median"] is None else f"{r['median'] * 100:+.2f}%"
        hit_s = "—" if r["hit_rate"] is None else f"{r['hit_rate'] * 100:.1f}"
        sh_s = "—" if r["sharpe"] is None else f"{r['sharpe']:.2f}"
        lines.append(
            f"{seg_label_disp:<30}{r['horizon_days']:>9}{r['count']:>7}"
            f"{mean_s:>10}{med_s:>10}{hit_s:>8}{sh_s:>9}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest insider trades.")
    parser.add_argument("--db", default=DB_FILE, help="path to sqlite DB")
    parser.add_argument(
        "--since", default=None,
        help="only include trades on/after this YYYY-MM-DD date",
    )
    parser.add_argument(
        "--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="comma-separated forward-return horizons in days",
    )
    parser.add_argument(
        "--market-adjust", action="store_true",
        help="subtract SPY's same-window return (requires SPY in stock_prices)",
    )
    parser.add_argument(
        "--out", default=None,
        help="optional path to write a JSON dump of segment rows",
    )
    args = parser.parse_args(argv)

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        log.error("pandas is required to run backtest.py (pip install -r requirements.txt)")
        return 2

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    since = parse_trade_date(args.since) if args.since else None

    log.info("loading trades from %s (since=%s)", args.db, since)
    conn = _connect(args.db)
    try:
        trades = load_trades(conn, since=since)
        log.info("loaded %d trades with parseable dates", len(trades))
        if trades.empty:
            log.warning("no trades to score; exiting")
            return 0

        prices = load_prices(conn, trades["ticker"].unique())
        log.info("loaded prices for %d tickers", len([t for t, p in prices.items() if p]))

        spy_prices: list[tuple[date, float]] = []
        if args.market_adjust:
            spy = load_prices(conn, ["SPY"]).get("SPY", [])
            if not spy:
                log.warning("--market-adjust requested but no SPY rows in stock_prices; skipping")
            else:
                spy_prices = spy

        compute_returns_table(trades, prices, horizons)
        add_segmentation_columns(trades)
        if spy_prices:
            market_adjust(trades, spy_prices, horizons)
    finally:
        conn.close()

    overall: list[dict] = []
    for h in horizons:
        persp = _flip_sign_for_sells(trades, f"ret_{h}d")
        stats = summarise(persp)
        stats.update({"segment_field": "ALL", "segment_value": "trades",
                      "horizon_days": h, "adjusted": False})
        overall.append(stats)

    by_side = segment_report(trades, "side", horizons, adjusted=False)
    by_source = segment_report(trades, "source", horizons, adjusted=False)
    by_trader_freq = segment_report(trades, "trader_freq", horizons, adjusted=False)
    by_trade_size = segment_report(trades, "trade_size_q", horizons, adjusted=False)

    all_rows: list[dict] = (
        list(overall) + by_side + by_source + by_trader_freq + by_trade_size
    )
    if spy_prices:
        all_rows += segment_report(trades, "side", horizons, adjusted=True)
        all_rows += segment_report(trades, "source", horizons, adjusted=True)

    print(render_text_report(overall, "OVERALL  (insider perspective; sells inverted)"))
    print()
    print(render_text_report(by_side, "BY SIDE  (insider perspective)"))
    print()
    print(render_text_report(by_source, "BY SOURCE  (insider perspective)"))
    print()
    print(render_text_report(
        by_trader_freq,
        "BY TRADER FREQUENCY  (does repeat-offender weighting earn its keep?)",
    ))
    print()
    print(render_text_report(
        by_trade_size,
        "BY TRADE SIZE QUARTILE  (does the 'large trade' rule have edge?)",
    ))

    if spy_prices:
        by_side_adj = segment_report(trades, "side", horizons, adjusted=True)
        by_source_adj = segment_report(trades, "source", horizons, adjusted=True)
        print()
        print(render_text_report(
            by_side_adj,
            "BY SIDE — SPY-ADJUSTED  (excess return over SPY in same window)",
        ))
        print()
        print(render_text_report(
            by_source_adj,
            "BY SOURCE — SPY-ADJUSTED  (excess return over SPY in same window)",
        ))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_rows, f, indent=2, default=str)
        log.info("wrote %d rows to %s", len(all_rows), args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
