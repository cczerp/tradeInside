"""FastAPI surface for the tradeInside analytics.

Endpoints (all read-only):
  GET /health                          liveness probe
  GET /stats                           DB-level counts
  GET /trades                          paginated trade lookup
  GET /trader/{name}                   trader profile (auth-gated when API_KEY set)
  GET /backtest/summary                forward-return segments (auth-gated)

Auth: when TRADEINSIDE_API_KEY is set in the environment, the gated
endpoints require an X-API-Key header. When unset, the gates are
no-ops — useful for local dev.

Run locally:
    pip install -r requirements.txt
    uvicorn api:app --reload

Run via docker compose (preferred):
    docker compose up --build
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from logging_config import get_logger

log = get_logger(__name__)

DB_FILE = os.getenv("TRADEINSIDE_DB", "tradeinsider.db")
API_KEY = os.getenv("TRADEINSIDE_API_KEY")

app = FastAPI(
    title="tradeInside API",
    version="0.1.0",
    description="Read-only analytics over the insider-trading DB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If TRADEINSIDE_API_KEY is set, demand a matching X-API-Key header.

    A None API_KEY explicitly means 'open access' (dev mode). Production
    deployments must set the env var.
    """
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/stats")
def stats():
    """High-level DB counts."""
    from database import Database

    db = Database(DB_FILE)
    if not db.connect():
        raise HTTPException(status_code=503, detail="database unavailable")
    try:
        return db.get_database_stats()
    finally:
        db.close()


@app.get("/trades")
def trades(
    ticker: Optional[str] = Query(None, description="exact ticker match (case-insensitive)"),
    trader: Optional[str] = Query(None, description="substring match on insider/politician/fund name"),
    since: Optional[str] = Query(None, description="ISO date, e.g. 2025-01-01"),
    side: Optional[str] = Query(None, description="buy / sell / other"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Trades feed with simple filters. Always ordered by transaction_date desc."""
    where: list[str] = []
    params: list = []

    if ticker:
        where.append("ticker = ?")
        params.append(ticker.upper())
    if trader:
        where.append(
            "(insider_name LIKE ? OR politician_name LIKE ? OR fund_name LIKE ?)"
        )
        params.extend([f"%{trader}%"] * 3)
    if since:
        where.append("transaction_date >= ?")
        params.append(since)

    sql = "SELECT * FROM trades"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY transaction_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = _connect()
    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

    if side:
        from backtest import normalise_transaction_type
        wanted = side.lower()
        rows = [r for r in rows if normalise_transaction_type(r.get("transaction_type")) == wanted]

    return {"count": len(rows), "limit": limit, "offset": offset, "rows": rows}


# ---------------------------------------------------------------------------
# Auth-gated endpoints
# ---------------------------------------------------------------------------


@app.get("/trader/{name}", dependencies=[Depends(require_api_key)])
def trader_profile(name: str, limit: int = Query(50, ge=1, le=500)):
    """Profile for a single trader by name substring."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM trades "
            "WHERE insider_name LIKE ? OR politician_name LIKE ? OR fund_name LIKE ? "
            "ORDER BY transaction_date DESC LIMIT ?",
            [f"%{name}%", f"%{name}%", f"%{name}%", limit],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"no trades for trader matching '{name}'")

    rows_d = [dict(r) for r in rows]
    tickers = sorted({r["ticker"] for r in rows_d if r.get("ticker")})
    dates = [r.get("transaction_date") for r in rows_d if r.get("transaction_date")]

    return {
        "trader": name,
        "trade_count": len(rows_d),
        "distinct_tickers": len(tickers),
        "tickers": tickers,
        "first_trade": min(dates) if dates else None,
        "last_trade": max(dates) if dates else None,
        "recent_trades": rows_d[: min(10, len(rows_d))],
    }


@app.get("/backtest/summary", dependencies=[Depends(require_api_key)])
def backtest_summary(
    since: Optional[str] = Query(None, description="ISO date filter"),
    horizons: str = Query("7,30", description="comma-separated horizon days"),
    market_adjust: bool = Query(False, description="subtract SPY return"),
    sector_adjust: bool = Query(False, description="subtract sector ETF return"),
):
    """Run a backtest and return per-side segment statistics as JSON."""
    from backtest import (
        compute_returns_table,
        load_prices,
        load_trades,
        market_adjust as do_market_adjust,
        parse_trade_date,
        sector_adjust as do_sector_adjust,
        segment_report,
        add_segmentation_columns,
    )

    try:
        horizons_t = tuple(int(h) for h in horizons.split(",") if h.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="horizons must be comma-separated integers")
    if not horizons_t:
        raise HTTPException(status_code=400, detail="at least one horizon required")
    since_d = parse_trade_date(since) if since else None

    conn = sqlite3.connect(DB_FILE)
    try:
        df = load_trades(conn, since=since_d)
        if df.empty:
            return {
                "horizons": list(horizons_t),
                "since": since,
                "adjusted": "raw",
                "segments": [],
            }

        prices = load_prices(conn, df["ticker"].unique())
        compute_returns_table(df, prices, horizons_t)
        add_segmentation_columns(df)

        prefix = "ret_"
        adj_label = "raw"
        if sector_adjust:
            from analyzer import SECTOR_MAP
            from benchmarks import all_benchmark_symbols, benchmark_for_ticker

            needed = all_benchmark_symbols(SECTOR_MAP)
            sector_benchmarks = load_prices(conn, needed)
            if any(sector_benchmarks.values()):
                bench_for_row = [benchmark_for_ticker(t, SECTOR_MAP) for t in df["ticker"]]
                do_sector_adjust(df, sector_benchmarks, bench_for_row, horizons_t)
                prefix = "sector_excess_"
                adj_label = "sector_adj"
        elif market_adjust:
            spy = load_prices(conn, ["SPY"]).get("SPY", [])
            if spy:
                do_market_adjust(df, spy, horizons_t)
                prefix = "excess_"
                adj_label = "spy_adj"

        segments = segment_report(df, "side", horizons_t, prefix=prefix)
    finally:
        conn.close()

    return {
        "horizons": list(horizons_t),
        "since": since,
        "adjusted": adj_label,
        "segments": segments,
    }
