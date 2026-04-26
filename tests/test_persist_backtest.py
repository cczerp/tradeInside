"""Tests for backtest.persist_backtest_run.

Uses a temp SQLite DB seeded with the database.sql schema. No third-party
deps beyond stdlib + the project's own modules — these run anywhere.
"""
import os
import sqlite3
from datetime import date, datetime

import pytest

from backtest import persist_backtest_run
from database import Database

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "database.sql")


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "test.db"
    db = Database(db_file=str(path))
    assert db.connect()
    db.create_tables(schema_file=SCHEMA_FILE)
    db.close()
    c = sqlite3.connect(str(path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def _row(**over):
    base = {
        "horizon_days": 7,
        "segment_field": "side",
        "segment_value": "buy",
        "adjusted": "sector_adj",
        "count": 100,
        "mean": 0.034,
        "median": 0.012,
        "std": 0.08,
        "hit_rate": 0.65,
        "sharpe": 0.42,
    }
    base.update(over)
    return base


def test_persist_writes_all_rows(conn):
    rows = [_row(segment_value="buy"), _row(segment_value="sell", count=50)]
    inserted = persist_backtest_run(conn, rows, run_id="r1")
    assert inserted == 2

    n = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
    assert n == 2


def test_persist_records_metadata(conn):
    when = datetime(2026, 1, 15, 12, 0, 0)
    persist_backtest_run(conn, [_row()], run_id="r-meta",
                        since_filter=date(2025, 1, 1), run_at=when)

    row = conn.execute("SELECT * FROM backtest_runs").fetchone()
    assert row["run_id"] == "r-meta"
    assert row["run_at"].startswith("2026-01-15T12:00:00")
    assert row["since_filter"] == "2025-01-01"
    assert row["adjusted"] == "sector_adj"
    assert row["n"] == 100


def test_persist_empty_list_is_noop(conn):
    assert persist_backtest_run(conn, [], run_id="r-empty") == 0
    assert conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 0


def test_persist_groups_rows_under_one_run_id(conn):
    persist_backtest_run(conn, [_row(), _row(segment_value="sell")], run_id="r-A")
    persist_backtest_run(conn, [_row()], run_id="r-B")

    a = conn.execute("SELECT COUNT(*) FROM backtest_runs WHERE run_id = 'r-A'").fetchone()[0]
    b = conn.execute("SELECT COUNT(*) FROM backtest_runs WHERE run_id = 'r-B'").fetchone()[0]
    assert a == 2
    assert b == 1


def test_persist_handles_none_values(conn):
    """summarise() returns None for mean/sharpe on empty/zero-std data —
    persistence must accept those without raising."""
    row = _row(mean=None, median=None, std=None, hit_rate=None, sharpe=None, count=0)
    persist_backtest_run(conn, [row], run_id="r-none")

    db_row = conn.execute("SELECT * FROM backtest_runs").fetchone()
    assert db_row["mean_return"] is None
    assert db_row["sharpe"] is None
