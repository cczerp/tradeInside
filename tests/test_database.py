"""Tests for database.Database against an in-memory SQLite DB.

These tests avoid touching the real tradeinsider.db or any network calls, and
have no third-party dependencies beyond the stdlib.
"""
import os
import sqlite3

import pytest

from database import Database

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "database.sql")


@pytest.fixture
def db(tmp_path):
    """Database backed by a fresh temp-file SQLite DB with schema applied."""
    path = tmp_path / "test.db"
    d = Database(db_file=str(path))
    assert d.connect()
    d.create_tables(schema_file=SCHEMA_FILE)
    yield d
    d.close()


def _trade(**over):
    base = {
        "source": "openinsider",
        "ticker": "AAPL",
        "transaction_date": "2024-01-15",
        "transaction_type": "Purchase",
        "insider_name": "Jane Doe",
        "politician_name": None,
        "fund_name": None,
        "shares": 1000,
        "price": 180.5,
        "filed_date": "2024-01-17",
    }
    base.update(over)
    return base


def test_bulk_insert_trades_inserts_new_rows(db):
    ok = db.bulk_insert_trades([
        _trade(ticker="AAPL"),
        _trade(ticker="MSFT", insider_name="John Smith"),
    ])
    assert ok
    stats = db.get_database_stats()
    assert stats["total_trades"] == 2
    assert stats["unique_tickers"] == 2
    assert stats["unique_traders"] == 2
    assert stats["by_source"] == {"openinsider": 2}


def test_bulk_insert_trades_deduplicates(db):
    rows = [_trade(), _trade()]  # identical
    db.bulk_insert_trades(rows)
    assert db.get_database_stats()["total_trades"] == 1

    # A second call with the same payload is still a no-op.
    db.bulk_insert_trades(rows)
    assert db.get_database_stats()["total_trades"] == 1


def test_bulk_insert_trades_skips_rows_without_required_fields(db):
    ok = db.bulk_insert_trades([
        _trade(ticker=None),
        _trade(transaction_date=None),
        _trade(ticker="VALID"),
    ])
    assert ok
    assert db.get_database_stats()["total_trades"] == 1


def test_bulk_insert_trades_empty_list_is_ok(db):
    assert db.bulk_insert_trades([]) is True
    assert db.get_database_stats()["total_trades"] == 0


def test_insert_trade_single_row_path_still_works(db):
    db.insert_trade(_trade())
    assert db.get_database_stats()["total_trades"] == 1


def test_indexes_are_created(db):
    conn = sqlite3.connect(db.db_file)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    conn.close()
    names = {r[0] for r in rows}
    assert "idx_trades_ticker" in names
    assert "idx_trades_transaction_date" in names
    assert "idx_trades_insider_name" in names
