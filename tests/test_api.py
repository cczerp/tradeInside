"""Smoke tests for the FastAPI surface.

Each test points the app at a temp SQLite DB so we don't touch the real
tradeinsider.db. Auth-gated endpoints are exercised both ways: open
(API_KEY unset) and closed (API_KEY set, header missing/wrong/correct).
"""
from __future__ import annotations

import importlib
import os
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from database import Database


SCHEMA_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "database.sql")


def _seed_db(path: str) -> None:
    db = Database(db_file=path)
    assert db.connect()
    db.create_tables(schema_file=SCHEMA_FILE)
    db.bulk_insert_trades([
        {
            "source": "openinsider", "ticker": "AAPL",
            "transaction_date": "2025-02-01", "transaction_type": "P - Purchase",
            "insider_name": "Jane Doe", "shares": 100, "price": 180.0,
        },
        {
            "source": "openinsider", "ticker": "MSFT",
            "transaction_date": "2025-02-05", "transaction_type": "S - Sale",
            "insider_name": "John Smith", "shares": 50, "price": 410.0,
        },
        {
            "source": "quiver", "ticker": "AAPL",
            "transaction_date": "2025-03-01", "transaction_type": "Purchase",
            "politician_name": "Pelosi Nancy", "shares": 1000, "price": 195.0,
        },
    ])
    db.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient pointed at a fresh temp DB, with auth disabled."""
    db_path = str(tmp_path / "test.db")
    _seed_db(db_path)

    monkeypatch.setenv("TRADEINSIDE_DB", db_path)
    monkeypatch.delenv("TRADEINSIDE_API_KEY", raising=False)
    # Reload module so the env vars take effect at import time
    import api
    importlib.reload(api)
    return TestClient(api.app)


@pytest.fixture
def gated_client(tmp_path, monkeypatch):
    """Same as `client` but with auth enabled."""
    db_path = str(tmp_path / "test.db")
    _seed_db(db_path)

    monkeypatch.setenv("TRADEINSIDE_DB", db_path)
    monkeypatch.setenv("TRADEINSIDE_API_KEY", "secret-token")
    import api
    importlib.reload(api)
    return TestClient(api.app)


# --- public endpoints -----------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats_returns_expected_shape(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_trades"] == 3
    assert body["unique_tickers"] == 2
    assert "by_source" in body


def test_trades_filters_by_ticker(client):
    r = client.get("/trades", params={"ticker": "AAPL"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert all(row["ticker"] == "AAPL" for row in body["rows"])


def test_trades_filters_by_trader_substring(client):
    r = client.get("/trades", params={"trader": "Pelosi"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["rows"][0]["politician_name"] == "Pelosi Nancy"


def test_trades_filters_by_side(client):
    r = client.get("/trades", params={"side": "buy"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert all("Purchase" in row["transaction_type"] for row in rows)


def test_trades_pagination_offsets(client):
    r1 = client.get("/trades", params={"limit": 1, "offset": 0})
    r2 = client.get("/trades", params={"limit": 1, "offset": 1})
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["rows"][0]["id"] != r2.json()["rows"][0]["id"]


# --- auth-gated endpoints -------------------------------------------------

def test_trader_profile_open_when_no_api_key_set(client):
    r = client.get("/trader/Jane")
    assert r.status_code == 200
    body = r.json()
    assert body["trade_count"] == 1
    assert "AAPL" in body["tickers"]


def test_trader_profile_404_when_no_match(client):
    r = client.get("/trader/Nobody")
    assert r.status_code == 404


def test_trader_profile_requires_key_when_set(gated_client):
    r = gated_client.get("/trader/Jane")
    assert r.status_code == 401

    r = gated_client.get("/trader/Jane", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401

    r = gated_client.get("/trader/Jane", headers={"X-API-Key": "secret-token"})
    assert r.status_code == 200


def test_backtest_summary_validates_horizons(client):
    r = client.get("/backtest/summary", params={"horizons": "abc,xyz"})
    assert r.status_code == 400
