"""Tests for pipeline.py durability primitives.

Covers _record_run_start, _record_run_end, _acquire_lock, and the
build_success_summary string. The actual subprocess steps aren't
exercised here — those are integration concerns.
"""
import json
import os
import sqlite3
import time

import pytest

import pipeline
from database import Database

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "database.sql")


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "test.db")
    db = Database(db_file=path)
    assert db.connect()
    db.create_tables(schema_file=SCHEMA_FILE)
    db.close()
    monkeypatch.setattr(pipeline, "DB_FILE", path)
    return path


def _row(conn, run_id):
    return conn.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()


def test_record_run_start_inserts_running_row(db_path):
    pipeline._record_run_start("r-1", "cli")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = _row(conn, "r-1")
        assert row is not None
        assert row["status"] == "running"
        assert row["triggered_by"] == "cli"
        assert row["finished_at"] is None
    finally:
        conn.close()


def test_record_run_end_marks_success_with_duration(db_path):
    pipeline._record_run_start("r-2", "api")
    started = time.time() - 5  # pretend the run took 5 seconds
    pipeline._record_run_end(
        "r-2", "success", started,
        step_results=[{"step": "scrape", "ok": True, "attempts": 1}],
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = _row(conn, "r-2")
        assert row["status"] == "success"
        assert row["finished_at"] is not None
        assert row["duration_secs"] >= 4.5
        steps = json.loads(row["step_results"])
        assert steps == [{"step": "scrape", "ok": True, "attempts": 1}]
    finally:
        conn.close()


def test_record_run_end_records_failed_step(db_path):
    pipeline._record_run_start("r-3", "n8n")
    pipeline._record_run_end(
        "r-3", "failed", time.time(),
        failed_step="Scrape Latest Insider Trades",
        step_results=[{"step": "Scrape Latest Insider Trades", "ok": False}],
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = _row(conn, "r-3")
        assert row["status"] == "failed"
        assert row["failed_step"] == "Scrape Latest Insider Trades"
    finally:
        conn.close()


def test_acquire_lock_excludes_concurrent_holder(tmp_path):
    lock = str(tmp_path / "test.lock")
    fd = pipeline._acquire_lock(lock)
    try:
        with pytest.raises(pipeline._PipelineLockTaken):
            pipeline._acquire_lock(lock)
    finally:
        pipeline._release_lock(fd)

    # After release, a fresh acquire must succeed.
    fd2 = pipeline._acquire_lock(lock)
    pipeline._release_lock(fd2)


def test_build_success_summary_includes_run_id_and_db_counts(db_path):
    summary = pipeline.build_success_summary(run_id="r-summary", duration_secs=12.3)
    assert "r-summary" in summary
    assert "Duration:" in summary
    assert "Trades in DB:" in summary


def test_heal_stale_runs_marks_old_running_as_aborted(db_path):
    """A row that was started 24h ago and never finished should be auto-
    marked 'aborted' so dashboards don't lie about pipeline health."""
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO pipeline_runs (run_id, started_at, status, triggered_by) "
            "VALUES (?, ?, 'running', 'cli')",
            [("r-stale", old), ("r-fresh", fresh)],
        )
        conn.commit()
    finally:
        conn.close()

    healed = pipeline._heal_stale_runs(stale_after_secs=6 * 3600)
    assert healed == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        stale = _row(conn, "r-stale")
        fresh_row = _row(conn, "r-fresh")
        assert stale["status"] == "aborted"
        assert "stale" in (stale["notes"] or "")
        assert fresh_row["status"] == "running"
    finally:
        conn.close()


def test_heal_stale_runs_no_op_when_no_stale_rows(db_path):
    """Empty / all-fresh table shouldn't touch anything."""
    assert pipeline._heal_stale_runs() == 0
