# pipeline.py
"""End-to-end pipeline: scrape → fetch → analyze → backtest → alert.

Designed to be run on a daily schedule (cron / systemd timer / n8n
HTTP trigger). Every invocation:

  1. acquires an OS file-lock so overlapping runs are impossible;
  2. records start/finish state in the pipeline_runs table;
  3. runs each step as a subprocess, retrying once on transient
     non-zero exit before declaring failure;
  4. emails a one-line success summary with top alerts when the run
     completes cleanly, or a failure email naming the failed step;
  5. exits 0 / 1 so cron-style supervisors can detect failures.

Importable: call `run_pipeline(triggered_by="api")` from the API for
on-demand runs, or run as a script for scheduled runs.
"""
from __future__ import annotations

import errno
import fcntl
import glob
import json
import os
import smtplib
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv

from logging_config import get_logger

load_dotenv()

log = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_FILE = os.getenv("TRADEINSIDE_DB", os.path.join(os.path.dirname(__file__), "tradeinsider.db"))

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
GMAIL_USER = os.getenv("EMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("EMAIL_PASS")
ALERT_EMAIL = os.getenv("EMAIL_TO")

ALERT_ON_SAME_COMPANY_CLUSTER = True
ALERT_ON_REPEAT_OFFENDER = True
ALERT_ON_LARGE_TRADE = True
ALERT_ON_PRE_EVENT_TIMING = True

MIN_RISK_SCORE_FOR_ALERT = 150

# How many seconds to wait before retrying a failed step (one retry only).
RETRY_BACKOFF_SECS = int(os.getenv("PIPELINE_RETRY_BACKOFF_SECS", 30))

# Lock file used to prevent overlapping pipeline runs on a single host.
LOCK_FILE = os.getenv("PIPELINE_LOCK_FILE", "/tmp/tradeinside.pipeline.lock")

# Per-step subprocess timeout (seconds). 0 disables the timeout.
STEP_TIMEOUT_SECS = int(os.getenv("PIPELINE_STEP_TIMEOUT_SECS", 0))


# ============================================================================
# EMAIL
# ============================================================================

def send_email_alert(subject, body):
    """Send email via Gmail SMTP. Returns True on success, False on failure."""
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and ALERT_EMAIL):
        log.warning("email creds not set; skipping send for %r", subject)
        return False
    try:
        recipients = [e.strip() for e in ALERT_EMAIL.split(",")]

        msg = MIMEMultipart()
        msg["From"] = GMAIL_USER
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, recipients, msg.as_string())
        server.quit()

        print(f"[✓] Email alert sent to {len(recipients)} recipient(s): {subject}")
        return True
    except Exception as e:
        log.exception("failed to send email alert")
        print(f"[✗] Failed to send email: {e}")
        return False


def check_for_alerts(report_path):
    """Parse the latest top-10 report and email any alert-worthy patterns."""
    if not os.path.exists(report_path):
        log.warning("report not found: %s", report_path)
        return
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    alerts = _parse_alerts(content)
    if alerts:
        body = build_alert_body(alerts)
        send_email_alert(
            f"🚨 INSIDER TRADING ALERT: {len(alerts)} Suspicious Pattern(s) Detected",
            body,
        )
    else:
        print("[✓] No alert-worthy patterns detected")


def _parse_alerts(content):
    """Walk the top-10 report text and return a list of alert dicts."""
    alerts = []
    current_trader, current_score, current_patterns = None, 0, []
    for line in content.split("\n"):
        if line.startswith("#") and " - " in line:
            if current_trader and should_alert(current_trader, current_score, current_patterns):
                alerts.append({
                    "trader": current_trader,
                    "score": current_score,
                    "patterns": current_patterns,
                })
            parts = line.split(" - ", 1)
            if len(parts) > 1:
                current_trader = parts[1].split("(")[0].strip()
                current_patterns = []
        elif line.startswith("Risk Score:"):
            try:
                current_score = int(line.split(":")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                current_score = 0
        elif line.strip().startswith("•"):
            current_patterns.append(line.strip())
    if current_trader and should_alert(current_trader, current_score, current_patterns):
        alerts.append({"trader": current_trader, "score": current_score, "patterns": current_patterns})
    return alerts


def should_alert(trader, score, patterns):
    if score < MIN_RISK_SCORE_FOR_ALERT:
        return False
    text = " ".join(patterns)
    if ALERT_ON_SAME_COMPANY_CLUSTER and "Same-company cluster" in text:
        return True
    if ALERT_ON_REPEAT_OFFENDER and "Repeat Offender" in text:
        return True
    if ALERT_ON_LARGE_TRADE and "Large trade" in text:
        return True
    if ALERT_ON_PRE_EVENT_TIMING and "days before event" in text:
        return True
    return False


def build_alert_body(alerts):
    body = "Insider Trading Pattern Detection Alert\n"
    body += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    body += "=" * 80 + "\n\n"
    for i, alert in enumerate(alerts, 1):
        body += f"#{i} - {alert['trader']}\nRisk Score: {alert['score']}\nPatterns:\n"
        for pattern in alert["patterns"][:5]:
            body += f"  {pattern}\n"
        body += "\n" + "-" * 80 + "\n\n"
    body += f"\nTotal Alerts: {len(alerts)}\n"
    body += "Check the full report in ./reports/ for complete details.\n"
    return body


# ============================================================================
# RUN-LEVEL DURABILITY (lock + DB tracking)
# ============================================================================

class _PipelineLockTaken(Exception):
    """Raised when another pipeline run already holds the lock."""


def _acquire_lock(path: str = LOCK_FILE):
    """fcntl-based exclusive lock. Returns the open fd; caller must release.

    On a busy host, two scheduled triggers landing simultaneously would
    otherwise both run scrape/analyze in parallel — corrupting the report
    folder and racing on writes. The lock makes the second invocation
    exit cleanly with status 'aborted'.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            raise _PipelineLockTaken(
                f"another pipeline run holds {path}; not starting a parallel run"
            ) from e
        raise
    return fd


def _release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_db():
    return sqlite3.connect(DB_FILE)


def _heal_stale_runs(stale_after_secs: int = 6 * 3600) -> int:
    """Mark any 'running' row older than ``stale_after_secs`` as 'aborted'.

    Pipelines that crash hard (OOM, host reboot, kill -9) leave a row stuck
    on 'running' forever, which corrupts dashboards and makes 'is the
    pipeline healthy?' impossible to answer. Self-heal at the start of the
    next run instead of requiring manual cleanup. Returns the number of
    rows touched.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - stale_after_secs
    conn = _open_db()
    try:
        rows = conn.execute(
            "SELECT id, run_id, started_at FROM pipeline_runs WHERE status = 'running'"
        ).fetchall()
        healed = 0
        for row_id, run_id, started_at in rows:
            try:
                ts = datetime.fromisoformat(started_at).timestamp()
            except (TypeError, ValueError):
                continue
            if ts < cutoff:
                conn.execute(
                    "UPDATE pipeline_runs SET status = 'aborted', "
                    "       finished_at = ?, "
                    "       notes = COALESCE(notes,'') || 'auto-marked stale' "
                    "WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"), row_id),
                )
                healed += 1
                log.warning("healed stale run row run_id=%s started_at=%s", run_id, started_at)
        if healed:
            conn.commit()
        return healed
    finally:
        conn.close()


def _record_run_start(run_id: str, triggered_by: str) -> None:
    conn = _open_db()
    try:
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, started_at, status, triggered_by) "
            "VALUES (?, ?, 'running', ?)",
            (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), triggered_by),
        )
        conn.commit()
    finally:
        conn.close()


def _record_run_end(
    run_id: str,
    status: str,
    started_at: float,
    failed_step: Optional[str] = None,
    step_results: Optional[list] = None,
    notes: Optional[str] = None,
) -> None:
    conn = _open_db()
    try:
        conn.execute(
            "UPDATE pipeline_runs SET finished_at = ?, status = ?, duration_secs = ?, "
            "failed_step = ?, step_results = ?, notes = ? WHERE run_id = ?",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status,
                round(time.time() - started_at, 2),
                failed_step,
                json.dumps(step_results) if step_results is not None else None,
                notes,
                run_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# STEP EXECUTION
# ============================================================================

def run_command(command, step_name, timeout_secs=None):
    """Run a shell command, capture output, return (success, stdout, stderr).

    A non-zero exit yields success=False; the caller decides whether to
    retry. Subprocess timeout is opt-in via STEP_TIMEOUT_SECS.
    """
    print(f"\n{'=' * 80}\nSTEP: {step_name}\n{'=' * 80}")
    timeout = timeout_secs if timeout_secs and timeout_secs > 0 else None
    try:
        result = subprocess.run(
            command, shell=True, check=True,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        log.error("step %s timed out after %ss", step_name, timeout)
        return False, e.stdout or "", f"timeout after {timeout}s"
    except subprocess.CalledProcessError as e:
        print(f"[✗] {step_name} failed (exit {e.returncode})")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, e.stdout or "", e.stderr or ""


def _run_step_with_retry(command, step_name, max_retries: int = 1):
    """Run a step, retry once on failure with backoff. Returns a dict
    summarising the attempt(s) for the run-history record."""
    attempts = 0
    started = time.time()
    while True:
        attempts += 1
        ok, out, err = run_command(command, step_name, timeout_secs=STEP_TIMEOUT_SECS)
        if ok:
            return {
                "step": step_name, "ok": True, "attempts": attempts,
                "duration_secs": round(time.time() - started, 2),
            }
        if attempts > max_retries:
            return {
                "step": step_name, "ok": False, "attempts": attempts,
                "duration_secs": round(time.time() - started, 2),
                "stderr_tail": (err or "")[-500:],
            }
        log.warning("step %s failed; retrying in %ds", step_name, RETRY_BACKOFF_SECS)
        time.sleep(RETRY_BACKOFF_SECS)


# ============================================================================
# SUCCESS SUMMARY
# ============================================================================

def build_success_summary(run_id: str, duration_secs: float) -> str:
    """One-page summary emailed on a clean run. Pulls headline numbers from
    the just-persisted backtest snapshot + the trades table."""
    conn = _open_db()
    try:
        total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        latest_scrape = conn.execute(
            "SELECT MAX(created_at) FROM trades"
        ).fetchone()[0]
        bt_run = conn.execute(
            "SELECT run_id, run_at FROM backtest_runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        side_rows = []
        if bt_run:
            side_rows = conn.execute(
                "SELECT segment_value, horizon_days, n, mean_return, hit_rate, sharpe "
                "FROM backtest_runs WHERE run_id = ? AND segment_field = 'side' "
                "AND adjusted IN ('sector_adj','spy_adj','raw') "
                "ORDER BY adjusted DESC, segment_value, horizon_days",
                (bt_run[0],),
            ).fetchall()
    finally:
        conn.close()

    lines = []
    lines.append(f"tradeInside daily run — {run_id}")
    lines.append("=" * 60)
    lines.append(f"Duration:       {duration_secs:.1f}s")
    lines.append(f"Trades in DB:   {total_trades:,}")
    lines.append(f"Latest scrape:  {latest_scrape or 'unknown'}")
    if bt_run:
        lines.append(f"Backtest run:   {bt_run[0]}  (run_at={bt_run[1]})")

    if side_rows:
        lines.append("")
        lines.append("Latest backtest — by side (best benchmark adjustment first):")
        lines.append(f"  {'side':<6}{'h':>4}{'n':>7}{'mean':>10}{'hit%':>8}{'sharpe':>9}")
        for sv, h, n, mean, hit, sharpe in side_rows[:8]:  # cap to keep email small
            mean_s = "—" if mean is None else f"{mean * 100:+.2f}%"
            hit_s = "—" if hit is None else f"{hit * 100:5.1f}"
            sh_s = "—" if sharpe is None else f"{sharpe:.2f}"
            lines.append(f"  {sv:<6}{h:>4}{n:>7}{mean_s:>10}{hit_s:>8}{sh_s:>9}")

    lines.append("")
    lines.append("Full report in ./reports/. Pull /backtest/latest for the JSON.")
    return "\n".join(lines)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

PIPELINE_STEPS = (
    ("python scrape.py --auto",                                  "Scrape Latest Insider Trades"),
    ("python fetch_data.py all",                                 "Fetch Stock Prices & Events"),
    ("python analyzer.py",                                       "Analyze Trading Patterns"),
    ("python backtest.py --sector-adjust --by-pattern --persist","Run + Persist Backtest Snapshot"),
)


def run_pipeline(
    triggered_by: str = "cli",
    send_summary_on_success: bool = True,
    skip_lock: bool = False,
) -> dict:
    """Execute the full pipeline with durability tracking.

    Returns a dict with keys: run_id, status ('success' | 'failed' | 'aborted'),
    duration_secs, failed_step (or None), step_results.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.time()

    print("\n" + "=" * 80)
    print("INSIDER TRADING DETECTION PIPELINE")
    print(f"run_id={run_id} triggered_by={triggered_by}")
    print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    lock_fd = None
    if not skip_lock:
        try:
            lock_fd = _acquire_lock()
        except _PipelineLockTaken as e:
            log.warning("not running: %s", e)
            print(f"[!] {e}")
            return {
                "run_id": run_id, "status": "aborted",
                "duration_secs": 0.0, "failed_step": None,
                "step_results": [], "notes": str(e),
            }

    healed = _heal_stale_runs()
    if healed:
        print(f"[i] healed {healed} stale 'running' row(s) from previous crash(es)")

    _record_run_start(run_id, triggered_by)

    step_results: list = []
    failed_step: Optional[str] = None
    try:
        for command, step_name in PIPELINE_STEPS:
            result = _run_step_with_retry(command, step_name, max_retries=1)
            step_results.append(result)
            if not result["ok"]:
                failed_step = step_name
                break

        status = "success" if failed_step is None else "failed"
        duration = time.time() - started

        if status == "success":
            reports = glob.glob("./reports/insider_TOP10_*.txt")
            if reports:
                latest_report = max(reports, key=os.path.getctime)
                print(f"\n[✓] Latest report: {latest_report}")
                check_for_alerts(latest_report)
            if send_summary_on_success:
                send_email_alert(
                    f"✅ tradeInside daily run OK — {run_id}",
                    build_success_summary(run_id, duration),
                )
        else:
            send_email_alert(
                "❌ tradeInside Pipeline Failure",
                f"Pipeline run {run_id} failed at step: {failed_step}\n\n"
                f"step_results = {json.dumps(step_results, indent=2, default=str)}\n",
            )

        _record_run_end(run_id, status, started,
                        failed_step=failed_step, step_results=step_results)
        return {
            "run_id": run_id, "status": status,
            "duration_secs": round(duration, 2),
            "failed_step": failed_step, "step_results": step_results,
        }
    except Exception as exc:
        log.exception("unhandled pipeline error")
        _record_run_end(run_id, "failed", started,
                        failed_step="unhandled_exception",
                        step_results=step_results, notes=str(exc))
        send_email_alert(
            "❌ tradeInside Pipeline Crash",
            f"Pipeline {run_id} crashed: {exc}",
        )
        return {
            "run_id": run_id, "status": "failed",
            "duration_secs": round(time.time() - started, 2),
            "failed_step": "unhandled_exception", "step_results": step_results,
            "notes": str(exc),
        }
    finally:
        if lock_fd is not None:
            _release_lock(lock_fd)
        print("\n" + "=" * 80)
        print(f"PIPELINE COMPLETE — run_id={run_id}")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and ALERT_EMAIL):
        print("\n[!] WARNING: email creds missing in .env — emails will be skipped\n")
    elif GMAIL_USER == "your_email@gmail.com":
        print("\n[!] WARNING: placeholder email creds in .env — update before production\n")

    triggered_by = "cron" if os.getenv("INVOCATION_ID") else "cli"  # systemd sets INVOCATION_ID
    result = run_pipeline(triggered_by=triggered_by)
    sys.exit(0 if result["status"] == "success" else 1)
