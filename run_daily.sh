#!/usr/bin/env bash
# run_daily.sh — wrapper around `python pipeline.py` for scheduled runs.
#
# Designed to be invoked by systemd timer (or cron / Task Scheduler):
#   systemd: ExecStart=/home/<you>/tradeInside/run_daily.sh
#   cron:    0 7 * * 1-5  cd /home/<you>/tradeInside && ./run_daily.sh >> ~/tradeinside.log 2>&1
#
# Behaviour:
#   - cd's to the project directory regardless of where it was invoked from
#   - activates ./venv if it exists (no-op otherwise; system python is used)
#   - runs pipeline.py and propagates its exit code so supervisors see failures
#   - structured per-run log goes to ./logs/run_YYYY-MM-DD.log; the most
#     recent stdout is also tee'd to ./logs/last.log for quick checking
#
# Idempotent: the pipeline itself takes a flock at /tmp/tradeinside.pipeline.lock
# so two overlapping invocations won't corrupt anything; the second exits
# cleanly with status='aborted'.

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Activate the project's virtualenv if present. Standard locations:
#   ./venv  ./.venv
for candidate in venv .venv; do
    if [[ -f "$candidate/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$candidate/bin/activate"
        break
    fi
done

DATE_TAG="$(date -u +%Y-%m-%d)"
LOG_FILE="logs/run_${DATE_TAG}.log"

{
    echo "==============================================================="
    echo "tradeInside daily run"
    echo "started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "user:    ${USER:-unknown}"
    echo "host:    $(hostname)"
    echo "python:  $(python3 --version 2>&1)"
    echo "cwd:     $(pwd)"
    echo "==============================================================="

    python3 pipeline.py
    EXIT=$?

    echo
    echo "==============================================================="
    echo "finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")  exit=$EXIT"
    echo "==============================================================="
    exit "$EXIT"
} 2>&1 | tee "$LOG_FILE" | tee logs/last.log
exit "${PIPESTATUS[0]}"
