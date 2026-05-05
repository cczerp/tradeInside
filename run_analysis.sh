#!/bin/bash
# Automated Insider Trading Analysis Script (Linux/WSL)
# Run this script daily to detect insider trading patterns

echo "========================================"
echo "INSIDER TRADING DETECTION - AUTOMATED RUN"
echo "Started: $(date)"
echo "========================================"
echo

cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

echo "[1/3] Running scraper..."
python3 scrape.py --auto
if [ $? -ne 0 ]; then
    echo "ERROR: Scraper failed"
    exit 1
fi

echo
echo "[2/3] Fetching stock data..."
python3 fetch_data.py all
if [ $? -ne 0 ]; then
    echo "ERROR: Data fetch failed"
    exit 1
fi

echo
echo "[3/3] Running analysis..."
python3 analyzer.py
if [ $? -ne 0 ]; then
    echo "ERROR: Analysis failed"
    exit 1
fi

echo
echo "[4/3] Sending email alerts..."
python3 pipeline.py --email-only "$@"
if [ $? -ne 0 ]; then
    echo "ERROR: Pipeline failed"
    exit 1
fi

echo
echo "========================================"
echo "ANALYSIS COMPLETE"
echo "Finished: $(date)"
echo "========================================"
