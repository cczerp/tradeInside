@echo off
REM Automated Insider Trading Analysis Script
REM Run this script daily to detect insider trading patterns

echo ========================================
echo INSIDER TRADING DETECTION - AUTOMATED RUN
echo Started: %date% %time%
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Running scraper...
python scrape.py --auto
if errorlevel 1 (
    echo ERROR: Scraper failed
    exit /b 1
)

echo.
echo [2/3] Fetching stock data...
python fetch_data.py all
if errorlevel 1 (
    echo ERROR: Data fetch failed
    exit /b 1
)

echo.
echo [3/3] Running analysis...
python analyzer.py
if errorlevel 1 (
    echo ERROR: Analysis failed
    exit /b 1
)

echo.
echo [4/3] Sending email alerts...
python pipeline.py
if errorlevel 1 (
    echo ERROR: Pipeline failed
    exit /b 1
)

echo.
echo ========================================
echo ANALYSIS COMPLETE
echo Finished: %date% %time%
echo ========================================

REM Keep window open if run manually
if "%1"=="" pause
