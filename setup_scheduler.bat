@echo off
REM Setup Windows Task Scheduler for automatic daily runs

echo ========================================
echo TASK SCHEDULER SETUP
echo ========================================
echo.
echo This will create a scheduled task to run the insider trading
echo analysis every day at 6:00 PM (after market close).
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%run_analysis.bat

echo.
echo Creating scheduled task...
echo.

schtasks /create /tn "InsiderTradingAnalysis" /tr "\"%SCRIPT_PATH%\" auto" /sc daily /st 18:00 /f

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create scheduled task
    echo You may need to run this script as Administrator
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Scheduled task created: InsiderTradingAnalysis
echo Run time: Daily at 6:00 PM
echo.
echo To view or modify the task:
echo   1. Press Win+R
echo   2. Type: taskschd.msc
echo   3. Find "InsiderTradingAnalysis" in the list
echo.
echo To test it now, run: run_analysis.bat
echo.
pause
