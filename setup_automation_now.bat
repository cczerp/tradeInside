@echo off
echo ========================================
echo SETTING UP AUTOMATED INSIDER TRADING DETECTION
echo ========================================
echo.
echo This will create a Windows scheduled task to run daily at 6:00 PM
echo.

set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%run_analysis.bat

echo Creating scheduled task...
echo.

schtasks /create /tn "InsiderTradingAnalysis" /tr "\"%SCRIPT_PATH%\" auto" /sc daily /st 18:00 /f

if errorlevel 1 (
    echo.
    echo ========================================
    echo NOTICE: Administrator privileges required
    echo ========================================
    echo.
    echo To set up automation:
    echo   1. Right-click this file: setup_automation_now.bat
    echo   2. Select "Run as Administrator"
    echo.
    echo OR manually create the task:
    echo   1. Press Win+R
    echo   2. Type: taskschd.msc
    echo   3. Create task pointing to: run_analysis.bat
    echo   4. Schedule: Daily at 6:00 PM
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo The system will now run automatically every day at 6:00 PM
echo.
echo You will receive email alerts at:
echo   • little.cee.zers@gmail.com
echo   • ajaywtsn@gmail.com
echo   • colangelocourtney@gmail.com
echo.
echo To test it now, run: run_analysis.bat
echo.
echo Reports will be saved in: %SCRIPT_DIR%reports\
echo.
pause
