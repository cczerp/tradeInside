@echo off
echo ========================================
echo INSIDER TRADING SYSTEM - STATUS CHECK
echo ========================================
echo.

REM Check if scheduled task exists
echo [1/4] Checking automation...
schtasks /query /tn "InsiderTradingAnalysis" >nul 2>&1
if errorlevel 1 (
    echo   Status: NOT CONFIGURED
    echo   Action: Run setup_automation.ps1 as Administrator
) else (
    echo   Status: ACTIVE
    schtasks /query /tn "InsiderTradingAnalysis" /fo LIST | findstr /C:"Next Run Time" /C:"Last Run Time" /C:"Status"
)

echo.
echo [2/4] Checking database...
if exist "tradeinsider.db" (
    echo   Status: EXISTS
    for %%I in (tradeinsider.db) do echo   Size: %%~zI bytes
) else (
    echo   Status: NOT FOUND
    echo   Action: Run analyzer.py to create database
)

echo.
echo [3/4] Checking latest report...
if exist "reports\" (
    dir /b /o-d "reports\insider_TOP10_*.txt" 2>nul | findstr /r ".*" >nul
    if errorlevel 1 (
        echo   Status: NO REPORTS YET
        echo   Action: Run run_analysis.bat to generate first report
    ) else (
        for /f "delims=" %%i in ('dir /b /o-d "reports\insider_TOP10_*.txt" 2^>nul') do (
            echo   Latest: %%i
            goto :foundreport
        )
        :foundreport
    )
) else (
    echo   Status: Reports folder not found
)

echo.
echo [4/4] Checking email configuration...
findstr /C:"EMAIL_USER" .env >nul 2>&1
if errorlevel 1 (
    echo   Status: NOT CONFIGURED
    echo   Action: Add EMAIL_USER, EMAIL_PASS, EMAIL_TO to .env
) else (
    for /f "tokens=2 delims==" %%a in ('findstr /C:"EMAIL_USER" .env') do echo   From: %%a
    for /f "tokens=2 delims==" %%a in ('findstr /C:"EMAIL_TO" .env') do echo   To: %%a
    echo   Status: CONFIGURED
    echo   Test: Run python test_email.py
)

echo.
echo ========================================
echo QUICK ACTIONS
echo ========================================
echo.
echo Test system now:     run_analysis.bat
echo Send test email:     python test_email.py
echo View latest report:  notepad reports\insider_TOP10_*.txt
echo.
pause
