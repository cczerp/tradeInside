# PowerShell version of system check
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "INSIDER TRADING SYSTEM - STATUS CHECK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check scheduled task
Write-Host "[1/4] Checking automation..." -ForegroundColor Yellow
try {
    $task = Get-ScheduledTask -TaskName "InsiderTradingAnalysis" -ErrorAction Stop
    Write-Host "  Status: " -NoNewline
    Write-Host "ACTIVE" -ForegroundColor Green
    Write-Host "  Next Run: $($task.Triggers[0].StartBoundary)"

    $taskInfo = Get-ScheduledTaskInfo -TaskName "InsiderTradingAnalysis"
    Write-Host "  Last Run: $($taskInfo.LastRunTime)"
    Write-Host "  Last Result: $($taskInfo.LastTaskResult) (0 = Success)"
} catch {
    Write-Host "  Status: " -NoNewline
    Write-Host "NOT CONFIGURED" -ForegroundColor Red
    Write-Host "  Action: Run setup_automation.ps1 as Administrator"
}

Write-Host ""

# Check database
Write-Host "[2/4] Checking database..." -ForegroundColor Yellow
if (Test-Path "tradeinsider.db") {
    $dbSize = (Get-Item "tradeinsider.db").Length
    Write-Host "  Status: " -NoNewline
    Write-Host "EXISTS" -ForegroundColor Green
    Write-Host "  Size: $([math]::Round($dbSize/1MB, 2)) MB"
} else {
    Write-Host "  Status: " -NoNewline
    Write-Host "NOT FOUND" -ForegroundColor Red
    Write-Host "  Action: Run analyzer.py to create database"
}

Write-Host ""

# Check latest report
Write-Host "[3/4] Checking latest report..." -ForegroundColor Yellow
if (Test-Path "reports") {
    $latestReport = Get-ChildItem "reports\insider_TOP10_*.txt" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1

    if ($latestReport) {
        Write-Host "  Latest: " -NoNewline
        Write-Host "$($latestReport.Name)" -ForegroundColor Green
        Write-Host "  Created: $($latestReport.LastWriteTime)"

        $ageHours = ((Get-Date) - $latestReport.LastWriteTime).TotalHours
        if ($ageHours -lt 24) {
            Write-Host "  Age: " -NoNewline
            Write-Host "$([math]::Round($ageHours, 1)) hours (FRESH)" -ForegroundColor Green
        } else {
            Write-Host "  Age: " -NoNewline
            Write-Host "$([math]::Round($ageHours/24, 1)) days" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Status: " -NoNewline
        Write-Host "NO REPORTS YET" -ForegroundColor Yellow
        Write-Host "  Action: Run run_analysis.bat"
    }
} else {
    Write-Host "  Status: Reports folder not found" -ForegroundColor Red
}

Write-Host ""

# Check email config
Write-Host "[4/4] Checking email configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    $envContent = Get-Content ".env"
    $emailUser = $envContent | Select-String "EMAIL_USER=" | ForEach-Object { $_ -replace "EMAIL_USER=", "" }
    $emailTo = $envContent | Select-String "EMAIL_TO=" | ForEach-Object { $_ -replace "EMAIL_TO=", "" }

    if ($emailUser) {
        Write-Host "  From: " -NoNewline
        Write-Host "$emailUser" -ForegroundColor Green

        if ($emailTo) {
            $recipients = $emailTo -split ","
            Write-Host "  To: " -NoNewline
            Write-Host "$($recipients.Count) recipient(s)" -ForegroundColor Green
            foreach ($email in $recipients) {
                Write-Host "      - $($email.Trim())"
            }
        }

        Write-Host "  Status: " -NoNewline
        Write-Host "CONFIGURED" -ForegroundColor Green
        Write-Host "  Test: python test_email.py"
    } else {
        Write-Host "  Status: " -NoNewline
        Write-Host "NOT CONFIGURED" -ForegroundColor Red
    }
} else {
    Write-Host "  Status: " -NoNewline
    Write-Host ".env file not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "QUICK ACTIONS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test system now:     " -NoNewline
Write-Host ".\run_analysis.bat" -ForegroundColor Yellow
Write-Host "Send test email:     " -NoNewline
Write-Host "python test_email.py" -ForegroundColor Yellow
Write-Host "View latest report:  " -NoNewline
Write-Host "notepad reports\insider_TOP10_*.txt" -ForegroundColor Yellow
Write-Host "Fix email config:    " -NoNewline
Write-Host "notepad .env" -ForegroundColor Yellow
Write-Host ""
