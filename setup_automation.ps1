# PowerShell script to set up automated insider trading detection
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "INSIDER TRADING DETECTION - AUTOMATION SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Join-Path $PSScriptRoot "run_analysis.bat"
$taskName = "InsiderTradingAnalysis"

Write-Host "Creating scheduled task..." -ForegroundColor Yellow
Write-Host "  Task Name: $taskName"
Write-Host "  Script: $scriptPath"
Write-Host "  Schedule: Daily at 6:00 PM"
Write-Host ""

try {
    # Check if task exists
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

    if ($existingTask) {
        Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }

    # Create action
    $action = New-ScheduledTaskAction -Execute $scriptPath -Argument "auto"

    # Create trigger (daily at 6 PM)
    $trigger = New-ScheduledTaskTrigger -Daily -At "6:00PM"

    # Create settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

    # Register task
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Automated insider trading pattern detection and email alerts" | Out-Null

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The system will now run automatically:" -ForegroundColor Green
    Write-Host "  • Every day at 6:00 PM (after market close)" -ForegroundColor White
    Write-Host "  • Scrapes latest insider trades" -ForegroundColor White
    Write-Host "  • Analyzes with 12 pattern detectors" -ForegroundColor White
    Write-Host "  • Sends email alerts for high-risk patterns" -ForegroundColor White
    Write-Host ""
    Write-Host "Email alerts will be sent to:" -ForegroundColor Yellow
    Write-Host "  • little.cee.zers@gmail.com" -ForegroundColor White
    Write-Host "  • ajaywtsn@gmail.com" -ForegroundColor White
    Write-Host "  • colangelocourtney@gmail.com" -ForegroundColor White
    Write-Host ""
    Write-Host "To view the task:" -ForegroundColor Cyan
    Write-Host "  1. Press Win+R" -ForegroundColor White
    Write-Host "  2. Type: taskschd.msc" -ForegroundColor White
    Write-Host "  3. Find 'InsiderTradingAnalysis'" -ForegroundColor White
    Write-Host ""
    Write-Host "To test it now, run: .\run_analysis.bat" -ForegroundColor Cyan
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "  1. Right-click PowerShell" -ForegroundColor White
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "  3. Run: .\setup_automation.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
