# Automated Insider Trading Detection Setup

## Quick Start

### Option 1: Windows Task Scheduler (Recommended for Windows)

1. **Right-click** on `setup_scheduler.bat` and select **"Run as Administrator"**
2. Press any key when prompted
3. Done! The system will now run automatically every day at 6:00 PM

### Option 2: Manual Windows Setup

1. Press `Win + R`
2. Type `taskschd.msc` and press Enter
3. Click **"Create Basic Task"**
4. Name: `InsiderTradingAnalysis`
5. Trigger: **Daily** at **6:00 PM** (after market close)
6. Action: **Start a program**
7. Program: Browse to `run_analysis.bat`
8. Click **Finish**

### Option 3: Linux/WSL Cron Job

1. Make the script executable:
   ```bash
   chmod +x run_analysis.sh
   ```

2. Edit your crontab:
   ```bash
   crontab -e
   ```

3. Add this line (runs daily at 6 PM):
   ```
   0 18 * * * /path/to/tradeInsider/run_analysis.sh >> /path/to/tradeInsider/cron.log 2>&1
   ```

4. Save and exit

## What Gets Automated

The automation runs these steps daily:

1. **Scrape** latest insider trades from sources
2. **Fetch** current stock prices and corporate events
3. **Analyze** patterns with enhanced detection:
   - Recurring buy patterns
   - Sustained accumulation (no sells)
   - Volume spike correlation
   - After-hours/weekend filings
   - Statistical outliers
   - Multi-ticker accumulation
   - Same-company clustering
   - Perfect timing detection
   - And more...
4. **Send email alerts** for high-risk patterns (score ≥ 100)

## Email Configuration

Before running automation, configure your `.env` file:

```env
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
EMAIL_TO=your-email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Getting a Gmail App Password

1. Go to https://myaccount.google.com/apppasswords
2. Select "Mail" and "Windows Computer"
3. Click "Generate"
4. Copy the 16-character password into `.env`

## Testing

### Test the automation manually:

**Windows:**
```cmd
run_analysis.bat
```

**Linux/WSL:**
```bash
./run_analysis.sh
```

### Check if scheduled task is working:

**Windows:**
1. Open Task Scheduler (`taskschd.msc`)
2. Find "InsiderTradingAnalysis"
3. Right-click → **Run**
4. Check the "Last Run Result" column

**Linux:**
```bash
tail -f cron.log
```

## Customization

### Change Run Time

**Windows:**
- Open Task Scheduler → Edit the task → Triggers → Edit

**Linux:**
- Edit crontab and change the time (format: `minute hour * * *`)
  - 9 AM: `0 9 * * *`
  - 6 PM: `0 18 * * *`
  - 11 PM: `0 23 * * *`

### Change Alert Threshold

Edit `pipeline.py`:
```python
MIN_RISK_SCORE_FOR_ALERT = 100  # Lower = more alerts, Higher = fewer alerts
```

### Run Multiple Times Per Day

**Windows:** Create multiple scheduled tasks with different times

**Linux:** Add multiple cron entries:
```
0 9 * * * /path/to/run_analysis.sh
0 18 * * * /path/to/run_analysis.sh
```

## Troubleshooting

### "Task failed to start"
- Make sure Python is in your PATH
- Run `setup_scheduler.bat` as Administrator

### "No email received"
- Check `.env` file has correct credentials
- Test with: `python pipeline.py`
- Check spam folder

### "Database locked" error
- Make sure analyzer isn't already running
- Close any database browsers (DB Browser for SQLite, etc.)

### "Module not found" error
- Install dependencies: `pip install -r requirements.txt`

## Monitoring

### Check Reports

Reports are saved in `./reports/` folder:
- `insider_TOP10_YYYYMMDD_HHMMSS.txt` - Latest top 10 suspicious traders

### Check Logs (Linux)

```bash
tail -f cron.log
```

### View Task History (Windows)

1. Open Task Scheduler
2. Click on "Task Scheduler Library"
3. Find "InsiderTradingAnalysis"
4. Click "History" tab at the bottom

## Performance

The enhanced system includes:
- ✅ Database indexes for 3-5x faster queries
- ✅ Parallel processing for timing detection
- ✅ Noise filtering (removes trades <$5K)
- ✅ Optimized volume spike detection
- ✅ Smart caching of price data (24-hour refresh)

Expected runtime: 2-10 minutes depending on data size

## Support

If you encounter issues:
1. Check the error message in the console/log
2. Verify `.env` configuration
3. Test each script individually:
   - `python scrape.py --auto`
   - `python fetch_data.py all`
   - `python analyzer.py`
   - `python pipeline.py`

Good luck catching those insider traders! 🎯
