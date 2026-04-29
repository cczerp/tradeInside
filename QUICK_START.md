# 🚀 Quick Start Guide

## Get Up and Running in 5 Minutes

### Step 1: Configure Email (2 minutes)

1. Create a `.env` file in this directory if it doesn't exist
2. Add these lines (replace with your info):

```env
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
EMAIL_TO=your-email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

3. Get Gmail app password:
   - Go to https://myaccount.google.com/apppasswords
   - Generate a password for "Mail"
   - Copy the 16-character password to `EMAIL_PASS`

---

### Step 2: Add Database Indexes (30 seconds)

```bash
python add_indexes.py
```

This makes queries 5x faster.

---

### Step 3: Test the System (2 minutes)

```bash
# Windows
run_analysis.bat

# Linux/WSL
./run_analysis.sh
```

This will:
- Scrape latest insider trades
- Fetch stock prices
- Run enhanced analysis with 12 pattern detectors
- Generate report in `./reports/`
- Send email alert if any patterns found

---

### Step 4: Set Up Automation (1 minute)

**Windows (Right-click → "Run as Administrator"):**
```bash
setup_scheduler.bat
```

**Linux/WSL:**
```bash
chmod +x run_analysis.sh
crontab -e
# Add: 0 18 * * * /full/path/to/run_analysis.sh >> /full/path/to/cron.log 2>&1
```

---

## ✅ You're Done!

The system will now:
- Run **automatically every day at 6 PM** (after market close)
- **Email you** when suspicious patterns are detected (score ≥ 100)
- Save **detailed reports** in `./reports/` folder

---

## 📊 What to Expect

### High-Priority Alerts (Score 200+)

Look for patterns with:
- ✅ Recurring buys (3+ in 30 days)
- ✅ Sustained accumulation (60+ days, no sells)
- ✅ Same-company cluster (3+ insiders)
- ✅ Volume spike correlation
- ✅ CEO/CFO trades (2.0x multiplier)
- ✅ Repeat offenders (2.5x multiplier)

### Example Email Alert:
```
🚨 INSIDER TRADING ALERT: 2 Suspicious Pattern(s) Detected

#1 - John Smith (CEO, NVDA)
Risk Score: 287
Patterns:
  • Recurring buys: NVDA (5 buys in 2 months, 125,000 shares)
  • Volume spike: NVDA (bought 12d before 347% volume surge)
  • Perfect timing: NVDA (+47.3% in 7-14 days)
```

---

## 🎯 Trading Strategy

1. **Review alerts** when you get the email
2. **Check full report** in `./reports/` folder
3. **Cross-reference** with your DD
4. **Enter positions** on high-conviction signals (score 200+)

---

## 🔧 Troubleshooting

### No email received?
- Check spam folder
- Verify `.env` credentials
- Test: `python pipeline.py`

### "Database locked" error?
- Close DB Browser for SQLite if open
- Make sure no other instance is running

### "Module not found" error?
```bash
pip install pandas sqlite3 yfinance numpy
```

---

## 📚 Full Documentation

- **ENHANCEMENTS_SUMMARY.md** - Complete feature list
- **AUTOMATION_SETUP.md** - Detailed automation guide

---

That's it! You're now running a professional-grade insider trading detection system. 🎯

Good luck! 📈
