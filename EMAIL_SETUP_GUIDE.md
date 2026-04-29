# 📧 Email Setup Guide - 2 Minutes

## Current Configuration

**Emails ARE configured and ready to go!**

Alerts will be sent to:
- ✅ little.cee.zers@gmail.com
- ✅ ajaywtsn@gmail.com
- ✅ colangelocourtney@gmail.com

**Just need to refresh the Gmail app password (takes 2 minutes).**

---

## 🔧 Fix Email in 2 Minutes

### Step 1: Generate New Gmail App Password

1. **Go to:** https://myaccount.google.com/apppasswords
2. **Sign in** with: cczerp@gmail.com
3. **Select:**
   - App: Mail
   - Device: Windows Computer
4. **Click "Generate"**
5. **Copy the 16-character password** (looks like: "abcd efgh ijkl mnop")

### Step 2: Update .env File

1. **Open:** `C:\Users\Dragon\Desktop\projettccs\tradeInsider\.env`
2. **Find this line:**
   ```
   EMAIL_PASS=tams ciww tirt dowv
   ```
3. **Replace with your new password:**
   ```
   EMAIL_PASS=abcd efgh ijkl mnop
   ```
4. **Save the file**

### Step 3: Test It

Open PowerShell or Command Prompt:
```bash
cd C:\Users\Dragon\Desktop\projettccs\tradeInsider
python test_email.py
```

You should see:
```
✅ SUCCESS!
Test email sent successfully to 3 recipient(s)!
```

**Check your inbox!** (and spam folder)

---

## ➕ Add More Email Addresses (Optional)

Want to add more people to the alerts? Easy!

### Edit .env File

Find this line:
```
EMAIL_TO=little.cee.zers@gmail.com,ajaywtsn@gmail.com,colangelocourtney@gmail.com
```

Add more emails (comma-separated):
```
EMAIL_TO=little.cee.zers@gmail.com,ajaywtsn@gmail.com,colangelocourtney@gmail.com,newemail@gmail.com,anotheremail@gmail.com
```

**No limit!** Add as many as you want.

---

## 📩 What Emails Will Look Like

### Alert Email Example:

```
From: cczerp@gmail.com
To: little.cee.zers@gmail.com, ajaywtsn@gmail.com, colangelocourtney@gmail.com
Subject: 🚨 INSIDER TRADING ALERT: 3 Suspicious Pattern(s) Detected

Insider Trading Pattern Detection Alert
Generated: 2026-04-29 18:05:23
================================================================================

#1 - Piconi Robert (CEO, NRGV)
Risk Score: 2387
Patterns:
  • Recurring buys: NRGV (6 buys in 1 month, 185,000 shares)
  • Volume spike: NRGV (bought 4d before 105% volume surge)
  • Perfect timing: NRGV (+36.0% in 7-14 days)

--------------------------------------------------------------------------------

#2 - Altman Peter (CEO, BCDA)
Risk Score: 1700
Patterns:
  • Same-company cluster: 3 insiders buying together
  • Recurring buys: BCDA (3 buys in 1 month, 144,000 shares)
  • Volume spike: BCDA (bought 4d before 162% volume surge)

--------------------------------------------------------------------------------

Total Alerts: 3

Check the full report in ./reports/ for complete details.
```

---

## 🔔 When You'll Get Emails

### Daily Schedule:
- **6:00 PM** - System runs automatically
- **6:05 PM** - Analysis completes
- **6:05 PM** - Email sent (if any patterns found)

### Email Triggers:
You get an email when ANY trader scores ≥ 100 points

**Current threshold:** 100 points (catches most patterns)

**Want fewer emails?** Edit `pipeline.py`:
```python
MIN_RISK_SCORE_FOR_ALERT = 200  # Higher = fewer emails
```

**Want more emails?** Lower it:
```python
MIN_RISK_SCORE_FOR_ALERT = 50  # Lower = more emails
```

---

## ✅ How to Know It's Working

### Method 1: Check Latest Report

Every day after 6 PM, check for new report:
```
C:\Users\Dragon\Desktop\projettccs\tradeInsider\reports\
```

**File format:** `insider_TOP10_YYYYMMDD_HHMMSS.txt`

**If you see a new file:** System is working! ✅

---

### Method 2: Run Status Check

**Windows:**
```cmd
cd C:\Users\Dragon\Desktop\projettccs\tradeInsider
powershell -ExecutionPolicy Bypass -File check_system.ps1
```

**PowerShell:**
```powershell
cd C:\Users\Dragon\Desktop\projettccs\tradeInsider
.\check_system.ps1
```

You'll see:
```
[✅] Automation:   ACTIVE - Next run: Tomorrow 6:00 PM
[✅] Database:     EXISTS - 61.42 MB
[✅] Latest Report: Today at 6:05 PM
[✅] Email Config: 3 recipients
```

---

### Method 3: Check Task Scheduler

1. Press `Win + R`
2. Type: `taskschd.msc`
3. Find: **"InsiderTradingAnalysis"**
4. Check:
   - **Status:** Ready ✅
   - **Next Run Time:** Tomorrow 6:00 PM ✅
   - **Last Run Result:** 0x0 (success) ✅

---

### Method 4: Check Email Inbox

**Tomorrow at 6:05 PM:**
- Check inbox for alert email
- If no email: Either (a) no patterns found, or (b) email needs fixing

**To test email NOW:**
```bash
python test_email.py
```

---

## 🚨 Troubleshooting

### No Email Received?

**Check 1: Email password**
```bash
python test_email.py
```

If you see `SMTPAuthenticationError`:
- Gmail app password expired or wrong
- Follow "Step 1: Generate New Gmail App Password" above

**Check 2: Spam folder**
- Gmail might filter as spam first time
- Mark as "Not Spam" to train filter

**Check 3: System ran?**
```powershell
.\check_system.ps1
```

Look for "Latest Report" - if it's fresh (< 2 hours old), system ran successfully.

**Check 4: Any patterns found?**
Open latest report. If no one scored ≥ 100, no email was sent (this is normal).

---

### Email Works But No Alerts?

**This is NORMAL!**

Not every day has suspicious patterns. You might get:
- **0-2 alerts per week** during slow periods
- **5-10 alerts per week** during busy periods
- **15-30 alerts per week** during earnings season

**If you want to test:** Lower the threshold temporarily:

Edit `pipeline.py`:
```python
MIN_RISK_SCORE_FOR_ALERT = 50  # Very sensitive (for testing)
```

Run manually:
```bash
python pipeline.py
```

You should get an email with more patterns.

**Then change it back to 100 for production.**

---

### Want to Test Right Now?

**Force a manual run:**
```bash
cd C:\Users\Dragon\Desktop\projettccs\tradeInsider
run_analysis.bat
```

This will:
1. Scrape latest trades
2. Run analysis
3. Generate report
4. Send email (if patterns found)

**Time:** 2-5 minutes

---

## 📊 Email Settings Reference

All email settings are in `.env`:

```env
# From address (your Gmail)
EMAIL_USER=cczerp@gmail.com

# Gmail app password (16 characters from Google)
EMAIL_PASS=your-app-password-here

# Recipients (comma-separated, no spaces)
EMAIL_TO=email1@gmail.com,email2@gmail.com,email3@gmail.com

# SMTP server (don't change)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**To add recipients:** Just add more emails to `EMAIL_TO`, separated by commas.

**No limit on recipients!** Send to 1 person or 100 people.

---

## 🎯 Quick Commands

### Send test email:
```bash
python test_email.py
```

### Check system status:
```powershell
.\check_system.ps1
```

### Run analysis manually:
```bash
.\run_analysis.bat
```

### View latest report:
```bash
notepad reports\insider_TOP10_*.txt
```

### Edit email config:
```bash
notepad .env
```

---

## ✅ Summary

**Current Setup:**
- ✅ 3 recipients configured
- ✅ Automation running daily at 6 PM
- ⚠️ Email password needs refresh (2-minute fix)

**To finish setup:**
1. Generate new Gmail app password (link above)
2. Update `.env` file
3. Run `python test_email.py`
4. Check inbox

**Then:**
- Wait for tomorrow 6 PM for first real alert
- Or run `run_analysis.bat` to test now

**You're almost done!** 🎯
