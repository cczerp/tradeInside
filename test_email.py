#!/usr/bin/env python3
"""
Test email configuration
Send a test email to verify everything is working
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email settings
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
GMAIL_USER = os.getenv('EMAIL_USER')
GMAIL_APP_PASSWORD = os.getenv('EMAIL_PASS')
ALERT_EMAIL = os.getenv('EMAIL_TO')

def send_test_email():
    """Send a test email to verify configuration"""

    print("=" * 80)
    print("INSIDER TRADING DETECTION - EMAIL TEST")
    print("=" * 80)
    print()

    # Validate configuration
    if not GMAIL_USER or not GMAIL_APP_PASSWORD or not ALERT_EMAIL:
        print("[ERROR] Email not configured in .env file")
        print()
        print("Your .env file needs:")
        print("  EMAIL_USER=your-email@gmail.com")
        print("  EMAIL_PASS=your-app-password")
        print("  EMAIL_TO=recipient1@gmail.com,recipient2@gmail.com")
        print()
        return False

    print(f"[*] From: {GMAIL_USER}")
    print(f"[*] To: {ALERT_EMAIL}")
    print(f"[*] Server: {SMTP_SERVER}:{SMTP_PORT}")
    print()

    # Parse multiple recipients
    recipients = [email.strip() for email in ALERT_EMAIL.split(',')]
    print(f"[*] Sending to {len(recipients)} recipient(s):")
    for email in recipients:
        print(f"    - {email}")
    print()

    # Build test email
    subject = "🎯 Test Alert - Insider Trading Detection System"

    body = f"""
Insider Trading Detection System - Test Email

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
✅ EMAIL SYSTEM IS WORKING!
========================================

This is a test email from your insider trading detection system.

Configuration:
  • From: {GMAIL_USER}
  • Server: {SMTP_SERVER}
  • Recipients: {len(recipients)}

What happens next:
  • System runs automatically every day at 6:00 PM
  • Analyzes 1,000-5,000 insider trades
  • Detects 12 different suspicious patterns
  • Sends email alerts when risk score ≥ 100

Example alert (you'll get this format):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 INSIDER TRADING ALERT: 3 Suspicious Patterns Detected

#1 - John Smith (CEO, NVDA)
Risk Score: 2387
Patterns:
  • Recurring buys: 6x in 1 month
  • Volume spike: bought before 391% surge
  • Perfect timing: +69.4%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your system is configured and ready!

Next alert: Tonight at 6:00 PM (if patterns found)

Questions? Check these docs:
  • HOW_IT_WORKS.md - Complete guide
  • QUICK_START.md - 5-minute setup
  • SETUP_COMPLETE.md - What was done

Good hunting! 🎯

---
Automated Insider Trading Detection System
Powered by 12 pattern detection algorithms
"""

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        print("[*] Connecting to SMTP server...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()

        print("[*] Logging in...")
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)

        print("[*] Sending email...")
        text = msg.as_string()
        server.sendmail(GMAIL_USER, recipients, text)
        server.quit()

        print()
        print("=" * 80)
        print("✅ SUCCESS!")
        print("=" * 80)
        print()
        print(f"Test email sent successfully to {len(recipients)} recipient(s)!")
        print()
        print("Check your inbox for the test email.")
        print("(Don't forget to check spam folder)")
        print()
        print("Your system is ready to send daily alerts!")
        print()
        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR: Failed to send email")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("  1. Wrong email/password in .env")
        print("  2. Need to generate Gmail app password:")
        print("     https://myaccount.google.com/apppasswords")
        print("  3. Less secure app access disabled")
        print()
        return False

if __name__ == "__main__":
    send_test_email()
