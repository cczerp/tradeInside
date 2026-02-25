# pipeline.py
import subprocess
import sys
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Email settings from .env
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
GMAIL_USER = os.getenv('EMAIL_USER')
GMAIL_APP_PASSWORD = os.getenv('EMAIL_PASS')
ALERT_EMAIL = os.getenv('EMAIL_TO')

# Alert thresholds
ALERT_ON_SAME_COMPANY_CLUSTER = True
ALERT_ON_REPEAT_OFFENDER = True
ALERT_ON_LARGE_TRADE = True
ALERT_ON_PRE_EVENT_TIMING = True

# Minimum risk score to alert (prevents spam)
MIN_RISK_SCORE_FOR_ALERT = 150

# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

def send_email_alert(subject, body):
    """Send email via Gmail SMTP"""
    try:
        # Parse multiple email recipients
        recipients = [email.strip() for email in ALERT_EMAIL.split(',')]
        
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = ', '.join(recipients)  # Join for display in email header
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(GMAIL_USER, recipients, text)  # Send to list of recipients
        server.quit()
        
        print(f"[✓] Email alert sent to {len(recipients)} recipient(s): {subject}")
        return True
    except Exception as e:
        print(f"[✗] Failed to send email: {e}")
        return False

def check_for_alerts(report_path):
    """Parse the report and check for alert-worthy patterns"""
    if not os.path.exists(report_path):
        print(f"[!] Report not found: {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    alerts = []
    
    # Parse top 10 traders
    lines = content.split('\n')
    current_trader = None
    current_score = 0
    current_patterns = []
    
    for i, line in enumerate(lines):
        if line.startswith('#') and ' - ' in line:
            # New trader entry
            if current_trader and should_alert(current_trader, current_score, current_patterns):
                alerts.append({
                    'trader': current_trader,
                    'score': current_score,
                    'patterns': current_patterns
                })
            
            # Reset for new trader
            parts = line.split(' - ', 1)
            if len(parts) > 1:
                current_trader = parts[1].split('(')[0].strip()
                current_patterns = []
        
        elif line.startswith('Risk Score:'):
            try:
                current_score = int(line.split(':')[1].split('(')[0].strip())
            except:
                current_score = 0
        
        elif line.strip().startswith('•'):
            current_patterns.append(line.strip())
    
    # Check last trader
    if current_trader and should_alert(current_trader, current_score, current_patterns):
        alerts.append({
            'trader': current_trader,
            'score': current_score,
            'patterns': current_patterns
        })
    
    # Send alerts
    if alerts:
        subject = f"🚨 INSIDER TRADING ALERT: {len(alerts)} Suspicious Pattern(s) Detected"
        body = build_alert_body(alerts)
        send_email_alert(subject, body)
    else:
        print("[✓] No alert-worthy patterns detected")

def should_alert(trader, score, patterns):
    """Determine if this trader warrants an alert"""
    if score < MIN_RISK_SCORE_FOR_ALERT:
        return False
    
    pattern_text = ' '.join(patterns)
    
    # Check for alert triggers
    triggers = []
    
    if ALERT_ON_SAME_COMPANY_CLUSTER and 'Same-company cluster' in pattern_text:
        triggers.append('Same-Company Cluster')
    
    if ALERT_ON_REPEAT_OFFENDER and 'Repeat Offender' in pattern_text:
        triggers.append('Repeat Offender')
    
    if ALERT_ON_LARGE_TRADE and 'Large trade' in pattern_text:
        triggers.append('Large Trade')
    
    if ALERT_ON_PRE_EVENT_TIMING and 'days before event' in pattern_text:
        triggers.append('Pre-Event Timing')
    
    return len(triggers) > 0

def build_alert_body(alerts):
    """Build formatted email body"""
    body = f"Insider Trading Pattern Detection Alert\n"
    body += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    body += "=" * 80 + "\n\n"
    
    for i, alert in enumerate(alerts, 1):
        body += f"#{i} - {alert['trader']}\n"
        body += f"Risk Score: {alert['score']}\n"
        body += f"Patterns:\n"
        for pattern in alert['patterns'][:5]:  # Limit to 5 patterns per trader
            body += f"  {pattern}\n"
        body += "\n" + "-" * 80 + "\n\n"
    
    body += f"\nTotal Alerts: {len(alerts)}\n"
    body += "\nCheck the full report in ./reports/ for complete details.\n"
    
    return body

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_command(command, step_name):
    """Run a shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[✗] {step_name} failed!")
        print(f"Error: {e.stderr}")
        return False

def run_pipeline():
    """Execute the complete pipeline"""
    print("\n" + "="*80)
    print("INSIDER TRADING DETECTION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    steps = [
        ("python scrape.py --auto", "Scrape Latest Insider Trades"),
        ("python fetch_data.py all", "Fetch Stock Prices & Events"),
        ("python analyzer.py", "Analyze Trading Patterns")
    ]
    
    # Execute steps
    for command, step_name in steps:
        if not run_command(command, step_name):
            print(f"\n[✗] Pipeline failed at: {step_name}")
            send_email_alert(
                "❌ Pipeline Failure", 
                f"The insider trading pipeline failed at step: {step_name}\n\nCheck logs for details."
            )
            sys.exit(1)
    
    # Find most recent report
    import glob
    reports = glob.glob("./reports/insider_TOP10_*.txt")
    if reports:
        latest_report = max(reports, key=os.path.getctime)
        print(f"\n[✓] Latest report: {latest_report}")
        
        # Check for alerts
        check_for_alerts(latest_report)
    else:
        print("\n[!] No reports found")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Validate configuration
    if not GMAIL_USER or not GMAIL_APP_PASSWORD or not ALERT_EMAIL:
        print("\n[✗] ERROR: Email credentials not configured in .env file")
        print("\nMake sure your .env file contains:")
        print("  EMAIL_USER=your-email@gmail.com")
        print("  EMAIL_PASS=your-app-password")
        print("  EMAIL_TO=your-email@gmail.com")
        print("\nGenerate app password at: https://myaccount.google.com/apppasswords")
        sys.exit(1)
    
    if GMAIL_USER == "your_email@gmail.com":
        print("\n[!] WARNING: Using default/placeholder email in .env file")
        print("Update your .env file with real credentials before production use.\n")
    
    print(f"[✓] Email configured: {GMAIL_USER} -> {ALERT_EMAIL}\n")
    
    run_pipeline()