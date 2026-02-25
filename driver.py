import os
import re
import zipfile
import requests
import subprocess
import sys
import shutil
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def get_chrome_version():
    """Get installed Google Chrome version (Windows)."""
    try:
        output = subprocess.check_output(
            r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version',
            shell=True
        ).decode()
        version = re.search(r"(\d+\.\d+\.\d+\.\d+)", output).group(1)
        return version
    except Exception as e:
        print("[-] Could not detect Chrome version:", e)
        sys.exit(1)

def download_chromedriver(chrome_version, target_dir="drivers"):
    """Download and extract matching ChromeDriver for the installed Chrome version."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "chromedriver.zip"

    base_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/win64/chromedriver-win64.zip"

    print(f"[+] Downloading ChromeDriver {chrome_version}...")
    r = requests.get(base_url, stream=True)
    if r.status_code != 200:
        print(f"[-] Failed to download driver: {r.status_code}")
        sys.exit(1)

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    extracted_driver = target_dir / "chromedriver-win64" / "chromedriver.exe"
    final_driver = target_dir / "chromedriver.exe"
    shutil.move(str(extracted_driver), str(final_driver))

    shutil.rmtree(target_dir / "chromedriver-win64")
    os.remove(zip_path)

    print(f"[✓] Installed ChromeDriver at: {final_driver}")
    return str(final_driver)

def ensure_chromedriver():
    """Ensure a matching ChromeDriver exists, otherwise download it. SILENT."""
    chrome_version = get_chrome_version()
    driver_path = Path("drivers/chromedriver.exe")
    if driver_path.exists():
        return str(driver_path)
    return download_chromedriver(chrome_version)

def get_driver(headless=False):
    """
    Create and return a fully stealthed Chrome WebDriver.
    Hides all automation indicators.
    """
    options = Options()
    
    # === STEALTH MODE ===
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Remove "Chrome is being controlled" banner
    options.add_argument("--disable-infobars")
    
    # Realistic window size
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    
    # Real user agent
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    
    # SSL/Security
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--allow-insecure-localhost")
    
    # Stability
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # Suppress all logs
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    
    # Performance
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--mute-audio")
    
    # Set preferences to look more human
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)
    
    if headless:
        options.add_argument("--headless=new")
    
    # Get driver path silently
    driver_path = ensure_chromedriver()
    service = Service(driver_path)
    service.log_path = os.devnull  # Suppress service logs completely
    
    # Create driver
    driver = webdriver.Chrome(service=service, options=options)
    
    # === JAVASCRIPT STEALTH INJECTION ===
    stealth_js = """
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    window.chrome = {runtime: {}};
    Object.defineProperty(navigator, 'permissions', {
        get: () => ({
            query: () => Promise.resolve({state: 'granted'})
        })
    });
    """
    
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": stealth_js})
    
    # Set timeouts
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(10)
    
    return driver