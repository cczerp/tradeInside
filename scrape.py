# scrape.py
import os, sys, asyncio, json
import pandas as pd
import questionary
from datetime import datetime
from playwright.async_api import async_playwright
from io import StringIO

from sites import SITES
from database import Database
from aliases import ALIASES
from normalize import normalize_columns, get_standard_columns

BASE = os.path.dirname(__file__)
DIR_HTML = os.path.join(BASE, "data", "html")
DIR_CSV  = os.path.join(BASE, "data", "csv")
DIR_JSON = os.path.join(BASE, "data", "json")
for d in (DIR_HTML, DIR_CSV, DIR_JSON): 
    os.makedirs(d, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def ts():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")

def normalize_key(display_name: str) -> str:
    """Convert a friendly menu label → actual sites.py key"""
    return ALIASES.get(display_name, display_name)

def export_table(df, category, sub, page_idx):
    """Save DataFrame with STANDARDIZED columns"""
    df = normalize_columns(df)
    
    df["source"] = category
    df["section"] = sub
    df["page"] = page_idx
    df["scraped_at"] = ts()
    
    standard_cols = get_standard_columns()
    existing_standard = [c for c in standard_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in standard_cols]
    df = df[existing_standard + other_cols]

    csvp = os.path.join(DIR_CSV, f"{category}_{sub}_page{page_idx}.csv")
    
    try:
        df.to_csv(csvp, index=False)
        print(f"   CSV: {csvp} ({len(df)} rows)")
    except PermissionError:
        print(f"   WARNING: Could not save CSV (file is open) - skipping {csvp}")

    jsonp = os.path.join(DIR_JSON, f"{category}_{sub}_page{page_idx}.jsonl")
    with open(jsonp, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=str, ensure_ascii=False)+"\n")
    print(f"   JSONL: {jsonp}")
    
    return df

def insert_to_database(df, db, category, sub):
    """Insert normalized data into database"""
    if not db or not db.conn:
        return
    
    for _, row in df.iterrows():
        if 'ticker' not in row or pd.isna(row.get('ticker')):
            continue
        
        trade = {
            "source": category,
            "ticker": str(row['ticker']).upper().strip(),
            "transaction_date": str(row.get('transaction_date', '')),
            "transaction_type": str(row.get('transaction_type', '')),
            "insider_name": str(row.get('trader_name', '')),
            "politician_name": None,
            "fund_name": None,
            "shares": row.get('shares'),
            "price": row.get('price'),
            "filing_date": str(row.get('filing_date', '')),
        }
        
        try:
            db.insert_trade(trade)
        except:
            pass

# -----------------------
# Parsers
# -----------------------
def parse_all_tables(html):
    """Extract ALL tables from HTML"""
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
        if not tables:
            return None
        return max(tables, key=lambda t: t.shape[0] * t.shape[1])
    except:
        return None

# -----------------------
# Core Scraping
# -----------------------
async def paginate_and_scrape(page, category, sub, url, db):
    """Scrape a site with pagination support (except OpenInsider)"""
    page_idx = 1
    
    single_page_only = (category == "openinsider")
    
    while True:
        print(f"\n{category} -> {sub} | Page {page_idx}: {url}")
        
        try:
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"   Failed to load page: {e}")
            break

        html = await page.content()
        fp = os.path.join(DIR_HTML, f"{category}_{sub}_page{page_idx}.html")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"   HTML: {fp}")

        df = parse_all_tables(html)

        if df is not None and not df.empty:
            df = export_table(df, category, sub, page_idx)
            insert_to_database(df, db, category, sub)
        else:
            print("   No parseable table found")

        if single_page_only:
            print("   No pagination for this site.")
            break

        next_btn = await page.query_selector("a[rel=next], a:has-text('Next'):not([class*='disabled'])")
        if next_btn:
            href = await next_btn.get_attribute("href")
            if href:
                if href.startswith("http"):
                    url = href
                elif href.startswith("/"):
                    from urllib.parse import urljoin
                    url = urljoin(page.url, href)
                elif href.startswith("?"):
                    base = url.split("?")[0]
                    url = base + href
                else:
                    url = url.rstrip("/") + "/" + href
                
                page_idx += 1
                continue
            else:
                print("   Next button has no valid href.")
                break
        else:
            print("   No more pages.")
            break


async def scrape_site(site, subs):
    """Main scraper entry point"""
    targets = SITES[site]
    db = Database()
    db.connect()
    db.migrate()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False, 
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            locale="en-US", 
            viewport={"width":1366, "height":768}
        )
        page = await context.new_page()

        for sub in subs:
            if sub not in targets:
                print(f"Unknown subpage key: {sub}, skipping.")
                continue
            
            url = targets[sub]
            
            if "your-ticker" in url or "your-target-page" in url:
                print(f"Skipping placeholder URL: {url}")
                continue
                
            await paginate_and_scrape(page, site, sub, url, db)

        await context.close()
        await browser.close()

    db.close()

# -----------------------
# Menu
# -----------------------
def menu():
    while True:
        site_choices = ["All Sites", "---"] + list(SITES.keys()) + ["Exit"]
        
        site = questionary.select(
            "Pick a site:",
            choices=site_choices
        ).ask()

        if site == "Exit" or site is None:
            print("Exiting menu.")
            sys.exit(0)

        if site == "All Sites":
            for site_name in SITES.keys():
                print(f"\n{'='*60}")
                print(f"SCRAPING: {site_name.upper()}")
                print('='*60)
                
                all_subs = list(SITES[site_name].keys())
                asyncio.run(scrape_site(site_name, all_subs))
            
            print(f"\n{'='*60}")
            print("ALL SITES SCRAPED")
            print("="*60)
            print("\nNext steps:")
            print("  1. Run: python fetch_data.py all")
            print("  2. Run: python anal.py")
            continue

        if site == "---":
            continue
            
        subs_display = []
        for friendly_name, real_key in ALIASES.items():
            if site in SITES and real_key in SITES[site]:
                subs_display.append(friendly_name)

        if not subs_display:
            subs_display = list(SITES[site].keys())

        subs_display_with_all = ["Select All"] + subs_display

        subs = questionary.checkbox(
            f"Pick pages from {site}:",
            choices=subs_display_with_all
        ).ask()

        if not subs:
            print("No selection.\n")
            continue

        if "Select All" in subs:
            subs = subs_display

        real_subs = [normalize_key(sub) for sub in subs]
        asyncio.run(scrape_site(site, real_subs))

if __name__ == "__main__":
    import sys
    
    # Check for --auto flag
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("Running in automated mode - scraping all sites...")
        for site_name in SITES.keys():
            print(f"\n{'='*60}")
            print(f"SCRAPING: {site_name.upper()}")
            print('='*60)
            
            all_subs = list(SITES[site_name].keys())
            asyncio.run(scrape_site(site_name, all_subs))
        
        print(f"\n{'='*60}")
        print("ALL SITES SCRAPED")
        print("="*60)
    else:
        # Run interactive menu
        menu()