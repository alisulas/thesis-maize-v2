"""Download BPS Indonesia corn/maize yield data by province, 2003-2023.

Strategy:
1. Primary: BPS WebAPI (requires API key — register at webapi.bps.go.id)
2. Fallback A: BPS static table direct download via requests
3. Fallback B: Use FAOSTAT national + note province data needs manual download

Without BPS API key, this script uses the fallback approach and saves a
detailed manual download guide.

Outputs:
    data/raw/indonesia/bps_corn_province_*.csv (if automated works)
    data/raw/indonesia/MANUAL_DOWNLOAD_GUIDE.md (always saved)
    data/processed/indonesia/yield_indonesia_2003_2023.parquet (when data is ready)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "indonesia"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "indonesia"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# BPS WebAPI base URL
BPS_API_BASE = "https://webapi.bps.go.id/v1/api"

# Known BPS table IDs for corn/jagung (may need verification)
# These are the indicator IDs for agricultural production tables
BPS_CORN_TABLE_IDS = {
    "production_ton": 1715,   # Produksi Jagung menurut Provinsi
    "planted_ha": 1716,       # Luas Tanam Jagung menurut Provinsi
    "harvested_ha": 1717,     # Luas Panen Jagung menurut Provinsi
}

# BPS province codes and English names (34 provinces as of 2013+)
BPS_PROVINCE_MAP: dict[str, str] = {
    "11": "Aceh",
    "12": "North Sumatra",
    "13": "West Sumatra",
    "14": "Riau",
    "15": "Jambi",
    "16": "South Sumatra",
    "17": "Bengkulu",
    "18": "Lampung",
    "19": "Bangka Belitung Islands",
    "21": "Riau Islands",
    "31": "DKI Jakarta",
    "32": "West Java",
    "33": "Central Java",
    "34": "DI Yogyakarta",
    "35": "East Java",
    "36": "Banten",
    "51": "Bali",
    "52": "West Nusa Tenggara",
    "53": "East Nusa Tenggara",
    "61": "West Kalimantan",
    "62": "Central Kalimantan",
    "63": "South Kalimantan",
    "64": "East Kalimantan",
    "65": "North Kalimantan",    # Created 2012, data from 2013
    "71": "North Sulawesi",
    "72": "Central Sulawesi",
    "73": "South Sulawesi",
    "74": "Southeast Sulawesi",
    "75": "Gorontalo",
    "76": "West Sulawesi",
    "81": "Maluku",
    "82": "North Maluku",
    "91": "West Papua",
    "94": "Papua",
}


def try_bps_api(api_key: str, table_id: int) -> pd.DataFrame | None:
    """Attempt to fetch data from BPS WebAPI.

    Args:
        api_key: BPS API key from webapi.bps.go.id.
        table_id: BPS static table ID.

    Returns:
        Dataframe if successful, None otherwise.
    """
    url = f"{BPS_API_BASE}/view/model/statictable/domain/0000/lang/ind/id/{table_id}/key/{api_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            return pd.DataFrame(data.get("data", {}).get("data", []))
        logger.warning(f"BPS API returned status: {data.get('status')}")
        return None
    except Exception as e:
        logger.warning(f"BPS API call failed: {e}")
        return None


def try_bps_static_download() -> dict[str, pd.DataFrame]:
    """Attempt to download BPS static HTML tables via requests + BeautifulSoup.

    Uses bs4 html.parser (already in requirements.txt) to avoid lxml dependency.

    Returns:
        Dict of {metric: dataframe} if successful, empty dict otherwise.
    """
    from bs4 import BeautifulSoup

    results = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.bps.go.id/",
    }

    table_urls = {
        "production_ton": "https://www.bps.go.id/en/statistics-table/2/MTk4MyMy/corn-production-by-province.html",
    }

    for metric, url in table_urls.items():
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            logger.info(f"  BPS response: HTTP {resp.status_code} for {metric}")
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                tables = soup.find_all("table")
                logger.info(f"  Found {len(tables)} HTML tables on page")
                if tables:
                    # Parse largest table (likely the data table)
                    largest = max(tables, key=lambda t: len(t.find_all("tr")))
                    rows = largest.find_all("tr")
                    data = [[cell.get_text(strip=True) for cell in row.find_all(["th", "td"])] for row in rows]
                    df = pd.DataFrame(data[1:], columns=data[0] if data else None)
                    results[metric] = df
                    logger.info(f"  Parsed {metric}: {len(df)} rows, cols: {df.columns.tolist()[:5]}")
            else:
                logger.warning(f"  BPS returned HTTP {resp.status_code} for {metric}")
        except Exception as e:
            logger.warning(f"  BPS web parse failed for {metric}: {e}")

    return results


def save_manual_guide() -> None:
    """Save a detailed manual download guide for when automation fails."""
    guide = """# BPS Indonesia Manual Download Guide
Generated by: src/data/download_indonesia.py
Date: see file timestamp

## Why Manual Download is Needed

BPS (Badan Pusat Statistik) Indonesia requires either:
1. A BPS WebAPI key (register free at webapi.bps.go.id)
2. Manual download from BPS website (no programmatic access)

## Step-by-Step Manual Download

### Step 1: Register BPS API Key (5-10 minutes)
1. Go to: https://webapi.bps.go.id
2. Click "Daftar" (Register)
3. Fill in the form (email, name, etc.)
4. Verify email
5. Get API key from dashboard
6. Add to .env: BPS_API_KEY=your_key_here
7. Re-run: python src/data/download_indonesia.py

### Step 2: Manual Website Download (if API not available)

#### Corn Production by Province:
URL: https://www.bps.go.id/id/statistics-table/2/MTk4MyMy/produksi-jagung-menurut-provinsi.html
- Click "Download Excel" or "Unduh"
- Save as: data/raw/indonesia/bps_corn_production_by_province.xlsx

#### Corn Planted Area by Province:
URL: https://www.bps.go.id/id/statistics-table/2/MTgzMiMy/luas-panen-jagung-menurut-provinsi.html
- Save as: data/raw/indonesia/bps_corn_harvested_by_province.xlsx

#### Alternative URL (English):
https://www.bps.go.id/en/statistics-table?subject=55

### Step 3: After Manual Download
Run the cleaning script:
    python src/data/clean_indonesia.py

## Expected Data Format from BPS

BPS tables are typically WIDE FORMAT:
| Provinsi | 2003 | 2004 | ... | 2023 |
|----------|------|------|-----|------|
| Aceh     | 1234 | 1456 | ... | 2000 |

Will need pd.melt() to convert to long format.

## Known Data Issues
- Unit: Usually in ton (not thousand tonnes — verify on download)
- Pre-2013: Only 33 provinces (no Kalimantan Utara)
- 2018: BPS changed methodology (KSA) — slight discontinuity expected
- Province names in Indonesian: normalize to English using BPS_PROVINCE_MAP

## Alternative Sources If BPS Fails

1. FAOSTAT national (already downloaded):
   data/processed/indonesia/yield_indonesia_national_faostat_2003_2023.csv
   (National level only — not province level)

2. Indonesia Geospatial Portal: https://tanahair.indonesia.go.id
3. One Map Policy: https://portalksp.ina-sdi.or.id/
4. Academic datasets: Check Mendeley Data / Harvard Dataverse
   Search: "Indonesia maize province yield" or "jagung provinsi"

5. Contact: BPS Agricultural Statistics Division
   Email: stat.tanaman-pangan@bps.go.id
"""
    guide_path = RAW_DIR / "MANUAL_DOWNLOAD_GUIDE.md"
    with open(guide_path, "w") as f:
        f.write(guide)
    logger.info(f"Manual download guide saved → {guide_path}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Indonesia Province Corn Data Download (BPS) ===")

    # Always save the manual guide
    save_manual_guide()

    # Attempt 1: BPS WebAPI (needs key)
    bps_api_key = None
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(PROJECT_ROOT / ".env")
        bps_api_key = os.getenv("BPS_API_KEY")
    except ImportError:
        pass

    if bps_api_key:
        logger.info("BPS API key found — attempting WebAPI download...")
        prod_df = try_bps_api(bps_api_key, BPS_CORN_TABLE_IDS["production_ton"])
        if prod_df is not None and not prod_df.empty:
            prod_path = RAW_DIR / "bps_corn_production_by_province.csv"
            prod_df.to_csv(prod_path, index=False)
            logger.info(f"  Production data → {prod_path}")
    else:
        logger.info("No BPS_API_KEY found — skipping WebAPI (see MANUAL_DOWNLOAD_GUIDE.md)")

    # Attempt 2: BPS static web tables
    logger.info("Attempting BPS static web table download...")
    web_results = try_bps_static_download()

    if web_results:
        for metric, df in web_results.items():
            path = RAW_DIR / f"bps_corn_{metric}_web.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Web parse: {metric} → {path}")
    else:
        logger.info("  Web table download failed (BPS blocks automated access)")

    # Summary
    downloaded = list(RAW_DIR.glob("bps_corn*.csv"))
    if downloaded:
        logger.info(f"\nBPS files downloaded: {len(downloaded)}")
        for f in downloaded:
            logger.info(f"  {f.name}")
        logger.info("\nNEXT STEP: Run src/data/clean_indonesia.py to process raw files")
    else:
        logger.warning(
            "\nNo BPS province data downloaded automatically.\n"
            "ACTION NEEDED: Follow MANUAL_DOWNLOAD_GUIDE.md to download BPS Excel files.\n"
            "National-level FAOSTAT data IS available at:\n"
            f"  {PROCESSED_DIR / 'yield_indonesia_national_faostat_2003_2023.csv'}"
        )

    logger.info("=== Indonesia download attempt complete ===")


if __name__ == "__main__":
    main()
