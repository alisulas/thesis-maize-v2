# Data Directory

## Status Overview

| Country | Level | Status | Records | Source | File |
|---------|-------|--------|---------|--------|------|
| USA | County | ✅ Done | 41,349 rows, 2,280 counties | USDA NASS QuickStats API | `raw/usa/nass_corn_county_yield_2003_2023.csv` |
| USA | National | ✅ Done | 21 years | OWID (FAOSTAT) | `raw/usa/owid_usa_corn_national_2003_2023.csv` |
| Indonesia | National | ✅ Done | 21 years | OWID (FAOSTAT) | `raw/indonesia/owid_indonesia_corn_national_2003_2023.csv` |
| Vietnam | National | ✅ Done | 21 years | OWID (FAOSTAT) | `raw/vietnam/owid_vietnam_corn_national_2003_2023.csv` |
| Thailand | National | ✅ Done | 21 years | OWID (FAOSTAT) | `raw/thailand/owid_thailand_corn_national_2003_2023.csv` |
| Indonesia | Province | ⏳ Manual needed | 34 prov × 21 yr | BPS (JS-rendered) | See guide below |
| Vietnam | Province | ⏳ Manual needed | 63 prov × 21 yr | GSO (inaccessible) | See guide below |
| Thailand | Province | ⏳ Manual needed | 77 prov × 21 yr | OAE (JS-rendered) | See guide below |

**Downloaded**: 2026-05-07

---

## What Was Downloaded Automatically

### USA County Yield (PRIMARY — model training source)
- **File**: `raw/usa/nass_corn_county_yield_2003_2023.csv` (14.6 MB)
- **Processed**: `processed/usa/yield_usa_2003_2023.parquet` (225 KB)
- **Script**: `src/data/download_usa.py`
- **Stats**: 41,349 rows | 2,280 counties | 21 years (2003–2023) | 0% missing
- **Unit**: `yield_bu_acre` (original) + `yield_ton_ha` (converted, 1 bu/acre = 0.06277 t/ha)
- **Notes**: (D) suppressed values → NaN. "OTHER COMBINED" aggregate rows excluded.

### National Reference Data (all countries)
- **File**: `processed/owid_all_countries_national.csv`
- **Script**: `src/data/download_faostat.py`
- **Purpose**: Cross-validation against province totals; NOT used directly for training
- **Source**: Our World in Data (processed from FAOSTAT/FAO)
- **Unit**: yield already in t/ha

| Country | Yield range (t/ha) | Mean |
|---------|-------------------|------|
| USA | 7.73 – 11.13 | 10.11 |
| Indonesia | 3.24 – 6.14 | 4.75 |
| Vietnam | 3.18 – 5.04 | 4.28 |
| Thailand | 3.82 – 4.67 | 4.26 |

---

## Province-Level Data — Manual Download Required

BPS, GSO, and OAE all use JavaScript rendering or require API authentication.
Automated scraping attempted but failed. Manual download needed before model training.

### Indonesia (BPS) — PRIORITY

**Why BPS specifically**: Only official source for province-level jagung data.
**Estimated data**: 34 provinces × 21 years = ~714 rows (small!)

**Steps**:
1. Go to: https://www.bps.go.id/id/statistics-table/2/MTk4MyMy/produksi-jagung-menurut-provinsi.html
2. Click "Unduh" (Download) → Excel
3. Save to: `data/raw/indonesia/bps_corn_production_ton_by_province.xlsx`
4. Repeat for planted area: https://www.bps.go.id/id/statistics-table/2/MTgzMiMy/luas-panen-jagung.html
5. Save to: `data/raw/indonesia/bps_corn_harvested_ha_by_province.xlsx`
6. Run: `python src/data/clean_indonesia.py` (to be written after download)

**BPS API alternative** (faster): Register at https://webapi.bps.go.id → add key to `.env` as `BPS_API_KEY` → re-run `src/data/download_indonesia.py`

**Known quirks**:
- Data format: wide (provinces as rows, years as columns) → needs pd.melt()
- Units: tonnes (NOT thousand tonnes — verify on download page)
- Pre-2013: 33 provinces (no Kalimantan Utara)
- 2018: BPS changed to KSA methodology → slight break in series

### Vietnam (GSO) — SECONDARY

**Steps**:
1. Go to: https://www.gso.gov.vn/en/agriculture-forestry-and-fishery/
2. Search for "maize" or "corn" statistics by province
3. Alternatively: https://www.gso.gov.vn/px-web-2/?pxid=V0717&theme=Nong--lam-nghiep-va-thuy-san
4. Save to: `data/raw/vietnam/gso_corn_province_*.xlsx`

**Fallback**: GSO Statistical Yearbook PDF → extract Table on corn/maize production
**GSO website note**: Requires VPN in some regions (503/connection refused from non-VN IPs)

**Known quirks**:
- Yield in quintal/ha → convert: `yield_ton_ha = yield_quintal_ha / 10`
- Province names have Vietnamese diacritics → normalize with unidecode
- 2025 administrative reform merged some provinces — use pre-2025 boundaries

### Thailand (OAE) — SECONDARY

**Steps**:
1. Go to: https://www.oae.go.th/view/1/DownLoad/TH-TH
2. Look for "Corn" or "ข้าวโพด" (khao phot) statistics
3. Find annual Excel files for crop area/production by province
4. Save to: `data/raw/thailand/oae_corn_province_*.xlsx`

**Alternative**: OAE publishes annual PDF report "Agricultural Statistics of Thailand"
URL pattern: https://www.oae.go.th/assets/portlet/book_stat_th/book_stat_th_{year}.pdf
Then extract Table on corn/maize

**Known quirks**:
- Crop year vs calendar year (OAE uses crop year e.g. 2020/21)
- Province names in Thai script → match against GADM English names
- Main corn provinces: Nakhon Ratchasima, Lopburi, Chiang Rai (~20 northern provinces)

---

## Data Provenance & Version

| File | Source | Downloaded | Script |
|------|--------|-----------|--------|
| `raw/usa/nass_corn_county_yield_2003_2023.csv` | USDA NASS QuickStats | 2026-05-07 | `download_usa.py` |
| `raw/usa/nass_corn_county_yield_meta.json` | USDA NASS (metadata) | 2026-05-07 | `download_usa.py` |
| `raw/*/owid_*_corn_national_2003_2023.csv` | Our World in Data | 2026-05-07 | `download_faostat.py` |

---

## Processed Files Schema

All processed files in `processed/*/` use this standard schema:

```
region_id       str    FIPS (USA) or province code (ASEAN)
region_name     str    English name
country         str    ISO3: USA, IDN, VNM, THA
year            int    2003–2023
yield_ton_ha    float  Tonnes per hectare (primary metric)
production_ton  float  Total production (where available)
planted_ha      float  Planted/harvested area (where available)
data_source     str    USDA_NASS | BPS | GSO | OAE | OWID_FAOSTAT
```

## Rules

- `raw/` directory: NEVER modify. These are immutable originals.
- `processed/` directory: Output of cleaning scripts only. Reproducible from raw.
- Files >50MB: Do NOT commit to git. Add to `.gitignore` if needed.
