---
name: Data Download Status
description: Status of yield data download for each country in the thesis pipeline
type: project
---

As of 2026-05-07, the following data has been downloaded:

**DONE (automated)**
- USA county yield: 41,349 records, 2,280 counties, 2003–2023 via NASS API
  - Raw: `data/raw/usa/nass_corn_county_yield_2003_2023.csv` (14.6 MB)
  - Processed: `data/processed/usa/yield_usa_2003_2023.parquet` (225 KB)
- National reference (all 4 countries): `data/processed/owid_all_countries_national.csv` via OWID/FAOSTAT

**BLOCKED — manual download needed**
- Indonesia province (BPS): JS-rendered site, needs BPS API key or manual XLS download from bps.go.id
  - Guide: `data/raw/indonesia/MANUAL_DOWNLOAD_GUIDE.md`
  - Script ready to run once BPS_API_KEY is in .env: `src/data/download_indonesia.py`
- Vietnam province (GSO): GSO website inaccessible (connection refused from non-VN IP, may need VPN)
- Thailand province (OAE): JS-rendered SPA, no direct file URLs working

**Why:** FAOSTAT API (fenixservices.fao.org) was returning 521 (Cloudflare down), switched to OWID as fallback for national-level.

**How to apply:** Next session priorities are: (1) get BPS API key and re-run Indonesia script, (2) manually download Vietnam GSO and Thailand OAE data, (3) run full data-validation on USA data, (4) build GEE satellite extraction pipeline.
