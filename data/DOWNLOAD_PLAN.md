# Yield Data Download Plan — Thesis Maize Transfer Learning
**Date**: 2026-05-07  
**Status**: Draft — awaiting approval before scripting

---

## Summary

| Country | Method | API Key Needed | Est. Time | Blocker? |
|---------|--------|---------------|-----------|----------|
| USA | NASS QuickStats API (auto) | YES — free, ~5 min | 30 min | Need to register key first |
| Indonesia | BPS manual download | NO (manual) | 60 min | BPS API too complex for discovery |
| Vietnam | FAOSTAT API (proxy) | NO | 30 min | GSO website has no clean API |
| Thailand | FAOSTAT API (proxy) | NO | 30 min | OAE website has no clean API |

---

## USA — USDA NASS QuickStats

### API Overview
- **Base URL**: `https://quickstats.nass.usda.gov/api/api_GET/`
- **API Key**: Required. **Free signup** at quickstats.nass.usda.gov/api — key emailed instantly.
- **Response formats**: JSON (default), XML, or CSV — will use **CSV**.
- **Record limit**: Max 50,000 per request. County corn yield 2003–2023 ≈ 21,000–31,500 records → safely under limit.

### Key Query Parameters
```
commodity_desc   = CORN
statisticcat_desc = YIELD
agg_level_desc   = COUNTY
year__GE         = 2003
year__LE         = 2023
format           = CSV
key              = <YOUR_API_KEY>
```

### Fields returned by NASS
| NASS field | Our target field | Notes |
|-----------|-----------------|-------|
| `state_alpha` | — | State abbreviation |
| `state_name` | `region_name` (state part) | |
| `county_name` | `region_name` | |
| `county_ansi` | `region_id` | 5-digit FIPS code (state+county) |
| `year` | `year` | |
| `Value` | `yield_bu_acre` | **Unit: BU / ACRE** (not tonnes/ha!) |
| — | `yield_ton_ha` | Computed: × 0.0628 (1 bu/acre = 62.77 kg/ha) |

> **Unit conversion note**: NASS reports in bushels/acre. For comparison with ASEAN data (tonnes/ha), convert: `yield_ton_ha = yield_bu_acre × 0.06277`. Keep both columns in raw file.

### Output file
```
data/raw/usa/
  nass_corn_county_yield_2003_2023.csv   ← raw NASS download
  nass_corn_county_yield_2003_2023_meta.json  ← query params + API version logged
```

### Estimated records
~1,000–1,500 corn-producing counties × 21 years = ~21,000–31,500 rows. Some counties have gaps (drought/no crop), so expect missing years — that's OK, data-validation skill handles this.

---

## Indonesia — BPS (Badan Pusat Statistik)

### Why not BPS API?
BPS does have an API (`webapi.bps.go.id`) but it requires:
1. Registering for an API key
2. Discovering the variable ID for jagung (corn) production — which requires calling multiple metadata endpoints first
3. The dynamic data endpoint structure is underdocumented for agricultural commodities

**Decision**: Manual download is more reliable for first-pass data collection. Script BPS API only if we need to refresh data frequently.

### Manual Download Method
BPS publishes static tables for jagung production by province. Two options:

**Option A — BPS Website Static Tables (preferred)**
- URL: `https://www.bps.go.id/en/statistics-table` → search "corn production by province"
- Download as XLS/XLSX, one table per ~5 years, manual cleanup needed
- Typical columns: Province Name, [year1], [year2], ..., [yearN] in wide format
- Coverage: Usually 2000–present, but early years (2003–2005) may need separate table

**Option B — FAOSTAT (fallback)**
- FAOSTAT tracks Indonesia national-level data, NOT province-level
- Use only if BPS province-level data is unavailable for some years

**Recommended approach for today**: Download from BPS website (XLS), convert manually, validate against FAOSTAT national totals.

### Fields in BPS data
| BPS column | Our target field | Notes |
|-----------|-----------------|-------|
| Province name (Indonesian) | `region_name` | Need mapping to English + BPS province code |
| `kode_provinsi` | `region_id` | BPS 2-digit province code |
| Year (wide format) | `year` | Needs reshape wide→long |
| Production (1000 ton) | `production_ton` | Multiply by 1000 to get tonnes |
| Planted area (1000 ha) | `planted_ha` | Same unit conversion |
| — | `yield_ton_ha` | Computed: `production_ton / planted_ha` |

> **BPS quirk**: Data is often in wide format (provinces as rows, years as columns). Will need `pd.melt()` to reshape.

> **Province count**: 34 provinces (post-2013 split). Pre-2013 data uses 33 provinces (Kalimantan Utara didn't exist). Handle split carefully.

### Output file
```
data/raw/indonesia/
  bps_corn_province_production_2003_2023.xlsx   ← raw BPS download (DO NOT MODIFY)
  bps_corn_province_planted_2003_2023.xlsx      ← planted area from BPS
  bps_province_codes.csv                         ← BPS province code → name mapping
```

### Estimated records
34 provinces × 21 years = 714 rows. Small dataset — fast to validate manually.

---

## Vietnam — GSO (General Statistics Office)

### API Situation
GSO (`gso.gov.vn`) has no public API. Website is browsable but requires manual XLS download. FAOSTAT is a better programmatic source for Vietnam national data, but for **province-level** data, GSO is the only official source.

### Recommended approach
1. Try FAOSTAT API for province-level Vietnam data (FAOSTAT does have sub-national data for some countries)
2. If unavailable: manual download from GSO website
3. Fallback: Contact GSO directly or use published research datasets

> **Risk**: Vietnam province-level maize data coverage may be sparse before 2010. Flag this as a potential scope limitation.

### Output file
```
data/raw/vietnam/
  gso_corn_province_yield_2003_2023.xlsx   ← raw GSO download
```

---

## Thailand — OAE (Office of Agricultural Economics)

### API Situation
OAE (`oae.go.th`) has no public API. Data is published as PDF reports or XLS per year.

### Recommended approach
1. Try FAOSTAT API for province-level Thailand data
2. If unavailable: download OAE annual reports (XLS format, one file per year → concatenate)
3. Thailand has 77 provinces, but maize production concentrated in ~20 northern provinces

### Output file
```
data/raw/thailand/
  oae_corn_province_yield_2003_2023.xlsx   ← raw OAE download (or per-year files)
```

---

## Standard Schema (All Countries — Processed Output)

After cleaning, every country's processed file must have these columns:

```
region_id       str    Country-specific code (FIPS for USA, BPS code for ID, etc.)
region_name     str    English name
country         str    "USA" | "IDN" | "VNM" | "THA"
year            int    2003–2023
yield_ton_ha    float  Tonnes per hectare (primary metric for comparison)
production_ton  float  Total production in metric tonnes (if available)
planted_ha      float  Planted area in hectares (if available)
harvested_ha    float  Harvested area in hectares (if available)
data_source     str    "NASS" | "BPS" | "GSO" | "OAE" | "FAOSTAT"
```

**Processed output location**: `data/processed/<country>/yield_<country>_2003_2023.parquet`

---

## Blockers & Action Items

### Before scripting USA:
- [ ] **Register NASS API key** at `https://quickstats.nass.usda.gov/api` — takes 5 min, key arrives by email
- [ ] Confirm email for registration: sulashidayat@gmail.com (or institutional email?)

### Before downloading Indonesia:
- [ ] Open BPS website, find corn production table: `https://www.bps.go.id/en/statistics-table`
- [ ] Check if single XLS covers 2003–2023 or need multiple downloads
- [ ] Confirm: does BPS have separate table for "planted area" vs "production"? (Need both for yield calculation)

### Vietnam & Thailand (next session):
- [ ] Test FAOSTAT API for sub-national data coverage
- [ ] If FAOSTAT has no province data: manual download from GSO/OAE

---

## Time Estimate — Today's Session

| Task | Est. Time |
|------|-----------|
| Register NASS API key | 5 min |
| Write + test USA download script | 30 min |
| Run download, validate output | 15 min |
| Manual BPS Indonesia download + format check | 30 min |
| Write Indonesia cleaning script | 30 min |
| Validate Indonesia data | 15 min |
| **Total** | **~2h 15 min** |

---

## Data Quality Flags to Check After Download (data-validation skill)

1. **USA**: Missing counties (expected — many counties don't grow corn)
2. **USA**: Outlier years (drought years like 2012 will show dips — keep them, they're real)
3. **Indonesia**: Pre-2013 province split (Kalimantan Utara missing from older data)
4. **All**: yield_ton_ha range sanity check — maize typically 1–15 t/ha globally
5. **All**: Cross-validate national total against FAOSTAT (`production_ton.sum()` by year should match FAOSTAT ±5%)

---

*Next step: Approve this plan → Claude Code writes `src/data/download_usa.py` first.*
