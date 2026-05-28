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

## Data Quality Flags to Check After Download (data-validation skill)

1. **USA**: Missing counties (expected — many counties don't grow corn)
2. **USA**: Outlier years (drought years like 2012 will show dips — keep them, they're real)
3. **Indonesia**: Pre-2013 province split (Kalimantan Utara missing from older data)
4. **All**: yield_ton_ha range sanity check — maize typically 1–15 t/ha globally
5. **All**: Cross-validate national total against FAOSTAT (`production_ton.sum()` by year should match FAOSTAT ±5%)
