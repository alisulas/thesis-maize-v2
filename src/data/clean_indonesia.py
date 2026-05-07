"""Clean BPS Indonesia corn province data → standard schema.

Handles the BPS wide-format table:
  "Luas Panen, Produksi, dan Produktivitas Jagung Menurut Provinsi"

Inputs (add new files to RAW_FILES as downloaded):
    data/raw/indonesia/bps_corn_province_2020_2024_raw.csv  ← KSA method
    data/raw/indonesia/bps_corn_province_2003_2019_raw.csv  ← to be added (pre-KSA)

Output:
    data/processed/indonesia/yield_indonesia_2003_2023.parquet

Unit conversion:
    Produktivitas (ku/ha) ÷ 10 → yield_ton_ha
    1 kuintal = 100 kg = 0.1 tonne
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "indonesia"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "indonesia"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# BPS province name → standard English name + BPS 2-digit code
# Source: BPS province classification
PROVINCE_MAP: dict[str, dict] = {
    "ACEH":                   {"name_en": "Aceh",                   "bps_code": "11"},
    "SUMATERA UTARA":         {"name_en": "North Sumatra",           "bps_code": "12"},
    "SUMATERA BARAT":         {"name_en": "West Sumatra",            "bps_code": "13"},
    "RIAU":                   {"name_en": "Riau",                    "bps_code": "14"},
    "JAMBI":                  {"name_en": "Jambi",                   "bps_code": "15"},
    "SUMATERA SELATAN":       {"name_en": "South Sumatra",           "bps_code": "16"},
    "BENGKULU":               {"name_en": "Bengkulu",                "bps_code": "17"},
    "LAMPUNG":                {"name_en": "Lampung",                 "bps_code": "18"},
    "KEP. BANGKA BELITUNG":   {"name_en": "Bangka Belitung Islands", "bps_code": "19"},
    "KEP. RIAU":              {"name_en": "Riau Islands",            "bps_code": "21"},
    "DKI JAKARTA":            {"name_en": "DKI Jakarta",             "bps_code": "31"},
    "JAWA BARAT":             {"name_en": "West Java",               "bps_code": "32"},
    "JAWA TENGAH":            {"name_en": "Central Java",            "bps_code": "33"},
    "DI YOGYAKARTA":          {"name_en": "DI Yogyakarta",           "bps_code": "34"},
    "JAWA TIMUR":             {"name_en": "East Java",               "bps_code": "35"},
    "BANTEN":                 {"name_en": "Banten",                  "bps_code": "36"},
    "BALI":                   {"name_en": "Bali",                    "bps_code": "51"},
    "NUSA TENGGARA BARAT":    {"name_en": "West Nusa Tenggara",      "bps_code": "52"},
    "NUSA TENGGARA TIMUR":    {"name_en": "East Nusa Tenggara",      "bps_code": "53"},
    "KALIMANTAN BARAT":       {"name_en": "West Kalimantan",         "bps_code": "61"},
    "KALIMANTAN TENGAH":      {"name_en": "Central Kalimantan",      "bps_code": "62"},
    "KALIMANTAN SELATAN":     {"name_en": "South Kalimantan",        "bps_code": "63"},
    "KALIMANTAN TIMUR":       {"name_en": "East Kalimantan",         "bps_code": "64"},
    "KALIMANTAN UTARA":       {"name_en": "North Kalimantan",        "bps_code": "65"},
    "SULAWESI UTARA":         {"name_en": "North Sulawesi",          "bps_code": "71"},
    "SULAWESI TENGAH":        {"name_en": "Central Sulawesi",        "bps_code": "72"},
    "SULAWESI SELATAN":       {"name_en": "South Sulawesi",          "bps_code": "73"},
    "SULAWESI TENGGARA":      {"name_en": "Southeast Sulawesi",      "bps_code": "74"},
    "GORONTALO":              {"name_en": "Gorontalo",               "bps_code": "75"},
    "SULAWESI BARAT":         {"name_en": "West Sulawesi",           "bps_code": "76"},
    "MALUKU":                 {"name_en": "Maluku",                  "bps_code": "81"},
    "MALUKU UTARA":           {"name_en": "North Maluku",            "bps_code": "82"},
    "PAPUA BARAT":            {"name_en": "West Papua",              "bps_code": "91"},
    "PAPUA BARAT DAYA":       {"name_en": "Southwest Papua",         "bps_code": "92"},
    "PAPUA":                  {"name_en": "Papua",                   "bps_code": "94"},
    "PAPUA SELATAN":          {"name_en": "South Papua",             "bps_code": "95"},
    "PAPUA TENGAH":           {"name_en": "Central Papua",           "bps_code": "96"},
    "PAPUA PEGUNUNGAN":       {"name_en": "Highland Papua",          "bps_code": "97"},
}

# Rows to exclude from province-level data
EXCLUDE_ROWS = {"INDONESIA"}


def parse_bps_wide(path: Path) -> pd.DataFrame:
    """Parse BPS wide-format CSV into long format.

    BPS table structure:
      Row 0: title
      Row 1: metric group headers (Luas Panen / Produktivitas / Produksi)
      Row 2: years
      Rows 3+: province data (with "INDONESIA" total at end)
      Last rows: notes (Catatan)

    Args:
        path: Path to raw BPS CSV file.

    Returns:
        Long-format dataframe with columns:
        [province_bps, year, harvested_ha, yield_ku_ha, production_ton]
    """
    raw = pd.read_csv(path, encoding="utf-8", header=None)

    # Row 1: identify metric groups (cols where group header appears)
    row1 = raw.iloc[1].tolist()
    row2 = raw.iloc[2].tolist()

    # Find column ranges for each metric
    group_starts: list[int] = []
    group_names: list[str] = []
    for i, val in enumerate(row1):
        if isinstance(val, str) and any(
            kw in val for kw in ["Luas Panen", "Produktivitas", "Produksi"]
        ):
            group_starts.append(i)
            group_names.append(val.strip())

    # n_years = columns per group = distance between group start columns
    n_years = (group_starts[1] - group_starts[0]) if len(group_starts) >= 2 else (len(row2) - 1) // 3
    # Years are in row 2, first group only (cols group_starts[0] .. group_starts[0]+n_years)
    years = [
        int(float(v))
        for v in row2[group_starts[0]: group_starts[0] + n_years]
        if v not in (None, "") and str(v).replace(".", "").replace("-", "").isdigit()
    ]

    # Province data rows: skip rows 0-2 (headers), stop at "INDONESIA" + notes
    data_rows = []
    for _, row in raw.iloc[3:].iterrows():
        province = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if not province or province == "nan" or province == "Catatan":
            continue
        if province in EXCLUDE_ROWS:
            continue
        if province not in PROVINCE_MAP:
            logger.warning(f"  Unknown province: '{province}' — skipped")
            continue

        vals = row.iloc[1:].tolist()
        data_rows.append({"province_bps": province, "values": vals})

    # Build wide dataframe
    n_years = len(years)
    records = []
    for row_data in data_rows:
        prov = row_data["province_bps"]
        v = row_data["values"]
        for i, year in enumerate(years):
            def safe_float(x: object) -> float | None:
                s = str(x).strip()
                if s in ("-", "nan", "", "None"):
                    return None
                try:
                    return float(s.replace(",", ""))
                except ValueError:
                    return None

            harvested_ha = safe_float(v[i])                    # group 0 offset
            yield_ku_ha  = safe_float(v[i + n_years])          # group 1 offset
            production_ton = safe_float(v[i + 2 * n_years])    # group 2 offset

            records.append({
                "province_bps": prov,
                "year": year,
                "harvested_ha": harvested_ha,
                "yield_ku_ha": yield_ku_ha,
                "production_ton": production_ton,
            })

    return pd.DataFrame(records)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply province mapping, unit conversion, add standard columns.

    Args:
        df: Output of parse_bps_wide().

    Returns:
        Dataframe in project standard schema.
    """
    rows = []
    for _, row in df.iterrows():
        info = PROVINCE_MAP.get(row["province_bps"], {})
        yield_ton_ha = row["yield_ku_ha"] / 10.0 if row["yield_ku_ha"] is not None else None

        rows.append({
            "region_id":      "IDN-" + info.get("bps_code", "??"),
            "region_name":    info.get("name_en", row["province_bps"]),
            "province_bps":   row["province_bps"],
            "country":        "IDN",
            "year":           int(row["year"]),
            "yield_ku_ha":    row["yield_ku_ha"],      # original unit (keep for audit)
            "yield_ton_ha":   round(yield_ton_ha, 4) if yield_ton_ha is not None else None,
            "harvested_ha":   row["harvested_ha"],
            "production_ton": row["production_ton"],
            "data_source":    "BPS",
            "method_note":    "KSA" if int(row["year"]) >= 2020 else "eye_estimate",
        })

    return pd.DataFrame(rows).sort_values(["region_id", "year"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Quick sanity checks on cleaned data."""
    logger.info("--- Validation ---")
    logger.info(f"  Provinces: {df['region_id'].nunique()}")
    logger.info(f"  Years: {sorted(df['year'].unique())}")
    logger.info(f"  Total rows: {len(df)}")

    valid = df["yield_ton_ha"].dropna()
    lo, hi = 1.0, 8.0  # Indonesia maize: 1–8 t/ha expected
    out_pct = ((valid < lo) | (valid > hi)).mean() * 100
    logger.info(f"  Yield range: {valid.min():.2f}–{valid.max():.2f} t/ha | mean {valid.mean():.2f}")
    if out_pct > 5:
        logger.warning(f"  {out_pct:.1f}% yields outside [{lo}, {hi}] t/ha — check units")
    else:
        logger.info(f"  {out_pct:.1f}% outside [{lo}, {hi}] t/ha — OK")

    missing_pct = df["yield_ton_ha"].isna().mean() * 100
    logger.info(f"  Missing yield: {missing_pct:.1f}% (new provinces before split expected)")

    # Cross-check: production / area ≈ yield_ton_ha / 10 (yield in ku/ha)
    # production_ton / harvested_ha = yield in ton/ha directly
    df2 = df.dropna(subset=["production_ton", "harvested_ha", "yield_ton_ha"])
    df2 = df2[df2["harvested_ha"] > 0]
    computed = df2["production_ton"] / df2["harvested_ha"]
    diff = (computed - df2["yield_ton_ha"]).abs()
    if diff.max() > 0.1:
        logger.warning(f"  Max production/area vs yield_ton_ha discrepancy: {diff.max():.4f} t/ha")
    else:
        logger.info(f"  Production/area cross-check: max diff {diff.max():.4f} t/ha — OK")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== BPS Indonesia Corn Data Cleaning ===")

    # Collect all raw files (add more here as downloaded)
    raw_files = sorted(RAW_DIR.glob("bps_corn_province_*_raw.csv"))
    if not raw_files:
        logger.error(f"No raw BPS files found in {RAW_DIR}")
        sys.exit(1)

    all_dfs: list[pd.DataFrame] = []
    for raw_path in raw_files:
        logger.info(f"Parsing: {raw_path.name}")
        df_raw = parse_bps_wide(raw_path)
        df_std = standardize(df_raw)
        logger.info(f"  → {len(df_std)} rows | years {sorted(df_std.year.unique())}")
        all_dfs.append(df_std)

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .drop_duplicates(subset=["region_id", "year"], keep="last")
        .sort_values(["region_id", "year"])
        .reset_index(drop=True)
    )

    validate(combined)

    # Save
    out_path = PROCESSED_DIR / "yield_indonesia_2003_2023.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"\nSaved → {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Also save CSV for quick inspection
    csv_path = PROCESSED_DIR / "yield_indonesia_2003_2023.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved → {csv_path}")

    logger.info("=== Done. NOTE: Only covers 2020-2024. Download pre-2020 data to extend. ===")
    logger.info("Pre-2020 BPS source: https://www.bps.go.id/id/statistics-table/1/OTgxIzE=/produksi-jagung-menurut-provinsi.html")


if __name__ == "__main__":
    main()
