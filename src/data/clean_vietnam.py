"""
Clean and standardize Vietnam GSO province-level maize data.
Merges planted area (E06.25), yield (E06.26), and production (E06.27).

Units:
  - Area: thousand ha → ha (×1000)
  - Yield: quintal/ha → ton/ha (÷10)
  - Production: thousand tons → tons (×1000)

Source: https://www.nso.gov.vn/en/px-web/
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/Manually Download/Vietnam")
OUTPUT_DIR = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/processed/vietnam")
RAW_DIR = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/raw/vietnam")

# Regional aggregate rows to skip
AGGREGATE_ROWS = {
    "WHOLE COUNTRY",
    "Red River Delta",
    "Northern midlands and mountain areas",
    "Northern Central area and Central coastal area",
    "Central Highlands",
    "South East",
    "Mekong River Delta",
}

# Province name corrections (double spaces, typos)
NAME_CORRECTIONS = {
    "Thai  Nguyen": "Thai Nguyen",
    "Quang  Nam": "Quang Nam",
    "Quang  Ngai": "Quang Ngai",
    "Khanh  Hoa": "Khanh Hoa",
    "Ninh  Thuan": "Ninh Thuan",
    "Binh  Duong": "Binh Duong",
    "Kien  Giang": "Kien Giang",
    "Ho Chi Minh city": "Ho Chi Minh City",
}


def _parse_gso_file(path: Path, var_name: str, multiplier: float) -> pd.DataFrame:
    """Parse a single GSO Excel file.

    Args:
        path: Path to Excel file
        var_name: Output column name for the variable
        multiplier: Conversion factor (e.g., 1000 for thousand→raw, 0.1 for quintal→ton)

    Returns:
        Long-format DataFrame with columns: region_name, year, {var_name}
    """
    logger.info(f"  Reading: {path.name}")

    df = pd.read_excel(path, header=None)

    # Row 0 = title, Row 1 = empty/header label, Row 2 = year headers
    # Data starts from row 3
    years_raw = df.iloc[2, 1:].tolist()
    years = []
    for y in years_raw:
        try:
            years.append(int(float(y)))
        except (ValueError, TypeError):
            years.append(None)  # "Prel. 2024" or similar
    year_cols = list(zip(range(1, len(years) + 1), years))
    year_cols = [(c, y) for c, y in year_cols if y is not None]

    records = []
    for row_idx in range(3, len(df)):
        name = str(df.iloc[row_idx, 0]).strip()
        if not name or name == "nan":
            continue
        if name in AGGREGATE_ROWS:
            continue

        name = NAME_CORRECTIONS.get(name, name)

        for col_idx, year in year_cols:
            val = df.iloc[row_idx, col_idx]
            if val is None or str(val).strip() in ("..", "nan", ""):
                continue
            try:
                num_val = float(val) * multiplier
            except (ValueError, TypeError):
                continue
            records.append(
                {
                    "region_name": name,
                    "year": year,
                    var_name: round(num_val, 4),
                }
            )

    result = pd.DataFrame(records)
    logger.info(f"    → {len(result)} rows, {result['region_name'].nunique()} provinces, "
                 f"{result['year'].nunique()} years")
    return result


def _combine_ha_noi(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Ha Tay into Ha Noi for consistent pre/post-2008 series.

    From 2008 onward, Ha Noi includes Ha Tay. For training consistency,
    we combine pre-2008 Ha Noi + Ha Tay into a single "Ha Noi" series.
    """
    ha_tay = df[df["region_name"] == "Ha Tay"].copy()
    ha_noi = df[df["region_name"] == "Ha Noi"].copy()

    if len(ha_tay) == 0:
        return df

    # Combine: Ha Tay data pre-2008 merged into Ha Noi
    pre_2008 = ha_tay[ha_tay["year"] < 2008]
    if len(pre_2008) == 0:
        return df[df["region_name"] != "Ha Tay"]

    logger.info(f"  Merging Ha Tay → Ha Noi for {len(pre_2008)} pre-2008 rows")

    # Create combined Ha Noi
    combined = pd.concat([ha_noi, pre_2008], ignore_index=True)
    combined = combined.groupby(["year"], as_index=False).agg({
        "region_name": "first",
        "planted_ha": "sum",
        "harvested_ha": "sum",
        "production_ton": "sum",
    })

    # Update region_name
    combined["region_name"] = "Ha Noi"

    # Recalculate yield
    combined["yield_ton_ha"] = round(
        combined["production_ton"] / combined["planted_ha"], 4
    )
    combined["yield_ton_ha"] = combined["yield_ton_ha"].replace(float("inf"), None)

    # Remove old Ha Noi pre-2008 rows and all Ha Tay, add combined
    df = df[~((df["region_name"] == "Ha Noi") & (df["year"] < 2008))]
    df = df[df["region_name"] != "Ha Tay"]
    df = pd.concat([df, combined], ignore_index=True)

    return df.sort_values(["year", "region_name"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse each file
    logger.info("Parsing GSO files...")

    area_df = _parse_gso_file(
        BASE_DIR / "E06.25.xlsx",
        var_name="planted_ha",
        multiplier=1000.0,  # thousand ha → ha
    )

    yield_df = _parse_gso_file(
        BASE_DIR / "E06.26.xlsx",
        var_name="yield_ton_ha_raw",
        multiplier=0.1,  # quintal/ha → ton/ha
    )

    prod_df = _parse_gso_file(
        BASE_DIR / "E06.27.xlsx",
        var_name="production_ton",
        multiplier=1000.0,  # thousand tons → tons
    )

    # Merge on region_name + year
    logger.info("Merging variables...")
    merged = area_df.merge(yield_df, on=["region_name", "year"], how="outer")
    merged = merged.merge(prod_df, on=["region_name", "year"], how="outer")

    # Use planted area as harvested (GSO only reports planted)
    merged["harvested_ha"] = merged["planted_ha"]

    # Recalculate yield from production/area (more reliable than reported yield)
    merged["yield_ton_ha"] = round(
        merged["production_ton"] / merged["planted_ha"], 4
    )
    merged["yield_ton_ha"] = merged["yield_ton_ha"].replace(float("inf"), None)

    # Drop raw yield column
    merged = merged.drop(columns=["yield_ton_ha_raw"])

    # Handle Ha Noi/Ha Tay merge
    merged = _combine_ha_noi(merged)

    # Add metadata
    merged["country"] = "VNM"
    merged["data_source"] = "GSO"
    merged = merged.sort_values(["year", "region_name"]).reset_index(drop=True)

    # Drop rows with zero planted area (data errors)
    bad_mask = (merged["planted_ha"] == 0) | (merged["planted_ha"].isna())
    if bad_mask.any():
        logger.warning(f"  Dropping {bad_mask.sum()} rows with zero/missing planted area")
        merged = merged[~bad_mask]

    # Drop zero production rows (likely data errors)
    zero_prod = merged["production_ton"] == 0
    if zero_prod.any():
        logger.warning(f"  Dropping {zero_prod.sum()} rows with zero production")
        merged = merged[~zero_prod]

    # Reorder columns to standard schema
    merged = merged[
        ["region_name", "year", "country", "planted_ha", "harvested_ha",
         "production_ton", "yield_ton_ha", "data_source"]
    ]

    # Save
    out_path = OUTPUT_DIR / "yield_vietnam_province_1995_2023.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")
    logger.info(f"Total: {len(merged)} rows, {merged['region_name'].nunique()} provinces, "
                 f"{merged['year'].nunique()} years ({merged['year'].min()}–{merged['year'].max()})")

    # Summary
    print("\n=== VIETNAM DATA SUMMARY ===")
    print(f"Rows: {len(merged)}")
    print(f"Provinces: {merged['region_name'].nunique()}")
    print(f"Years: {merged['year'].min()}–{merged['year'].max()} ({merged['year'].nunique()} years)")
    print(f"Yield range: {merged['yield_ton_ha'].dropna().min():.2f} – {merged['yield_ton_ha'].dropna().max():.2f} ton/ha")
    print(f"Yield mean: {merged['yield_ton_ha'].dropna().mean():.2f} ton/ha")
    print(f"Missing rows: {merged.isnull().any(axis=1).sum()}/{len(merged)}")

    # Per-year coverage
    print("\n=== Coverage per year (sample) ===")
    for year in range(1995, 2024, 5):
        n = len(merged[merged["year"] == year])
        total_prod = merged[merged["year"] == year]["production_ton"].sum()
        print(f"  {year}: {n} provinces, total prod={total_prod/1e6:.2f}M tons")

    # Top provinces
    print("\n=== Top 10 provinces (avg production) ===")
    top = merged.groupby("region_name")["production_ton"].mean().nlargest(10)
    for prov, val in top.items():
        print(f"  {prov}: {val:,.0f} tons")

    # Cross-validate vs OWID
    _cross_validate(merged)
    return merged


def _cross_validate(df: pd.DataFrame) -> None:
    """Cross-validate aggregated province data against OWID/FAOSTAT national totals."""
    owid_path = RAW_DIR / "owid_vietnam_corn_national_2003_2023.csv"
    if not owid_path.exists():
        logger.warning("  OWID reference file not found, skipping cross-validation")
        return

    owid = pd.read_csv(owid_path)
    print("\n=== Cross-validation vs OWID/FAOSTAT ===")
    for year in sorted(df["year"].unique()):
        total_prod = df[df["year"] == year]["production_ton"].sum()
        total_area = df[df["year"] == year]["planted_ha"].sum()
        avg_yield = total_prod / total_area if total_area else 0

        owid_row = owid[owid["year"] == year]
        if len(owid_row) == 0:
            continue
        owid_yield = owid_row.iloc[0].get("yield_ton_ha", 0)
        owid_prod = owid_row.iloc[0].get("production_ton", 0)

        if owid_prod and owid_prod > 0:
            prod_diff = abs(total_prod - owid_prod) / owid_prod * 100
            yield_diff = abs(avg_yield - owid_yield) / owid_yield * 100 if owid_yield else 0
            if prod_diff > 5 or yield_diff > 5:
                print(f"  {year}: Prod diff={prod_diff:.1f}%, Yield diff={yield_diff:.1f}%")
    print("  (Only showing years with >5% discrepancy)")


if __name__ == "__main__":
    main()
