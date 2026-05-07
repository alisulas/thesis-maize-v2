"""Download USDA NASS corn yield data by county, 2003-2023.

Outputs:
    data/raw/usa/nass_corn_county_yield_2003_2023.csv
    data/raw/usa/nass_corn_county_yield_meta.json
    data/processed/usa/yield_usa_2003_2023.parquet
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "usa"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "usa"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NASS_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
# 1 bu/acre corn (56 lb/bu, 1 acre = 4046.86 m²) = 62.77 kg/ha
BU_ACRE_TO_TON_HA: float = 0.06277


def fetch_nass(api_key: str) -> pd.DataFrame:
    """Fetch corn yield records from NASS QuickStats API.

    Args:
        api_key: NASS API key from .env.

    Returns:
        Raw dataframe with all NASS fields intact.

    Raises:
        RuntimeError: If API returns an error or empty response.
    """
    params = {
        "key": api_key,
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "agg_level_desc": "COUNTY",
        "unit_desc": "BU / ACRE",
        "domain_desc": "TOTAL",
        "year__GE": "2003",
        "year__LE": "2023",
        "format": "JSON",
    }

    logger.info("Querying NASS QuickStats API (may take 30–90 sec)...")
    resp = requests.get(NASS_URL, params=params, timeout=300)
    resp.raise_for_status()

    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"NASS API error: {payload['error']}")

    records = payload.get("data", [])
    if not records:
        raise RuntimeError("NASS returned 0 records. Check query parameters or API key.")

    logger.info(f"Received {len(records):,} raw records from NASS.")
    if len(records) >= 49_000:
        logger.warning("Near 50,000 record limit — some counties may be missing. Consider paginating.")

    return pd.DataFrame(records)


def clean_nass(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NASS data to project schema.

    Handles:
    - Removes aggregate "OTHER (COMBINED) COUNTIES" rows
    - Converts suppressed values "(D)" → NaN
    - Builds 5-digit FIPS code (state + county)
    - Converts bu/acre → ton/ha

    Args:
        df: Raw dataframe from fetch_nass().

    Returns:
        Cleaned dataframe in project standard schema.
    """
    logger.info(f"Cleaning {len(df):,} raw records...")
    n_raw = len(df)

    # Remove aggregate county rows (not real counties)
    agg_mask = df["county_name"].str.upper().str.contains("OTHER", na=False)
    df = df[~agg_mask].copy()
    logger.info(f"  Removed {agg_mask.sum()} 'OTHER (COMBINED)' aggregate rows")

    # Suppress confidential values: "(D)" = withheld, "(Z)" = rounds to zero
    df["Value"] = df["Value"].str.strip()
    df.loc[df["Value"].isin(["(D)", "(Z)", "(NA)", ""]), "Value"] = None
    df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    n_suppressed = df["Value"].isna().sum()
    logger.info(f"  Suppressed/missing yield values: {n_suppressed:,} ({n_suppressed/len(df)*100:.1f}%)")

    # 5-digit FIPS: state_fips_code (2 digits) + county_code (3 digits)
    df["region_id"] = (
        df["state_fips_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
    )
    df["region_name"] = df["county_name"].str.title() + ", " + df["state_alpha"]

    # Unit conversion
    df["yield_bu_acre"] = df["Value"]
    df["yield_ton_ha"] = (df["Value"] * BU_ACRE_TO_TON_HA).round(4)

    result = (
        df.assign(
            year=df["year"].astype(int),
            country="USA",
            data_source="USDA_NASS",
            state=df["state_alpha"],
        )[[
            "region_id", "region_name", "state", "county_name",
            "year", "yield_bu_acre", "yield_ton_ha", "country", "data_source",
        ]]
        .sort_values(["region_id", "year"])
        .reset_index(drop=True)
    )

    logger.info(
        f"  Cleaned: {n_raw:,} → {len(result):,} rows | "
        f"{result['region_id'].nunique():,} counties | "
        f"{result['year'].nunique()} years"
    )
    return result


def validate_quick(df: pd.DataFrame) -> bool:
    """Quick sanity check — full validation done separately.

    Args:
        df: Cleaned dataframe.

    Returns:
        True if all checks pass, False if any warnings.
    """
    logger.info("--- Quick validation ---")
    passed = True

    valid_yield = df["yield_ton_ha"].dropna()
    lo, hi = 1.0, 20.0
    out_pct = ((valid_yield < lo) | (valid_yield > hi)).mean() * 100
    logger.info(f"  Yield range: {valid_yield.min():.2f}–{valid_yield.max():.2f} t/ha | mean: {valid_yield.mean():.2f}")
    if out_pct > 5:
        logger.warning(f"  {out_pct:.1f}% yields outside [{lo}, {hi}] t/ha — check conversion")
        passed = False
    else:
        logger.info(f"  {out_pct:.1f}% outside [{lo}, {hi}] t/ha — OK")

    years = sorted(df["year"].unique())
    expected = list(range(2003, 2024))
    missing_years = [y for y in expected if y not in years]
    if missing_years:
        logger.warning(f"  Missing years: {missing_years}")
        passed = False
    else:
        logger.info(f"  All {len(expected)} years present (2003–2023) — OK")

    missing_pct = df["yield_ton_ha"].isna().mean() * 100
    logger.info(f"  Missing yield: {missing_pct:.1f}% (expected: suppressed counties)")

    return passed


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("NASS_API_KEY")
    if not api_key:
        logger.error("NASS_API_KEY not found in .env — cannot proceed.")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== USA Corn Yield Download — USDA NASS QuickStats ===")

    raw_df = fetch_nass(api_key)
    clean_df = clean_nass(raw_df)
    validate_quick(clean_df)

    # Raw CSV — preserve all NASS fields, DO NOT MODIFY after this
    raw_path = RAW_DIR / "nass_corn_county_yield_2003_2023.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Raw CSV → {raw_path} ({raw_path.stat().st_size / 1024:.0f} KB)")

    # Metadata — log exactly what was downloaded
    meta: dict = {
        "downloaded_at": datetime.now().isoformat(),
        "source": "USDA NASS QuickStats API",
        "endpoint": NASS_URL,
        "query_params": {
            "commodity_desc": "CORN",
            "statisticcat_desc": "YIELD",
            "agg_level_desc": "COUNTY",
            "unit_desc": "BU / ACRE",
            "domain_desc": "TOTAL",
            "year_range": "2003–2023",
        },
        "raw_record_count": len(raw_df),
        "clean_record_count": len(clean_df),
        "counties": int(clean_df["region_id"].nunique()),
        "states": int(clean_df["state"].nunique()),
        "years": sorted(clean_df["year"].unique().tolist()),
        "unit_conversion": "yield_ton_ha = yield_bu_acre × 0.06277",
        "notes": [
            "(D) values suppressed by NASS for confidentiality → NaN",
            "OTHER (COMBINED) COUNTIES rows excluded",
            "Full 5-digit FIPS code = state_fips (2) + county_code (3)",
        ],
    }
    meta_path = RAW_DIR / "nass_corn_county_yield_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadata → {meta_path}")

    # Processed Parquet — standard schema for model training
    proc_path = PROCESSED_DIR / "yield_usa_2003_2023.parquet"
    clean_df.to_parquet(proc_path, index=False)
    logger.info(f"Processed Parquet → {proc_path} ({proc_path.stat().st_size / 1024:.0f} KB)")

    logger.info("=== Download complete. Next: run data-validation skill for full validation. ===")


if __name__ == "__main__":
    main()
