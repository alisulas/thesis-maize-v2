"""Download national-level corn data for all countries, 2003-2023.

Used as cross-validation reference for province-level data.
Primary source: Our World in Data (OWID) — processed from FAOSTAT/FAO.
OWID is used because FAOSTAT's API (fenixservices.fao.org) is unreliable.

Outputs:
    data/raw/{country}/owid_{country}_corn_national_2003_2023.csv
    data/processed/{country}/yield_{country}_national_owid_2003_2023.csv
    data/processed/owid_all_countries_national.csv  ← combined reference
"""

import io
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (thesis research)"}

# OWID dataset URLs — data sourced from FAOSTAT, already in standard units
OWID_URLS: dict[str, str] = {
    "yield_ton_ha": "https://ourworldindata.org/grapher/maize-yields.csv",
    "production_ton": "https://ourworldindata.org/grapher/maize-production.csv",
    "harvested_ha": "https://ourworldindata.org/grapher/maize-area-harvested.csv",
}

# Target countries: OWID entity name → project key + iso3
COUNTRIES: dict[str, dict] = {
    "United States": {"key": "usa",       "iso3": "USA", "name": "United States"},
    "Indonesia":     {"key": "indonesia", "iso3": "IDN", "name": "Indonesia"},
    "Vietnam":       {"key": "vietnam",   "iso3": "VNM", "name": "Vietnam"},
    "Thailand":      {"key": "thailand",  "iso3": "THA", "name": "Thailand"},
}


def fetch_owid(metric: str) -> pd.DataFrame:
    """Download one OWID maize dataset and filter to target countries + years.

    Args:
        metric: Key in OWID_URLS ('yield_ton_ha', 'production_ton', 'harvested_ha').

    Returns:
        Filtered dataframe with columns [country_key, iso3, year, <metric>].
    """
    url = OWID_URLS[metric]
    logger.info(f"Downloading OWID {metric} from {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    # OWID columns: Entity, Code, Year, <metric name>
    value_col = [c for c in df.columns if c not in ("Entity", "Code", "Year")][0]

    df = df[df["Entity"].isin(COUNTRIES)].copy()
    df = df[(df["Year"] >= 2003) & (df["Year"] <= 2023)].copy()

    df["country_key"] = df["Entity"].map(lambda x: COUNTRIES[x]["key"])
    df["iso3"] = df["Entity"].map(lambda x: COUNTRIES[x]["iso3"])
    df = df.rename(columns={"Year": "year", value_col: metric})

    logger.info(f"  {metric}: {len(df)} rows for {df['Entity'].unique().tolist()}")
    return df[["country_key", "iso3", "Entity", "year", metric]]


def build_national_table() -> pd.DataFrame:
    """Merge yield + production + area into one table per country-year.

    Returns:
        Wide dataframe: one row per (country, year) with all metrics.
    """
    dfs: dict[str, pd.DataFrame] = {}
    for metric in OWID_URLS:
        try:
            dfs[metric] = fetch_owid(metric)
        except Exception as e:
            logger.error(f"Failed to fetch {metric}: {e}")

    if not dfs:
        raise RuntimeError("All OWID downloads failed.")

    # Merge on country + year
    base = list(dfs.values())[0][["country_key", "iso3", "Entity", "year"]].copy()
    for metric, df in dfs.items():
        base = base.merge(df[["country_key", "year", metric]], on=["country_key", "year"], how="left")

    base = base.rename(columns={"Entity": "region_name"})
    base["country"] = base["iso3"]
    base["region_id"] = base["iso3"]
    base["data_source"] = "OWID_FAOSTAT"
    base["level"] = "national"

    return base.sort_values(["country_key", "year"]).reset_index(drop=True)


def main() -> None:
    logger.info("=== National Corn Data Download — OWID (FAOSTAT source) ===")
    logger.info("Countries: USA, Indonesia, Vietnam, Thailand | Years: 2003–2023")

    combined = build_national_table()

    # Save per-country files
    for country_key, grp in combined.groupby("country_key"):
        raw_dir = PROJECT_ROOT / "data" / "raw" / str(country_key)
        proc_dir = PROJECT_ROOT / "data" / "processed" / str(country_key)
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        raw_path = raw_dir / f"owid_{country_key}_corn_national_2003_2023.csv"
        grp.to_csv(raw_path, index=False)

        proc_path = proc_dir / f"yield_{country_key}_national_owid_2003_2023.csv"
        grp.to_csv(proc_path, index=False)

        v = grp["yield_ton_ha"].dropna()
        logger.info(
            f"  {country_key.upper()}: {len(grp)} years | "
            f"yield {v.min():.2f}–{v.max():.2f} t/ha | mean {v.mean():.2f}"
        )

    # Save combined reference
    combined_path = PROJECT_ROOT / "data" / "processed" / "owid_all_countries_national.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"\nCombined national reference → {combined_path}")
    logger.info("=== National download complete ===")
    logger.info("NOTE: This is NATIONAL-level data. Province-level still needed for model training.")


if __name__ == "__main__":
    main()
