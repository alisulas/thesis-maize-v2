"""Download Indonesia maize yield data from BDSP Kementan.

Source: https://bdsp2.pertanian.go.id/bdsp/id/lokasi
Covers: all 38 provinces, 2003-2023
Indicators:
    0103 - Luas Panen (Ha)
    0104 - Produksi (Ton)
    0105 - Produktivitas (Kuintal/Ha) → converted to ton/ha

Output:
    data/raw/indonesia/bdsp_jagung_provinsi_2003_2023_raw.csv  (raw, wide format)
    data/processed/indonesia/yield_indonesia_province_2003_2023.csv  (tidy, thesis schema)

Usage:
    source .venv/bin/activate
    python src/data/download_bdsp_indonesia.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUT = PROJECT_ROOT / "data" / "raw" / "indonesia"
PROCESSED_OUT = PROJECT_ROOT / "data" / "processed" / "indonesia"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://bdsp2.pertanian.go.id/bdsp/id"

INDICATORS = {
    "luas_panen":     {"cd": "0103", "nm": "LUAS PANEN",    "satuan": "3",  "unit": "Ha"},
    "produksi":       {"cd": "0104", "nm": "PRODUKSI",       "satuan": "20", "unit": "Ton"},
    "produktivitas":  {"cd": "0105", "nm": "PRODUKTIVITAS",  "satuan": "9",  "unit": "Kuintal/Ha"},
}

YEAR_START = 2003
YEAR_END   = 2023


def fetch_indicator(indicator_key: str) -> pd.DataFrame:
    """Fetch one indicator for all provinces, all years. Returns wide-format DataFrame."""
    ind = INDICATORS[indicator_key]
    logger.info(f"  Fetching: {ind['nm']} ({ind['unit']}) ...")

    payload = {
        "subsektor":    "01",
        "subsektorcd":  "01",
        "subsektornm":  "Tanaman Pangan",
        "level":        "02",
        "levelnm":      "Provinsi",
        "prov":         "00",
        "satuan":       ind["satuan"],
        "satuannm":     ind["unit"],
        "sts_angka":    "6",
        "sts_angkanm":  "Angka Tetap",
        "sumb_data":    "00",
        "sumb_datanm":  "-- Pilih Sumber Data --",
        "tahunAwal":    str(YEAR_START),
        "tahunAkhir":   str(YEAR_END),
        "komoditas":    "01027",
        "komoditasnm":  "JAGUNG",
        "indikator":    ind["cd"],
        "indikatornm":  ind["nm"],
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer":      f"{BASE_URL}/lokasi",
        "User-Agent":   "Mozilla/5.0",
    }

    resp = requests.post(f"{BASE_URL}/lokasi/result", data=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "example"})
    if table is None:
        raise RuntimeError(f"No table found in response for {indicator_key}")

    headers_row = table.find("thead").find_all("td")
    cols = [td.get_text(strip=True) for td in headers_row]  # ['No', 'Lokasi', '2003', ...]

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(cols):
            rows.append(cells)

    df = pd.DataFrame(rows, columns=cols)
    df = df.drop(columns=["No"])

    # Clean numeric: replace comma decimal separator, strip whitespace
    year_cols = [c for c in df.columns if c.isdigit()]
    for col in year_cols:
        df[col] = df[col].str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse province code + name from "11 - Aceh"
    df[["prov_code", "prov_name"]] = df["Lokasi"].str.extract(r"^(\d+)\s*-\s*(.+)$")
    df = df.drop(columns=["Lokasi"])

    # Add indicator column
    df["indicator"] = indicator_key

    logger.info(f"    → {len(df)} provinces, {len(year_cols)} years")
    return df, year_cols


def wide_to_tidy(
    luas_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    prod_kv_df: pd.DataFrame,
    year_cols: list[str],
) -> pd.DataFrame:
    """Melt three wide-format DataFrames into one tidy thesis-schema DataFrame."""

    def melt_one(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        return df.melt(
            id_vars=["prov_code", "prov_name"],
            value_vars=year_cols,
            var_name="year",
            value_name=value_name,
        )

    luas_tidy  = melt_one(luas_df,    "harvested_ha")
    prod_tidy  = melt_one(prod_df,    "production_ton")
    yield_tidy = melt_one(prod_kv_df, "yield_kuha")

    merged = luas_tidy.merge(prod_tidy,  on=["prov_code", "prov_name", "year"]) \
                      .merge(yield_tidy, on=["prov_code", "prov_name", "year"])

    merged["year"] = merged["year"].astype(int)

    # Convert ku/ha → ton/ha  (1 ku = 100 kg = 0.1 ton)
    merged["yield_ton_ha"] = merged["yield_kuha"] / 10.0

    # Thesis standard schema
    merged["region_id"]  = "IDN-" + merged["prov_code"].str.strip()
    merged["region_name"] = merged["prov_name"].str.strip()
    merged["country"]    = "IDN"
    merged["data_source"] = "BDSP-Kementan"

    out = merged[[
        "region_id", "region_name", "country", "year",
        "harvested_ha", "production_ton", "yield_ton_ha", "data_source",
    ]].sort_values(["region_id", "year"]).reset_index(drop=True)

    return out


def main() -> None:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading BDSP Kementan — Jagung Provinsi Indonesia")
    logger.info(f"  Years: {YEAR_START}–{YEAR_END}")

    luas_df,   year_cols = fetch_indicator("luas_panen")
    time.sleep(1)
    prod_df,   _         = fetch_indicator("produksi")
    time.sleep(1)
    prod_kv_df, _        = fetch_indicator("produktivitas")

    # Save raw wide-format (all 3 indicators stacked)
    raw_df = pd.concat([luas_df, prod_df, prod_kv_df], ignore_index=True)
    raw_path = RAW_OUT / "bdsp_jagung_provinsi_2003_2023_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"\nRaw saved → {raw_path}")

    # Build tidy thesis-schema CSV
    tidy_df = wide_to_tidy(luas_df, prod_df, prod_kv_df, year_cols)

    # Drop rows with no yield data
    n_before = len(tidy_df)
    tidy_df = tidy_df.dropna(subset=["yield_ton_ha"])
    tidy_df = tidy_df[tidy_df["yield_ton_ha"] > 0]
    n_after = len(tidy_df)
    logger.info(f"Dropped {n_before - n_after} zero/null rows ({n_after} kept)")

    processed_path = PROCESSED_OUT / "yield_indonesia_province_2003_2023.csv"
    tidy_df.to_csv(processed_path, index=False)
    logger.info(f"Tidy saved → {processed_path}")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"  Provinces: {tidy_df['region_id'].nunique()}")
    logger.info(f"  Years: {sorted(tidy_df['year'].unique())}")
    logger.info(f"  Yield range: {tidy_df['yield_ton_ha'].min():.2f} – {tidy_df['yield_ton_ha'].max():.2f} ton/ha")
    logger.info(f"  Total rows: {len(tidy_df)}")

    # Check for provinces with sparse data
    coverage = tidy_df.groupby("region_id")["year"].count()
    sparse = coverage[coverage < 10]
    if len(sparse):
        logger.warning(f"  Provinces with <10 years of data: {sparse.to_dict()}")


if __name__ == "__main__":
    main()
