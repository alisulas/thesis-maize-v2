"""Download Indonesia maize data at kabupaten (district) level from BDSP Kementan.

Source: https://bdsp2.pertanian.go.id/bdsp/id/lokasi
Level: Kabupaten (level=03), loops over all 38 provinces
Indicators:
    0103 - Luas Panen (Ha)
    0104 - Produksi (Ton)
    0105 - Produktivitas (Kuintal/Ha) → converted to ton/ha in processed output

Output:
    data/raw/indonesia/bdsp_jagung_kabupaten_2003_2023_raw.csv  (raw, stacked wide)
    data/processed/indonesia/yield_indonesia_kabupaten_2003_2023.csv  (tidy schema)

Usage:
    source .venv/bin/activate
    python src/data/download_bdsp_indonesia_kab.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUT       = PROJECT_ROOT / "data" / "raw" / "indonesia"
PROCESSED_OUT = PROJECT_ROOT / "data" / "processed" / "indonesia"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://bdsp2.pertanian.go.id/bdsp/id"

INDICATORS = {
    "luas_panen":    {"cd": "0103", "nm": "LUAS PANEN",   "satuan": "3",  "unit": "Ha"},
    "produksi":      {"cd": "0104", "nm": "PRODUKSI",      "satuan": "20", "unit": "Ton"},
    "produktivitas": {"cd": "0105", "nm": "PRODUKTIVITAS", "satuan": "9",  "unit": "Kuintal/Ha"},
}

YEAR_START = 2003
YEAR_END   = 2023

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer":      f"{BASE_URL}/lokasi",
    "User-Agent":   "Mozilla/5.0",
}


def get_provinces() -> list[dict]:
    """Fetch all province codes and names from the API."""
    resp = requests.get(f"{BASE_URL}/lokasi/getProv", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()   # [{"fkode_prop": "11", "nama_prop": "Aceh"}, ...]


def fetch_kabupaten(prov_code: str, prov_name: str, indicator_key: str) -> pd.DataFrame | None:
    """Fetch one indicator for all kabupaten in one province. Returns wide DataFrame or None."""
    ind = INDICATORS[indicator_key]

    payload = {
        "subsektor":   "01",
        "subsektorcd": "01",
        "subsektornm": "Tanaman Pangan",
        "level":       "03",
        "levelnm":     "Kabupaten",
        "prov":        prov_code,
        "provnm":      prov_name,
        "satuan":      ind["satuan"],
        "satuannm":    ind["unit"],
        "sts_angka":   "6",
        "sts_angkanm": "Angka Tetap",
        "sumb_data":   "00",
        "sumb_datanm": "-- Pilih Sumber Data --",
        "tahunAwal":   str(YEAR_START),
        "tahunAkhir":  str(YEAR_END),
        "komoditas":   "01027",
        "komoditasnm": "JAGUNG",
        "indikator":   ind["cd"],
        "indikatornm": ind["nm"],
    }

    resp = requests.post(f"{BASE_URL}/lokasi/result", data=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "example"})
    if table is None:
        return None

    header_cells = table.find("thead").find_all("td")
    cols = [td.get_text(strip=True) for td in header_cells]

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(cols):
            rows.append(cells)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=cols).drop(columns=["No"])

    year_cols = [c for c in df.columns if c.isdigit()]
    for col in year_cols:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".").str.strip(), errors="coerce"
        )

    # "3301 - Kab. Cilacap" → kab_code="3301", kab_name="Kab. Cilacap"
    parsed = df["Lokasi"].str.extract(r"^(\d+)\s*-\s*(.+)$")
    df["kab_code"] = parsed[0]
    df["kab_name"] = parsed[1].str.strip()
    df["prov_code"] = prov_code
    df["prov_name"] = prov_name
    df["indicator"] = indicator_key
    df = df.drop(columns=["Lokasi"])

    return df, year_cols


def build_tidy(all_wide: dict[str, pd.DataFrame], year_cols: list[str]) -> pd.DataFrame:
    """Merge three indicator DataFrames into tidy thesis-schema."""

    def melt_one(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        return df.melt(
            id_vars=["kab_code", "kab_name", "prov_code", "prov_name"],
            value_vars=year_cols,
            var_name="year",
            value_name=value_name,
        )

    luas  = melt_one(all_wide["luas_panen"],    "harvested_ha")
    prod  = melt_one(all_wide["produksi"],       "production_ton")
    yields = melt_one(all_wide["produktivitas"], "yield_kuha")

    merged = luas.merge(prod,   on=["kab_code", "kab_name", "prov_code", "prov_name", "year"]) \
                 .merge(yields, on=["kab_code", "kab_name", "prov_code", "prov_name", "year"])

    merged["year"]         = merged["year"].astype(int)
    merged["yield_ton_ha"] = merged["yield_kuha"] / 10.0
    merged["region_id"]    = "IDN-" + merged["kab_code"].str.strip()
    merged["region_name"]  = merged["kab_name"]
    merged["country"]      = "IDN"
    merged["data_source"]  = "BDSP-Kementan"

    out = merged[[
        "region_id", "region_name", "prov_code", "prov_name",
        "country", "year", "harvested_ha", "production_ton", "yield_ton_ha", "data_source",
    ]].sort_values(["region_id", "year"]).reset_index(drop=True)

    return out


def main() -> None:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)

    provinces = get_provinces()
    logger.info(f"Provinces loaded: {len(provinces)}")
    logger.info(f"Years: {YEAR_START}–{YEAR_END}  |  Indicators: {list(INDICATORS.keys())}")
    logger.info(f"Total requests: {len(provinces) * len(INDICATORS)}\n")

    all_chunks: dict[str, list[pd.DataFrame]] = {k: [] for k in INDICATORS}
    year_cols_ref: list[str] = []

    for i, prov in enumerate(provinces, 1):
        code = prov["fkode_prop"]
        name = prov["nama_prop"]
        logger.info(f"[{i:2d}/{len(provinces)}] {code} – {name}")

        for ind_key in INDICATORS:
            result = fetch_kabupaten(code, name, ind_key)
            if result is None:
                logger.warning(f"    {ind_key}: no table returned, skipping")
                continue
            df, year_cols = result
            if not year_cols_ref:
                year_cols_ref = year_cols
            n_kab = df["kab_code"].notna().sum()
            logger.info(f"    {ind_key}: {n_kab} kabupaten")
            all_chunks[ind_key].append(df)
            time.sleep(0.4)   # be polite to the server

    logger.info("\nMerging all chunks...")
    merged_wide = {k: pd.concat(v, ignore_index=True) for k, v in all_chunks.items() if v}

    # Save raw stacked wide
    raw_frames = []
    for k, df in merged_wide.items():
        raw_frames.append(df)
    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_path = RAW_OUT / "bdsp_jagung_kabupaten_2003_2023_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Raw saved → {raw_path}  ({len(raw_df)} rows)")

    # Build tidy
    tidy = build_tidy(merged_wide, year_cols_ref)

    n_before = len(tidy)
    tidy = tidy.dropna(subset=["yield_ton_ha"])
    tidy = tidy[tidy["yield_ton_ha"] > 0]
    logger.info(f"Dropped {n_before - len(tidy)} zero/null rows, {len(tidy)} kept")

    processed_path = PROCESSED_OUT / "yield_indonesia_kabupaten_2003_2023.csv"
    tidy.to_csv(processed_path, index=False)
    logger.info(f"Tidy saved → {processed_path}")

    logger.info("\n=== Summary ===")
    logger.info(f"  Kabupaten: {tidy['region_id'].nunique()}")
    logger.info(f"  Provinces covered: {tidy['prov_code'].nunique()}")
    logger.info(f"  Years: {sorted(tidy['year'].unique())}")
    logger.info(f"  Yield range: {tidy['yield_ton_ha'].min():.2f} – {tidy['yield_ton_ha'].max():.2f} ton/ha")
    logger.info(f"  Total rows: {len(tidy)}")

    coverage = tidy.groupby("region_id")["year"].count()
    sparse = coverage[coverage < 5]
    if len(sparse):
        logger.info(f"  Kabupaten with <5 years: {len(sparse)}")


if __name__ == "__main__":
    main()
