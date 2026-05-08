"""
Clean and standardize BPS Indonesia province-level maize data (2020-2024).

Source: BPS - Produksi Jagung Menurut Provinsi
Format: Wide CSV with 3 variable groups (Luas Panen, Produktivitas, Produksi)
Units:
  - Luas Panen: already in hectares (ha)
  - Produktivitas: kuintal/ha → convert to ton/ha (÷10)
  - Produksi: already in tons

Output: Standard schema in data/processed/indonesia/
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path(
    "/Users/alisulas/ClaudeQ/thesis_maize/data/Manually Download/Indonesia/"
    "BPS data Panen Jagung.csv"
)
OUTPUT_DIR = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/processed/indonesia")
YEARS = [2020, 2021, 2022, 2023, 2024]

# Aggregate rows to skip
SKIP_PROVINCES = {"INDONESIA"}

# Province name normalization
NAME_MAP = {
    "DI YOGYAKARTA": "DI Yogyakarta",
    "JAWA BARAT": "Jawa Barat",
    "JAWA TENGAH": "Jawa Tengah",
    "JAWA TIMUR": "Jawa Timur",
    "DKI JAKARTA": "DKI Jakarta",
    "NUSA TENGGARA BARAT": "Nusa Tenggara Barat",
    "NUSA TENGGARA TIMUR": "Nusa Tenggara Timur",
    "KEP. BANGKA BELITUNG": "Kep. Bangka Belitung",
    "KEP. RIAU": "Kep. Riau",
    "KALIMANTAN BARAT": "Kalimantan Barat",
    "KALIMANTAN TENGAH": "Kalimantan Tengah",
    "KALIMANTAN SELATAN": "Kalimantan Selatan",
    "KALIMANTAN TIMUR": "Kalimantan Timur",
    "KALIMANTAN UTARA": "Kalimantan Utara",
    "SULAWESI UTARA": "Sulawesi Utara",
    "SULAWESI TENGAH": "Sulawesi Tengah",
    "SULAWESI SELATAN": "Sulawesi Selatan",
    "SULAWESI TENGGARA": "Sulawesi Tenggara",
    "SULAWESI BARAT": "Sulawesi Barat",
    "SUMATERA UTARA": "Sumatera Utara",
    "SUMATERA BARAT": "Sumatera Barat",
    "SUMATERA SELATAN": "Sumatera Selatan",
    "MALUKU": "Maluku",
    "MALUKU UTARA": "Maluku Utara",
    "PAPUA BARAT": "Papua Barat",
    "PAPUA BARAT DAYA": "Papua Barat Daya",
    "PAPUA SELATAN": "Papua Selatan",
    "PAPUA TENGAH": "Papua Tengah",
    "PAPUA PEGUNUNGAN": "Papua Pegunungan",
}


def _parse_num(val: str) -> float | None:
    """Parse Indonesian number string to float. Returns None for missing."""
    if val is None:
        return None
    val = str(val).strip().replace(",", ".")
    if val in ("-", "", "nan", "0") or not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading: {INPUT_PATH.name}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Column layout (verified from file):
    # 0: Province name
    # 1-5: Luas Panen (ha) 2020-2024
    # 6-10: Produktivitas (ku/ha) 2020-2024
    # 11-15: Produksi (ton) 2020-2024

    records = []
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        if not name or name == "nan" or "Catatan" in name:
            continue
        if name in SKIP_PROVINCES:
            continue

        name = NAME_MAP.get(name, name.title())

        for i, year in enumerate(YEARS):
            area_ha = _parse_num(row.iloc[1 + i])
            yld_ku = _parse_num(row.iloc[6 + i])
            prod_ton = _parse_num(row.iloc[11 + i])

            # Skip rows with no data (e.g., DKI Jakarta)
            if area_ha is None and prod_ton is None:
                continue

            yld_ton_ha = yld_ku / 10.0 if yld_ku is not None else None

            # Recalculate yield from production/area (more reliable)
            if prod_ton is not None and area_ha is not None and area_ha > 0:
                yld_ton_ha = round(prod_ton / area_ha, 4)

            records.append(
                {
                    "region_name": name,
                    "year": year,
                    "country": "IDN",
                    "planted_ha": area_ha,
                    "harvested_ha": area_ha,
                    "production_ton": prod_ton,
                    "yield_ton_ha": yld_ton_ha,
                    "data_source": "BPS",
                }
            )

    result = pd.DataFrame(records)
    result = result.sort_values(["year", "region_name"]).reset_index(drop=True)

    # Drop rows with no meaningful data
    result = result.dropna(subset=["yield_ton_ha"])
    result = result[result["production_ton"] > 0]  # DKI Jakarta has 0 prod

    out_path = OUTPUT_DIR / "yield_indonesia_province_2020_2024.csv"
    result.to_csv(out_path, index=False)

    logger.info(f"Saved: {out_path}")
    logger.info(f"Total: {len(result)} rows, {result['region_name'].nunique()} provinces, "
                 f"{result['year'].nunique()} years")

    # Summary
    print("\n=== INDONESIA BPS DATA ===")
    print(f"Rows: {len(result)}")
    print(f"Provinces: {result['region_name'].nunique()}")
    print(f"Years: {result['year'].min()}–{result['year'].max()}")
    print(f"Yield: {result['yield_ton_ha'].min():.2f} – {result['yield_ton_ha'].max():.2f} ton/ha "
          f"(mean: {result['yield_ton_ha'].mean():.2f})")
    print(f"Total production 2024: {result[result['year']==2024]['production_ton'].sum()/1e6:.2f}M tons")

    # Per-year coverage
    print("\n=== Per-year coverage ===")
    for year in YEARS:
        sub = result[result["year"] == year]
        total = sub["production_ton"].sum()
        print(f"  {year}: {len(sub)} provinces, total prod={total/1e6:.2f}M tons, "
              f"mean yield={sub['yield_ton_ha'].mean():.2f} t/ha")

    return result


if __name__ == "__main__":
    main()
