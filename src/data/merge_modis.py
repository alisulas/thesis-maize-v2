"""Merge MODIS CSVs with yield labels → training tensors.

Reads all per-year CSVs from data/raw/modis/, aligns with yield labels via
explicit region name mappings, and outputs one .npz per country:
  X: (N, T=46, F=10)  float32  — satellite features
  y: (N,)             float32  — yield in ton/ha
  region_ids: (N,)   str
  years: (N,)         int

Features (F=10, EVI dropped due to export overflow):
    b01 b02 b03 b04 b05 b06 b07  ndvi  LST_Day_1km  LST_Night_1km

Usage:
    python src/data/merge_modis.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_MODIS = PROJECT_ROOT / "data" / "raw" / "modis"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "modis"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
    "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07",
    "ndvi", "LST_Day_1km", "LST_Night_1km",
]
N_FEATURES = len(FEATURE_COLS)  # 10
N_TIMESTEPS = 46


# ---------------------------------------------------------------------------
# Region name maps (GAUL ADM1_NAME → region identifier in yield data)
# ---------------------------------------------------------------------------

IDN_GAUL_TO_REGION_ID: dict[str, str] = {
    "Nangroe Aceh Darussalam": "IDN-11",
    "Sumatera Utara":          "IDN-12",
    "Sumatera Barat":          "IDN-13",
    "Riau":                    "IDN-14",
    "Jambi":                   "IDN-15",
    "Sumatera Selatan":        "IDN-16",
    "Bengkulu":                "IDN-17",
    "Lampung":                 "IDN-18",
    "Bangka Belitung":         "IDN-19",
    "Kepulauan-riau":          "IDN-21",
    "Dki Jakarta":             "IDN-31",
    "Jawa Barat":              "IDN-32",
    "Jawa Tengah":             "IDN-33",
    "Daerah Istimewa Yogyakarta": "IDN-34",
    "Jawa Timur":              "IDN-35",
    "Banten":                  "IDN-36",
    "Bali":                    "IDN-51",
    "Nusatenggara Barat":      "IDN-52",
    "Nusatenggara Timur":      "IDN-53",
    "Kalimantan Barat":        "IDN-61",
    "Kalimantan Tengah":       "IDN-62",
    "Kalimantan Selatan":      "IDN-63",
    "Kalimantan Timur":        "IDN-64",
    "Sulawesi Utara":          "IDN-71",
    "Sulawesi Tengah":         "IDN-72",
    "Sulawesi Selatan":        "IDN-73",
    "Sulawesi Tenggara":       "IDN-74",
    "Gorontalo":               "IDN-75",
    "Sulawesi Barat":          "IDN-76",
    "Maluku":                  "IDN-81",
    "Maluku Utara":            "IDN-82",
    "Papua Barat":             "IDN-91",
    "Papua":                   "IDN-94",
    # Kalimantan Utara (IDN-65) and 4 new Papua provinces not in GAUL 2015
}

# GAUL ADM1_NAME → yield region_name (None = drop, Ha Tay merged into Ha Noi 2008)
VNM_NAME_FIX: dict[str, str | None] = {
    "Ba Ria-Vung Tau":    "Ba Ria - Vung Tau",
    "Can Tho city":       "Can Tho",
    "Da Nang City":       "Da Nang",
    "Ha Noi City":        "Ha Noi",
    "Hai Phong City":     "Hai Phong",
    "Thua Thien - Hue":   "Thua Thien-Hue",
    "Ha Tay":             None,
}

# GAUL ADM1_NAME → yield region_name_en (only the mismatched ones)
THA_NAME_FIX: dict[str, str] = {
    "Buriram":             "Buri Ram",
    "Chainat":             "Chai Nat",
    "Kampaeng Phet":       "Kamphaeng Phet",
    "Lopburi":             "Lop Buri",
    "Nong Bua Lamphu":     "Nong Bua Lam Phu",
    "Phachinburi":         "Prachin Buri",
    "Prachuap Khilikhan":  "Prachuap Khiri Khan",
    "Si Saket":            "Si Sa Ket",
    "Suphanburi":          "Suphan Buri",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fill_feature_nans(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """Forward-fill then backward-fill all feature NaNs per (region, year) block.

    Remaining NaN after ffill+bfill (e.g. all-cloud year) are filled with 0.
    """
    df = df.copy()
    for col in FEATURE_COLS:
        df[col] = (
            df.groupby([region_col, "year"])[col]
            .transform(lambda s: s.ffill().bfill().fillna(0.0))
        )
    return df


def build_tensor(
    modis_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    region_col: str,
    yield_region_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Build (X, y, region_ids, years) arrays by matching MODIS rows with yield.

    Args:
        modis_df: MODIS DataFrame with region_col, year, date, feature cols.
        yield_df: Yield DataFrame with yield_region_col, year, yield_ton_ha.
        region_col: Column in modis_df identifying region (already normalized).
        yield_region_col: Column in yield_df identifying region.

    Returns:
        X: (N, 46, 10) float32
        y: (N,) float32
        region_ids: list of N strings
        years: list of N ints
    """
    modis_df = modis_df.sort_values([region_col, "year", "date"])

    X_list, y_list, rid_list, yr_list = [], [], [], []
    missing_yield, missing_ts = 0, 0

    yield_lookup = (
        yield_df.set_index([yield_region_col, "year"])["yield_ton_ha"]
        .to_dict()
    )

    for (region_id, year), grp in modis_df.groupby([region_col, "year"]):
        # Enforce exactly 46 timesteps
        if len(grp) != N_TIMESTEPS:
            missing_ts += 1
            continue

        # Look up yield
        y_val = yield_lookup.get((region_id, year))
        if y_val is None or pd.isna(y_val):
            missing_yield += 1
            continue

        x = grp[FEATURE_COLS].values.astype(np.float32)  # (46, 10)
        # Clip NDVI to [-1, 1]
        ndvi_idx = FEATURE_COLS.index("ndvi")
        x[:, ndvi_idx] = np.clip(x[:, ndvi_idx], -1.0, 1.0)

        X_list.append(x)
        y_list.append(float(y_val))
        rid_list.append(region_id)
        yr_list.append(int(year))

    if missing_yield:
        logger.info(f"    Dropped {missing_yield} region-years: no yield label")
    if missing_ts:
        logger.info(f"    Dropped {missing_ts} region-years: wrong timestep count")

    X = np.stack(X_list).astype(np.float32)   # (N, 46, 10)
    y = np.array(y_list, dtype=np.float32)
    return X, y, rid_list, yr_list


# ---------------------------------------------------------------------------
# Per-country processors
# ---------------------------------------------------------------------------

def process_usa() -> None:
    logger.info("=== USA ===")
    files = sorted(RAW_MODIS.glob("modis_usa_*.csv"))
    logger.info(f"  Loading {len(files)} CSVs...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = fill_feature_nans(df, "GEOID")
    df["region_id"] = df["GEOID"].astype(str)

    yield_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "usa" / "yield_usa_2003_2023.parquet"
    )
    yield_df["region_id"] = yield_df["region_id"].astype(str)

    X, y, rids, yrs = build_tensor(df, yield_df, "region_id", "region_id")
    logger.info(f"  X: {X.shape}, y: {y.shape}  yield [{y.min():.2f}, {y.max():.2f}] t/ha")

    out = OUT_DIR / "usa_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out}")


def process_idn() -> None:
    logger.info("=== IDN ===")
    files = sorted(RAW_MODIS.glob("modis_idn_*.csv"))
    logger.info(f"  Loading {len(files)} CSVs...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = fill_feature_nans(df, "ADM1_NAME")

    # Map GAUL name → region_id
    df["region_id"] = df["ADM1_NAME"].map(IDN_GAUL_TO_REGION_ID)
    unmapped = df[df["region_id"].isna()]["ADM1_NAME"].unique()
    if len(unmapped):
        logger.warning(f"  Unmapped GAUL names: {unmapped}")
    df = df.dropna(subset=["region_id"])

    yield_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "indonesia" / "yield_indonesia_2003_2023.parquet"
    )

    X, y, rids, yrs = build_tensor(df, yield_df, "region_id", "region_id")
    logger.info(f"  X: {X.shape}, y: {y.shape}  yield [{y.min():.2f}, {y.max():.2f}] t/ha")

    out = OUT_DIR / "idn_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out}")


def process_vnm() -> None:
    logger.info("=== VNM ===")
    files = sorted(RAW_MODIS.glob("modis_vnm_*.csv"))
    logger.info(f"  Loading {len(files)} CSVs...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = fill_feature_nans(df, "ADM1_NAME")

    # Normalize name: apply fixes, then use as region_id
    df["region_id"] = df["ADM1_NAME"].apply(
        lambda n: VNM_NAME_FIX.get(n, n)  # apply fix; keep original if not in map
    )
    df = df[df["region_id"].notna()]  # drop Ha Tay (None)

    yield_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "vietnam" / "yield_vietnam_province_1995_2023.csv"
    )

    X, y, rids, yrs = build_tensor(df, yield_df, "region_id", "region_name")
    logger.info(f"  X: {X.shape}, y: {y.shape}  yield [{y.min():.2f}, {y.max():.2f}] t/ha")

    out = OUT_DIR / "vnm_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out}")


def process_tha() -> None:
    logger.info("=== THA ===")
    files = sorted(RAW_MODIS.glob("modis_tha_*.csv"))
    logger.info(f"  Loading {len(files)} CSVs...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = fill_feature_nans(df, "ADM1_NAME")

    df["region_id"] = df["ADM1_NAME"].apply(
        lambda n: THA_NAME_FIX.get(n, n)
    )

    yield_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "thailand" / "thailand_province_yield_2021_2023.csv"
    )

    X, y, rids, yrs = build_tensor(df, yield_df, "region_id", "region_name_en")
    logger.info(f"  X: {X.shape}, y: {y.shape}  yield [{y.min():.2f}, {y.max():.2f}] t/ha")

    out = OUT_DIR / "tha_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    process_usa()
    process_idn()
    process_vnm()
    process_tha()
    logger.info("\n=== All tensors saved to data/processed/modis/ ===")


if __name__ == "__main__":
    main()
