"""Merge MODIS CSVs with USDA yield labels → USA training tensor.

Reads all per-year CSVs from data/raw/modis/ (modis_usa_*.csv), aligns with
USDA NASS yield data via GEOID, and outputs:
    usa_modis.npz: X (N, T=46, F=10), y (N,), region_ids (N,), years (N,)

Features (F=10, EVI dropped due to export overflow):
    b01 b02 b03 b04 b05 b06 b07  ndvi  LST_Day_1km  LST_Night_1km

Usage:
    python src/data/02_merge_usa.py
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
    # GEOID and year come out as float64 after concat (NaN rows in some CSVs promote dtype)
    # Drop rows where GEOID or year is NaN, then cast to int before converting to strings
    df = df.dropna(subset=["GEOID", "year"])
    df["GEOID"] = df["GEOID"].astype(int)
    df["year"]  = df["year"].astype(int)
    df = fill_feature_nans(df, "GEOID")
    df["region_id"] = df["GEOID"].astype(str).str.zfill(5)

    yield_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "usa" / "clean_yield_usa_2003_2023.parquet"
    )
    yield_df["region_id"] = yield_df["region_id"].astype(str)

    X, y, rids, yrs = build_tensor(df, yield_df, "region_id", "region_id")
    logger.info(f"  X: {X.shape}, y: {y.shape}  yield [{y.min():.2f}, {y.max():.2f}] t/ha")

    out = OUT_DIR / "usa_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    process_usa()
    logger.info("Done → data/processed/modis/usa_modis.npz")


if __name__ == "__main__":
    main()
