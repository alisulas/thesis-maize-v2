"""Merge MODIS CSVs with BDSP kabupaten yield → Indonesia training tensor.

Reads per-year CSVs from data/raw/modis/ (modis_idn_*.csv), aligns with BDSP
kabupaten yield data via administrative code mapping, and outputs:
    idn_modis.npz: X (N, T=46, F=10), y (N,), region_ids (N,), years (N,)

Mapping: Kemendagri KODE_KKN → "IDN-{code}" → BDSP region_id

Features (F=10, EVI dropped due to export overflow):
    b01 b02 b03 b04 b05 b06 b07  ndvi  LST_Day_1km  LST_Night_1km

Usage:
    python src/data/02_merge_idn.py
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
N_FEATURES = len(FEATURE_COLS)
N_TIMESTEPS = 46

# Kolom dari GEE extraction (01_extract_modis_idn.py) — LapakGIS shapefile
GEE_KAB_CODE_COL = "KDPKAB"    # format: "11.01" → will be mapped to "IDN-1101"
GEE_KAB_NAME_COL = "WADMKK"    # e.g. "Aceh Selatan"


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


def build_kabupaten_mapping(modis_df: pd.DataFrame, yield_df: pd.DataFrame) -> dict:
    """Build lookup mapping GEE kab_code → BDSP region_id.

    GEE KDPKAB format: "11.01", "32.72" (dotted, LapakGIS/BIG codes)
    BDSP kab_code:     "1101", "3272"   (4-digit, BPS codes)
    BDSP region_id:    "IDN-1101"

    Strategy:
        1. Strip dots from GEE code → direct match on BDSP kab_code_raw
        2. Manual override for known BPS↔BIG code divergences (pemekaran Papua, etc.)
        3. Name-based fuzzy match as fallback
    """
    # Known code mismatches: BIG shapefile code → BPS BDSP code
    # Caused by pemekaran provinsi (Papua) and BPS↔BIG code numbering differences
    MANUAL_CODE_OVERRIDE = {
        # Kalimantan: BPS vs BIG numbering (Kutai group + Tanah Bumbu/Balangan)
        "6402": "6403",  # Kutai Kartanegara
        "6407": "6402",  # Kutai Barat
        "6408": "6404",  # Kutai Timur
        "6403": "6405",  # Berau
        "6310": "6311",  # Tanah Bumbu
        "6311": "6312",  # Balangan
        "1472": "1473",  # Kota Dumai
        "7324": "7325",  # Luwu Timur
        # Papua pemekaran: BDSP old codes → shapefile new codes
        "9103": "9403",  # Jayapura (kab)
        "9104": "9404",  # Nabire
        "9106": "9409",  # Biak Numfor
        "9108": "9410",  # Paniai (also 9403=Paniai in Papua Tengah)
        "9110": "9419",  # Sarmi
        "9111": "9420",  # Keerom
        "9115": "9426",  # Waropen
        "9119": "9427",  # Supiori
        "9171": "9471",  # Kota Jayapura
        "9201": "9107",  # Sorong (kab)
        "9203": "9101",  # Fak-Fak
        "9204": "9106",  # Sorong Selatan
        "9205": "9108",  # Raja Ampat
        "9208": "9102",  # Kaimana
        "9209": "9109",  # Tambrauw
        "9210": "9110",  # Maybrat
        "9211": "9111",  # Manokwari Selatan
        "9212": "9112",  # Pegunungan Arfak
        "9271": "9171",  # Kota Sorong
        "9302": "9413",  # Boven Digoel
        "9303": "9414",  # Mappi
        "9304": "9415",  # Asmat
        "9402": "9411",  # Puncak Jaya
        "9403": "9410",  # Paniai (duplicate entry)
        "9404": "9412",  # Mimika
        "9405": "9433",  # Puncak
        "9501": "9402",  # Jayawijaya
        "9502": "9417",  # Pegunungan Bintang
        "9503": "9416",  # Yahukimo
        "9504": "9418",  # Tolikara
        "9507": "9430",  # Lanny Jaya
        # Sisa 2: Teluk group
        "9206": "9104",  # Teluk Bintuni
        "9207": "9103",  # Teluk Wondama
    }

    gee_codes = set(modis_df[GEE_KAB_CODE_COL].dropna().unique())

    yield_df = yield_df.copy()
    yield_df["kab_code_raw"] = yield_df["region_id"].str.replace("IDN-", "")

    mapping = {}

    for code in sorted(gee_codes):
        code_str = str(code).strip()
        code_clean = code_str.replace(".", "")

        # 1. Manual override FIRST (to fix BPS↔BIG code conflicts)
        bdsp_code = MANUAL_CODE_OVERRIDE.get(code_clean)
        if bdsp_code and bdsp_code in yield_df["kab_code_raw"].values:
            mapping[code] = f"IDN-{bdsp_code}"
            continue

        # 2. Direct code match (dot-stripped)
        match = yield_df[yield_df["kab_code_raw"] == code_clean]
        if len(match) > 0:
            mapping[code] = match.iloc[0]["region_id"]
            continue

    # 3. Name-based matching for remaining unmatched
    unmapped_codes = gee_codes - set(mapping.keys())
    if unmapped_codes:
        yield_df["name_clean"] = (
            yield_df["region_name"]
            .str.lower()
            .str.replace("kab. ", "")
            .str.replace("kota ", "")
            .str.strip()
        )
        modis_name_lookup = dict(zip(
            modis_df[GEE_KAB_CODE_COL],
            modis_df[GEE_KAB_NAME_COL].str.lower().str.strip()
        ))

        for code in sorted(unmapped_codes):
            gee_name = modis_name_lookup.get(code, "")
            match = yield_df[yield_df["name_clean"] == gee_name]
            if len(match) > 0:
                mapping[code] = match.iloc[0]["region_id"]
                continue
            # Partial match (name contains or is contained)
            match = yield_df[
                yield_df["name_clean"].str.contains(gee_name.replace("-", " "), na=False)
                | yield_df["name_clean"].apply(lambda n: n in gee_name if isinstance(n, str) else False)
            ]
            if len(match) > 0:
                mapping[code] = match.iloc[0]["region_id"]

    unmapped = gee_codes - set(mapping.keys())
    if unmapped:
        logger.warning(f"  Unmapped GEE kab_codes ({len(unmapped)}): {sorted(unmapped)[:20]}...")

    return mapping


def build_tensor(
    modis_df: pd.DataFrame,
    yield_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Build (X, y, region_ids, years) arrays by matching MODIS rows with yield.

    Returns:
        X: (N, 46, 10) float32
        y: (N,) float32
        region_ids: list of N strings
        years: list of N ints
    """
    modis_df = modis_df.sort_values(["region_id", "year", "date"])

    X_list, y_list, rid_list, yr_list = [], [], [], []
    missing_yield, missing_ts = 0, 0

    yield_lookup = (
        yield_df.set_index(["region_id", "year"])["yield_ton_ha"]
        .to_dict()
    )

    for (region_id, year), grp in modis_df.groupby(["region_id", "year"]):
        if len(grp) != N_TIMESTEPS:
            missing_ts += 1
            continue

        y_val = yield_lookup.get((region_id, year))
        if y_val is None or pd.isna(y_val):
            missing_yield += 1
            continue

        x = grp[FEATURE_COLS].values.astype(np.float32)

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

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, rid_list, yr_list


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Indonesia Kabupaten MODIS Merge ===")

    files = sorted(RAW_MODIS.glob("modis_idn_*.csv"))
    if not files:
        logger.error(f"No modis_idn_*.csv found in {RAW_MODIS}")
        logger.error("Run 01_extract_modis_idn.py first, then download from Google Drive")
        sys.exit(1)

    logger.info(f"  Loading {len(files)} CSVs...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    logger.info(f"  Raw rows: {len(df):,}")

    yield_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "indonesia" / "yield_indonesia_kabupaten_2003_2023.csv"
    )
    logger.info(f"  Yield rows: {len(yield_df):,}, kabupaten: {yield_df['region_id'].nunique()}")

    mapping = build_kabupaten_mapping(df, yield_df)

    df["region_id"] = df[GEE_KAB_CODE_COL].map(mapping)
    before_map = len(df)
    df = df.dropna(subset=["region_id"])
    logger.info(f"  Mapped: {len(df):,} rows ({before_map - len(df)} unmapped dropped)")

    df = fill_feature_nans(df, "region_id")

    X, y, rids, yrs = build_tensor(df, yield_df)
    logger.info(f"  Tensor: X={X.shape}, y={y.shape}")
    logger.info(f"  Yield range: [{y.min():.2f}, {y.max():.2f}] t/ha")
    logger.info(f"  Regions: {len(set(rids))} kabupaten")
    logger.info(f"  Years: {sorted(set(yrs))}")

    # Drop zero/near-zero yield samples
    valid = y > 0.1
    n_dropped = (~valid).sum()
    if n_dropped:
        logger.info(f"  Dropping {n_dropped} samples with yield <= 0.1 t/ha")
        X = X[valid]
        y = y[valid]
        rids = np.array(rids)[valid].tolist()
        yrs = np.array(yrs)[valid].tolist()

    out = OUT_DIR / "idn_modis.npz"
    np.savez_compressed(out, X=X, y=y,
                        region_ids=np.array(rids), years=np.array(yrs))
    logger.info(f"  Saved → {out} ({len(y)} samples)")


if __name__ == "__main__":
    main()
