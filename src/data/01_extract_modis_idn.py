"""Extract MODIS features for Indonesia kabupaten via Google Earth Engine.

User must first upload Kemendagri shapefile to GEE as an asset.
Reference: https://map.kemendagri.go.id/ → Zipped Shapefile → upload ke GEE

Usage:
    python src/data/01_extract_modis_idn.py --test       # 1 province, 1 year
    python src/data/01_extract_modis_idn.py              # all years 2003–2023
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import ee

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── GEE Configuration ───────────────────────────────────────────────────────
GEE_PROJECT = "alamat-413120"
DRIVE_FOLDER = "thesis_maize_idn"

# User-uploaded Kemendagri kabupaten boundaries (update after upload)
KEMENDAGRI_ASSET = "projects/alamat-413120/assets/lapakgis"
KAB_CODE_COL = "KDPKAB"     # kabupaten code (format: "11.01")
KAB_NAME_COL = "WADMKK"     # kabupaten name (e.g. "Aceh Selatan")

# ─── MODIS Datasets ──────────────────────────────────────────────────────────
MOD09A1 = "MODIS/061/MOD09A1"
MYD11A2 = "MODIS/061/MYD11A2"

SR_BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
LST_BANDS = ["LST_Day_1km", "LST_Night_1km"]

YEARS = list(range(2003, 2024))   # 2003–2023


def preprocess_sr(image: ee.Image) -> ee.Image:
    scaled = image.select(SR_BANDS).multiply(0.0001)
    b1, b2, b7 = (scaled.select(b) for b in
                  ["sur_refl_b01", "sur_refl_b02", "sur_refl_b07"])
    ndvi = b2.subtract(b1).divide(b2.add(b1)).rename("ndvi")
    return scaled.addBands(ndvi).copyProperties(
        image, ["system:time_start"]
    )


def preprocess_lst(image: ee.Image) -> ee.Image:
    return (
        image.select(LST_BANDS).multiply(0.02).add(-273.15)
        .copyProperties(image, ["system:time_start"])
    )


def get_joined_collection(year: int) -> ee.ImageCollection:
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    sr_col = ee.ImageCollection(MOD09A1).filterDate(start, end).map(preprocess_sr)
    lst_col = ee.ImageCollection(MYD11A2).filterDate(start, end).map(preprocess_lst)

    time_filter = ee.Filter.maxDifference(
        difference=16 * 24 * 60 * 60 * 1000,
        leftField="system:time_start",
        rightField="system:time_start",
    )
    joined = ee.Join.saveBest("lst_match", "time_diff").apply(
        sr_col, lst_col, time_filter
    )

    def merge(feature) -> ee.Image:
        img = ee.Image(feature)
        return img.addBands(ee.Image(img.get("lst_match")))

    return ee.ImageCollection(joined.map(merge))


def get_kabupaten() -> ee.FeatureCollection:
    """Load Kemendagri kabupaten boundaries from user asset."""
    fc = ee.FeatureCollection(KEMENDAGRI_ASSET)
    n = fc.size().getInfo()
    logger.info(f"  Kabupaten loaded: {n} features from {KEMENDAGRI_ASSET}")
    return fc


def build_export_task(year: int, kabupaten: ee.FeatureCollection) -> ee.batch.Task:
    col = get_joined_collection(year)

    def reduce_to_kabupaten(image: ee.Image) -> ee.FeatureCollection:
        date_str = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=kabupaten,
            reducer=ee.Reducer.mean(),
            scale=500,
            crs="EPSG:4326",
        )
        return reduced.map(lambda f: f.set("date", date_str, "year", year))

    result = col.map(reduce_to_kabupaten).flatten()

    return ee.batch.Export.table.toDrive(
        collection=result,
        description=f"modis_idn_{year}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"modis_idn_{year}",
        fileFormat="CSV",
        selectors=[
            KAB_CODE_COL, KAB_NAME_COL, "year", "date",
            "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07",
            "ndvi",
            "LST_Day_1km", "LST_Night_1km",
        ],
    )


def main(test_mode: bool = False) -> None:
    logger.info(f"Initializing GEE project: {GEE_PROJECT}")
    ee.Initialize(project=GEE_PROJECT)

    logger.info(f"Loading boundary: {KEMENDAGRI_ASSET}")
    kabupaten = get_kabupaten()

    if test_mode:
        logger.info("=== TEST MODE: Jawa Barat 2020 ===")
        # Filter to one province for testing
        test_fc = kabupaten.filter(ee.Filter.stringStartsWith(KAB_CODE_COL, "32"))
        logger.info(f"  Test kabupaten: {test_fc.size().getInfo()}")
        task = build_export_task(2020, test_fc)
        task.start()
        logger.info(f"  Task submitted: {task.id}")
        logger.info("  Polling every 30s (Ctrl+C to stop)...")
        try:
            while task.active():
                logger.info(f"    Status: {task.status()['state']}")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
        logger.info(f"  Final: {task.status()['state']}")
        return

    tasks: list[tuple[int, ee.batch.Task]] = []
    for year in YEARS:
        task = build_export_task(year, kabupaten)
        task.start()
        tasks.append((year, task))
        logger.info(f"  Submitted: IDN {year} → task {task.id}")
        time.sleep(0.5)

    logger.info(f"\nTotal tasks: {len(tasks)}")
    logger.info("Monitor: https://code.earthengine.google.com/tasks")

    task_log = PROJECT_ROOT / "experiments" / "logs" / "gee_tasks_idn.csv"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([{"year": y, "task_id": t.id, "status": "SUBMITTED"}
                  for y, t in tasks]).to_csv(task_log, index=False)
    logger.info(f"Task IDs saved → {task_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(test_mode=args.test)
