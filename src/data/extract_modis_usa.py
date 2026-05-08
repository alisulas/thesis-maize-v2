"""Extract MODIS features for USA counties via Google Earth Engine.

Submits export tasks to Google Drive. One CSV per year.

Usage:
    python src/data/extract_modis_usa.py --test    # 1 state, 1 year
    python src/data/extract_modis_usa.py           # all years
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import ee
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

GEE_PROJECT = "alamat-413120"
DRIVE_FOLDER = "thesis_maize_gee"

MOD09A1 = "MODIS/061/MOD09A1"
MYD11A2 = "MODIS/061/MYD11A2"
TIGER_COUNTIES = "TIGER/2018/Counties"

SR_BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
LST_BANDS = ["LST_Day_1km", "LST_Night_1km"]

# Years from yield data (2003-2023 for training, 2024-2025 are bonus)
YEARS = list(range(2003, 2024))


def preprocess_sr(image: ee.Image) -> ee.Image:
    scaled = image.select(SR_BANDS).multiply(0.0001)
    b1, b2, b7 = (scaled.select(b) for b in
                  ["sur_refl_b01", "sur_refl_b02", "sur_refl_b07"])
    ndvi = b2.subtract(b1).divide(b2.add(b1)).rename("ndvi")
    evi = (b2.subtract(b1).multiply(2.5)
           .divide(b2.add(b1.multiply(6)).subtract(b7.multiply(7.5)).add(1))
           .rename("evi"))
    return scaled.addBands(ndvi).addBands(evi).copyProperties(
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
    joined = ee.Join.saveBest("lst_match", "time_diff").apply(sr_col, lst_col, time_filter)

    def merge(feature) -> ee.Image:
        img = ee.Image(feature)  # Join returns FeatureCollection; cast to Image
        return img.addBands(ee.Image(img.get("lst_match")))

    return ee.ImageCollection(joined.map(merge))


def get_usa_counties(state_fips_list: list[str]) -> ee.FeatureCollection:
    """Return TIGER county boundaries for corn-producing states only."""
    return ee.FeatureCollection(TIGER_COUNTIES).filter(
        ee.Filter.inList("STATEFP", state_fips_list)
    )


def build_export_task(
    year: int,
    counties: ee.FeatureCollection,
) -> ee.batch.Task:
    col = get_joined_collection(year)

    def reduce_to_counties(image: ee.Image) -> ee.FeatureCollection:
        date_str = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=counties,
            reducer=ee.Reducer.mean(),
            scale=500,
            crs="EPSG:4326",
        )
        return reduced.map(lambda f: f.set("date", date_str, "year", year))

    result = col.map(reduce_to_counties).flatten()

    return ee.batch.Export.table.toDrive(
        collection=result,
        description=f"modis_usa_{year}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"modis_usa_{year}",
        fileFormat="CSV",
        selectors=[
            "GEOID", "NAME", "STATEFP", "year", "date",
            "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07",
            "ndvi", "evi",
            "LST_Day_1km", "LST_Night_1km",
        ],
    )


def main(test_mode: bool = False) -> None:
    logger.info(f"Initializing GEE project: {GEE_PROJECT}")
    ee.Initialize(project=GEE_PROJECT)

    # Load state FIPS from yield data
    yield_path = PROJECT_ROOT / "data" / "processed" / "usa" / "yield_usa_2003_2023.parquet"
    yield_df = pd.read_parquet(yield_path)
    state_fips = yield_df["region_id"].str[:2].unique().tolist()
    logger.info(f"Corn-producing states: {len(state_fips)} states → filtering TIGER counties")

    counties = get_usa_counties(state_fips)

    if test_mode:
        # Test: Iowa only (STATEFP=19), year 2020
        logger.info("=== TEST MODE: Iowa 2020 only ===")
        iowa_counties = ee.FeatureCollection(TIGER_COUNTIES).filter(
            ee.Filter.eq("STATEFP", "19")
        )
        task = build_export_task(2020, iowa_counties)
        task.start()
        logger.info(f"Task submitted: {task.id}")
        logger.info("Polling every 30s (Ctrl+C to stop)...")
        try:
            while task.active():
                logger.info(f"  Status: {task.status()['state']}")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
        logger.info(f"Final: {task.status()['state']}")
        return

    # Submit all years
    tasks: list[tuple[int, ee.batch.Task]] = []
    for year in YEARS:
        task = build_export_task(year, counties)
        task.start()
        tasks.append((year, task))
        logger.info(f"  Submitted: USA {year} → task {task.id}")
        time.sleep(0.5)

    logger.info(f"\nTotal tasks: {len(tasks)}")
    logger.info("Monitor: https://code.earthengine.google.com/tasks")

    task_log = PROJECT_ROOT / "experiments" / "logs" / "gee_tasks_usa.csv"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"year": y, "task_id": t.id, "status": "SUBMITTED"}
                  for y, t in tasks]).to_csv(task_log, index=False)
    logger.info(f"Task IDs saved → {task_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(test_mode=args.test)
