"""Extract MODIS features for ASEAN provinces via Google Earth Engine.

Submits export tasks to Google Drive. Download CSVs after tasks complete.

Collections:
    MOD09A1 (8-day SR, 500m) → 7 bands + NDVI + EVI
    MYD11A2 (8-day LST, 1km) → LST_day + LST_night

Output per task (Drive folder: thesis_maize_gee):
    modis_{country}_{year}.csv
    Columns: region_id, region_name, country, year, date,
             b1-b7, ndvi, evi, lst_day, lst_night

Usage:
    # Test first (1 country, 1 year):
    python src/data/extract_modis_asean.py --test

    # Submit all tasks:
    python src/data/extract_modis_asean.py
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
MCD12Q1 = "MODIS/061/MCD12Q1"
GAUL_L1 = "FAO/GAUL/2015/level1"

DRIVE_FOLDER_V2 = "thesis_maize_gee_v2"  # output folder for cropland-masked extraction

SR_BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
SR_SCALE = 0.0001

LST_BANDS = ["LST_Day_1km", "LST_Night_1km"]
LST_SCALE = 0.02
LST_OFFSET = -273.15  # K → °C

# Countries: GAUL name, our ISO3, year range from yield data
COUNTRIES: dict[str, dict] = {
    "IDN": {
        "gaul_name": "Indonesia",
        "years": list(range(2020, 2025)),
    },
    "VNM": {
        "gaul_name": "Viet Nam",
        "years": list(range(2003, 2024)),  # MODIS reliable from 2003
    },
    "THA": {
        "gaul_name": "Thailand",
        "years": list(range(2021, 2024)),
    },
}


def get_cropland_mask(year: int) -> ee.Image:
    """MCD12Q1 IGBP class 12 (Croplands) binary mask for a given year.

    MCD12Q1 is available up to ~2022 in GEE; years beyond that use 2022.
    """
    lc_year = min(year, 2022)
    lc = (
        ee.ImageCollection(MCD12Q1)
        .filter(ee.Filter.calendarRange(lc_year, lc_year, "year"))
        .first()
    )
    return lc.select("LC_Type1").eq(12)


def preprocess_sr(image: ee.Image) -> ee.Image:
    """Scale MOD09A1 surface reflectance and add NDVI, EVI."""
    scaled = image.select(SR_BANDS).multiply(SR_SCALE)
    b1 = scaled.select("sur_refl_b01")  # Red
    b2 = scaled.select("sur_refl_b02")  # NIR
    b7 = scaled.select("sur_refl_b07")  # SWIR2

    ndvi = b2.subtract(b1).divide(b2.add(b1)).rename("ndvi")
    evi = (
        b2.subtract(b1)
        .multiply(2.5)
        .divide(b2.add(b1.multiply(6)).subtract(b7.multiply(7.5)).add(1))
        .rename("evi")
    )
    return (
        scaled.addBands(ndvi).addBands(evi)
        .copyProperties(image, ["system:time_start"])
    )


def preprocess_lst(image: ee.Image) -> ee.Image:
    """Scale MYD11A2 LST from Kelvin to Celsius."""
    return (
        image.select(LST_BANDS)
        .multiply(LST_SCALE)
        .add(LST_OFFSET)
        .copyProperties(image, ["system:time_start"])
    )


def get_joined_collection(year: int) -> ee.ImageCollection:
    """Return SR collection with LST joined by nearest date (±16 days)."""
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    sr_col = (
        ee.ImageCollection(MOD09A1)
        .filterDate(start, end)
        .map(preprocess_sr)
    )
    lst_col = (
        ee.ImageCollection(MYD11A2)
        .filterDate(start, end)
        .map(preprocess_lst)
    )

    # Temporal join: each SR image gets nearest LST within 16 days
    time_filter = ee.Filter.maxDifference(
        difference=16 * 24 * 60 * 60 * 1000,  # 16 days in ms
        leftField="system:time_start",
        rightField="system:time_start",
    )
    join = ee.Join.saveBest(matchKey="lst_match", measureKey="time_diff")
    joined = join.apply(sr_col, lst_col, time_filter)

    def merge_bands(feature) -> ee.Image:
        img = ee.Image(feature)  # Join returns FeatureCollection; cast to Image
        lst = ee.Image(img.get("lst_match"))
        return img.addBands(lst)

    return ee.ImageCollection(joined.map(merge_bands))


def build_export_task(
    country_iso: str,
    gaul_name: str,
    year: int,
    use_cropland_mask: bool = False,
) -> ee.batch.Task:
    """Build one GEE export task for country × year.

    Args:
        country_iso: ISO3 code (e.g. 'IDN').
        gaul_name: FAO GAUL country name (e.g. 'Indonesia').
        year: Calendar year.
        use_cropland_mask: If True, apply MCD12Q1 class-12 mask before aggregation.

    Returns:
        GEE export task (not yet started).
    """
    provinces = ee.FeatureCollection(GAUL_L1).filter(
        ee.Filter.eq("ADM0_NAME", gaul_name)
    )

    col = get_joined_collection(year)

    if use_cropland_mask:
        crop_mask = get_cropland_mask(year)
        col = col.map(lambda img: img.updateMask(crop_mask))

    folder = DRIVE_FOLDER_V2 if use_cropland_mask else DRIVE_FOLDER
    iso_lower = country_iso.lower()
    desc = f"modis_{iso_lower}_v2_{year}" if use_cropland_mask else f"modis_{iso_lower}_{year}"

    def reduce_to_provinces(image: ee.Image) -> ee.FeatureCollection:
        date_str = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=provinces,
            reducer=ee.Reducer.mean(),
            scale=500,
            crs="EPSG:4326",
        )
        return reduced.map(
            lambda f: f.set(
                "date", date_str,
                "year", year,
                "country", country_iso,
            )
        )

    result = col.map(reduce_to_provinces).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=desc,
        folder=folder,
        fileNamePrefix=f"modis_{iso_lower}_{year}",
        fileFormat="CSV",
        selectors=[
            "ADM1_CODE", "ADM1_NAME", "country", "year", "date",
            "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
            "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07",
            "ndvi", "evi",
            "LST_Day_1km", "LST_Night_1km",
        ],
    )
    return task


def main(test_mode: bool = False, use_cropland_mask: bool = False) -> None:
    logger.info(f"Initializing GEE project: {GEE_PROJECT}")
    ee.Initialize(project=GEE_PROJECT)

    if use_cropland_mask:
        logger.info("Cropland mask ENABLED → MCD12Q1 IGBP class 12")
        logger.info(f"Output Drive folder: {DRIVE_FOLDER_V2}")
    else:
        logger.info(f"Output Drive folder: {DRIVE_FOLDER}")

    if test_mode:
        logger.info("=== TEST MODE: IDN 2022 only ===")
        task = build_export_task("IDN", "Indonesia", 2022, use_cropland_mask=use_cropland_mask)
        task.start()
        logger.info(f"Task submitted: {task.id}")
        logger.info("Polling every 30s (Ctrl+C to stop and monitor manually)...")
        try:
            while task.active():
                status = task.status()["state"]
                logger.info(f"  Status: {status}")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
        logger.info(f"Final status: {task.status()['state']}")
        return

    # Submit all tasks
    tasks: list[tuple[str, int, ee.batch.Task]] = []

    for country_iso, cfg in COUNTRIES.items():
        gaul_name = cfg["gaul_name"]
        for year in cfg["years"]:
            task = build_export_task(country_iso, gaul_name, year, use_cropland_mask=use_cropland_mask)
            task.start()
            tasks.append((country_iso, year, task))
            logger.info(f"  Submitted: {country_iso} {year} → task {task.id}")
            time.sleep(0.5)

    logger.info(f"\nTotal tasks submitted: {len(tasks)}")
    logger.info(f"Monitor at: https://code.earthengine.google.com/tasks")

    suffix = "_v2" if use_cropland_mask else ""
    task_log = PROJECT_ROOT / "experiments" / "logs" / f"gee_tasks_asean{suffix}.csv"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"country": c, "year": y, "task_id": t.id, "status": "SUBMITTED"}
            for c, y, t in tasks]
    pd.DataFrame(rows).to_csv(task_log, index=False)
    logger.info(f"Task IDs saved → {task_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: submit IDN 2022 only and poll")
    parser.add_argument("--masked", action="store_true",
                        help="Apply MCD12Q1 cropland mask; outputs to thesis_maize_gee_v2")
    args = parser.parse_args()
    main(test_mode=args.test, use_cropland_mask=args.masked)
