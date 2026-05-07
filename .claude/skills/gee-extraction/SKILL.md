---
name: gee-extraction
description: Use this skill whenever the task involves Google Earth Engine (GEE) operations for extracting MODIS satellite data, NDVI/EVI time series, or remote sensing features for crop yield prediction. Triggers include any mention of MODIS, GEE, Google Earth Engine, satellite imagery extraction, NDVI extraction, vegetation indices, surface reflectance, land surface temperature, cropland masks, or aggregating satellite data per administrative boundary (county, province, district). Use this skill before writing any GEE Python API code. Do NOT use for general remote sensing concepts unrelated to actual data extraction.
---

# Google Earth Engine Extraction Skill

This skill covers patterns and best practices for extracting MODIS satellite data via Google Earth Engine Python API for the maize yield prediction thesis project.

## Critical Constraints (Read First)

- **Memory limits**: GEE has strict memory limits per computation. Always batch by year and region.
- **Computation timeout**: Default 5 minutes; complex aggregations will fail silently.
- **Export limits**: Free tier ~250,000 rows per export. Plan exports accordingly.
- **Projection issues**: ALWAYS specify CRS explicitly (`'EPSG:4326'`) to avoid silent reprojection errors.
- **Authentication**: GEE requires `ee.Authenticate()` once per environment, then `ee.Initialize(project='your-project-id')`.

## Standard Asset IDs for This Project

```python
# MODIS Surface Reflectance (8-day, 500m, 7 bands) — primary input
MOD09A1 = "MODIS/061/MOD09A1"

# MODIS Land Surface Temperature (8-day, 1km) — secondary input
MYD11A2 = "MODIS/061/MYD11A2"

# MODIS Land Cover (annual, 500m) — for cropland masking
MCD12Q1 = "MODIS/061/MCD12Q1"

# CHIRPS Precipitation (daily, 5km) — optional weather feature
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"

# ERA5 Reanalysis (monthly, 27km) — alternative weather
ERA5 = "ECMWF/ERA5_LAND/MONTHLY_AGGR"

# GADM administrative boundaries
GADM_LEVEL1 = "FAO/GAUL/2015/level1"  # Province/State
GADM_LEVEL2 = "FAO/GAUL/2015/level2"  # County/District
```

## Standard Workflow Pattern

Every GEE extraction follows this 5-step pattern:

```python
import ee
ee.Initialize(project='your-project-id')

# 1. Define spatial filter (region of interest)
region = ee.FeatureCollection(GADM_LEVEL2).filter(
    ee.Filter.eq('ADM0_NAME', 'United States of America')
)

# 2. Define temporal filter (date range)
start_date = '2020-01-01'
end_date = '2020-12-31'

# 3. Load and filter image collection
collection = (
    ee.ImageCollection(MOD09A1)
    .filterDate(start_date, end_date)
    .filterBounds(region)
)

# 4. Apply preprocessing (cloud masking, scaling)
def preprocess_modis(image):
    # Apply scale factor
    bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03',
             'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07']
    return image.select(bands).multiply(0.0001).copyProperties(
        image, ['system:time_start']
    )

processed = collection.map(preprocess_modis)

# 5. Aggregate per region
def aggregate_region(image):
    stats = image.reduceRegions(
        collection=region,
        reducer=ee.Reducer.mean(),
        scale=500,
        crs='EPSG:4326'
    )
    return stats.map(lambda f: f.set('date', image.date().format('YYYY-MM-dd')))

result = processed.map(aggregate_region).flatten()

# Export to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=result,
    description='usa_modis_2020',
    folder='thesis_data',
    fileFormat='CSV'
)
task.start()
```

## Histogram Conversion (You et al. 2017 Method)

This project requires converting per-pixel reflectance values into histograms for CNN input. The standard configuration:

- **Bins**: 32 per band
- **Bands**: 9 (7 reflectance + 2 LST day/night)
- **Timesteps**: 30 (8-day composites covering growing season)
- **Output shape**: `(timesteps, bands, bins)` = `(30, 9, 32)` per region-year

```python
def compute_histogram(image, region, num_bins=32):
    """Compute pixel value histogram per band per region."""
    histogram = image.reduceRegion(
        reducer=ee.Reducer.fixedHistogram(min=0, max=10000, steps=num_bins),
        geometry=region.geometry(),
        scale=500,
        maxPixels=1e9
    )
    return histogram
```

## Cropland Masking

Apply maize-specific cropland mask before aggregation. Two options:

**Option 1: MODIS Land Cover (simpler, lower resolution)**
```python
landcover = ee.ImageCollection(MCD12Q1).filter(ee.Filter.eq('system:index', '2020_01_01')).first()
cropland_mask = landcover.select('LC_Type1').eq(12)  # Class 12 = Croplands
masked_image = original_image.updateMask(cropland_mask)
```

**Option 2: USDA Cropland Data Layer (USA only, very accurate)**
```python
# CDL has maize as class 1
cdl = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.calendarRange(2020, 2020, 'year')).first()
maize_mask = cdl.select('cropland').eq(1)
```

**For ASEAN**: Use MapSPAM 2020 or MODIS Land Cover (cropland class) since CDL is USA-only.

## Region Filtering for This Project

```python
# USA county-level (filter to corn-belt states for efficiency)
corn_belt_states = ['Iowa', 'Illinois', 'Indiana', 'Nebraska', 'Minnesota',
                    'Ohio', 'Wisconsin', 'Missouri', 'South Dakota', 'Kansas']
usa_counties = ee.FeatureCollection(GADM_LEVEL2).filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'United States of America'),
        ee.Filter.inList('ADM1_NAME', corn_belt_states)
    )
)

# Indonesia provinces
indonesia_provinces = ee.FeatureCollection(GADM_LEVEL1).filter(
    ee.Filter.eq('ADM0_NAME', 'Indonesia')
)

# Vietnam provinces
vietnam_provinces = ee.FeatureCollection(GADM_LEVEL1).filter(
    ee.Filter.eq('ADM0_NAME', 'Viet Nam')  # Note: spelling in GADM
)

# Thailand provinces
thailand_provinces = ee.FeatureCollection(GADM_LEVEL1).filter(
    ee.Filter.eq('ADM0_NAME', 'Thailand')
)
```

## Common Pitfalls & Solutions

### Pitfall 1: Silent Memory Failures
**Symptom**: Export task says "complete" but output is empty/truncated.
**Solution**: Reduce scope. Process one year at a time, one country at a time.

### Pitfall 2: Wrong Coordinate System
**Symptom**: Aggregations don't match administrative boundaries.
**Solution**: Always pass `crs='EPSG:4326'` explicitly to `reduceRegions`.

### Pitfall 3: Cloud Contamination in Tropical Regions
**Symptom**: NDVI values look noisy/inconsistent for ASEAN data.
**Solution**: Apply MODIS QA bits to mask clouds:
```python
def mask_modis_clouds(image):
    qa = image.select('StateQA')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)  # Bit 10: cloud state
    return image.updateMask(cloud_mask)
```

### Pitfall 4: Mismatched Band Names Between MODIS Versions
**Symptom**: KeyError when selecting bands.
**Solution**: Use `MODIS/061/...` (Collection 6.1) consistently throughout project.

### Pitfall 5: Asynchronous Export Tasks
**Symptom**: Script finishes but data isn't in Drive yet.
**Solution**: Monitor task status:

```python
import time
while task.active():
    print(f'Polling for task: {task.id}')
    time.sleep(30)
print(f'Task done: {task.status()}')
```

## Output Format Standards for This Project

Every extraction script outputs CSV with these columns:
region_id, region_name, country, year, doy_8day,
b1, b2, b3, b4, b5, b6, b7,    # 7 reflectance bands
lst_day, lst_night,              # 2 temperature bands
ndvi, evi                        # 2 vegetation indices

After CSV download, convert to histogram tensors with:
```python
# Output shape: (n_regions, n_timesteps, n_bands, n_bins)
hist_tensor = compute_histograms_from_csv('usa_modis_2020.csv', num_bins=32)
np.save(f'data/processed/usa_2020_histograms.npy', hist_tensor)
```

## Performance Tips

1. **Always test on small region first**: Single county or province, single year. If it works, scale up.
2. **Use `.aside(print)` for debugging**: GEE's lazy evaluation makes debugging tricky. `.aside(print, 'message')` lets you log intermediate values.
3. **Batch exports per year**: One export task per year per country. Don't try to export 21 years × 4 countries in one task.
4. **Save intermediate results**: After each successful extraction, immediately save raw CSV to `data/raw/` and never modify it.

## Validation Checklist (Run After Every Extraction)

- [ ] Number of unique regions matches expected (e.g., USA corn-belt = ~700 counties)
- [ ] Number of timesteps per region-year is consistent (8-day composites = 46 per year)
- [ ] No nulls in critical columns (region_id, year, b1-b7)
- [ ] NDVI values in expected range [-1, 1]
- [ ] Plot one region's NDVI time series to visually verify seasonal pattern
- [ ] Cross-check region count against known administrative boundary count

## Related Files in Project

- Extraction scripts: `src/data/extract_modis.py`, `src/data/extract_weather.py`
- Histogram conversion: `src/data/histograms.py`
- Validation: `src/data/validate_satellite.py`
- Notebooks: `notebooks/02_gee_extraction.ipynb`

## When NOT to Use This Skill

- General remote sensing theory questions (no GEE code involved)
- Visualization-only tasks (use matplotlib/folium directly)
- Non-satellite data (yield CSVs, weather station data) — use `data-validation` skill instead
