# Cropland Mask Impact Estimation

*Generated: 2026-05-08 — estimated from literature, no GEE re-run needed*

## Iowa as Typical Corn Belt County

Iowa USDA CDL statistics (well-documented public data):

| Land cover class | % of state area |
|------------------|-----------------|
| Corn             | ~34%            |
| Soybean          | ~28%            |
| Total cropland   | ~67%            |
| Pasture + hay    | ~12%            |
| Forest/urban/water | ~21%          |

**Within a typical Iowa corn county**: corn occupies ~34–45% of pixels.

## Signal-to-Noise Analysis (Naive Estimate)

Without cropland mask, each pixel value is:
```
observed_mean = α × corn_signal + (1-α) × non_corn_signal
```
where α ≈ 0.35 (fraction of corn pixels).

If corn_signal ≠ non_corn_signal by ~2× (typical NDVI: corn peak 0.8, non-crop ~0.4):

| Scenario | NDVI mean | Corn contribution | Non-crop noise |
|----------|-----------|-------------------|----------------|
| Without mask (Iowa) | ~0.55 | 0.35 × 0.80 = 0.28 | 0.65 × 0.42 = 0.27 |
| With corn mask      | ~0.80 | 1.00 × 0.80 = 0.80 | 0 |

**Estimated SNR improvement: ~2.8× higher corn signal contribution after masking.**

## Expected R² Improvement

Literature benchmark (You et al. 2017, county-level USA with CDL mask): R² ≈ 0.70–0.75.
Our current result (no mask): R² ≈ 0.39 on test.

If we assume the gap is primarily due to masking:
- **Estimated R² with cropland mask: 0.60–0.70** (aligns with thesis target ≥0.60)

## ASEAN Cropland Fractions (MapSPAM 2020 estimates)

| Country | Province avg % maize pixels |
|---------|-----------------------------|
| Indonesia | 2–15% (highly variable; Java higher) |
| Vietnam | 5–25% (northern highlands higher) |
| Thailand | 3–20% (northeastern provinces higher) |

**ASEAN impact is even larger**: with only 2–15% of pixels being maize, current features are ~85–98% non-crop noise. Masking would dramatically improve signal quality.

## Recommended Fix

Re-run GEE extraction scripts with:
```python
# Add to preprocess_sr() in extract_modis_*.py:
landcover = ee.ImageCollection('MODIS/061/MCD12Q1')\
    .filterDate(f'{year}-01-01', f'{year}-12-31').first()
cropland_mask = landcover.select('LC_Type1').eq(12)  # class 12 = croplands
return scaled.addBands(ndvi).addBands(evi).updateMask(cropland_mask)\
    .copyProperties(image, ['system:time_start'])
```
This adds ~1 line per extraction script and requires re-submitting 50 GEE tasks (~1 day).
