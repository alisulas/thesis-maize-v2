---
name: data-validation
description: Use this skill whenever the task involves validating, cleaning, or quality-checking yield data, weather data, or processed datasets for the maize yield prediction thesis. Triggers include any mention of data validation, sanity check, data quality, outlier detection, missing data handling, cross-source validation (e.g., BPS vs FAOSTAT), data exploration, or generating summary statistics. Use this skill before training any model on a new dataset and after every data extraction step. Do NOT use for satellite data extraction (use gee-extraction) or model training (use pytorch-training).
---

# Data Validation Skill

This skill covers patterns for validating and quality-checking datasets in the maize yield prediction thesis project. **Skipping data validation is the #1 cause of unpublishable thesis results.**

## Why Data Validation Matters

In yield prediction, common silent errors include:
- Wrong unit conversions (bushels/acre vs ton/ha) → 2.5x systematic error
- Mismatched region codes (BPS vs GADM) → silent data loss
- Outlier yields from war zones, natural disasters → bias model
- Missing years filled with zero → trains model to predict zero
- Train/test leakage from year overlap → inflated accuracy

**Rule**: Never train a model on data you haven't validated.

## Standard Validation Workflow

For every new dataset, run these 5 stages:

### Stage 1: Schema Validation

```python
import pandas as pd
from typing import Optional

def validate_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    dtypes: Optional[dict] = None,
) -> dict:
    """Validate dataframe has required columns with correct types."""
    issues = []

    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        issues.append(f'Missing columns: {missing}')

    # Check dtypes
    if dtypes:
        for col, expected_dtype in dtypes.items():
            if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, expected_dtype):
                issues.append(f'Column {col}: expected {expected_dtype}, got {df[col].dtype}')

    return {'passed': len(issues) == 0, 'issues': issues}
```

### Stage 2: Range Validation

```python
def validate_yield_ranges(df: pd.DataFrame, country: str) -> dict:
    """Check yield values are in plausible range per country."""
    issues = []

    # Country-specific maize yield bounds (ton/ha)
    bounds = {
        'usa': (3.0, 15.0),       # USA corn belt: 8-12 typical
        'indonesia': (1.0, 8.0),  # Tropical: 4-6 typical
        'vietnam': (1.0, 7.0),    # Tropical: 4-5 typical
        'thailand': (0.5, 6.0),   # Tropical: 3-5 typical
    }

    if country not in bounds:
        issues.append(f'Unknown country bounds: {country}')
        return {'passed': False, 'issues': issues}

    lo, hi = bounds[country]
    out_of_range = df[(df['yield_ton_ha'] < lo) | (df['yield_ton_ha'] > hi)]
    if len(out_of_range) > 0:
        pct = len(out_of_range) / len(df) * 100
        issues.append(f'{len(out_of_range)} rows ({pct:.1f}%) out of range [{lo}, {hi}]')
        if pct > 5:
            issues.append('CRITICAL: >5% out of range. Check unit conversions.')

    return {'passed': len(issues) == 0, 'issues': issues}
```

### Stage 3: Temporal Coverage

```python
def validate_temporal_coverage(
    df: pd.DataFrame,
    expected_years: range,
    region_col: str = 'region_id',
    year_col: str = 'year',
) -> dict:
    """Verify each region has data for all expected years."""
    issues = []
    coverage = df.groupby(region_col)[year_col].nunique()
    expected = len(expected_years)
    incomplete = coverage[coverage < expected]
    if len(incomplete) > 0:
        issues.append(f'{len(incomplete)} regions with incomplete coverage:')
        for region, n_years in incomplete.head(10).items():
            issues.append(f'  - {region}: {n_years}/{expected} years')

    return {'passed': len(issues) == 0, 'issues': issues, 'coverage': coverage}
```

### Stage 4: Cross-Source Validation

```python
def cross_validate_with_faostat(
    local_df: pd.DataFrame,
    faostat_df: pd.DataFrame,
    country: str,
    tolerance: float = 0.15,
) -> dict:
    """Compare aggregated local data against FAOSTAT national totals."""
    issues = []

    # Aggregate local to national
    local_national = local_df.groupby('year').agg({
        'production_ton': 'sum',
        'planted_ha': 'sum',
    }).reset_index()
    local_national['yield_local'] = local_national['production_ton'] / local_national['planted_ha']

    # Compare
    merged = local_national.merge(
        faostat_df[['year', 'yield_faostat']],
        on='year',
    )
    merged['relative_diff'] = (merged['yield_local'] - merged['yield_faostat']).abs() / merged['yield_faostat']
    discrepant = merged[merged['relative_diff'] > tolerance]

    if len(discrepant) > 0:
        issues.append(f'{len(discrepant)} years with >{tolerance*100}% discrepancy vs FAOSTAT')
        for _, row in discrepant.iterrows():
            issues.append(
                f'  - Year {row["year"]}: local={row["yield_local"]:.2f}, '
                f'faostat={row["yield_faostat"]:.2f}, diff={row["relative_diff"]*100:.1f}%'
            )

    return {'passed': len(issues) == 0, 'issues': issues}
```

### Stage 5: Visual Inspection

```python
import matplotlib.pyplot as plt

def visualize_yield_trends(df: pd.DataFrame, country: str, save_dir: str) -> None:
    """Plot yield trends per region — visual sanity check."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. National average over time
    national = df.groupby('year')['yield_ton_ha'].mean()
    axes[0, 0].plot(national.index, national.values, marker='o')
    axes[0, 0].set_title(f'{country.upper()} National Avg Yield Trend')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Yield (ton/ha)')

    # 2. Distribution histogram
    axes[0, 1].hist(df['yield_ton_ha'], bins=50)
    axes[0, 1].set_title(f'{country.upper()} Yield Distribution')
    axes[0, 1].set_xlabel('Yield (ton/ha)')

    # 3. Top 10 regions
    top_regions = df.groupby('region_name')['yield_ton_ha'].mean().nlargest(10)
    axes[1, 0].barh(top_regions.index, top_regions.values)
    axes[1, 0].set_title('Top 10 Regions by Avg Yield')

    # 4. Heatmap region × year
    pivot = df.pivot_table(index='region_name', columns='year', values='yield_ton_ha')
    im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='viridis')
    axes[1, 1].set_title('Yield Heatmap (region × year)')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{country}_validation.png', dpi=100)
    plt.close()
```

## Standard Validation Output Format

Every validation script outputs a JSON report:

```python
import json
from datetime import datetime

def generate_validation_report(country: str, results: dict, output_path: str) -> None:
    report = {
        'country': country,
        'validated_at': datetime.now().isoformat(),
        'overall_passed': all(r['passed'] for r in results.values()),
        'stages': results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Validation report: {output_path}')
    print(f'Overall: {"PASSED" if report["overall_passed"] else "FAILED"}')
```

## Country-Specific Gotchas

### USA (USDA NASS)
- Yield reported in **bushels/acre** for corn. Convert to ton/ha:
  - `yield_ton_ha = yield_bu_acre * 0.0628`
- Some counties report "D" (suppressed for confidentiality). Treat as missing, NOT zero.
- "OTHER (COMBINED) COUNTIES" rows are aggregates — exclude from county-level analysis.

### Indonesia (BPS)
- Yield in **kuintal/ha** at some sources. Convert: `yield_ton_ha = yield_kuintal_ha / 10`
- Province name encoding: "DI YOGYAKARTA" vs "Daerah Istimewa Yogyakarta" — normalize.
- 2018+ uses KSA methodology (different from pre-2018) — note this in paper.
- Some provinces split (e.g., new provinces in Papua). Handle aggregation carefully.

### Vietnam (GSO)
- Yield often in **quintal/ha** (10 quintal = 1 ton).
- Province names in Vietnamese with diacritics — normalize using `unidecode`.
- "Whole country" rows present — exclude from province-level data.
- Recent administrative reorganization (2025) — use pre-reorganization boundaries for historical consistency.

### Thailand (OAE)
- Province names in Thai script — match against GADM English names carefully.
- Bangkok metro area has near-zero maize — exclude or treat carefully.
- Crop year vs calendar year confusion: OAE often uses "crop year" (e.g., 2020/2021).

## Outlier Detection Patterns

### Pattern 1: Statistical Outliers (per region)

```python
def flag_temporal_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Flag yield values >n_std from regional mean."""
    df = df.copy()
    df['yield_zscore'] = df.groupby('region_id')['yield_ton_ha'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df['is_outlier'] = df['yield_zscore'].abs() > n_std
    return df
```

### Pattern 2: Year-over-Year Anomalies

```python
def flag_yoy_jumps(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Flag >50% year-over-year changes (suspicious)."""
    df = df.sort_values(['region_id', 'year']).copy()
    df['yield_yoy_change'] = df.groupby('region_id')['yield_ton_ha'].pct_change()
    df['suspicious_yoy'] = df['yield_yoy_change'].abs() > threshold
    return df
```

### Pattern 3: Known Disaster Years

Document and exclude (or flag) known agricultural disasters:
- 2015-2016 El Niño drought (Indonesia, Thailand, Vietnam)
- 2019 Fall Armyworm outbreak (ASEAN-wide for maize)
- 2020 COVID disruption (variable impact)

```python
DISASTER_YEARS = {
    'el_nino_2015': {'years': [2015, 2016], 'countries': ['indonesia', 'thailand', 'vietnam']},
    'faw_2019': {'years': [2019], 'countries': ['indonesia', 'thailand', 'vietnam']},
}
```

**Decision**: Don't auto-exclude. Flag, then discuss in paper Limitations section.

## Missing Data Strategies

### Decision Tree

1. **Is missing < 5% of total?** → Drop those rows
2. **Is missing localized (specific region/year)?** → Investigate cause
3. **Is missing systematic (entire region, all years)?** → Drop region
4. **Is data MAR (missing at random)?** → Imputation may be OK
5. **Default**: Document missingness in paper, don't impute yield values

```python
def analyze_missingness(df: pd.DataFrame) -> dict:
    """Report missingness pattern."""
    total = len(df)
    missing_per_col = df.isnull().sum()
    return {
        'total_rows': total,
        'missing_by_column': missing_per_col[missing_per_col > 0].to_dict(),
        'pct_missing_by_column': (missing_per_col / total * 100).to_dict(),
        'rows_with_any_missing': df.isnull().any(axis=1).sum(),
    }
```

## Sanity Checks for Satellite Data

After GEE extraction, validate:

```python
def validate_modis_features(features: np.ndarray, country: str) -> dict:
    """Validate extracted MODIS feature tensor."""
    issues = []

    # Shape check: (n_regions*n_years, n_timesteps, n_bands, n_bins)
    if features.ndim != 4:
        issues.append(f'Wrong dim: expected 4, got {features.ndim}')

    # NaN check
    nan_pct = np.isnan(features).mean() * 100
    if nan_pct > 1.0:
        issues.append(f'High NaN rate: {nan_pct:.2f}%')

    # NDVI range (assuming index 7 is NDVI band, scaled 0-1)
    if features.shape[2] >= 8:
        ndvi = features[:, :, 7, :]  # adjust index based on your band order
        ndvi_min, ndvi_max = np.nanmin(ndvi), np.nanmax(ndvi)
        if ndvi_min < -1 or ndvi_max > 1:
            issues.append(f'NDVI out of [-1, 1]: [{ndvi_min}, {ndvi_max}]')

    # Histogram should sum to ~1 per (timestep, band) if normalized
    sums = features.sum(axis=-1)
    sum_dev = np.abs(sums - 1.0).mean()
    if sum_dev > 0.1:
        issues.append(f'Histogram normalization issue: avg deviation {sum_dev:.3f}')

    return {'passed': len(issues) == 0, 'issues': issues}
```

## Quick Validation Notebook Template

For exploratory validation, use this template in `notebooks/01_eda_yield.ipynb`:

```python
# Cell 1: Imports & load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/indonesia_yield.csv')

# Cell 2: Basic info
print(df.info())
print(df.describe())
print(df.head())

# Cell 3: Missing data
print(df.isnull().sum())

# Cell 4: Distribution
df['yield_ton_ha'].hist(bins=50)
plt.title('Yield Distribution')

# Cell 5: Trend
df.groupby('year')['yield_ton_ha'].mean().plot(marker='o')
plt.title('National Avg Yield Over Time')

# Cell 6: Per-region heatmap
pivot = df.pivot_table(index='province', columns='year', values='yield_ton_ha')
sns.heatmap(pivot, cmap='viridis')

# Cell 7: Outliers
outliers = df[df['yield_ton_ha'] > df['yield_ton_ha'].quantile(0.99)]
print(outliers)
```

## Validation Checklist for New Dataset

Before declaring data "ready for training":

- [ ] Schema validated (columns, dtypes)
- [ ] Range validated (per-country yield bounds)
- [ ] Temporal coverage complete (all expected years per region)
- [ ] Cross-source check passed (vs FAOSTAT or other authority)
- [ ] Visualizations generated and reviewed
- [ ] Outliers flagged and documented
- [ ] Missingness pattern analyzed
- [ ] Region names normalized (matched to GADM)
- [ ] Unit conversions verified (yield in ton/ha consistently)
- [ ] Validation report saved to `data/processed/{country}_validation.json`
- [ ] Decision documented in `data/README.md`

## Related Files in Project

- Validation scripts: `src/data/validate_*.py`
- Data documentation: `data/README.md`
- Notebooks: `notebooks/01_eda_yield.ipynb`
- Validation reports: `data/processed/*_validation.json`

## When NOT to Use This Skill

- Initial data download (use direct API/scraping scripts)
- Satellite data extraction (use `gee-extraction` skill)
- Model evaluation metrics (use `pytorch-training` skill)
- Statistical hypothesis testing for results (different domain)