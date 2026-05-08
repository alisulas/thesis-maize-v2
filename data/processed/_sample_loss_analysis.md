# Sample Loss Analysis
*Generated: 2026-05-08*

## 1. Raw Yield Rows vs Tensor Samples

| Country | Raw Yield Rows | Regions | Years | Tensor Samples | Tensor Regions | Dropped | % Loss | Primary Reason |
|---------|---------------|---------|-------|---------------|----------------|---------|--------|---------------|
| USA | 41,349 | 2280 | 2003–2025 | 32,296 | 2111 | 9,053 | 21.9% | Counties not growing corn (yield suppressed '(D)' or missing in NASS) |
| IDN | 190 | 38 | 2020–2024 | 162 | 33 | 28 | 14.7% | 5 provinces not in GAUL 2015 boundaries (4 new Papua + Kalimantan Utara) |
| VNM | 1,331 | 63 | 2003–2023 | 1,315 | 63 | 16 | 1.2% | Ha Tay province dropped (merged into Ha Noi 2008); 8 region-years with no yield label |
| THA | 126 | 43 | 2021–2023 | 126 | 43 | 0 | 0.0% | 34 provinces filtered out — yield data only covers 43 corn-producing provinces (76 GAUL total) |

### Notes
- **USA**: Many NASS rows are counties outside corn-producing areas, or counties where yield data exists in other years but MODIS extraction found no match.
- **IDN**: 5 provinces not in GAUL 2015 (new Papua provinces split in 2022 + Kalimantan Utara). These have minimal corn production.
- **VNM**: Ha Tay merged into Ha Noi administratively in 2008. GAUL 2015 still lists it but GSO does not.
- **THA**: OAE yield data only covers 43 corn-producing provinces out of 76. 34 GAUL provinces dropped (no yield label = no maize production).

## 2. USA Zero Yield Investigation

- **Total USA tensor samples**: 32,296
- **Samples with yield = 0.0 t/ha**: 206 (0.6%)
- **Samples with yield > 0 but < 0.5 t/ha** (very low): 0 (0.0%)

### Sample of zero-yield counties

|   region_id |   year |   yield |
|------------:|-------:|--------:|
|       12029 |   2003 |       0 |
|       12029 |   2004 |       0 |
|       12041 |   2004 |       0 |
|       12073 |   2004 |       0 |
|       12075 |   2004 |       0 |
|       12107 |   2005 |       0 |
|       16011 |   2003 |       0 |
|       16011 |   2004 |       0 |
|       16043 |   2004 |       0 |
|       16051 |   2003 |       0 |
|       16051 |   2004 |       0 |
|       26043 |   2003 |       0 |
|       26103 |   2007 |       0 |
|       31009 |   2007 |       0 |
|       31031 |   2004 |       0 |
|       31045 |   2004 |       0 |
|       31045 |   2007 |       0 |
|       31149 |   2007 |       0 |
|       35005 |   2003 |       0 |
|       35005 |   2004 |       0 |

### Zero-yield count by year

|      |   n_zero |
|-----:|---------:|
| 2003 |       51 |
| 2004 |       43 |
| 2005 |       36 |
| 2006 |       40 |
| 2007 |       26 |
| 2008 |        8 |
| 2009 |        2 |

### Decision
**REMOVE from training.** Zero yield from NASS can mean:
1. County planted corn but 100% crop failure (very rare, ~1-2 counties/year)
2. Data entry error / rounding in suppressed counties (more likely)
3. County harvested 0 acres (crop abandoned)

Recommendation: **filter out 206 zero-yield samples** before final training. This removes 0.0% of USA tensor samples. Also consider filtering yield > 16 t/ha (3 samples look like unit errors based on max=16.96 t/ha which is biologically possible but extreme).

## Domain Gap Distance Metrics

### NDVI (mean per region-year)

| Pair | KL(A→B) | KL(B→A) | Wasserstein |
|------|---------|---------|-------------|
| USA↔IDN | 10.7087 | 1.0055 | 0.1302 |
| USA↔VNM | 1.0288 | 0.2253 | 0.0412 |
| USA↔THA | 11.5118 | 0.8747 | 0.1001 |
| IDN↔VNM | 2.8467 | 11.2331 | 0.1112 |

### LST Daytime (mean per region-year)

| Pair | KL(A→B) | Wasserstein |
|------|---------|-------------|
| USA↔IDN | 18.2761 | 9.1756 |
| USA↔VNM | 16.5390 | 9.2246 |
| USA↔THA | 18.9611 | 12.2352 |
| IDN↔VNM | 0.4011 | 0.4169 |

**Interpretation:** Higher Wasserstein distance = larger domain gap.
NDVI Wasserstein: USA↔IDN probably largest (tropical evergreen vs temperate seasonal).
