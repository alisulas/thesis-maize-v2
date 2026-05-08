# Vietnam Negative Transfer Analysis

*Generated: 2026-05-08*

## Overview

Test set: 2023, 63 provinces

| Model | R² | RMSE | Mean Error (bias) | Std Error |
|-------|-----|------|-------------------|----------|
| Transfer (USA→VNM) | -0.1855 | 1.4586 | -0.5825 t/ha | 1.3372 |
| From Scratch       | -0.0919 | 1.3998 | -0.6052 t/ha | 1.2622 |

## Hypothesis Testing

### (a) Province-specific errors

Top 5 worst-predicted provinces (transfer model):

| province   |   Mean |Error| (t/ha) |
|:-----------|----------------------:|
| Dong Thap  |               3.86881 |
| Kien Giang |               3.31436 |
| Dong Nai   |               3.30843 |
| An Giang   |               2.96078 |
| Da Nang    |               2.77441 |

### (b) Systematic bias

**YES — systematic bias detected.** Transfer model systematically under-predicted by 0.58 t/ha on average.

This is a classic domain shift symptom: the USA-trained model learned a relationship between spectral features and yield that doesn't hold in Vietnam's tropical climate.

### (c) Error distribution

- Error range: -3.87 to 2.41 t/ha
- 90th percentile |error|: 2.49 t/ha

## 10 Best Predictions (Transfer Model)

| province    |   actual_t_ha |   pred_transfer |   err_transfer |
|:------------|--------------:|----------------:|---------------:|
| Tuyen Quang |        4.617  |         4.60538 |     -0.0116186 |
| Quang Ninh  |        4.7119 |         4.68283 |     -0.0290723 |
| Son La      |        4.502  |         4.54227 |      0.040266  |
| Vinh Phuc   |        4.7333 |         4.66952 |     -0.0637755 |
| Kon Tum     |        4.34   |         4.19068 |     -0.149322  |
| Ha Tinh     |        4.6875 |         4.50091 |     -0.186588  |
| Nghe An     |        4.7024 |         4.47934 |     -0.223059  |
| Hoa Binh    |        4.7303 |         4.49255 |     -0.237746  |
| Lao Cai     |        3.8966 |         4.16032 |      0.263719  |
| Thai Nguyen |        4.9669 |         4.68874 |     -0.278159  |

## 10 Worst Predictions (Transfer Model)

| province   |   actual_t_ha |   pred_transfer |   err_transfer |
|:-----------|--------------:|----------------:|---------------:|
| Dong Thap  |        8.7083 |         4.83949 |       -3.86881 |
| Kien Giang |        8      |         4.68564 |       -3.31436 |
| Dong Nai   |        8.1652 |         4.85677 |       -3.30843 |
| An Giang   |        7.6545 |         4.69372 |       -2.96078 |
| Da Nang    |        7      |         4.22559 |       -2.77441 |
| Bac Lieu   |        7      |         4.41899 |       -2.58101 |
| Dak Nong   |        6.7525 |         4.23862 |       -2.51388 |
| Binh Duong |        2.3333 |         4.74625 |        2.41295 |
| Khanh Hoa  |        2.1765 |         4.49411 |        2.31761 |
| Dak Lak    |        6.3333 |         4.33272 |       -2.00059 |

## Root Cause Hypothesis

1. **Climate mismatch**: USA corn grows in temperate, rain-fed conditions. Vietnam corn is largely in mountainous northern provinces with monsoon climate. NDVI and LST seasonal patterns are fundamentally different (see seasonal_profiles.png).
2. **Scale mismatch**: USA county (~1,000 km²) vs Vietnam province (~5,000–15,000 km²). Province-level mean dilutes the crop signal further.
3. **No cropland masking**: Both USA and VNM include non-agricultural pixels. Domain gap in non-crop land cover (forest, water) adds unrelated spectral noise.
4. **Small fine-tuning set**: 1,189 VNM training samples (63 provinces × ~19 years) may be insufficient to overcome the pretrained USA bias.

**Recommendation for thesis**: This negative transfer *supports H2* (domain gap causes negative transfer). DANN or domain-invariant feature learning should mitigate this.
