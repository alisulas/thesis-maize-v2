# Knowledge Base: Data Pipeline — USA

Dokumen ini menjelaskan pipeline data dari awal sampai siap training, fokus ke USA sebagai source domain. Ditulis dari hasil diskusi mendetail.

---

## Daftar Isi

1. [download_usa.py — Dapat Label Yield](#1-download_usapy--dapat-label-yield)
2. [Apa itu county? Kenapa county level?](#2-apa-itu-county-kenapa-county-level)
3. [Angka 41.349 dari mana?](#3-angka-41349-dari-mana)
4. [extract_modis_usa.py — Dapat Fitur Satelit](#4-extract_modis_usapy--dapat-fitur-satelit)
5. [Kenapa MOD09A1 + MYD11A2?](#5-kenapa-mod09a1--myd11a2)
6. [7 Band Reflektansi MODIS](#6-7-band-reflektansi-modis)
7. [Proses GEE: Step-by-Step](#7-proses-gee-step-by-step)
8. [Kenapa 46 timestep? Kenapa rata-rata per county?](#8-kenapa-46-timestep-kenapa-rata-rata-per-county)
9. [10 Fitur Final](#9-10-fitur-final)
10. [Yang Tidak Dipakai](#10-yang-tidak-dipakai)

---

## 1. download_usa.py — Dapat Label Yield

### Cara kerja

| Tahap | Apa yang terjadi |
|-------|-----------------|
| **Input** | NASS API key dari `.env` |
| **Query** | USDA NASS QuickStats API: `COMMODITY=CORN`, `STATISTIC=YIELD`, `LEVEL=COUNTY`, `2003–2023` |
| **Cleaning** | Buang "OTHER (COMBINED)" aggregate rows → suppressed values "(D)" → NaN → bangun 5-digit FIPS `region_id` |
| **Konversi** | `bu/acre × 0.06277 = ton/ha` |
| **Output** | `data/processed/usa/yield_usa_2003_2023.parquet` |

### Output detail

- **Baris**: 41.349
- **County**: 2.280
- **Tahun**: 2003–2023 (21 tahun)
- **Kolom**: `region_id, region_name, state, county_name, year, yield_bu_acre, yield_ton_ha, country, data_source`

---

## 2. Apa itu county? Kenapa county level?

**County** = level administrasi di bawah state (provinsi). Struktur USA:

```
Negara → State (50) → County (~3.143)
```

County setara dengan **kabupaten** di Indonesia.

### Kenapa county, bukan level lain?

| Level | Kenapa tidak |
|-------|-------------|
| Per petani/farm | USDA tidak publikasi data farm-level (privasi) |
| Per state | Terlalu kasar — satu state bisa punya iklim mikro berbeda (Illinois utara vs selatan) |
| **Per county** | Granularitas terkecil yang dipublikasikan USDA secara konsisten → **yang dipakai** |
| Sub-county | Tidak ada data yield sub-county; MODIS 500m juga kurang resolusi |

---

## 3. Angka 41.349 dari mana?

```
2.280 county × 21 tahun = 47.880 (seharusnya)
Yang ada: 41.349 baris
Hilang:    ~6.500 baris
```

### Kenapa hilang?

USDA NASS **menyensor** (suppress) data county yang cuma punya sedikit petani jagung — demi melindungi privasi bisnis. Data yang disensor ditulis "(D)" di API response. Ini normal dan expected.

Selain itu, ada county yang nanam jagung di beberapa tahun tapi tidak di tahun lain (rotasi tanaman).

---

## 4. extract_modis_usa.py — Dapat Fitur Satelit

### Input

1. `yield_usa_2003_2023.parquet` — dipakai untuk **dapat daftar state FIPS** yang nanam jagung
2. GEE authentication (project `alamat-413120`)

### Kenapa baca parquet buat dapat state FIPS?

GEE perlu tahu county mana saja yang harus diambil citra satelitnya. Kalau diambil semua 3.143 county USA, buang-buang komputasi. Jadi script-nya:

```
region_id = "17031"  → ambil 2 digit pertama = "17" = Illinois
region_id = "19001"  → "19" = Iowa
...
```

Hasilnya: daftar ~48 state corn-producing. Lalu GEE filter TIGER county boundaries hanya untuk state-state ini.

### Output

21 CSV di Google Drive folder `thesis_maize_gee`: `modis_usa_2003.csv` ... `modis_usa_2023.csv`

Tiap CSV berisi:
```
GEOID, NAME, STATEFP, year, date,
sur_refl_b01, sur_refl_b02, sur_refl_b03, sur_refl_b04,
sur_refl_b05, sur_refl_b06, sur_refl_b07,
ndvi, evi,
LST_Day_1km, LST_Night_1km
```

~128.000 baris per tahun (46 timestep × ~2.800 county).

---

## 5. Kenapa MOD09A1 + MYD11A2?

Tanaman jagung "terlihat" dari satelit lewat dua sinyal utama:

| Sinyal | Satelit | Jam lewat | Apa yang diukur | Kenapa penting |
|--------|---------|-----------|-----------------|----------------|
| **Reflektansi** | MOD09A1 (Terra) | ~10:30 pagi | Seberapa banyak cahaya dipantulkan tanaman di 7 panjang gelombang | Tanaman sehat = banyak pantul NIR, banyak serap Red → NDVI |
| **Suhu permukaan** | MYD11A2 (Aqua) | ~13:30 siang | Suhu tanah/daun (°C), siang & malam | Jagung stres kalau >35°C → hasil panen turun |

### Kenapa dua satelit?

- **Terra** lewat pagi: cahaya optimal, atmosfer lebih bersih → ideal buat reflektansi
- **Aqua** lewat siang bolong: suhu puncak harian → ideal buat LST

Keduanya digabung (join) per 8-hari sehingga satu sampel punya fitur spektral + termal lengkap.

### Kenapa tidak data lain?

- **Sentinel-2**: Resolusi lebih tinggi (10m) tapi cuma tersedia sejak 2015 — nggak bisa 2003-2023
- **Landsat**: Resolusi 30m tapi siklus 16 hari — terlalu jarang untuk time series 8-harian
- **MODIS**: Resolusi 500m, gratis, tersedia kontinu sejak 2003 — ideal untuk 21 tahun time series per county

---

## 6. 7 Band Reflektansi MODIS

MODIS MOD09A1 punya **7 sensor** (band), bukan cuma RGB seperti kamera:

| Band | Nama | Panjang gelombang | Fungsi |
|------|------|-------------------|--------|
| b01 | Red | 620–670 nm | Diserap tanaman sehat (klorofil) |
| b02 | NIR | 841–876 nm | Dipantulkan kuat oleh daun sehat |
| b03 | Blue | 459–479 nm | Koreksi atmosfer |
| b04 | Green | 545–565 nm | Referensi vegetasi |
| b05 | SWIR 1 | 1230–1250 nm | Kelembaban daun & tanah |
| b06 | SWIR 2 | 1628–1652 nm | Kelembaban |
| b07 | SWIR 3 | 2105–2155 nm | Kelembaban |

**SWIR** (Short-Wave Infrared) penting karena sensitif ke kadar air di daun dan tanah — indikasi kekeringan yang nggak terlihat dari NDVI.

### Kenapa dikali 0.0001?

MODIS menyimpan reflektansi sebagai integer 0–10.000 (bukan 0–1):

```
Nilai mentah: 0–10.000  →  × 0.0001  →  0.0–1.0 (reflektansi sebenarnya)
```

---

## 7. Proses GEE: Step-by-Step

Untuk **satu tahun** (contoh: 2020):

### Step 1: Ambil 46 gambar MOD09A1

```
46 komposit 8-harian sepanjang 2020
Jan 1-8, Jan 9-16, ..., Dec 25-31
```

### Step 2: Preprocessing reflektansi → `preprocess_sr()`

```
Input:  1 gambar MOD09A1 mentah (7 band, nilai 0–10.000)

1. Scale 7 band × 0.0001                   → nilai 0–1
2. Hitung NDVI = (b02 - b01) / (b02 + b01) → vegetasi index
3. Hitung EVI = formula kompleks           → vegetasi index (alternatif)

Output: 1 gambar dengan 9 band
        [b01, b02, b03, b04, b05, b06, b07, ndvi, evi]
```

Kode:
```python
scaled = image.select(SR_BANDS).multiply(0.0001)
b1, b2, b7 = scaled.select("sur_refl_b01"), ...
ndvi = b2.subtract(b1).divide(b2.add(b1)).rename("ndvi")
evi = (b2.subtract(b1).multiply(2.5)
       .divide(b2.add(b1.multiply(6)).subtract(b7.multiply(7.5)).add(1))
       .rename("evi")
return scaled.addBands(ndvi).addBands(evi)
```

### Step 3: Ambil 46 gambar MYD11A2 + preprocessing

```
Input:  1 gambar MYD11A2 mentah (LST_Day, LST_Night, format: Kelvin × 50)

Konversi: nilai × 0.02 - 273.15  →  Celsius

Contoh: 15000 × 0.02 - 273.15 = 26.85°C

Output: 1 gambar dengan 2 band (LST_Day, LST_Night) dalam °C
```

Kode:
```python
image.select(LST_BANDS).multiply(0.02).add(-273.15)
```

### Step 4: Join Terra + Aqua → `get_joined_collection()`

```
Untuk setiap timestep (1-8 Jan, 9-16 Jan, ...):

"Cari gambar Aqua yang waktunya paling dekat dengan gambar Terra ini
 (maksimal selisih 16 hari)"

Gabung → 1 gambar dengan 11 band:
[b01, b02, b03, b04, b05, b06, b07, ndvi, evi, LST_Day, LST_Night]
```

Kode:
```python
join = ee.Join.saveBest("lst_match", "time_diff")
joined = join.apply(sr_col, lst_col, time_filter)

def merge(feature):
    img = ee.Image(feature)
    return img.addBands(ee.Image(img.get("lst_match")))
```

### Step 5: Cropland masking (opsional, flag `--masked`)

```
Dari seluruh piksel dalam county, ambil hanya yang
diklasifikasi sebagai "Croplands" (MCD12Q1 kelas 12)

Hutan, kota, sawah non-jagung → dibuang
```

### Step 6: Zonal statistics → rata-rata per county

```
Untuk setiap gambar (46 kali per tahun):

  "Ambil semua piksel dalam batas county ini → hitung rata-rata setiap band"

Hasil: 1 baris per county per tanggal = 11 nilai rata-rata
```

Kode:
```python
image.reduceRegions(
    collection=counties,
    reducer=ee.Reducer.mean(),
    scale=500,    # resolusi MODIS = 500m per piksel
)
```

### Step 7: Flatten & export ke Google Drive

```
46 gambar × ~2.800 county → ~128.000 baris CSV → Google Drive
```

**Semua komputasi ini dijalankan di server Google Earth Engine**, bukan di laptop. Laptop cuma kirim perintah dan terima file CSV hasil.

---

## 8. Kenapa 46 timestep? Kenapa rata-rata per county?

### 46 timestep

365 hari ÷ 8 hari (komposit MODIS) = ~46 periode observasi per tahun.

Setiap komposit adalah rata-rata 8 hari pengamatan — ini mengurangi noise dari awan (MODIS optis tidak bisa melihat menembus awan).

### Rata-rata per county, bukan per piksel

Karena label yield hanya tersedia per county ("County X panen 8.5 ton/ha tahun 2020"). Jadi fitur satelit juga harus diagregasi ke level yang sama supaya bisa dipasangkan:

```
Input model:  rata-rata reflektansi semua piksel dalam county
Label:        yield county tersebut dalam ton/ha
```

---

## 9. 10 Fitur Final

Semua fitur berasal dari MODIS:

| # | Fitur | Sumber | Deskripsi |
|---|-------|--------|-----------|
| 1 | sur_refl_b01 | MOD09A1 (Terra) | Red — diserap klorofil |
| 2 | sur_refl_b02 | MOD09A1 (Terra) | NIR — dipantulkan daun sehat |
| 3 | sur_refl_b03 | MOD09A1 (Terra) | Blue — koreksi atmosfer |
| 4 | sur_refl_b04 | MOD09A1 (Terra) | Green — referensi |
| 5 | sur_refl_b05 | MOD09A1 (Terra) | SWIR 1 — kelembaban |
| 6 | sur_refl_b06 | MOD09A1 (Terra) | SWIR 2 — kelembaban |
| 7 | sur_refl_b07 | MOD09A1 (Terra) | SWIR 3 — kelembaban |
| 8 | ndvi | Dihitung | (NIR-Red)/(NIR+Red) — indeks vegetasi |
| 9 | LST_Day_1km | MYD11A2 (Aqua) | Suhu permukaan siang (°C) |
| 10 | LST_Night_1km | MYD11A2 (Aqua) | Suhu permukaan malam (°C) |

Shape tensor final: **(N_samples, 46 timesteps, 10 features)**

---

## 10. Yang Tidak Dipakai

| Variabel | Kenapa |
|----------|--------|
| **EVI** | Dihitung di GEE, tapi dibuang saat merge — nilai overflow (±10¹¹) karena penyebut formula EVI mendekati nol saat dirata-rata per county. Lihat `memory/project_decisions.md` |
| **Curah hujan** | Tidak ada di pipeline. Bisa jadi future work (CHIRPS dataset) |
| **Kelembaban udara** | Tidak dipakai |
| **LAI / FPAR** | Produk MODIS lain (MCD15A2H, 4-hari), tidak diekstrak |
| **pH tanah, Nitrogen** | Tidak ada data spasial global yang granular per county/provinsi |
| **Elevasi** | Bisa dari SRTM, tapi tidak dimasukkan |
| **Cropland mask** | Sudah diimplementasikan di `extract_modis_usa.py --masked` tapi data hasilnya belum di-download untuk USA |

---

## Ringkasan Pipeline USA

```
USDA NASS API                    Google Earth Engine
     │                                     │
download_usa.py              extract_modis_usa.py
     │                                     │
yield_usa_2003_2023.parquet       50 CSV MODIS (Google Drive)
(label: yield ton/ha)             (fitur: 10 band × 46 timestep × county)
     │                                     │
     └──────────────┬──────────────────────┘
                    │
              merge_modis.py
                    │
          usa_modis.npz (X: 32296×46×10, y: 32296)
                    │
              dataset.py → DataLoader → train.py
```

---

*Ditulis dari hasil diskusi detail pipeline data. Update terakhir: 19 Mei 2026.*
