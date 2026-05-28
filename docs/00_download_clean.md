# Pipeline USA: Dari Raw Data ke Tensor Siap Training

> **Fokus:** USA saja. Indonesia dibahas terpisah setelah pipeline USA paham sepenuhnya.
> **Tujuan:** bisa menjelaskan sendiri bagaimana data mengalir dari CSV mentah sampai jadi tensor siap masuk model.
> **Cara belajar:** baca file → jalankan di `notebooks/00_coba.ipynb` → jawab pertanyaan.

---

## Peta Alur Data USA

```
USDA NASS API
    ↓ download_usa.py
data/raw/usa/nass_corn_county_yield_2003_2023.csv   ← 45.215 baris, 39 kolom (kotor)
    ↓ cleaning (filter, konversi unit)
data/processed/usa/yield_usa_2003_2023.parquet      ← 41.349 baris, 9 kolom (bersih)
                                                                        ↓
Google Earth Engine (GEE)                                               |
    ↓ extract_modis_usa.py (dikirim ke server Google)                   |
data/raw/modis/modis_usa_{2003..2023}.csv            ← 21 file, tiap file ~139k baris  |
    ↓ merge_modis.py (join yield + MODIS)                               |
data/processed/modis/usa_modis.npz                  ← X:(32296,46,10) y:(32296,) ←────┘
    ↓ dataset.py (PyTorch Dataset)
MaizeDataset("usa", split="train")                  ← siap masuk DataLoader → model
```

---

## LANGKAH 1: Raw Yield — USDA NASS API

**Script:** `src/data/download_usa.py`

**Yang terjadi:**
1. Kirim HTTP GET ke `https://quickstats.nass.usda.gov/api/api_GET/` dengan parameter: `commodity=CORN, statisticcat=YIELD, agg_level=COUNTY, year 2003–2023`
2. NASS balas dengan ~45.215 baris, 39 kolom — mayoritas kolom metadata tidak terpakai
3. Filter + cleaning → simpan ke parquet

**Input → Output:**
```
Input:  API key NASS (dari .env)
Proses: HTTP GET ke NASS dengan parameter:
          commodity = CORN
          statistic = YIELD
          level     = COUNTY
          unit      = BU / ACRE
          tahun     = 2003 s/d 2023
Output: data/raw/usa/nass_corn_county_yield_2003_2023.csv
        data/processed/usa/yield_usa_2003_2023.parquet
```




## Tahapan setelah download
Download raw data
        ↓
  Inspect (EDA)
  ├── shape: berapa row, kolom
  ├── dtypes: tipe data tiap kolom
  ├── value_counts: distribusi kategori
  ├── describe(): mean, min, max, quartile
  └── nunique(): berapa county unik, berapa tahun
        ↓
  Clean
  ├── filter tahun (2003-2023)
  ├── buang duplikat (prodn_practice_desc)
  ├── handle (D) suppressed → NaN
  └── handle (Z) zero → NaN
        ↓
  Wrangling
  ├── bangun FIPS code = state_fips + county_code
  ├── konversi unit: bu/acre × 0.06277 → ton/ha
  └── pilih + rename kolom
        ↓
  Anomaly Detection
  ├── yield = 0.0 (206 county USA)
  ├── yield < 0.1 → filter
  └── outlier ekstrem (IDN: 4109 t/ha)
        ↓
  Missing Data
  ├── berapa NaN per kolom?
  ├── NaN karena suppressed vs genuine missing
  └── strategi: drop row atau impute
        ↓
  Validate
  ├── jumlah row sesuai ekspektasi?
  ├── range yield masuk akal?
  ├── semua tahun ada?
  └── tidak ada duplikat county-year
        ↓
  Save → data/processed/
  ├── .parquet (untuk komputasi)
  └── .csv (untuk inspeksi manual)

  



### 1.1. Cek Manual: Buka CSV di Excel

| Apa yang dicek | Cara cek | Ekspektasi |
|----------------|----------|------------|
| **Jumlah baris** | Scroll ke bawah, lihat row count | ~45.000 baris |
| **Jumlah kolom** | Scroll ke kanan | 39 kolom |
| **Range tahun** | Filter kolom `year` | 2003 – 2023 (tapi bisa ada 2024–2025 jika re-download) |
| **Kolom penting** | `county_name`, `state_alpha`, `year`, `Value` | Inilah data inti |
| **Nilai "Value"** | Filter kolom `Value` | Ada angka (misal 180.5), ada "(D)", ada "(Z)" |

### 1.2. Validasi Jumlah Baris

Secara teori:
```
2.280 county × 21 tahun = 47.880 baris (kalau semua county nanam jagung tiap tahun)
```

Tapi data mentah cuma ~45.215 baris. Kenapa?
- NASS cuma ngasih data county yang **betul-betul ada petani jagung** di tahun itu
- County tanpa jagung = tidak muncul di query
- County yang disensor "(D)" = tetap muncul sebagai baris, tapi Value kosong

**Cek di Excel:** Pivot `county_name` × `year`. Pastikan:
- Tidak ada county yang muncul **lebih dari 1 kali di tahun yang sama** (duplikat — lihat 1.4)
- Maksimal 21 kemunculan per county (2003–2023)

### 1.3. "(D)" dan "(Z)" — Nilai yang Disensor

Filter kolom `Value` di Excel:
- **(D)** = *Disclosure suppressed* — USDA tidak boleh publikasi kalau cuma 1-2 petani di county itu (privasi)
- **(Z)** = Kurang dari unit pembulatan (<= 0.5 bu/acre)

Keduanya dikonversi jadi **NaN** (tidak bisa dipakai training — tidak ada label yield).

**Cek:** Hitung berapa baris yang Value-nya "(D)". Berapa persen dari total?

### 1.4. Temuan: Duplikat karena `prodn_practice_desc`

**Masalah:** Banyak county muncul 2-3× di tahun yang sama.

**Penyebab:** Query NASS tidak memfilter `prodn_practice_desc`. NASS melaporkan yield jagung dalam 3 kategori:

| prodn_practice_desc | Artinya |
|---------------------|---------|
| `ALL PRODUCTION PRACTICES` | Rata-rata tertimbang seluruh metode — **ini yang kita mau** |
| `IRRIGATED` | Yield dari lahan jagung irigasi saja |
| `NON-IRRIGATED` | Yield dari lahan jagung non-irigasi |

County yang punya irigasi (Colorado, Nebraska, Kansas) → 3 baris per tahun. Corn Belt (Iowa, Illinois) → cuma 1 baris.

**Fix:** Filter sebelum training:
```python
df = df[df["prodn_practice_desc"] == "ALL PRODUCTION PRACTICES"]
```

### 1.5. "OTHER (COMBINED) COUNTIES" — Data Sampah

Cari di `county_name` kata "OTHER". Ini county-county kecil yang digabung NASS jadi satu agregat — **bukan county beneran, tidak ada di peta**. Di cleaning, ini dibuang.

**Pertanyaan — harus bisa dijawab:**
- [ ] Apa itu county? Bedanya dengan state?
- [ ] Kenapa raw 45.215 baris tapi processed 41.349? Yang dibuang baris apa?
- [ ] Apa itu `"(D)"` di kolom `Value`? Kenapa dibuang, bukan diisi rata-rata?
- [ ] Konversi `bu/acre → ton/ha`: 1 bushel jagung berapa kg? 1 acre berapa hektar?

---



## LANGKAH 2: Processed Yield — Eksplorasi

**Script:** `download_usa.py` fungsi `clean_nass()`

**Yang dikerjakan:**
```
Input:  Raw CSV (45.215 baris × 39 kolom)
Proses: 1. Buang "OTHER (COMBINED)" rows
        2. "(D)" dan "(Z)" → NaN
        3. Bangun FIPS code: state_fips_code + county_code = 5-digit ID
        4. Konversi satuan: bu/acre × 0.06277 = ton/ha
        5. Pilih 9 kolom standar
        6. Sort by region_id, year
Output: data/processed/usa/yield_usa_2003_2023.parquet (41.349 × 9)
```

**Skema kolom:**

| Kolom | Tipe | Contoh |
|-------|------|--------|
| `region_id` | str | `"17019"` (FIPS 5-digit) |
| `region_name` | str | `"Champaign, Illinois"` |
| `state` | str | `"IL"` |
| `county_name` | str | `"CHAMPAIGN"` |
| `year` | int | `2020` |
| `yield_bu_acre` | float | `226.5` |
| `yield_ton_ha` | float | `14.22` |
| `country` | str | `"USA"` |
| `data_source` | str | `"USDA_NASS"` |

**Validasi di notebook:**

| Apa yang dicek | Cara cek | Ekspektasi |
|----------------|----------|------------|
| **Jumlah baris** | `len(df)` | 41.349 |
| **County unik** | `df['region_id'].nunique()` | 2.280 |
| **Range tahun** | `df['year'].min()` – `df['year'].max()` | 2003 – 2023 |
| **Yield range** | `df['yield_ton_ha'].describe()` | 0 – 17 t/ha |

**Validasi FIPS:**
```python
df['region_id'].str.len().value_counts()  # semua harus = 5
```

**Validasi konversi satuan:**
```python
(df['yield_bu_acre'] * 0.06277).round(4) == df['yield_ton_ha']
# Harusnya True semua
```

**Validasi yield anomali:**

| Cek | Temuan | Tindakan |
|-----|--------|----------|
| `yield == 0.0` | 206 sampel | Anomali — hapus sebelum training |
| `yield > 20` | 0 sampel | Aman (max corn yield dunia ~18 t/ha) |

**Pertanyaan:**
- [ ] Berapa range yield? Min dan max masuk akal?
- [ ] Berapa county per tahun? Kenapa bervariasi antar tahun?
- [ ] Tahun mana yield paling rendah? (Hint: kekeringan 2012)
- [ ] Kenapa ada 433 baris duplikat di processed? Apa artinya?

---




