# Pipeline Step 3: MODIS via Google Earth Engine

> **Fokus:** Ekstraksi fitur satelit MODIS per county USA menggunakan Google Earth Engine.
> **Tujuan:** Bisa menjelaskan sendiri: apa itu MODIS, kenapa GEE, bagaimana FIPS menghubungkan yield ke batas wilayah, dan apa artinya setiap kolom di output CSV.
> **Script:** `src/data/002_extract_modis_usa.py`

---

## Peta Alur Step 3

```
yield_usa_2003_2023.parquet       ← dari Step 2 (FIPS = kunci penghubung)
    ↓ ambil daftar state FIPS
TIGER/2018/Counties (GEE)         ← batas polygon county di server Google
    ↓ filter: hanya state yang nanam jagung
~2.800 county polygons
    ↓ untuk tiap county: ambil semua piksel MODIS di dalamnya
MOD09A1 (Terra) + MYD11A2 (Aqua) ← dua koleksi citra satelit
    ↓ hitung NDVI, EVI, konversi LST
    ↓ rata-rata semua piksel per county per tanggal (reduceRegions)
modis_usa_{year}.csv              ← 1 baris per county per tanggal (46 tanggal/tahun)
    ↓ 21 file (2003–2023) → Step 4: merge dengan yield
```

---

## BAGIAN 1: Kenapa Google Earth Engine?

### 1.1. Cara manual tanpa GEE (supaya paham bedanya)

Bayangkan kamu ingin mengambil data NDVI untuk seluruh county USA dari tahun 2003–2023 secara manual:

1. Pergi ke [NASA Earthdata](https://earthdata.nasa.gov/) atau [LP DAAC](https://lpdaac.usgs.gov/)
2. Pilih produk: **MOD09A1** (surface reflectance MODIS Terra, 8-day)
3. Filter area: USA continental (±48 state)
4. Filter waktu: 2003–2023 → 21 tahun × 46 composite = **966 file per tile**
5. MODIS pakai grid "tiles" — USA butuh ~20 tiles → 966 × 20 = **~19.000 file GeoTIFF**
6. Tiap file ~100–200 MB → **total ~2–3 TB data**
7. Download, simpan di hard disk, lalu untuk setiap file: potong per county, rata-ratakan piksel
8. Ulang untuk LST (MYD11A2): 19.000 file lagi
9. Total waktu: berhari-hari download + berhari-hari proses

**Dengan GEE:**
- Kamu kirim *kode* ke server Google
- Server Google yang download, potong, dan rata-ratakan
- Hasilnya cuma CSV kecil yang kamu download dari Google Drive
- Waktu: 5–20 menit per tahun, paralel untuk semua tahun

**Kenapa bisa begitu?** GEE menyimpan seluruh arsip MODIS di server mereka sendiri. Kamu cuma "menyewa" komputasinya.

### 1.2. Siapa yang butuh akun GEE?

GEE gratis untuk penelitian akademik. Daftar di [earthengine.google.com](https://earthengine.google.com) dengan email universitas atau jelaskan tujuan penelitian. Setelah approved, kamu dapat project ID — di project ini: `alamat-413120`.

---

## BAGIAN 2: MODIS — Dua Satelit, Dua Produk

### 2.1. Apa itu MODIS?

MODIS = **Moderate Resolution Imaging Spectroradiometer**. Sensor yang dipasang di dua satelit NASA:

| Satelit | Nama | Waktu Lewat | Produk yang dipakai |
|---------|------|-------------|---------------------|
| **Terra** | EOS-AM | Pagi (~10:30 pagi lokal) | MOD09A1 (reflektansi) |
| **Aqua** | EOS-PM | Siang (~13:30 siang lokal) | MYD11A2 (suhu) |

Keduanya mengorbit bumi dan mengambil gambar seluruh permukaan bumi setiap **1–2 hari sekali**. Tapi resolusinya bukan piksel per piksel yang tajam — MODIS dirancang untuk monitoring global, bukan detail lokal.

**Resolusi MODIS yang kita pakai:**
- MOD09A1: **500 meter per piksel** (artinya 1 piksel = area 500m × 500m)
- MYD11A2: **1 kilometer per piksel**

Bandingkan: Sentinel-2 (resolusi 10m) lebih tajam, tapi arsipnya baru mulai 2015 dan terlalu besar untuk diproses manual.

### 2.2. MOD09A1 — Surface Reflectance, 8-Day Composite

**Nama produk:** `MODIS/061/MOD09A1` di GEE

**Apa itu "surface reflectance"?**

Saat cahaya matahari mengenai tanaman, sebagian diserap (untuk fotosintesis) dan sebagian dipantulkan kembali. Reflektansi = persentase cahaya yang dipantulkan. MODIS mengukur ini dalam **7 panjang gelombang (band)** berbeda:

| Band | Nama | Panjang Gelombang | Yang diukur |
|------|------|-------------------|-------------|
| b01 | Red | 620–670 nm | Cahaya merah — diserap klorofil → tanaman sehat = nilai rendah |
| b02 | NIR | 841–876 nm | Inframerah dekat — dipantulkan mesofil daun → tanaman = nilai tinggi |
| b03 | Blue | 459–479 nm | Biru — diserap klorofil |
| b04 | Green | 545–565 nm | Hijau — sedikit dipantulkan |
| b05 | SWIR1 | 1230–1250 nm | Inframerah gelombang pendek — sensitivitas air di tanaman |
| b06 | SWIR2 | 1628–1652 nm | SWIR kedua — kandungan air & material tanah |
| b07 | SWIR3 | 2105–2155 nm | SWIR ketiga — identifikasi jenis tanah, lahan bakar |

**Nilai mentah vs nilai sebenarnya:**

GEE menyimpan nilai MODIS dalam integer (bilangan bulat) dengan scale factor. Nilai mentah b01 misalnya `800` artinya reflektansi = `800 × 0.0001 = 0.08` (8%). Di script:
```python
scaled = image.select(SR_BANDS).multiply(0.0001)
```
Setelah dikali 0.0001, semua nilai band harusnya di range **0.0 – 1.0**.

**Apa itu "8-day composite"?**

Daripada pakai satu gambar harian (yang sering tertutup awan), MODIS menggabungkan 8 hari foto menjadi satu gambar komposit — memilih piksel terbaik (paling bersih dari awan) dari 8 hari itu. Ini **MOD09A1**: 8-day Surface Reflectance.

Dalam 1 tahun (365 hari) ÷ 8 = **~46 komposit**. Itulah kenapa ada tepat 46 tanggal per county per tahun.

### 2.3. MYD11A2 — Land Surface Temperature (LST), dari Aqua

**Nama produk:** `MODIS/061/MYD11A2` di GEE

LST = suhu permukaan tanah/vegetasi (bukan suhu udara). Diukur dari panas yang dipancarkan permukaan ke satelit (inframerah termal).

**Nilai mentah vs nilai sebenarnya:**

Nilai mentah LST disimpan dalam Kelvin dengan scale factor 0.02. Konversi:
```
°C = nilai_mentah × 0.02 − 273.15
```

Di script:
```python
image.select(LST_BANDS).multiply(0.02).add(-273.15)
```

Contoh: nilai mentah `15000` → `15000 × 0.02 = 300.0 K` → `300.0 − 273.15 = 26.85°C`

**Kenapa ada dua: LST_Day dan LST_Night?**

Aqua melewati wilayah dua kali sehari (pagi dan siang). Suhu siang dan malam beda jauh dan keduanya punya makna agronomis berbeda:
- LST_Day: stress panas pada tanaman (>35°C bisa merusak jagung saat tasseling)
- LST_Night: laju respirasi malam (suhu malam tinggi = tanaman lebih boros energi)

---

## BAGIAN 3: TIGER Boundaries — Dari FIPS ke Polygon

### 3.1. Apa itu TIGER?

TIGER = **Topologically Integrated Geographic Encoding and Referencing** — database batas wilayah resmi dari US Census Bureau. Di GEE tersedia sebagai:
```
"TIGER/2018/Counties"
```

Ini adalah kumpulan **polygon** (batas berbentuk bidang datar) untuk setiap county di USA. Tiap polygon punya atribut:
- `GEOID` = **kode FIPS 5-digit** (2 digit state + 3 digit county)
- `NAME` = nama county
- `STATEFP` = kode state 2-digit

### 3.2. Kenapa FIPS adalah jembatan yang kritis?

Ingat dari Step 2: hasil cleaning NASS menghasilkan kolom `region_id` = FIPS 5-digit (misal `"17019"` untuk Champaign, Illinois).

Kolom ini sama persis dengan `GEOID` di TIGER. Ini yang memungkinkan kita:
1. **Di Step 3:** Filter county di GEE → hanya ambil county yang ada di yield data → tidak buang komputasi untuk county yang tidak nanam jagung
2. **Di Step 4 (merge):** Join yield (punya `region_id`) dengan MODIS (punya `GEOID`) menggunakan FIPS sebagai kunci

```
yield_usa.parquet: region_id = "17019"
                                  ↕ sama
TIGER di GEE:        GEOID   = "17019"  → polygon batas Champaign County
                                  ↕ dipakai untuk
MODIS output CSV:    GEOID   = "17019"  → rata-rata piksel di dalam polygon itu
```

### 3.3. Kenapa filter hanya state yang nanam jagung?

Di script:
```python
state_fips = yield_df["region_id"].str[:2].unique().tolist()
counties = get_usa_counties(state_fips)
```

Yield data (`yield_df`) hanya berisi state yang punya data jagung dari NASS. Dengan mengambil 2 digit pertama FIPS (= kode state), kita tahu state mana saja yang relevan. Kita filter TIGER untuk hanya state itu — mengurangi jumlah polygon yang harus diproses GEE dari ~3.100 county (semua USA) ke ~2.800 county (hanya yang nanam jagung).

---

## BAGIAN 4: Proses di GEE — Langkah per Langkah

### 4.1. Cara kerja script secara manual (seolah kamu kerjakan sendiri)

Bayangkan kamu di [code.earthengine.google.com](https://code.earthengine.google.com) (GEE Code Editor):

**Langkah A: Pilih area**
```javascript
// Di GEE Code Editor (JavaScript), ini ekuivalennya:
var counties = ee.FeatureCollection("TIGER/2018/Counties")
  .filter(ee.Filter.inList("STATEFP", ["17", "19", "18", ...]));
// → 2.800 polygon county
```

**Langkah B: Ambil koleksi citra MODIS untuk 1 tahun**
```javascript
var sr = ee.ImageCollection("MODIS/061/MOD09A1")
  .filterDate("2020-01-01", "2021-01-01");
// → ~46 gambar (tiap gambar = seluruh USA, 7 band, resolusi 500m)
```

**Langkah C: Hitung NDVI dan EVI dari band**

NDVI (Normalized Difference Vegetation Index) — formula dasar:
```
NDVI = (NIR - Red) / (NIR + Red)
     = (b02 - b01) / (b02 + b01)
```

Kenapa rumus ini? Tanaman sehat menyerap cahaya merah (b01 kecil) dan memantulkan inframerah (b02 besar) → hasil pembagian mendekati 1. Tanah kosong atau vegetasi mati: keduanya mirip → NDVI mendekati 0.

EVI (Enhanced Vegetation Index) — formula lebih kompleks:
```
EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
    = 2.5 × (b02 - b01) / (b02 + 6×b01 - 7.5×b07 + 1)
```

EVI mengkoreksi efek atmosfer dan tanah yang mengganggu NDVI di area dengan vegetasi sangat lebat atau tanah terang. Untuk Corn Belt USA, NDVI dan EVI korelasinya tinggi.

> **Catatan keputusan:** EVI dihitung di script tapi **dibuang saat merge** (Step 4). Kenapa? EVI dan NDVI sangat berkorelasi (r > 0.95 di jagung), jadi menyimpan keduanya hanya menambah redundansi fitur tanpa meningkatkan performa model. Keputusan ini ada di `memory/project_decisions.md`.

**Langkah D: Gabung Terra (SR) + Aqua (LST)**

MOD09A1 dan MYD11A2 punya jadwal berbeda (Terra pagi, Aqua siang). Untuk setiap gambar Terra, cari gambar Aqua yang **paling dekat waktunya** (dalam 16 hari):

```python
time_filter = ee.Filter.maxDifference(
    difference=16 * 24 * 60 * 60 * 1000,  # 16 hari dalam milidetik
    leftField="system:time_start",
    rightField="system:time_start",
)
joined = ee.Join.saveBest("lst_match", "time_diff").apply(sr_col, lst_col, time_filter)
```

Hasilnya: 46 gambar yang masing-masing punya 7 band reflektansi + NDVI + EVI + LST_Day + LST_Night = **11 band per gambar**.

**Langkah E: Rata-ratakan piksel per county (reduceRegions)**

Ini inti dari ekstraksi spasial. Untuk setiap gambar dan setiap county polygon:
- Identifikasi semua piksel 500m yang **jatuh di dalam** batas county
- Ambil rata-rata nilai semua piksel itu
- Hasilnya: **1 angka per band per county per tanggal**

```python
reduced = image.reduceRegions(
    collection=counties,
    reducer=ee.Reducer.mean(),
    scale=500,      # resolusi piksel (meter)
    crs="EPSG:4326",  # sistem koordinat WGS84
)
```

Kenapa `scale=500`? Karena MOD09A1 aslinya 500m/piksel. Kalau kamu set scale lebih kecil (misal 100m), GEE akan **oversample** (interpolasi) yang tidak menambah informasi tapi 5× lebih lambat.

**Langkah F: Export ke Google Drive**

Hasilnya bukan gambar tapi tabel (FeatureCollection) → export ke CSV di Google Drive folder `thesis_maize_gee`:

```
modis_usa_2020.csv
  GEOID, NAME, STATEFP, year, date, sur_refl_b01, ..., sur_refl_b07, ndvi, evi, LST_Day_1km, LST_Night_1km
  17019, Champaign, 17, 2020, 2020-01-01, 0.08, 0.16, 0.04, 0.06, 0.15, 0.09, 0.13, 0.33, 0.21, 18.5, 4.2
  17019, Champaign, 17, 2020, 2020-01-09, 0.07, 0.15, ...
  ... (46 baris untuk Champaign, lalu lanjut ke county berikutnya)
```

---

## BAGIAN 5: Cropland Mask — Fitur Tambahan (v2)

### 5.1. Masalah tanpa cropland mask

`reduceRegions` mengambil rata-rata SEMUA piksel dalam batas county — termasuk kota, danau, hutan, dan jalan. County seperti Cook County (Chicago) punya piksel bangunan/aspal lebih banyak daripada lahan pertanian. Rata-rata NDVI-nya akan jauh lebih rendah dari nilai sebenarnya di lahan jagung.

### 5.2. Solusi: MCD12Q1 — Land Cover Classification

**Nama produk:** `MODIS/061/MCD12Q1` — peta tutupan lahan tahunan dari MODIS.

Produk ini mengklasifikasikan setiap piksel 500m ke dalam kategori tutupan lahan berdasarkan sistem **IGBP (International Geosphere-Biosphere Programme)**:

| Kelas IGBP | Nomor | Artinya |
|------------|-------|---------|
| Croplands | **12** | Lahan pertanian (sawah, ladang, kebun) |
| Urban | 13 | Kota dan pemukiman |
| Forest | 1–5 | Berbagai jenis hutan |
| Water | 17 | Danau, sungai |
| ... | ... | ... |

**Di script:**
```python
def get_cropland_mask(year: int) -> ee.Image:
    lc_year = min(year, 2022)  # MCD12Q1 tersedia sampai ~2022
    lc = ee.ImageCollection(MCD12Q1)
        .filter(ee.Filter.calendarRange(lc_year, lc_year, "year"))
        .first()
    return lc.select("LC_Type1").eq(12)  # binary: 1 = cropland, 0 = bukan
```

Hasilnya: sebuah "topeng" (mask) di mana piksel lahan pertanian = 1, piksel lain = 0. Sebelum `reduceRegions`, kita terapkan mask ini:

```python
col = col.map(lambda img: img.updateMask(crop_mask))
```

Efeknya: rata-rata per county sekarang hanya menghitung piksel yang terklasifikasi sebagai lahan pertanian. Lebih representatif untuk prediksi yield.

**Output v2:** Folder `thesis_maize_gee_v2`, file `modis_usa_v2_{year}.csv`

---

## BAGIAN 6: Struktur Output CSV

### 6.1. Skema kolom

| Kolom | Tipe | Contoh | Keterangan |
|-------|------|--------|------------|
| `GEOID` | str | `"17019"` | FIPS 5-digit — kunci join ke yield |
| `NAME` | str | `"Champaign"` | Nama county |
| `STATEFP` | str | `"17"` | Kode state 2-digit |
| `year` | int | `2020` | Tahun |
| `date` | str | `"2020-01-01"` | Tanggal komposit 8-harian |
| `sur_refl_b01` | float | `0.082` | Band merah (0–1 setelah scaling) |
| `sur_refl_b02` | float | `0.164` | Band NIR |
| `sur_refl_b03` | float | `0.041` | Band biru |
| `sur_refl_b04` | float | `0.063` | Band hijau |
| `sur_refl_b05` | float | `0.148` | SWIR1 |
| `sur_refl_b06` | float | `0.092` | SWIR2 |
| `sur_refl_b07` | float | `0.131` | SWIR3 |
| `ndvi` | float | `0.334` | Vegetation index (-1 sampai 1) |
| `evi` | float | `0.218` | Enhanced vegetation index |
| `LST_Day_1km` | float | `18.5` | Suhu siang (°C) |
| `LST_Night_1km` | float | `4.2` | Suhu malam (°C) |

### 6.2. Berapa baris satu file CSV?

```
~2.280 county × 46 tanggal = ~104.880 baris per file (per tahun)
```

Sebenarnya bisa sedikit lebih atau kurang karena:
- Ada county yang semua pikselnya tertutup awan → `reduceRegions` tidak menghasilkan baris (NaN dihapus oleh GEE)
- Dengan cropland mask: county yang tidak punya piksel pertanian juga hilang

### 6.3. Validasi manual di Excel

**Buka `modis_usa_2020.csv` di Excel:**

| Apa yang dicek | Cara cek di Excel | Ekspektasi |
|----------------|-------------------|------------|
| Jumlah baris | Ctrl+End → lihat baris terakhir | ~100.000–140.000 |
| County unik | Data → Remove Duplicates (kolom GEOID) → hitung | ~2.000–2.800 |
| Tanggal unik | Filter kolom `date` → lihat pilihan | Tepat 46 tanggal |
| Satu county 46 baris | Filter GEOID = "17019" → hitung baris | Tepat 46 |
| Nilai b01–b07 | Sort kolom `sur_refl_b01` → cek min/max | 0.0 – 1.0 |
| NDVI range | Sort kolom `ndvi` → cek min/max | -1.0 – 1.0 (jagung tumbuh: > 0.3) |
| LST_Day range | Sort `LST_Day_1km` | 0 – 55°C wajar |
| LST_Night range | Sort `LST_Night_1km` | -10 – 30°C wajar |

**Cek musiman di Excel (pivot table):**
1. Buat Pivot Table: baris = `date`, nilai = rata-rata `ndvi`
2. Ekspektasi untuk Iowa (banyak jagung):
   - Januari: NDVI ~0.1–0.2 (lahan kosong/bersalju)
   - Mei–Juni: NDVI naik cepat (jagung tanam)
   - Juli–Agustus: NDVI puncak ~0.7–0.8 (jagung tumbuh penuh)
   - Oktober: NDVI turun cepat (jagung panen)
3. Kalau pola ini tidak muncul → ada masalah di ekstraksi

---

## BAGIAN 7: Cara Menjalankan Script

### 7.1. Prasyarat

1. Akun GEE aktif dan project ID dikonfigurasi
2. Autentikasi: `earthengine authenticate` di terminal
3. Yield data sudah ada: `data/processed/usa/yield_usa_2003_2023.parquet`

### 7.2. Test mode dulu (1 state, 1 tahun)

```bash
python src/data/002_extract_modis_usa.py --test
```

Ini kirim 1 task ke GEE untuk Iowa tahun 2020 saja. Buka [code.earthengine.google.com/tasks](https://code.earthengine.google.com/tasks) — kamu akan lihat task dengan status RUNNING lalu COMPLETED. File CSV akan muncul di Google Drive → folder `thesis_maize_gee`.

Download file itu secara manual, cek di Excel sesuai checklist 6.3.

### 7.3. Full extraction (semua tahun)

```bash
python src/data/002_extract_modis_usa.py
# Atau dengan cropland mask:
python src/data/002_extract_modis_usa.py --masked
```

Ini submit 21 task sekaligus (2003–2023). GEE menjalankan paralel. Waktu per task: 5–20 menit. Monitor di halaman Tasks GEE.

Setelah selesai: download semua 21 CSV dari Drive ke `data/raw/modis/`.

---

## BAGIAN 8: Pertanyaan Pemahaman

Pastikan kamu bisa menjawab ini tanpa melihat catatan:

- [ ] Apa bedanya Terra dan Aqua? Produk MODIS apa yang berasal dari masing-masing?
- [ ] Kenapa ada 46 tanggal per tahun, bukan 365? Apa itu "8-day composite"?
- [ ] Nilai raw b01 di GEE adalah `820`. Berapakah nilai setelah scaling?
- [ ] Hitung NDVI jika b01 = 0.08 dan b02 = 0.45. Artinya apa?
- [ ] Nilai raw LST_Day = `14500`. Berapakah dalam °C?
- [ ] Kenapa kita tidak pakai semua piksel dalam county, tapi filter dengan cropland mask?
- [ ] Apa itu TIGER boundaries? Mengapa FIPS adalah penghubung antara yield dan MODIS?
- [ ] Kolom `GEOID` di MODIS CSV = kolom apa di yield parquet? Kenapa ini bisa di-join?
- [ ] Kenapa EVI dihitung tapi kemudian dibuang di step merge?
- [ ] Kalau ada county yang `date` = "2020-07-04" tidak muncul di CSV, apa kemungkinan penyebabnya?
- [ ] `scale=500` di `reduceRegions` artinya apa? Apa yang terjadi kalau kita set ke 100?

---

## Ringkasan Keputusan Desain

| Keputusan | Pilihan | Alasan |
|-----------|---------|--------|
| Sumber data satelit | MODIS (500m, 8-day) | Arsip panjang (2000–sekarang), gratis, cakupan global, cocok untuk county-level |
| Komposit temporal | 8-hari | Mengurangi awan, tetap capture fenologi musiman |
| Timesteps per tahun | 46 | 365 ÷ 8 = 45.6, MODIS menghasilkan 46 komposit per kalender |
| Fitur reflektansi | b01–b07 + NDVI | 7 band SWIR/NIR/Visible + derived vegetation index |
| Fitur termal | LST_Day + LST_Night | Keduanya punya makna agronomis berbeda |
| EVI | Dihitung, lalu dibuang | Korelasi tinggi dengan NDVI → redundant |
| Batas wilayah | TIGER 2018 | Dataset resmi Census USA, tersedia di GEE, GEOID = FIPS |
| Agregasi spasial | Mean piksel per county | Sederhana, robust terhadap outlier piksel awan |
| Cropland mask | MCD12Q1 IGBP kelas 12 | Hanya piksel lahan pertanian yang relevan untuk prediksi yield |




## Manual Teknis
Buka NASA Earthdata Search:
  1. Search: MOD09A1
  2. Filter tanggal: Juli 2020
  3. Filter lokasi: gambar di peta, kotak di atas Illinois
  4. Download → dapat file: MOD09A1.A2020185.h10v05.061.2020194034158.hdf

  Format nama: [produk].[hari ke berapa di tahun itu].[tile].[versi].[tanggal proses]

  File ini berisi seluruh tile h10v05 — artinya mencakup Iowa, Illinois, Indiana, Wisconsin, dll. Ukuran: ~100 MB. Formatnya HDF (bukan TIFF langsung — perlu dikonversi dulu).

  Langkah 2: Buka di software GIS → "Potong per County"

  Buka QGIS (software GIS gratis):
  1. Load file HDF → pilih layer sur_refl_b02 (NIR) → kamu lihat gambar kotak besar seluruh Midwest
  2. Load shapefile batas county USA (dari Census/TIGER) → overlay di atas gambar
  3. Pilih polygon Champaign County
  4. Pakai tool Clip Raster by Mask Layer → "cookie cutter" polygon county ke raster GeoTIFF

  Hasilnya: file baru yang hanya berisi piksel-piksel yang jatuh di dalam batas Champaign County — mungkin ~200 piksel (area Champaign ±2.500 km², setiap piksel 0.25 km² = ~10.000 piksel, tapi
  jagung tidak penuh seluruh county).

  Langkah 3: Rata-ratakan piksel → "Zonal Statistics"

  Sekarang kamu punya 200 piksel, masing-masing punya nilai b01 dan b02.

  Hitung NDVI tiap piksel:
  NDVI_piksel_1 = (b02_1 - b01_1) / (b02_1 + b01_1) = (0.45 - 0.08) / (0.45 + 0.08) = 0.699
  NDVI_piksel_2 = (0.42 - 0.09) / (0.42 + 0.09) = 0.647
  ... (200 piksel)
    Kemudian ambil rata-rata semua 200 nilai NDVI → satu angka NDVI untuk seluruh Champaign County pada 4 Juli 2020 = 0.71.

  Di QGIS: pakai tool Zonal Statistics → pilih polygon county, pilih raster → otomatis hitung mean, min, max, std untuk setiap polygon.


### Download TIGER

  1. Buka: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
  2. Dropdown pertama: pilih tahun → 2023 (atau 2018, untuk konsistensi dengan script GEE yang pakai TIGER/2018)
  3. Dropdown kedua: pilih layer → "Counties (and equivalent)"
  4. Klik Submit
  5. Download file ZIP → extract → dapat folder berisi .shp, .dbf, .prj, dll.






# Pertanyaan Diskusi

oh iya, di akhir 03_GEE.md buatkan pertanyaan yang di akhir sesi harus bisa aku jawab (artinya aku sudah ngerti dan punya argument) di section # Pertanyaan Diskusi yang harus bisa dijawab




## Notes pertanyaan

di gee ku kan ada project alamat-413120, aku mau mulai from scratch dongs, nggak pakai alamanat-413120 ini, tapi mulai baru, thesis_maize_v1.



1.1. Cara manual tanpa GEE
web NASA Earthdata atau LP DAAC apa yang aku kungjungi? biar aku bisa paham bener, apa gimana, aku dapat sense on modis, apa yang harus aku lakukan agar aku paham betul. mungkin 1 loop download, 

milih Pilih produk: MOD09A1 dimana? kenapa kok nggak [ocean.nasa.gov](https://oceandata.sci.gsfc.nasa.gov/) atau proyek lain atau link lain?

biar aku paham sensesnya, seperti bahasa ini 46 composite = 966 file per tile, MODIS pakai grid "tiles" — USA butuh ~20 tiles → 966 × 20 = ~19.000 file GeoTIFF. nah composites itu apa, tile itu apa, grid, dll, kan aku harus mengerjakannya manual kan?

ini maksudnya apa : lalu untuk setiap file: potong per county, rata-ratakan piksel


"Server Google yang download, potong, dan rata-ratakan"
nah, di GEE aku nggak mau black-box seperti ini, aku mau tau prosesnya.



500 meter per piksel (artinya 1 piksel = area 500m × 500m) --> ini bisa aku pelajari dimana ya?



https://search.earthdata.nasa.gov/
https://oceandata.sci.gsfc.nasa.gov/





















































## Code coba-coba gee, gak penting, nanti dihapus
// 1 gambar MODIS Juli 2020
var oneImage = ee.ImageCollection('MODIS/061/MOD09A1')
  .filterDate('2020-07-04', '2020-07-12')
  .first()
  .multiply(0.0001);
var ndvi = oneImage.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('ndvi');
oneImage = oneImage.addBands(ndvi);

// Story County, Iowa
var storyCounty = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('GEOID', '19169'));
// ⭐ Clip: potong gambar MODIS sesuai batas Story County saja
var storyNdvi = oneImage.select('ndvi').clip(storyCounty);



// 🏙️ Cook County, Illinois — isinya Chicago
var cookCounty = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('GEOID', '17031'));

var cookNdvi = oneImage.select('ndvi').clip(cookCounty);
Map.addLayer(cookNdvi, {min: -1, max: 1, palette: ['red', 'yellow', 'green']}, 'NDVI Cook County');
Map.centerObject(cookCounty, 8);




// Tampilkan NDVI hanya di dalam Story County
Map.addLayer(storyNdvi, {min: -1, max: 1, palette: ['red', 'yellow', 'green']}, 'NDVI Story County');
Map.centerObject(storyCounty, 9);




// Histogram NDVI dalam Story County
var histogram = storyNdvi.reduceRegion({
  reducer: ee.Reducer.histogram(50),
  geometry: storyCounty.geometry(),
  scale: 500,
  maxPixels: 1e9
});
print('=== Distribusi piksel NDVI dalam Story County ===');
print(histogram.get('ndvi'));






MOD09A1 : 
- Composite 8 hari, total 46 gambar pertahun (365/8)

---

## BAGIAN 9: Dari Piksel ke Tensor — Ringkasan Konseptual

> Ini ringkasan seluruh proses GEE dalam satu mental model yang utuh.
> Bagian ini ditulis untuk bisa dijelaskan di sidang tanpa membuka catatan.

---

### 9.1. Satu Gambar MODIS = Jutaan Piksel

```
1 gambar MOD09A1 = citra raksasa menutup seluruh daratan (diproses tile per tile)
Resolusi         = 500m × 500m per piksel
1 piksel         = area 500m × 500m = 0.25 km² di muka bumi

USA continental  ≈ 8.000.000 km²
Jumlah piksel    = 8.000.000 ÷ 0.25 = ~32.000.000 piksel per gambar
```

Gambar sebesar ini tidak bisa langsung dipakai model — terlalu besar, dan setiap county butuh 1 angka representatif, bukan ribuan piksel.

---

### 9.2. Proses Inti: RASTER → TABULAR

Ini transformasi paling penting di seluruh pipeline.

```
RASTER (gambar piksel)          TABULAR (tabel angka)
────────────────────────        ─────────────────────────────────────
┌──┬──┬──┬──┬──┬──┐             GEOID  date        b01   ndvi  LST_D
│  │  │  │  │  │  │             19141  2020-08-04  0.05  0.79  31.2
├──┼──┼──┼──┼──┼──┤    →        19035  2020-08-04  0.06  0.81  30.8
│  │▓▓│▓▓│▓▓│  │  │             19153  2020-08-04  0.07  0.61  29.4
├──┼──┼──┼──┼──┼──┤             ...
│  │▓▓│▓▓│▓▓│  │  │
└──┴──┴──┴──┴──┴──┘
▓ = piksel dalam county
```

**Cara transformasinya (GEE `reduceRegions`):**
1. Ambil polygon batas county dari TIGER (kuncinya = GEOID = FIPS)
2. Identifikasi semua piksel yang tengahnya jatuh di dalam polygon
3. Rata-ratakan nilai semua piksel itu → 1 angka per band
4. Output: 1 baris tabel = 1 county × 1 tanggal

---

### 9.3. Hitung Piksel per County — dengan Angka Nyata

```
Rumus:  jumlah piksel = luas county (km²) ÷ 0.25 km²

County          Luas      Piksel teoritis   Piksel aktual*
──────────────  ────────  ───────────────   ──────────────
County A        1.000 km² 1.000 ÷ 0.25 = 4.000   ~3.800
O'Brien Iowa    1.484 km² 1.484 ÷ 0.25 = 5.936   ~5.700
Cherokee Iowa   1.490 km² 1.490 ÷ 0.25 = 5.960   ~5.700
Kossuth Iowa    2.526 km² 2.526 ÷ 0.25 = 10.104  ~9.800
County B        2.000 km² 2.000 ÷ 0.25 = 8.000   ~7.600

*Piksel aktual < teoritis karena centroid rule:
 piksel yang titik tengahnya tepat di tepi batas county dibuang
```

**Centroid rule** — GEE tidak bisa "belah" piksel setengah-setengah:
```
Untuk setiap piksel, GEE tanya:
  "Apakah TITIK TENGAH piksel ini ada di dalam polygon county?"
  YA  → piksel ikut rata-rata (100% nilainya dipakai)
  TIDAK → piksel dibuang (0% nilainya dipakai)

Konsekuensi di county tidak berbentuk kotak (misal Cherokee Iowa):
  - Piksel di tepi sungai: tengahnya di luar county → dibuang
    meski 50% areanya lahan jagung Cherokee
  - Piksel di county tetangga: tengahnya di dalam Cherokee → diambil
    meski 50% areanya bukan Cherokee

Dampak: error ~2–3% di county dengan banyak batas tidak lurus.
Solusi: cropland mask (MCD12Q1) mengurangi noise piksel tepi non-pertanian.
```

---

### 9.4. Dari 1 Piksel ke 1 Baris CSV

Untuk O'Brien County Iowa, tanggal 2020-08-04, ada ~5.700 piksel:

```
Piksel 1:    [b01=0.050, b02=0.430, ..., ndvi=0.791, LST_D=31.0, LST_N=17.5]
Piksel 2:    [b01=0.048, b02=0.442, ..., ndvi=0.803, LST_D=31.5, LST_N=18.0]
Piksel 3:    [b01=0.052, b02=0.418, ..., ndvi=0.778, LST_D=30.8, LST_N=17.2]
...
Piksel 5700: [b01=0.051, b02=0.435, ..., ndvi=0.789, LST_D=31.3, LST_N=17.9]

           ↓  rata-rata semua 5.700 baris

1 baris CSV:
GEOID  date        b01    b02    b03  ...  ndvi   LST_Day  LST_Night
19141  2020-08-04  0.051  0.430  0.028 ... 0.790  31.2     17.8
```

---

### 9.5. Skala: 1 Baris → 1 County → 1 Tahun → Semua Data

```
1 baris  = 1 county × 1 tanggal komposit 8-hari
         = hasil rata-rata ~5.700 piksel
         = 11 angka (GEOID, date, 9 fitur → tapi di tensor jadi 10 fitur tanpa EVI)

1 county = 46 baris (satu per komposit 8-hari sepanjang tahun)
         = time series lengkap 1 county selama 1 tahun

1 file CSV (1 tahun, semua county):
         = ~2.280 county × 46 tanggal
         = ~104.880 baris

21 file CSV (2003–2023):
         = ~104.880 × 21
         = ~2.202.480 baris total
```

---

### 9.6. FIPS sebagai Kunci Penghubung di Seluruh Pipeline

```
Sumber data        Nama kolom    Contoh nilai    Keterangan
─────────────────  ────────────  ──────────────  ─────────────────────────
yield parquet      region_id     "17019"         FIPS 5-digit (nama kolom: region_id)
TIGER di GEE       GEOID         "17019"         FIPS 5-digit (nama kolom: GEOID)
MODIS CSV output   GEOID         "17019"         FIPS 5-digit (diwariskan dari TIGER)

Proses:
  1. Script ambil region_id dari yield → ekstrak 2 digit pertama = state FIPS
  2. Filter TIGER county: hanya county dari state yang ada di yield
  3. GEE ekstrak MODIS per polygon → output GEOID = FIPS county itu
  4. Merge script: GEOID (MODIS) == region_id (yield) → pasangan X dan y
```

---

### 9.7. Dari CSV ke Tensor — Reshape Final

```
LANGKAH A: groupby (county, year) → ambil 46 baris

  19141 + 2020 → 46 baris CSV:
  [2020-01-01: b01=0.08, ndvi=0.12, LST_D=-8.5, ...]   ← t=0
  [2020-01-09: b01=0.08, ndvi=0.11, LST_D=-10.2, ...]  ← t=1
  ...
  [2020-12-27: b01=0.08, ndvi=0.11, LST_D=-4.1, ...]   ← t=45

LANGKAH B: .values → buang kolom tanggal/GEOID/year → ambil 10 fitur saja

  array shape (46, 10):
       b01   b02  ...  ndvi   LST_D   LST_N
  t=0  [0.08  0.15 ...  0.12  -8.5   -18.2]
  t=1  [0.08  0.15 ...  0.11 -10.2   -19.5]
  ...
  t=25 [0.05  0.43 ...  0.79  31.2    17.8]  ← puncak musim tanam
  ...
  t=45 [0.08  0.15 ...  0.11  -4.1   -12.3]

LANGKAH C: lookup yield → 1 angka label

  yield_lookup[("19141", 2020)] = 13.4 ton/ha

LANGKAH D: pasangkan X dan y, ulang untuk semua county-year valid

  np.stack semua X → (N, 46, 10)
  np.array semua y → (N,)
```

---

### 9.8. Tensor Final dan Maknanya

```
X = (N, T=46, F=10)
y = (N,)

N  = jumlah sampel VALID (county × tahun yang punya KEDUANYA: yield + MODIS)
T  = 46 timesteps (komposit 8-hari, Januari–Desember)
F  = 10 fitur per timestep (b01–b07, ndvi, LST_Day, LST_Night)
```

**Kenapa N bukan 2.280 × 21 = 47.880?**

```
2.280 county × 21 tahun = 47.880 kombinasi county-year teoritis
Dikurangi:
  - yield NaN (disensor NASS, "(D)") → ~11.000 county-year gugur
  - MODIS gagal export / < 46 timesteps → ~4.000 county-year gugur
  - yield = 0 (anomali data) → 206 sampel gugur

Tersisa: ~32.296 sampel valid → N = 32.296
```

---

### 9.9. Yang Dipelajari Model dari Tensor Ini

Model LSTM membaca X[i] dari t=0 sampai t=45 (seluruh tahun), lalu prediksi y[i].

Pola yang dicari model:

```
Pola 1 — yield tinggi:
  Jan–Apr:  NDVI rendah  (~0.1)  → lahan kosong/bersalju, normal
  Mei–Jun:  NDVI naik     (0.3→0.6) → jagung tanam, tumbuh
  Jul–Agu:  NDVI puncak   (~0.8) → kanopi penuh, sehat
  Sep–Okt:  NDVI turun    (0.8→0.3) → mature, mulai panen
  → Pola kurva lengkung jelas → model prediksi yield tinggi

Pola 2 — yield rendah (misal kekeringan):
  Jan–Apr:  NDVI rendah   (~0.1)  → normal
  Mei–Jun:  NDVI naik      (0.3→0.5) → mulai tanam
  Jul:      NDVI flat/turun (0.5→0.4) → kekeringan, tanaman stress
  Agu–Sep:  NDVI flat      (~0.4)  → tidak capai puncak optimal
  → Kurva terpotong, puncak lebih rendah → model prediksi yield rendah

Pola 3 — county banyak non-jagung (kota):
  NDVI flat sepanjang tahun (~0.2–0.4) → tidak ada siklus musiman jelas
  LST_Day tinggi terus → banyak aspal/bangunan menyerap panas
  → Model belajar ini bukan sinyal jagung yang reliable
```

**Pertanyaan wajib bisa dijawab:**
- [ ] Kenapa 1 piksel MODIS dirata-rata dengan ribuan piksel lain dalam 1 county?
- [ ] Apa yang dimaksud "RASTER → TABULAR"? Apa yang berubah?
- [ ] Kenapa N = 32.296, bukan 2.280 × 21 = 47.880?
- [ ] Kenapa pola NDVI musiman penting untuk prediksi yield?
- [ ] Apa yang terjadi pada county dengan sungai besar? Bagaimana cropland mask membantu?
- 



Pertanyaan

     1. Sebagai sesama researcher: Kalau 1 county = 46 baris CSV per tahun, dan yield cuma 1 angka — berarti 46 input mapping ke 1
        output. Apakah ini artinya model harus "menebak" kontribusi tiap tanggal terhadap yield? Bagaimana model tahu bahwa NDVI Juli
        lebih penting daripada NDVI Januari?

     2. Sebagai supervisor: Satu county punya ribuan piksel yang dirata-ratakan jadi 1 angka. Bukankah ini oversimplifikasi realitas
        agronomis — di mana dua ladang dalam county yang sama bisa berbeda pengelolaan, varietas, dan hasil panen? Bagaimana Anda
        mempertanggungjawabkan ini?

    1. Sesama researcher: Kenapa kamu pilih ee.Reducer.mean() bukan median() atau histogram()? You et al. (2017) yang kamu pakai sebagai referensi justru pakai histogram 32-bin, bukan
  mean. Apa tradeoff memilih mean untuk thesis ini?

  2. Supervisor: Scale scale=500 di reduceRegions() berarti kamu pakai resolusi native MODIS. Tapi county di USA ada yang sangat kecil (<100 km²) — hanya punya ~400 pixel. County besar
   (~5,000 km²) punya ~20,000 pixel. Apakah rata-rata dari 400 pixel sama reliablenya dengan rata-rata dari 20,000 pixel? Bagaimana kamu tangani county-county kecil ini di model?

  3. Sebagai murid ke pembimbing: Saya belum menerapkan cropland mask dan mengestimasi dampaknya ~+0.2 R². Apakah ekspektasi ini realistis, atau ada faktor lain yang bisa membuat mask
  justru menurunkan performa (misalnya: county dengan cropland mask sangat sedikit pixel → mean tidak stabil)?



        