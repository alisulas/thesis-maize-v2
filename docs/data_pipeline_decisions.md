# Keputusan Desain Data Pipeline — Justifikasi & Alternatif

> **Untuk apa dokumen ini?**
> Menjawab pertanyaan supervisor: *"Mengapa kamu membuat keputusan X, bukan Y?"*
> Setiap keputusan dicatat bersama alasannya, alternatif yang ada, dan risikonya.
>
> **Dokumen terkait:**
> - `docs/01_pipeline.md` — gambaran umum alur data
> - `docs/knowledge_datapipe.md` — penjelasan teknis tiap step
> - `src/data/001_download_usa.py` — implementasi step 1
> - `src/data/003_merge_modis.py` — implementasi step 3
> - `src/data/004_dataset.py` — implementasi step 4

---

## Peta Keputusan per Step

```
Step 1: Download + Cleaning Yield
  ├─ [K1]  Sumber data: USDA NASS QuickStats
  ├─ [K2]  Granularitas county (bukan farm, bukan state)
  ├─ [K3]  "(D)" → NaN, bukan imputasi
  ├─ [K4]  Yield < 0.1 ton/ha → NaN
  └─ [K5]  Satuan: bu/acre → ton/ha

Step 2: Ekstraksi Satelit (GEE)
  ├─ [K6]  MODIS dipilih, bukan Sentinel-2 atau Landsat
  ├─ [K7]  Interval 8-harian (46 timestep/tahun)
  ├─ [K8]  Agregasi: rata-rata piksel per county (mean, bukan histogram)
  └─ [K9]  EVI dibuang

Step 3: Merge MODIS + Yield
  ├─ [K10] NaN karena awan: forward-fill → backward-fill → isi 0
  └─ [K11] County-year tanpa yield label dibuang (bukan diimputasi)

Step 4: Dataset PyTorch
  ├─ [K12] Temporal split (bukan random)
  ├─ [K13] Rasio split: 18/2/1 tahun untuk train/val/test
  └─ [K14] Normalisasi Z-score, fit dari train saja
```

---

## Step 1: Download + Cleaning Yield

### [K1] Sumber Data: USDA NASS QuickStats

**Keputusan:** Menggunakan USDA NASS QuickStats API sebagai sumber yield jagung USA.

**Alasan:**
- Satu-satunya sumber resmi data yield jagung per county level di USA yang bersifat publik dan gratis
- Konsisten 2003–2023 tanpa gap metodologi (metodologi survei NASS konsisten sepanjang periode)
- Digunakan oleh hampir semua paper benchmark yang relevan (Wang et al. 2018, Khaki et al. 2021, Ma et al. 2021–2023)

**Alternatif yang ada:**
| Alternatif | Mengapa tidak dipilih |
|---|---|
| RMA (Risk Management Agency) | Data asuransi pertanian, hanya county yang ikut asuransi |
| State Agriculture Departments | Tidak konsisten antar state, banyak yang tidak publik |
| FAOSTAT | Hanya tersedia di level nasional, tidak per county |

**Risiko:** NASS menyensor data county kecil ("(D)") → ~15% data tidak tersedia. Sudah ditangani dengan NaN.

---

### [K2] Granularitas County (bukan farm atau state)

**Keputusan:** Menggunakan county sebagai unit spasial, setara kabupaten.

**Alasan:**
- Granularitas terkecil yang dipublikasikan USDA secara konsisten
- Setara dengan level data yang tersedia di Indonesia (kabupaten/kota dari BPS)
- MODIS (500m) cukup untuk representasi county (~1.000–5.000 km²), tapi terlalu kasar untuk farm-level
- Mayoritas paper transfer learning crop yield (Wang 2018, Khaki 2021, Ma 2021–2023) menggunakan county sebagai unit

**Alternatif yang ada:**
| Level | Mengapa tidak dipilih |
|---|---|
| Farm/field | USDA tidak publikasi, privasi petani; perlu data drone/Sentinel-2 per field |
| HUC watershed | Tidak align dengan batas administratif Indonesia |
| State | Terlalu kasar; kehilangan variasi spasial yang jadi sinyal utama model |

**Implikasi untuk transfer learning:** Indonesia menggunakan kabupaten yang secara konsep setara dengan county → domain transfer lebih valid secara metodologis.

---

### [K3] Nilai "(D)" → NaN, Bukan Imputasi

**Keputusan:** Nilai yang disensor NASS ("(D)" = disclosure suppressed) dikonversi ke NaN dan dibuang dari dataset, bukan diimputasi.

**Alasan:**
- "(D)" berarti USDA menyensor karena hanya 1–2 petani di county tersebut — tidak ada angka asli yang bisa dijadikan acuan imputasi
- Imputasi (rata-rata county sekitar, dll.) akan memasukkan **informasi palsu** sebagai label training — ini lebih berbahaya daripada kehilangan sampel
- ~15% missing rate masih aman; lebih dari cukup dengan 32.296 sampel tersisa

**Alternatif yang ada:**
| Alternatif | Masalah |
|---|---|
| Imputasi dengan mean county sekitar | Spatial leakage: county tetangga bukan prediktor yield yang valid |
| Imputasi dengan nilai tahun sebelumnya | Temporal leakage: data "masa depan" masuk ke training |
| Biarkan sebagai 0 | Model belajar bahwa 0 = "tidak diketahui" bukan "gagal panen" — bias besar |

---

### [K4] Yield < 0.1 ton/ha → NaN (dilakukan di Step 1)

**Keputusan:** Yield yang setelah konversi bernilai di bawah 0.1 ton/ha dikonversi ke NaN dan dibuang.

**Alasan:**
- 0.1 ton/ha ≈ 1.6 bu/acre. Yield jagung terendah yang pernah tercatat secara realistis di USA adalah ~20 bu/acre (kekeringan ekstrem)
- Nilai di bawah 0.1 ton/ha hampir pasti artifact pembulatan: nilai kecil (misal 0.001 bu/acre) yang lolos filter exact-zero karena `Value == 0` hanya menangkap nol persis
- Filter ini dilakukan di Step 1 (bukan di Step 4) agar data yang tersimpan di parquet sudah bersih — konsisten dengan prinsip *single responsibility*: cleaning di satu tempat

**Kenapa bukan di Step 4?**
- Kalau filter tersebar di beberapa file, ketika seseorang membaca `usa_modis.npz` tanpa menjalankan `004_dataset.py`, mereka akan mendapat data kotor
- Dokumen audit data lebih mudah: semua keputusan cleaning ada di `clean_nass()`

**Alternatif threshold:**
| Threshold | Risiko |
|---|---|
| 0.0 (hanya buang exact zero) | Lolos nilai 0.001 bu/acre yang tetap tidak masuk akal |
| 1.0 ton/ha | Terlalu agresif — tahun kekeringan parah seperti 2012 punya county di Kansas dengan yield 0.77 ton/ha (riil) |
| **0.1 ton/ha** ← pilihan | Sweet spot: menangkap artifact tanpa membuang kasus ekstrem yang valid |

---

### [K5] Satuan: bu/acre → ton/ha

**Keputusan:** Konversi dari bushel per acre ke ton per hektar.

**Alasan:**
- NASS melaporkan dalam bu/acre (unit Amerika), sedangkan Indonesia (BPS) dalam ton/ha
- Normalisasi unit wajib sebelum training — model tidak boleh belajar dari perbedaan satuan
- Faktor konversi: `1 bu/acre × 0.06277 = ton/ha` (1 bushel jagung = 56 lb; 1 acre = 4046.86 m²)

**Verifikasi manual:** 180 bu/acre × 0.06277 = 11.3 ton/ha ✓ (masuk akal untuk yield jagung Iowa)

---

## Step 2: Ekstraksi Satelit (GEE)

### [K6] MODIS Dipilih, Bukan Sentinel-2 atau Landsat

**Keputusan:** Menggunakan MODIS (MOD09A1 + MYD11A2) sebagai sumber data satelit.

**Alasan:**
- Satu-satunya produk satelit yang tersedia konsisten **2003–2023** dengan interval 8-harian
- Gratis dan terbuka lewat Google Earth Engine
- Digunakan oleh semua paper benchmark dalam domain ini

**Perbandingan:**
| Sensor | Resolusi | Interval | Tersedia sejak | Masalah |
|---|---|---|---|---|
| **MODIS** ← pilihan | 500m | 8 hari | 2003 | Resolusi rendah |
| Sentinel-2 | 10m | 5 hari | 2015 | Tidak bisa 2003–2014 |
| Landsat 8 | 30m | 16 hari | 2013 | Terlalu jarang, tidak ada 2003–2012 |
| Landsat 7 | 30m | 16 hari | 1999 | Scan Line Corrector failure 2003 |

**Trade-off yang diterima:** Resolusi 500m per piksel berarti satu piksel bisa mencakup beberapa lahan berbeda. Untuk county-level prediction, ini acceptable karena kita rata-ratakan semua piksel dalam county.

---

### [K7] Interval 8-Harian (46 Timestep/Tahun)

**Keputusan:** Menggunakan komposit 8-harian MODIS (46 observasi per tahun).

**Alasan:**
- MODIS memproduksi komposit 8-harian sebagai produk standar (MOD09A1) — bukan keputusan arbitrer, ini interval asli sensornya
- 46 timestep cukup untuk menangkap fenologi jagung: penanaman (April–Mei), pertumbuhan vegetatif (Juni–Juli), puncak (Agustus), panen (September–Oktober)
- Lebih jarang (misal bulanan = 12 timestep) akan kehilangan resolusi temporal fase kritis seperti silking

---

### [K8] Agregasi: Rata-rata Piksel per County (Mean)

**Keputusan:** Piksel dalam county diagregasi menggunakan rata-rata (mean) menjadi satu nilai per band per timestep.

**Alasan:**
- Label yield juga agregat county — satu angka untuk seluruh county → fitur juga harus satu angka per county agar dimensi cocok
- Mean adalah agregator yang paling umum dipakai dan paling mudang diinterpretasikan
- Secara komputasi paling efisien di GEE (`ee.Reducer.mean()`)

**Alternatif yang mungkin:**
| Metode | Kelebihan | Kelemahan |
|---|---|---|
| **Mean** ← pilihan | Sederhana, interpretable | Kehilangan informasi distribusi spasial |
| Histogram (32 bins) | Menangkap distribusi piksel (Khaki et al. 2021) | Dimensi jauh lebih besar, kompleks di GEE |
| Mean + Std + Q1/Q3 | Menangkap sebaran | Dimensi fitur bertambah 3–5x |
| Median | Robust terhadap outlier | Lebih lambat di GEE |

**Status:** Keputusan ini **terbuka untuk eksperimen**. Dosen menyarankan menambah statistik seperti median, std, Q1/Q3. Paper Khaki et al. (2021) menggunakan histogram. Ini adalah salah satu hyperparameter desain yang perlu dieksperimentasi.

---

### [K9] EVI Dibuang

**Keputusan:** EVI (Enhanced Vegetation Index) dihitung di GEE tetapi tidak dimasukkan ke tensor final.

**Alasan:** EVI menghasilkan nilai overflow (±10¹¹) saat dirata-rata per county. Ini terjadi karena formula EVI memiliki penyebut yang mendekati nol ketika rata-rata reflektansi dimasukkan, menyebabkan pembagian tidak stabil. NDVI (yang sudah ada) memberikan informasi vegetasi yang serupa tanpa masalah numerik ini.

**Referensi keputusan:** `memory/project_decisions.md`

---

## Step 3: Merge MODIS + Yield

### [K10] NaN karena Awan: Forward-fill → Backward-fill → Isi 0

**Keputusan:** Piksel yang NaN karena tutupan awan diisi dengan urutan: forward-fill dulu, lalu backward-fill, lalu isi 0 jika masih ada NaN.

**Alasan logika urutan:**
```
Timestep: Jan  Feb  Mar  Apr  ← awan ←  Jun
Raw NDVI:  0.20 NaN  NaN  NaN         0.55

Forward-fill (pakai nilai SEBELUM awan):
           0.20 0.20 0.20 0.20         0.55

Interpretasi: "kondisi vegetasi bulan lalu masih relevan untuk hari ini yang berawan"
```

**Kenapa forward-fill dulu, bukan backward-fill?**
- Forward-fill menggunakan masa lalu → tidak ada temporal leakage
- Backward-fill menggunakan masa depan → bisa dipakai hanya sebagai fallback untuk kasus NaN di awal tahun (belum ada nilai sebelumnya)
- Urutan ini meniru praktik di paper referensi (Wang et al. 2018)

**Kenapa isi 0 sebagai last resort?**
- County dengan NaN di SELURUH 46 timestep (misalnya county yang tertutup awan sepanjang tahun) tidak bisa di-fill → 0 menandai "tidak ada informasi"
- County-county ini kemungkinan besar akan ter-drop saat merge karena tidak punya yield label juga

**Alternatif:**
| Metode | Masalah |
|---|---|
| Interpolasi linear | Bisa "mengarang" puncak NDVI yang tidak pernah terjadi |
| Rata-rata tahun lain | Kompleks, butuh data lintas tahun |
| Buang timestep ber-NaN | County kehilangan dimensi → tensor tidak bisa di-stack (harus persegi) |

---

### [K11] County-Year tanpa Yield Label Dibuang

**Keputusan:** Jika MODIS data ada tapi yield label tidak tersedia (NaN), pasangan tersebut dibuang.

**Alasan:** Tanpa label, tidak bisa supervised learning. Tidak ada cara imputasi yield yang valid.

**Dampak:** 32.296 dari 34.180 yield records masuk ke tensor (~94.5%). Sekitar 5.5% hilang karena: MODIS CSV tidak tersedia untuk county-year tersebut, atau MODIS timestep tidak lengkap (≠ 46).

---

## Step 4: Dataset PyTorch

### [K12] Temporal Split (Bukan Random)

**Keputusan:** Data dibagi berdasarkan tahun, bukan secara acak.

```
Train : 2003–2020
Val   : 2021–2022
Test  : 2023
```

**Alasan:**
- Data yield bersifat temporal: yield 2023 dipengaruhi kondisi iklim dan teknologi yang terakumulasi dari tahun-tahun sebelumnya
- Random split akan mengizinkan data dari tahun yang sama masuk ke train dan test — model "melihat masa depan" saat evaluasi
- Temporal split mengukur kemampuan model untuk **generalize ke tahun yang belum pernah dilihat**, bukan hanya menghafal pola tahun tertentu
- Semua paper benchmark menggunakan temporal split untuk crop yield prediction

**Bukti masalah random split:** Simulasi di `notebooks/05_dataset.ipynb` Bagian 2 menunjukkan bahwa random split 80/10/10 menghasilkan semua 21 tahun overlap antara train dan test.

---

### [K13] Rasio Split: 18/2/1 Tahun

**Keputusan:** Train 18 tahun (2003–2020), Val 2 tahun (2021–2022), Test 1 tahun (2023).

**Alasan:**
- Test hanya 1 tahun (2023): ini adalah tahun paling baru yang belum "dilihat" model → paling representatif untuk evaluasi out-of-sample
- Val 2 tahun: cukup untuk tune hyperparameter tanpa mengorbankan terlalu banyak data training
- Train 18 tahun: memaksimalkan data training — model deep learning butuh data sebanyak mungkin untuk convergence

**Konsekuensi:**
- Test set hanya 1.349 sampel (4.2% dari total) → error bar evaluasi lebih lebar dibanding test set lebih besar
- Ini trade-off yang diterima karena temporal integrity lebih penting daripada ukuran test set

---

### [K14] Normalisasi Z-Score, Fit dari Train Saja

**Keputusan:** Fitur dinormalisasi menggunakan Z-score (`(X - mean) / std`), dengan mean dan std dihitung hanya dari training set.

**Alasan Z-score (bukan Min-Max):**
- Z-score robust terhadap outlier: satu county dengan NDVI ekstrem tidak akan menekan semua nilai lain ke range sempit
- Min-Max sangat sensitif terhadap outlier: satu nilai maksimum yang tidak biasa akan mengkompres sebagian besar data ke range kecil
- Z-score menghasilkan distribusi terpusat di 0 dengan std 1, yang cocok untuk inisialisasi weight neural network standar (glorot/he initialization mengasumsikan input ~N(0,1))

**Kenapa fit dari train saja:**
- Kalau mean/std dihitung dari semua data (termasuk val/test), informasi statistik test set "bocor" ke training → **data leakage**
- Contoh konkret: misal tahun 2023 punya anomali cuaca ekstrem → NDVI sangat rendah. Kalau std dihitung dari semua data termasuk 2023, nilai 2023 akan ter-normalisasi lebih "normal" dari seharusnya — model tidak "terkejut" dengan nilai ekstrem ini saat evaluasi

**Alternatif:**
| Metode | Masalah |
|---|---|
| Min-Max scaling | Sensitif outlier; range [0,1] tidak berubah kalau distribusi bergeser di target domain |
| **Z-score** ← pilihan | Stabil, distribusi tetap informatif |
| Robust scaling (median/IQR) | Lebih kuat, tapi tidak standar di literatur crop yield |
| Tanpa normalisasi | LST skala ratusan, NDVI skala 0–1 → gradient dominasi fitur berskala besar |

---

## Ringkasan: Keputusan yang Masih Terbuka untuk Eksperimen

Beberapa keputusan di atas bukan final — ini adalah *default* yang perlu dieksperimentasi:

| Keputusan | Status | Eksperimen yang direkomendasikan |
|---|---|---|
| [K8] Mean sebagai agregator | Terbuka | Bandingkan Mean vs Mean+Std vs Histogram-32-bins |
| [K13] Rasio split 18/2/1 | Terbuka | Coba 15/3/3 untuk test set lebih besar |
| [K14] Z-score normalisasi | Terbuka | Bandingkan Z-score vs Robust scaler |
| [K4] Threshold 0.1 ton/ha | Terbuka | Sensitivity analysis: coba 0.5 dan 1.0 |

---

*Dokumen ini harus diupdate setiap kali ada keputusan pipeline yang berubah.*
*Terakhir diupdate: 26 Mei 2026.*
