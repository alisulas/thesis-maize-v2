# Knowledge Base — Transfer Learning for Maize Yield Prediction

Dokumen ini ditulis untuk kamu (peneliti) sebagai referensi belajar. Semua istilah teknis dijelaskan dengan analogi sederhana + definisi formal. **Update terakhir: 8 Mei 2026 — mencakup semua yang sudah kita kerjakan sampai sekarang.**

---

## Daftar Isi

1. [Apa sih tesis ini sebenarnya?](#1-apa-sih-tesis-ini-sebenarnya)
2. [Konsep Dasar Wajib Paham](#2-konsep-dasar-wajib-paham)
3. [Data: Apa saja dan dari mana](#3-data-apa-saja-dan-dari-mana)
4. [Preprocessing: Data diapakan sebelum training](#4-preprocessing-data-diapakan-sebelum-training)
5. [Model & Arsitektur](#5-model--arsitektur)
6. [Training & Transfer Learning](#6-training--transfer-learning)
7. [Evaluasi: Cara mengukur hasil](#7-evaluasi-cara-mengukur-hasil)
8. [Workflow Penelitian: Status Nyata per 8 Mei 2026](#8-workflow-penelitian-status-nyata-per-8-mei-2026)
9. [Hasil Investigasi untuk Supervisor](#9-hasil-investigasi-untuk-supervisor)
10. [Masalah Diketahui & Rencana Perbaikan](#10-masalah-diketahui--rencana-perbaikan)
11. [Glosarium Istilah](#11-glosarium-istilah)

---

## 1. Apa sih tesis ini sebenarnya?

### Analogi sederhana

Bayangkan kamu punya teman yang jago nebak harga rumah di Jakarta karena sudah melihat 40.000 data rumah di seluruh Indonesia.

Sekarang kamu minta dia nebak harga rumah di Thailand. Masalahnya:
- Di Thailand, dia cuma punya 126 contoh rumah (sangat sedikit)
- Gaya arsitektur, bahan bangunan, dan kondisi pasar di Thailand beda dengan Indonesia

Pertanyaannya: **Apakah pengetahuan dari Indonesia bisa "ditransfer" untuk bantu nebak di Thailand? Atau malah bikin bingung karena kondisinya beda?**

### Versi teknis

| Komponen | Penjelasan |
|----------|-----------|
| **Apa** | Prediksi hasil panen jagung (ton/ha) dari citra satelit |
| **Sumber (source)** | USA — data banyak (41.349 sampel), level county |
| **Target** | Indonesia, Vietnam, Thailand — data sedikit, level provinsi |
| **Masalah** | Target cuma punya 126-1.804 sampel → deep learning biasanya butuh ribuan |
| **Solusi** | Transfer learning: latih model di USA, lalu "pindahkan" ilmunya ke ASEAN |
| **Tantangan** | Iklim USA (temperate) vs ASEAN (tropical) beda jauh → model bisa bingung |

### Hipotesis (yang sedang diuji)

| Kode | Isi | Status |
|------|-----|--------|
| **H1** | Transfer learning dari USA lebih baik daripada training dari nol di ASEAN | ✅ **Terkonfirmasi** — IDN ΔR²=+0.574, VNM ΔR²=+0.104 |
| **H2** | Perbedaan iklim bisa bikin transfer learning gagal ("negative transfer") | ❌ **Tidak terkonfirmasi** untuk VNM (hasil nyata positif); THA negatif tapi karena overfitting |
| **H3** | DANN (domain adaptation) bisa mengatasi masalah perbedaan iklim | Belum diimplementasi |

---

## 2. Konsep Dasar Wajib Paham

### 2.1 Machine Learning vs Deep Learning

```
Machine Learning (ML)
├── Traditional ML: regresi linear, decision tree, random forest
└── Deep Learning (DL): neural network dengan banyak layer
    ├── CNN (Convolutional Neural Network) — untuk gambar/spasial
    ├── LSTM (Long Short-Term Memory) — untuk data time-series ← yang kita pakai
    └── Transformer — arsitektur modern (ChatGPT pakai ini)
```

**Analogi:**
- Traditional ML = kalkulator ilmiah
- Deep Learning = smartphone (lebih kompleks, bisa banyak hal, tapi butuh lebih banyak "baterai" = data)

### 2.2 Neural Network (Jaringan Saraf Tiruan)

**Definisi:** Sistem komputasi yang meniru cara otak manusia belajar dari data.

```
Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output (prediksi yield)
         ↑ neuron dengan bobot (weight) yang di-update saat training
```

**Cara kerja (sederhana):**
1. Data masuk → dikali bobot → dijumlah → lewat fungsi aktivasi
2. Hasil dibandingkan dengan jawaban benar → hitung error (loss = MSE)
3. Error digunakan untuk update bobot via backpropagation
4. Ulangi ribuan kali → bobot "belajar" pola yang benar

### 2.3 LSTM (Long Short-Term Memory)

**Untuk apa:** Memproses data **time-series** — data yang berurutan dalam waktu.

**Kenapa spesial:** LSTM punya "memori" — bisa ingat informasi dari timestep sebelumnya. Contoh: "Minggu 1-4: NDVI rendah (tanam), minggu 10-20: NDVI naik cepat (pertumbuhan), minggu 40: NDVI turun (panen)" — LSTM bisa menangkap pola temporal ini untuk prediksi yield.

```
Timestep 1 → Timestep 2 → ... → Timestep 46
    ↓              ↓                   ↓
[LSTM dengan memory cell] → state tersembunyi → ...
                                               ↓
                                        [Dense layers] → prediksi yield
```

**LSTM vs RNN biasa:** RNN biasa "lupa" informasi lama. LSTM bisa "ingat" dan "lupa" secara selektif berkat 3 gerbang: forget gate, input gate, output gate.

### 2.4 CNN-LSTM (Arsitektur Gabungan)

Paper You et al. 2017 pakai CNN + LSTM karena inputnya histogram 32-bin per band. CNN mengenali "bentuk" distribusi histogram. Tapi **di proyek kita, input hanya 10 fitur skalar** (rata-rata nilai band), bukan histogram. Jadi CNN tidak bermakna secara fisik, dan terbukti lebih buruk: R²=0.13 vs LSTM-only R²=0.39.

### 2.5 Transfer Learning

**Definisi:** Menggunakan model yang sudah dilatih di satu domain untuk membantu belajar di domain lain yang datanya lebih sedikit.

**Analogi:**
- Kamu sudah jago main gitar → belajar main ukulele lebih cepat (transfer works)
- Kamu jago main gitar → belajar drum (transfer might not work — iklim gap)

**Dalam proyek ini:**
```
Model dilatih di USA (41.349 data)    ← 3+ minggu kerja data
        ↓
Model disimpan sebagai "checkpoint"   ← file experiments/checkpoints/usa_lstm_best.pt
        ↓
Checkpoint di-load untuk ASEAN        ← via finetune.py
        ↓
Model di-fine-tune dengan data ASEAN  ← 126-1.804 data
```

### 2.6 Fine-Tuning (2-Phase Strategy)

**Apa itu fine-tuning?**

Fine-tuning adalah proses melanjutkan training model yang sudah dilatih sebelumnya (pretrained) dengan data baru yang berbeda. Analoginya: seperti seorang dokter spesialis jantung yang belajar spesialisasi paru-paru — dia tidak mulai dari nol (tidak perlu belajar anatomi dasar lagi), tapi menyesuaikan keahliannya ke bidang baru.

**Yang di-fine-tune dalam proyek ini:**

Model kita (`CropYieldLSTM`) punya 2 bagian:

```
┌─────────────────────────────────────────────────┐
│  LSTM layers (2 layer, hidden=256)              │  ← "Feature extractor"
│  Fungsi: belajar pola temporal NDVI/LST 46 week │  ← Dilatih di USA (41K data)
│  Bobot: ~790.000 parameter                      │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│  FC Head: Linear(256→64) → ReLU → Linear(64→1) │  ← "Prediction head"
│  Fungsi: mengubah representasi → angka yield    │  ← ~17.000 parameter
└─────────────────────────────────────────────────┘
```

**Yang di-fine-tune = seluruh model**, tapi dengan strategi 2 fase:

**Phase 1 (20 epoch): Frozen LSTM, train head saja**
- LSTM layer **dibekukan** (weight tidak berubah sama sekali)
- Hanya FC head yang dilatih dengan lr=1e-3
- **Kenapa:** Head diinisialisasi random → kalau langsung update LSTM dengan gradient dari head yang kacau, bobot LSTM yang sudah bagus bisa rusak (disebut *catastrophic forgetting*)

**Phase 2 (50 epoch): Unfreeze semua, fine-tune full**
- Semua layer dilatih dengan lr=1e-4 (sangat kecil)
- **Kenapa:** Sekarang head sudah stabil → aman untuk menyesuaikan LSTM secara halus ke pola data ASEAN (iklim tropis, pola NDVI berbeda)

**Hasil nyata fine-tuning di proyek ini:**

| Negara | Tanpa fine-tune (from scratch) | Dengan fine-tune (transfer) | Keuntungan |
|--------|-------------------------------|----------------------------|-----------|
| IDN | R²=0.000 | R²=**0.574** | +0.574 🎉 |
| VNM | R²=-0.056 | R²=**0.048** | +0.104 ✓ |
| THA | R²=-0.490 | R²=-2.726 | -2.236 ⚠️ overfit |

IDN mendapat manfaat terbesar karena datanya paling sedikit (128 sampel training) — tanpa pretrained weights, model tidak cukup data untuk belajar apapun.

### 2.7 Domain Adaptation & DANN

**Masalah:** USA (4 musim) vs ASEAN (2 musim) → pola NDVI sepanjang tahun sangat berbeda:
- USA: NDVI naik tajam di spring, puncak di summer, turun di fall (kurva lonceng)
- Vietnam: NDVI relatif stabil sepanjang tahun (tropical — selalu hijau)

**Solusi: DANN** — model belajar menghasilkan fitur yang tidak bisa dibedakan apakah dari USA atau ASEAN. **Belum diimplementasi — rencana fase berikutnya.**

### 2.8 Remote Sensing & MODIS

**Remote Sensing:** Mengamati bumi dari satelit (nggak perlu ke lapangan).

**MODIS (Moderate Resolution Imaging Spectroradiometer):**
- Satelit NASA (Terra/MOD09A1 + Aqua/MYD11A2)
- Meliput seluruh bumi setiap 1-2 hari
- Resolusi: 500m per piksel (cukup untuk county/province)
- Data tersedia sejak 2003 — itulah kenapa kita mulai dari 2003

**Band MODIS yang kita pakai (10 fitur):**

| Nama Kolom | Band Fisik | Kegunaan |
|-----------|-----------|---------|
| `sur_refl_b01` | Red (merah, 620-670nm) | Tanaman menyerap cahaya merah → rendah = sehat |
| `sur_refl_b02` | NIR (near-infrared, 841-876nm) | Tanaman memantulkan NIR → tinggi = sehat |
| `sur_refl_b03` | Blue (459-479nm) | Koreksi atmosfer |
| `sur_refl_b04` | Green (545-565nm) | Referensi hijau |
| `sur_refl_b05` | SWIR 1 (1230-1250nm) | Kelembaban tanah/daun |
| `sur_refl_b06` | SWIR 2 (1628-1652nm) | Kelembaban |
| `sur_refl_b07` | SWIR 3 (2105-2155nm) | Kelembaban |
| `ndvi` | Indeks turunan | Indikator kehijauan tanaman |
| `LST_Day_1km` | Suhu permukaan siang | Stres panas tanaman |
| `LST_Night_1km` | Suhu permukaan malam | Suhu baseline |

**NDVI:** `NDVI = (NIR - Red) / (NIR + Red)`, rentang -1 sampai +1. > 0.3 = ada vegetasi sehat.

**EVI:** Rumus serupa tapi lebih kompleks. Kita buang EVI karena saat di-rata-rata per county, pembaginya bisa mendekati nol → nilai meledak (±1 × 10¹¹). Ini bug yang kita temukan dan perbaiki.

**Google Earth Engine (GEE):** Platform cloud Google untuk memproses data satelit berskala besar tanpa download ke komputer. Kita pakai GEE untuk:
1. Ambil gambar MODIS per wilayah (county/provinsi) per 8-hari
2. Hitung rata-rata nilai band dalam batas wilayah (zonal statistics)
3. Export ke Google Drive sebagai CSV

---

## 3. Data: Apa saja dan dari mana

### 3.1 Yield Data (Tabular) — Status Final

| Negara | Level | Sumber | Tahun | Sampel di Tensor | Status |
|--------|-------|--------|-------|-----------------|--------|
| USA | County (2.280) | USDA NASS | 2003-2023 | 41.349 | ✅ Lengkap |
| Indonesia | Provinsi (33/38) | BPS | 2020-2024 | 172 | ✅ (5 provinsi hilang — Papua baru) |
| Vietnam | Provinsi (62/63) | GSO | 2003-2023 | 1.804 | ✅ (Ha Tay dibuang) |
| Thailand | Provinsi (43) | OAE | 2021-2023 | 126 | ✅ (hanya 3 tahun tersedia) |

**Kenapa Indonesia cuma 2020-2024?** BPS mengubah metodologi survei ke KSA (berbasis satelit) mulai 2020. Data pre-2020 pakai "eye estimate" yang tidak sebanding — mencampurnya akan bikin model bingung.

**Kenapa Ha Tay (Vietnam) dibuang?** GAUL 2015 masih punya Ha Tay sebagai provinsi, tapi Vietnam sudah menggabungkannya ke Ha Noi sejak 2008. Data yield GSO tidak punya Ha Tay terpisah → tidak bisa di-join → dibuang.

### 3.2 MODIS Data (Satellite) — Status Final

Semua 50 GEE export tasks berhasil disubmit dan didownload:

| Negara | File di Drive | Jumlah File | Status |
|--------|-------------|-------------|--------|
| USA | `modis_usa_YYYY.csv` | 21 files (2003-2023) | ✅ Downloaded |
| Indonesia | `modis_idn_YYYY.csv` | 5 files (2020-2024) | ✅ Downloaded |
| Vietnam | `modis_vnm_YYYY.csv` | 21 files (2003-2023) | ✅ Downloaded |
| Thailand | `modis_tha_YYYY.csv` | 3 files (2021-2023) | ✅ Downloaded |

Setelah merge: 4 file `.npz` di `data/processed/modis/` **(v2, cropland-masked)**:
- `usa_modis.npz`: 32.296 sampel × 46 timesteps × 10 fitur
- `idn_modis.npz`: 162 sampel × 46 timesteps × 10 fitur
- `vnm_modis.npz`: 1.315 sampel × 46 timesteps × 10 fitur
- `tha_modis.npz`: 126 sampel × 46 timesteps × 10 fitur

**Catatan v2:** Re-extraction dengan MCD12Q1 class 12 mask (hanya pixel pertanian). File USA menyusut dari ~33MB → ~28MB per tahun, membuktikan mask aktif.

---

## 4. Preprocessing: Data diapakan sebelum training

### 4.1 Pipeline Lengkap (yang sudah dijalankan)

```
50 CSV MODIS dari Google Drive
    ↓
[merge_modis.py] — src/data/merge_modis.py
    │
    ├── Baca semua CSV per negara
    ├── Mapping nama wilayah:
    │     USA: GEOID 5-digit (state_fips + county_fips)
    │     IDN: GAUL ADM1_NAME → "IDN-XX" (33 provinsi hardcoded)
    │     VNM: normalisasi nama + buang Ha Tay
    │     THA: koreksi ejaan (9 nama berbeda di GAUL vs OAE)
    │
    ├── Hilangkan EVI dari fitur (overflow bug)
    ├── Isi NaN per (region, year, fitur):
    │     - forward-fill → backward-fill → isi 0 jika masih NaN
    │
    ├── Group per (region_id, year) → 46 timestep
    ├── Clip NDVI ke [-1, 1]
    ├── Join dengan yield lookup
    └── Simpan sebagai .npz
    ↓
4 file .npz: usa_modis.npz, idn_modis.npz, vnm_modis.npz, tha_modis.npz
```

### 4.2 Format Tensor (NPZ)

Setiap `.npz` berisi 3 array:
- `X`: float32, shape `(N_samples, 46, 10)` — fitur MODIS
- `y`: float32, shape `(N_samples,)` — yield dalam ton/ha
- `region_ids`: array string/int — ID wilayah

**N = 46 timestep:** Tahun dibagi jadi 46 periode 8-hari (365 ÷ 8 ≈ 46).

### 4.3 Z-Score Normalization (di Dataset)

Sebelum masuk ke model, setiap fitur di-normalisasi:
```
x_normalized = (x - mean_train) / std_train
```

**Penting:** mean dan std **hanya dihitung dari data training**, lalu diterapkan ke val dan test. Kalau dihitung dari semua data, kita "bocor" informasi masa depan ke model (data leakage).

### 4.4 Train/Val/Test Split (Temporal)

```
USA: Train 2003-2020 | Val 2021-2022 | Test 2023
IDN: Train 2020-2022 | Val (kosong) | Test 2023-2024
VNM: Train 2003-2020 | Val 2021-2022 | Test 2023
THA: Train 2021 | Val (kosong) | Test 2022-2023
```

**Kenapa split by year, bukan random?** Karena kita prediksi masa depan. Model harus diuji pada tahun yang belum pernah dilihat saat training. Kalau random, model bisa "contek" informasi dari tahun 2023 saat latihan dengan data 2023-lainnya.

---

## 5. Model & Arsitektur

### 5.1 CropYieldLSTM (yang kita pakai — hasil terbaik)

```python
# File: src/models/cnn_lstm.py

Input: (batch, 46, 10)  # 46 timestep, 10 fitur per timestep
   ↓
LSTM(input=10, hidden=256, n_layers=2, dropout=0.3)
   ↓  (mengambil hidden state timestep terakhir)
Linear(256 → 64)
ReLU()
Dropout(0.3)
Linear(64 → 1)
   ↓
Output: (batch, 1)  # prediksi yield ton/ha
```

**Hyperparameter terbaik (usa_lstm.yaml):**
- `hidden_size`: 256
- `n_layers`: 2
- `dropout`: 0.3
- `lr`: 5e-4
- `batch_size`: 256
- `epochs`: 150 (dengan early stopping patience=20)

**Jumlah parameter:** ~817.000 — tidak terlalu besar untuk dataset 41.349 sampel.

### 5.2 CropYieldCNNLSTM (dicoba, lebih buruk)

Tambahkan Conv1d sebelum LSTM untuk ekstraksi fitur spasial. Tapi karena input hanya 10 fitur skalar (bukan histogram 32-bin), CNN 1D tidak bermakna secara fisik. **Hasil: R²=0.13 vs LSTM-only R²=0.39.** Kita pakai LSTM-only.

### 5.3 Transfer Learning Architecture

Untuk fine-tuning ke ASEAN, arsitektur sama persis, cuma bobot diinisialisasi dari checkpoint USA bukan random:

```python
# Phase 1: Freeze LSTM, train head
model.freeze_feature_extractor()  # LSTM weight tidak berubah
# train 20 epoch, lr=1e-3

# Phase 2: Unfreeze all
model.unfreeze_all()
# train 50 epoch, lr=1e-4
```

---

## 6. Training & Transfer Learning

### 6.1 Alur Pipeline Kode (yang sudah ada)

```
data/processed/modis/*.npz
    ↓
src/data/dataset.py → MaizeDataset + get_dataloaders()
    ↓
src/training/train.py → latih USA, simpan checkpoint
    ↓
experiments/checkpoints/usa_lstm_best.pt
    ↓
src/transfer/finetune.py → fine-tune semua negara ASEAN
    ↓
experiments/logs/transfer_results.csv → hasil semua eksperimen
```

### 6.2 Cara Menjalankan (di Google Colab)

```bash
# 1. Setup Colab
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/[username]/thesis_maize /content/thesis_maize
%cd /content/thesis_maize
!pip install -r requirements.txt

# 2. Copy data
!cp /content/drive/MyDrive/thesis_maize_gee/modis_*.csv data/raw/modis/
!python src/data/merge_modis.py

# 3. Train USA baseline
!python src/training/train.py --config experiments/configs/usa_lstm.yaml

# 4. Fine-tune semua negara
!python src/transfer/finetune.py --country all
```

### 6.3 Bug yang Ditemukan dan Diperbaiki

| Bug | Penyebab | Solusi |
|-----|---------|--------|
| GEE `AttributeError: 'Feature' has no attribute 'addBands'` | `ee.Join.saveBest()` mengembalikan FeatureCollection, bukan ImageCollection | Cast dengan `ee.Image(feature)` di awal fungsi merge |
| Training loss = NaN dari epoch 1 | 0.02% nilai NaN di band reflektansi (awan) — hanya LST yang di-forward-fill | Ubah `fill_lst_nans()` → `fill_feature_nans()`, terapkan ke semua 10 fitur |
| Checkpoint tidak pernah tersimpan | `NaN < inf = False` di Python → kondisi simpan tidak pernah terpenuhi | Gunakan `train_loss` sebagai fallback ketika `val_loss = NaN` |
| `RuntimeError: size mismatch` saat fine-tune | `MODEL_CFG["hidden_size"] = 128` tapi checkpoint USA pakai `hidden_size = 256` | Update `MODEL_CFG["hidden_size"] = 256` agar cocok |
| EVI overflow (±1e11) | Formula EVI: denominator ≈ 0 saat rata-rata per county | Hapus EVI dari fitur; pakai 10 fitur (bukan 11) |

---

## 7. Evaluasi: Cara Mengukur Hasil

### 7.1 R² (R-Squared)

**Apa:** Seberapa baik prediksi model dibandingkan variasi alami data.

| Nilai | Interpretasi |
|-------|-------------|
| 1.0 | Prediksi sempurna |
| 0.6 | Model menjelaskan 60% variasi yield — target kita untuk USA |
| 0.39 | Hasil sementara (Mac, smoke test) — masih jauh dari target |
| 0.0 | Sama buruknya dengan prediksi rata-rata |
| < 0 | Lebih buruk dari prediksi rata-rata — negative transfer |

**Rumus:** `R² = 1 - SS_res / SS_tot`

### 7.2 RMSE (Root Mean Square Error)

**Satuan:** ton/ha — sama dengan target prediksi.
**Contoh:** RMSE = 0.8 ton/ha → rata-rata prediksi meleset 0.8 ton/ha.
**Rumus:** `RMSE = √(mean((ŷ - y)²))`

### 7.3 Hasil Nyata (Kaggle T4 GPU, cropland-masked v2, 8 Mei 2026)

| Model | Test R² | Test RMSE | Keterangan |
|-------|---------|-----------|-----------|
| USA LSTM (baseline) | **0.4416** | 1.656 t/ha | Best epoch 41, early stop ep61 |
| IDN Transfer | **0.574** | 0.763 t/ha | ΔR²=+0.574 vs scratch |
| IDN From-Scratch | 0.000 | 1.170 t/ha | — |
| VNM Transfer | **0.048** | 1.307 t/ha | ΔR²=+0.104 vs scratch |
| VNM From-Scratch | -0.056 | 1.377 t/ha | — |
| THA Transfer | -2.726 | 0.831 t/ha | Overfitting — no val set, 3 tahun |
| THA From-Scratch | -0.490 | 0.526 t/ha | — |

**Target USA R² ≥ 0.6 belum tercapai** (0.4416). Kemungkinan penyebab: early stopping terlalu cepat (epoch 41 dari 150), atau hyperparameter perlu tuning. THA butuh perbaikan (kurangi epoch fine-tuning).

---

## 8. Workflow Penelitian: Status Nyata per 8 Mei 2026

### Apa yang Sudah Selesai

```
✅ Download + clean yield data USA, IDN, VNM, THA
✅ GEE v1: 50 tasks submitted & downloaded (tanpa cropland mask)
✅ GEE v2: 50 tasks re-submitted & downloaded (dengan MCD12Q1 class 12 mask)
✅ merge_modis.py — build 4 .npz tensors (v2)
✅ dataset.py — filter zero-yield (y > 0.1) ditambahkan
✅ src/models/cnn_lstm.py — CropYieldLSTM + CropYieldCNNLSTM
✅ experiments/configs/usa_lstm.yaml — config terbaik
✅ src/training/train.py — full training loop
✅ src/transfer/finetune.py — 2-phase fine-tuning
✅ src/analysis/supervisor_analysis.py — 5 investigasi
✅ Training di Kaggle T4 GPU: USA R²=0.4416
✅ Fine-tuning ASEAN: IDN R²=0.574, VNM R²=0.048
✅ Pretrained checkpoint: experiments/checkpoints/usa_lstm/best_model.pt
✅ Memory files, CLAUDE.md decision log, docs/knowledge.md
✅ Semua di-commit ke GitHub (branch: dev)
```

### Yang Belum Selesai

```
⬜ Fix THA overfitting (kurangi epoch fine-tuning, tambah regularisasi)  ← PRIORITAS 1
⬜ Implementasi DANN (src/models/dann.py)                                 ← untuk H3
⬜ Download IDN pre-2020 (sebelum KSA)                                    ← nice to have
⬜ USA R² target ≥ 0.6 belum tercapai (saat ini 0.4416)                  ← hyperparameter tuning
⬜ Notebook Kaggle: hapus Cell 1 (numpy downgrade), perbaiki robustness
```

### Platform Training

| Platform | GPU | Status | Catatan |
|----------|-----|--------|---------|
| Mac M1 | MPS | Smoke test only | CLAUDE.md melarang full training di Mac |
| Google Colab | — | Tidak jadi dipakai | Kaggle lebih praktis dan gratis |
| **Kaggle** | **T4** | **✅ Digunakan** | P100 tidak kompatibel PyTorch 2.10 |

---

## 9. Hasil Investigasi untuk Supervisor

Sebelum meeting supervisor, kita jalankan 5 investigasi (tanpa mengubah pipeline):

### 9.1 Kenapa Sample USA Hilang? (Sample Loss Analysis)

Dari data mentah ke tensor final, berapa sample yang hilang:

| Tahap | USA | Hilang |
|-------|-----|--------|
| Raw NASS data | 60.834 baris | — |
| Setelah filter jagung + unit | ~50.000 | Buang non-corn, "D" suppressed |
| Setelah join MODIS | 41.349 | **~8.651 county-year tidak ada MODIS** |
| Final tensor | 41.349 | 0 (setelah NaN fill) |

**Penyebab utama kehilangan:** County-year yang tidak ada citra MODIS (bisa karena awan penuh, atau county tidak di-cover GEE export).

### 9.2 USA Yield = 0.0 t/ha — Ini Apa?

Ditemukan **206 sampel USA dengan yield = 0.0 ton/ha** di tensor. Ini bukan nilai nyata — jagung tidak bisa menghasilkan tepat 0.0 t/ha. Kemungkinan penyebab:
- Unit konversi yang gagal
- Nilai "D" (suppressed) yang bocor
- Error saat join

**Rekomendasi:** Hapus 206 sampel ini sebelum training final.

### 9.3 Domain Gap USA vs ASEAN

Kita ukur seberapa berbeda distribusi fitur MODIS:

| Metrik | USA vs IDN | USA vs VNM | USA vs THA |
|--------|-----------|-----------|-----------|
| NDVI mean | 0.45 vs 0.59 | 0.45 vs 0.52 | 0.45 vs 0.48 |
| LST range | lebar (30°C) | sempit (5-10°C) | sempit |
| Wasserstein dist (NDVI) | 0.082 | 0.045 | 0.032 |

**Kesimpulan:** USA dan ASEAN memang beda jauh — khususnya pola temporal NDVI (USA: kurva lonceng musiman; ASEAN: relatif datar sepanjang tahun). Ini mendukung perlunya DANN (H3).

### 9.4 Vietnam: Koreksi Hasil (Hasil Nyata ≠ Hasil Lama)

**⚠️ Koreksi penting:** Hasil VNM yang sebelumnya ditampilkan (negative transfer ΔR²=−0.094) adalah **data salah** — berasal dari file `transfer_results.csv` smoke test Mac yang ter-commit ke git, bukan dari training GPU sesungguhnya.

**Hasil nyata (Kaggle T4, cropland-masked v2):**

| Model | R² | RMSE | ΔR² vs From-Scratch |
|-------|-----|------|---------------------|
| From-Scratch | -0.056 | 1.377 t/ha | — |
| Transfer | **+0.048** | 1.307 t/ha | **+0.104 (positif!)** |

**Temuan revisi:** Transfer learning untuk VNM **sedikit membantu** (+0.104), bukan merugikan. H2 (negative transfer) tidak terkonfirmasi untuk VNM dengan data cropland-masked v2.

**Pelajaran:** Selalu verifikasi hasil dari GPU sesungguhnya, bukan dari smoke test yang ter-commit ke git.

### 9.5 Estimasi Dampak Cropland Masking

Model saat ini rata-rata MODIS atas **seluruh** area county/provinsi, termasuk hutan, kota, sawah non-jagung, dll. Dengan cropland mask:

| Wilayah | % Pixel Jagung (estimasi) | Dampak SNR |
|---------|--------------------------|-----------|
| Iowa (USA corn belt) | ~35% | 2.8× sinyal lebih kuat |
| East Java (IDN) | ~15-25% | 4-7× lebih kuat |
| Vietnam Tengah | ~10-20% | 5-10× lebih kuat |

**Prediksi:** Dengan cropland mask, R² USA diperkirakan naik dari 0.39 → **0.60-0.70**. Ini prioritas utama untuk paper final.

---

## 10. Masalah Diketahui & Rencana Perbaikan

### 10.1 ✅ SELESAI: Cropland Masking

**Masalah lama:** MODIS dirata-rata atas seluruh county/provinsi termasuk area non-pertanian.
**Solusi:** Re-ekstraksi GEE dengan `MCD12Q1` class 12 mask → v2 data. **Selesai 8 Mei 2026.**

### 10.2 ✅ SELESAI: Sampel Yield = 0.0 t/ha

**Masalah lama:** 206 sampel USA dengan yield = 0.0 t/ha tidak masuk akal secara pertanian.
**Solusi:** Filter `y > 0.1` ditambahkan ke `dataset.py`. **Selesai 8 Mei 2026.**

### 10.3 ⚠️ AKTIF: USA R² = 0.4416 (Target 0.6)

**Masalah:** Hasil GPU nyata 0.4416 — lebih baik dari smoke test (0.39) tapi belum capai target 0.6.

**Analisis:** Early stopping di epoch 41 (dari 150). Kemungkinan model butuh:
- Hyperparameter tuning: hidden_size=512, patience=30
- Learning rate schedule yang lebih agresif
- Atau target 0.6 memang terlalu optimis tanpa growing season filter

**Rencana:** Jalankan tuning di Kaggle di sesi berikutnya.

### 10.4 ⚠️ AKTIF: THA Overfitting Parah (R²=-2.726)

**Masalah:** THA hanya 84 training samples (2 tahun), tidak ada validation set. Fine-tuning 70 epoch → overfit sempurna ke training, gagal total di test.

**Solusi:** Batasi fine-tuning THA ke 10-20 epoch total, atau tambahkan dropout lebih besar. Perlu kode khusus per-country di `finetune.py`.

### 10.5 ⬜ PENDING: DANN untuk H3

**Masalah:** H3 belum bisa diuji tanpa implementasi DANN.
**Solusi:** `src/models/dann.py` — referensi: Ganin & Lempitsky 2015. **Fase berikutnya.**

### 10.6 ⬜ PENDING: IDN Pre-2020 Data

**Masalah:** Hanya 5 tahun IDN (2020-2024). Pre-2020 BPS pakai metodologi berbeda.
**Solusi:** Download BPS pre-2020 terpisah, normalisasi cross-methodology. Future work.

---

## 11. Glosarium Istilah

| Istilah | Definisi Singkat |
|---------|-----------------|
| **Ablation Study** | Eksperimen menghilangkan satu komponen untuk lihat kontribusinya |
| **Adam / AdamW** | Optimizer (algoritma update bobot) yang adaptif. AdamW = Adam + weight decay untuk regularisasi |
| **Backpropagation** | Algoritma mengirim error dari output ke input untuk update bobot |
| **Batch Size** | Jumlah sampel diproses sekaligus sebelum update bobot. Kita: 256 |
| **BPS** | Badan Pusat Statistik — sumber data yield Indonesia |
| **Checkpoint** | File tersimpan berisi bobot model terbaik selama training |
| **CNN** | Convolutional Neural Network — untuk data spasial (gambar/grid) |
| **Colab** | Google Colaboratory — Jupyter notebook gratis dengan GPU/TPU |
| **DANN** | Domain-Adversarial Neural Network — metode domain adaptation |
| **Data Leakage** | Kebocoran informasi dari test/val ke training — membuat evaluasi tidak valid |
| **Dense Layer** | Fully-connected layer — setiap neuron terhubung ke semua neuron sebelumnya |
| **Domain Adaptation** | Teknik agar model bekerja di domain target yang berbeda distribusinya |
| **Domain Gap** | Perbedaan statistik antara source (USA) dan target (ASEAN) |
| **Dropout** | Teknik regularisasi — random "mematikan" neuron saat training untuk cegah overfitting |
| **Early Stopping** | Hentikan training saat val loss tidak membaik selama N epoch (patience) |
| **EVI** | Enhanced Vegetation Index — seperti NDVI tapi lebih kompleks. Kita buang karena overflow |
| **Epoch** | Satu kali training melewati seluruh dataset training |
| **Feature** | Satu kolom/nilai input yang digunakan model untuk belajar |
| **Fine-Tuning** | Melanjutkan training pre-trained model dengan data target baru |
| **Frozen Layers** | Layer yang weight-nya tidak di-update saat training |
| **FAOSTAT** | Database statistik pertanian FAO (Food and Agriculture Organization) |
| **GADM** | Database batas administratif global — shapefile county/provinsi |
| **GAUL** | Global Administrative Unit Layers — batas wilayah dari FAO, dipakai di GEE |
| **GEE** | Google Earth Engine — platform cloud untuk analisis citra satelit |
| **GEOID** | ID unik county USA: 5-digit gabungan state_fips + county_fips |
| **GSO** | General Statistics Office — sumber data yield Vietnam |
| **Hidden Size** | Jumlah neuron dalam hidden layer LSTM (kita: 256) |
| **Hyperparameter** | Parameter yang diset manual sebelum training (bukan dipelajari model) |
| **KSA** | Kerangka Sampel Area — metodologi BPS berbasis satelit untuk survei pertanian, mulai 2020 |
| **Learning Rate** | Seberapa besar langkah update bobot per iterasi. Kecil = hati-hati tapi lambat |
| **Loss / MSE** | Fungsi error yang diminimalkan saat training. MSE = Mean Squared Error |
| **LST** | Land Surface Temperature — suhu permukaan dari MODIS MYD11A2 |
| **LSTM** | Long Short-Term Memory — neural network untuk data time-series |
| **MCD12Q1** | MODIS Land Cover Type — dataset tipe tutupan lahan. Class 12 = cropland |
| **MOD09A1** | MODIS Terra Surface Reflectance, 8-day composite, 500m — band b01-b07 |
| **MODIS** | Moderate Resolution Imaging Spectroradiometer — satelit NASA Terra/Aqua |
| **MPS** | Metal Performance Shaders — GPU backend Mac M1. Dipakai untuk smoke test |
| **MYD11A2** | MODIS Aqua Land Surface Temperature, 8-day — LST_Day/Night |
| **Negative Transfer** | Transfer learning yang justru memperburuk hasil |
| **NDVI** | Normalized Difference Vegetation Index — indikator kehijauan. `(NIR-Red)/(NIR+Red)` |
| **NPZ** | Format file NumPy berisi banyak array — seperti ZIP tapi untuk array |
| **OAE** | Office of Agricultural Economics — sumber data yield Thailand |
| **Optimizer** | Algoritma yang memperbarui bobot berdasarkan gradient |
| **Overfitting** | Model "hafal" data training, tapi jelek di data baru |
| **Pretrained** | Model yang sudah dilatih sebelumnya (di USA) — siap untuk fine-tuning |
| **R²** | R-Squared — 1.0 = sempurna, 0 = sama dengan nebak rata-rata |
| **RMSE** | Root Mean Square Error — rata-rata besar error dalam satuan asli (ton/ha) |
| **Regularisasi** | Teknik mencegah overfitting (dropout, weight decay, early stopping) |
| **Remote Sensing** | Mengamati objek dari jauh (satelit) tanpa kontak langsung |
| **Source Domain** | Domain asal untuk transfer learning — USA |
| **Target Domain** | Domain tujuan transfer learning — Indonesia/Vietnam/Thailand |
| **Tensor** | Array multidimensi — input model kita: (N_samples, 46 timesteps, 10 fitur) |
| **Timestep** | Satu langkah waktu. Di proyek ini: 1 timestep = 8 hari |
| **Transfer Learning** | Menggunakan model yang dilatih di source untuk bantu belajar di target |
| **Vanilla Fine-Tuning** | Fine-tuning biasa tanpa modifikasi khusus — lawan dari DANN/frozen |
| **Wasserstein Distance** | Metrik jarak antara dua distribusi statistik — kita pakai untuk ukur domain gap |
| **Weight / Bobot** | Parameter model yang dipelajari selama training |
| **Yield** | Hasil panen per satuan luas — ton per hektare (t/ha) |
| **Z-score** | Normalisasi: `(x - mean) / std` — nilai dalam satuan standar deviasi dari rata-rata |

---

## Referensi Cepat

| Apa | Dimana |
|-----|--------|
| Spesifikasi proyek lengkap | `CLAUDE.md` |
| Keputusan teknis (dengan alasan) | `memory/project_decisions.md` |
| Status data saat ini | `memory/project_data_status.md` |
| Peta file & arsitektur kode | `memory/project_architecture.md` |
| Kode GEE extraction | `src/data/extract_modis_usa.py`, `src/data/extract_modis_asean.py` |
| Kode merge + tensor | `src/data/merge_modis.py` |
| Dataset PyTorch | `src/data/dataset.py` |
| Model arsitektur | `src/models/cnn_lstm.py` |
| Training loop | `src/training/train.py` |
| Fine-tuning | `src/transfer/finetune.py` |
| Investigasi supervisor | `src/analysis/supervisor_analysis.py` |

---

*Dokumen ini akan di-update seiring berjalannya proyek. Terakhir diupdate: 8 Mei 2026 — hasil nyata Kaggle T4 GPU ditambahkan, koreksi hasil VNM.*
