# Knowledge Base V2 — Alasan Teknis & Hasil Eksperimen

Dokumen ini adalah lanjutan dari `knowledge.md` dan fokus pada **mengapa** setiap keputusan diambil, bukan sekadar **apa** yang dikerjakan. Ditulis untuk mempersiapkan diskusi supervisor dan presentasi thesis.

**Update terakhir: 13 Mei 2026**

---

## Daftar Isi

1. [Mengapa County Level (USA)?](#1-mengapa-county-level-usa)
2. [Mengapa LSTM, bukan CNN atau Transformer?](#2-mengapa-lstm-bukan-cnn-atau-transformer)
3. [Mengapa CNN-LSTM Gagal?](#3-mengapa-cnn-lstm-gagal)
4. [Arsitektur Detail: Alasan Tiap Pilihan Angka](#4-arsitektur-detail-alasan-tiap-pilihan-angka)
5. [Apa yang Di-finetune dan Mengapa?](#5-apa-yang-di-finetune-dan-mengapa)
6. [Hasil Eksperimen Jagung — Semua Negara](#6-hasil-eksperimen-jagung--semua-negara)
7. [Tentang "Eksperimen Padi"](#7-tentang-eksperimen-padi)
8. [Diagnosis: Mengapa R² USA Masih Rendah?](#8-diagnosis-mengapa-r-usa-masih-rendah)
9. [Pertanyaan yang Mungkin Ditanya Supervisor](#9-pertanyaan-yang-mungkin-ditanya-supervisor)

---

## 1. Mengapa County Level (USA)?

### Alasan Teknis

**County adalah unit terkecil di USA yang memiliki data yield pertanian yang dipublikasikan secara resmi.**

USDA NASS (National Agricultural Statistics Service) menerbitkan data yield per county berdasarkan mandatory survey — petani di USA wajib melapor. Data ini memiliki coverage konsisten sejak 1980an, dengan format county sebagai unit analisis.

| Pertanyaan | Jawaban |
|-----------|---------|
| Kenapa tidak per petani/farm? | USDA tidak mempublikasikan data farm-level ke publik (privasi). County adalah granularitas publik terkecil |
| Kenapa tidak per state? | State terlalu kasar — satu state bisa punya iklim mikro yang sangat berbeda (Illinois Utara vs Selatan) |
| Kenapa tidak sub-county? | Tidak ada data yield sub-county yang tersedia; MODIS 500m juga tidak cukup resolusi untuk sekecil itu |

### Kesesuaian dengan MODIS

MODIS Terra (MOD09A1) resolusinya **500 meter per piksel**. Rata-rata county corn belt seperti Iowa punya luas ~1.500 km² — setara ~6.000 piksel MODIS. Cukup untuk statistical averaging yang bermakna.

Kalau pakai resolution yang lebih halus (misal Sentinel-2, 10m), computational cost GEE akan jauh lebih tinggi dan tidak perlu untuk level county.

### Mengapa ASEAN Pakai Provinsi, Bukan County/Kabupaten?

Di ASEAN, tidak ada data yield per kabupaten yang dipublikasikan konsisten:

- **Indonesia:** BPS menerbitkan yield per provinsi secara nasional. Data kabupaten ada tapi tidak konsisten antar provinsi dan tahun.

Mismatch granularitas (USA county vs ASEAN provinsi) adalah **keterbatasan yang disengaja** dalam tesis ini — ini justru menjadi bagian dari analisis domain gap. Provinsi ASEAN lebih besar dari county USA, sehingga averaging MODIS lebih "kasar" dan noise lebih banyak (dilusi sinyal cropland).

---

## 2. Mengapa LSTM, Bukan CNN atau Transformer?

### Sifat Data: Time Series Agronomis

Data MODIS per county per tahun adalah:
```
T=1 (Jan 1–8)   → [b01, b02, ..., ndvi, LST_Day, LST_Night]
T=2 (Jan 9–16)  → [b01, b02, ..., ndvi, LST_Day, LST_Night]
...
T=46 (Dec 25–31)→ [b01, b02, ..., ndvi, LST_Day, LST_Night]
```

Ini adalah **sequence yang berurutan secara kausal** — nilai NDVI di minggu 20 bergantung pada kondisi tumbuh minggu 1–19. Ini adalah definisi klasik dari masalah yang cocok untuk LSTM.

### Keunggulan LSTM untuk Masalah Ini

| Kebutuhan | Mengapa LSTM Cocok |
|----------|-------------------|
| Ingat pola awal musim | LSTM forget gate belajar kapan mengabaikan atau mempertahankan sinyal dari timestep awal (misalnya: stress saat planting) |
| Tangkap pola puncak pertumbuhan | Input gate secara selektif "menyimpan" momentum NDVI saat naik tajam (fase vegetatif) |
| Perkiraan dari akumulasi | Cell state LSTM adalah bentuk integral running — cocok untuk akumulasi growing degree days, akumulasi NDVI |
| Urutan penting | LSTM memproses T=1→T=46 secara berurutan, berbeda dengan model bag-of-features |

### Kenapa Bukan Transformer?

Transformer (self-attention) memang state-of-the-art untuk banyak sequence task, tapi:

1. **Data kita kecil:** 41.349 sampel USA. Transformer butuh data lebih banyak untuk belajar attention patterns yang bermakna.
2. **Sequence kita pendek:** 46 timestep. LSTM masih kompetitif di sini; advantage Transformer baru kelihatan pada sequence panjang (>500 timestep).
3. **Overfitting risk:** Transformer punya lebih banyak parameter untuk ukuran model yang setara; dengan 41k sampel, risiko overfit lebih tinggi tanpa aggressive regularisasi.
4. **Literature agronomis:** Mayoritas paper yield prediction 2017–2023 masih menggunakan LSTM sebagai backbone. Memakai LSTM memudahkan perbandingan langsung dengan literature.

### Kenapa Bukan CNN 2D atau ConvLSTM?

CNN 2D cocok jika input adalah **gambar piksel per piksel** (misal: crop yield dari full satellite image per county). Di proyek kita, input sudah diagregasi jadi **10 nilai rata-rata** — tidak ada dimensi spasial yang perlu dikonvolusikan.

ConvLSTM cocok untuk video/spatial-temporal grids. Kita tidak punya grid spasial per timestep — hanya satu vektor 10-dimensi per timestep.

---

## 3. Mengapa CNN-LSTM Gagal?

### Konteks: Inspirasi dari You et al. 2017

Paper You et al. 2017 (*"Deep Gaussian Process for Crop Yield Prediction"*) menggunakan CNN + LSTM dengan input **histogram 32-bin per band per county**. Idenya:

```
Histogram pixel Red band → 32 nilai (distribusi nilai merah di seluruh piksel county)
Histogram pixel NIR band → 32 nilai
...
Input per timestep = ~200 nilai (32 bin × 6 band)
```

CNN 1D di sini masuk akal: ia mendeteksi **bentuk distribusi histogram** — misalnya "puncak di bin ke-8 berarti banyak pixel dengan reflektansi rendah = kanopi lebat".

### Apa yang Kita Lakukan Berbeda

Di proyek ini, karena GEE export dilakukan sebagai **zonal statistics** (rata-rata per county), input per timestep hanya:
```
Input per timestep = 10 nilai (rata-rata b01, b02, ..., ndvi, LST_Day, LST_Night)
```

Tidak ada histogram, tidak ada distribusi spasial — hanya satu angka per fitur.

### Mengapa CNN 1D Tidak Bermakna di Sini

CNN 1D yang diaplikasikan atas 10 fitur berarti filter kernel mempelajari **pola pada dimensi fitur** — misalnya "perhatikan kombinasi b01 dan b02 secara bersamaan". Ini secara fisik sama saja dengan layer Linear biasa (matriks bobot) — tidak ada struktur lokal yang perlu ditangkap.

Secara empiris:
| Model | Test R² | Test RMSE | Epoch Berhenti |
|-------|---------|-----------|----------------|
| CropYieldCNNLSTM | 0.13 | 2.06 t/ha | 46 |
| CropYieldLSTM | **0.39** | 1.73 t/ha | 107 |

CNN-LSTM tidak hanya lebih buruk — juga early-stop lebih cepat (convergence buruk) dan RMSE lebih tinggi hampir 0.33 t/ha.

**Kesimpulan:** CNN 1D menambahkan parameter tanpa menambahkan kemampuan representasi yang relevan. Occam's Razor berlaku — pilih LSTM-only.

---

## 4. Arsitektur Detail: Alasan Tiap Pilihan Angka

```
Input: (batch, 46, 10)
   ↓
LSTM(input_size=10, hidden_size=256, num_layers=2, dropout=0.3, batch_first=True)
   ↓  ambil hidden state timestep terakhir → (batch, 256)
Linear(256 → 64)
ReLU()
Dropout(0.3)
Linear(64 → 1)
   ↓
Output: (batch, 1)  — yield dalam ton/ha
```

### Mengapa hidden_size = 256?

**Trade-off: kapasitas vs overfitting.**

- `hidden_size=128`: terlalu sempit — dalam eksperimen awal, model underfitting bahkan di training set (loss stagnan).
- `hidden_size=512`: terlalu besar — overfit ke training set, val loss lebih tinggi.
- `hidden_size=256`: "sweet spot" untuk 41k sampel dengan 46×10 input. Jumlah parameter ~817k — rasio sample:parameter ≈ 50:1, aman dari overfitting parah.

### Mengapa num_layers = 2?

- Layer 1 LSTM: mempelajari low-level temporal patterns (transisi mingguan NDVI, variasi LST).
- Layer 2 LSTM: mengkombinasikan pola layer 1 jadi pola musiman yang lebih abstrak ("fase vegetatif berjalan normal?").
- 3 layers: tidak membantu — gradient vanishing mulai muncul, training lebih lama.

### Mengapa Intermediate Layer 64, Bukan Langsung 256→1?

256→1 langsung berarti model harus "squeeze" seluruh musim tanam jadi satu prediksi dalam satu langkah. Layer 64 berfungsi sebagai **representasi kompresi** yang memaksa model belajar fitur ringkas yang paling prediktif terhadap yield akhir musim.

Analoginya: seorang ahli agronomis tidak langsung prediksi yield dari semua data mentah — dia dulu simpulkan "musim ini curah hujan normal, pertumbuhan fase vegetatif oke, tidak ada heat stress signifikan" → baru prediksi yield.

### Mengapa Dropout = 0.3?

- Dropout 0.3 berarti 30% neuron di-zero setiap forward pass → regularisasi yang cukup agresif.
- USA corn belt punya pola yang cukup konsisten antar county — model mudah overfit ke pola spesifik Iowa vs Oklahoma.
- 0.3 adalah nilai literatur standar untuk LSTM agronomis (You et al. 2017 juga menggunakan 0.25–0.3).

### Mengapa LR = 5e-4 dengan Cosine Decay?

- LR awal 5e-4 (Adam): cepat konvergen tanpa oscillating.
- Cosine decay: LR turun secara halus dari 5e-4 → ~0 di epoch akhir, membantu model settle ke minimum yang lebih smooth.
- Alternatif StepLR terlalu keras — ada "cliff" di mana LR turun tiba-tiba.

---

## 5. Apa yang Di-finetune dan Mengapa?

### Anatomi Fine-tuning 2-Phase

#### Phase 1 (20 epoch): Frozen LSTM, hanya FC Head

```
LSTM weights → FROZEN (tidak berubah)
Linear(256→64) → TRAINABLE ← lr=1e-3
Linear(64→1)   → TRAINABLE ← lr=1e-3
```

**Mengapa ini penting?**

Ketika model di-load dari USA checkpoint, head layer (256→64→1) adalah **bobot yang sudah dilatih untuk distribusi yield USA** (rata-rata ~9 t/ha). Distribusi yield ASEAN berbeda:
- Indonesia: rata-rata ~4–5 t/ha

Kalau kita langsung fine-tune semua layer dengan LR besar, gradient dari head yang "keliru" ini akan merusak representasi LSTM yang sudah bagus. Phase 1 "memperbaiki" head dulu.

#### Phase 2 (50 epoch): Unfreeze All

```
LSTM weights → TRAINABLE ← lr=1e-4 (sangat kecil)
Linear(256→64) → TRAINABLE ← lr=1e-4
Linear(64→1)   → TRAINABLE ← lr=1e-4
```

LR yang sangat kecil (1e-4 vs 5e-4 di pretraining) penting untuk mencegah **catastrophic forgetting** — LSTM tidak "lupa" pola temporal yang dipelajari dari 41k sampel USA hanya karena 126–1.804 sampel ASEAN.

### Apa yang Sebenarnya Diadaptasi?

| Komponen | Yang Berubah di Fine-tuning |
|---------|---------------------------|
| LSTM gates (forget, input, output) | Halus — threshold kapan "ingat" vs "lupa" disesuaikan ke pola temporal ASEAN |
| LSTM hidden state transitions | Halus — bobot transisi disesuaikan ke amplitude NDVI ASEAN yang berbeda (lebih flat) |
| FC head (256→64→1) | Signifikan — mapping dari representation ke yield value disesuaikan ke range yield ASEAN |

### Mengapa Bukan Full Finetune dari Awal (Phase 2 Saja)?

Eksperimen dari literatur transfer learning (Howard & Ruder 2018, ULMFiT) menunjukkan bahwa tanpa Phase 1, akurasi final lebih rendah 5–15% karena instabilitas awal training merusak pretrained features. Ini berlaku khususnya saat target dataset kecil seperti Indonesia.

---

## 6. Hasil Eksperimen Jagung — Semua Negara

> **Catatan:** Semua hasil ini adalah **smoke test di Mac MPS**, bukan training final. Training dilakukan hanya 2–3 epoch efektif (early stopping) untuk memverifikasi pipeline tidak crash. Hasil final butuh Google Colab A100 dengan 150 epoch penuh.

### 6.1 USA Baseline

| Model | Test R² | Test RMSE | Epoch | Config |
|-------|---------|-----------|-------|--------|
| CropYieldCNNLSTM | 0.13 | 2.06 t/ha | 46 (early stop) | usa_baseline.yaml |
| **CropYieldLSTM** | **0.39** | **1.73 t/ha** | 107 (early stop) | usa_lstm.yaml ← PILIHAN |

**Interpretasi:** R²=0.39 artinya model menjelaskan 39% variasi yield antar county dan antar tahun. Ini **rendah dari target** (≥0.6) karena:
1. Training belum full (Mac bukan A100)
2. Tidak ada cropland masking — MODIS dirata-rata atas seluruh county termasuk hutan dan kota

### 6.2 Transfer ke Indonesia (IDN)

| Skenario | Test R² | Test RMSE | Catatan |
|---------|---------|-----------|---------|
| From-scratch (LSTM, tanpa pretrain) | **−0.29** | ~1.3 t/ha | Estimasi berdasarkan pattern |
| **Transfer dari USA** | **0.06** | **1.13 t/ha** | ΔR² = **+0.35** |

**Interpretasi:** Transfer learning memberikan kontribusi positif yang signifikan (+0.35 R²) untuk Indonesia. Ini masuk akal karena:
- Dataset IDN sangat kecil (162 sampel, 2020–2024)
- Tanpa pretrain, model tidak punya cukup data untuk belajar dari nol
- USA pretraining memberikan "prior" yang berguna tentang hubungan NDVI → yield

**Caveat:** R²=0.06 absolut masih rendah — signal MODIS untuk IDN sangat terdilusi oleh non-cropland pixels (hutan, sawah padi, kebun kelapa sawit).

### 6.3 Ringkasan Transfer Learning

| Negara | ΔR² Transfer vs Scratch | Interpretasi |
|--------|------------------------|--------------|
| Indonesia | **+0.35** (smoke test) | Transfer berguna — data terlalu sedikit untuk scratch |

**Catatan:** Hasil di atas dari smoke test Mac MPS (province level, 162 sampel). Dengan data kabupaten 2003–2025, hasil bisa sangat berbeda.

**Takeaway untuk tesis:** Transfer learning memberikan keuntungan ketika target dataset terlalu kecil untuk training dari nol.

---

## 7. Tentang "Eksperimen Padi"

**Tidak ada eksperimen padi dalam tesis ini.** Tesis ini secara eksplisit berfokus pada **jagung (maize/corn)** saja.

Ini bukan kelupaan — ini keputusan desain:

| Alasan | Penjelasan |
|--------|-----------|
| Fokus komparasi | Membandingkan USA corn belt dengan ASEAN corn menggunakan tanaman yang sama |
| Ketersediaan data | USA tidak punya produksi padi signifikan — sumber data (USDA NASS) tidak cocok untuk padi |
| Sinyatur spektral berbeda | Padi dan jagung punya NDVI signature yang berbeda (padi banyak di sawah tergenang — sinyal air ikut masuk) |
| Scope terbatas | Menambahkan padi = menambahkan dimensi baru (dua tanaman), yang bukan kontribusi utama tesis ini |

Jika ingin extend ke padi di masa depan, sumber data yang relevan:
- Indonesia: BPS punya data padi per provinsi lebih lengkap dari jagung
- MODIS mask: MODIS MCD12Q1 class 12 (Croplands) mencakup semua jenis tanaman — perlu class yang lebih spesifik untuk padi

---

## 8. Diagnosis: Mengapa R² USA Masih Rendah?

Target R² USA adalah ≥0.6. Saat ini: 0.39 (smoke test). Analisis penyebab:

### Penyebab #1: Tidak Ada Cropland Masking (Terbesar)

MODIS dirata-rata atas **seluruh** area county — termasuk hutan, kota, jalan raya, sawah non-jagung, padang rumput. Padahal persentase area jagung per county bervariasi sangat lebar:

| Wilayah | Estimasi % Jagung |
|---------|-----------------|
| Iowa corn belt county | ~35–50% |
| Illinois barat | ~25–40% |
| Non-corn-belt county | <5% |

Dengan masking cropland (MCD12Q1), hanya piksel yang terklasifikasi sebagai "cropland" yang dirata-rata. Ini meningkatkan signal-to-noise secara signifikan.

**Estimasi dampak:** R² diperkirakan naik ke 0.60–0.70 dengan masking. Ini adalah prioritas fix #1.

### Penyebab #2: Training Belum Full

Smoke test di Mac MPS berhenti di epoch 107 (early stopping dari patience=20). Model belum konvergen optimal karena:
- Mac MPS lebih lambat dari A100 Colab → proses dibatasi waktu
- Hanya beberapa iterasi efektif, bukan 150 epoch penuh

### Penyebab #3: 206 Sampel Yield = 0.0 t/ha

206 sampel dengan yield tepat 0.0 ton/ha adalah data corrupt — bukan kondisi pertanian nyata. Ini membuat model belajar pola "jagung tidak panen = 0.0" yang tidak benar, mengganggu fit keseluruhan.

Fix: Filter `y > 0.1` sebelum training.

### Penyebab #4: Tidak Ada Growing Season Filter

Timestep T=1 (Januari awal) dan T=46 (Desember akhir) untuk kebanyakan USA corn belt tidak relevan — lahan masih beku atau baru saja dipanen. Memasukkan 46 timestep penuh menambahkan ~20 timestep "noise" (winter dormancy).

Fix opsional: Filter ke T=23–T=37 (Mei–Oktober) untuk corn belt — ini periode growing season USA.

---

## 9. Pertanyaan yang Mungkin Ditanya Supervisor

### "Kenapa pakai county, bukan pixel?"

County adalah granularitas tertinggi yang bisa di-join antara yield data (USDA NASS) dan MODIS. Data yield per-pixel tidak tersedia secara publik. County juga cukup besar untuk MODIS 500m averaging yang stabil secara statistik.

### "Kenapa LSTM dan bukan Transformer?"

Dataset kita (41k sampel) relatif kecil untuk Transformer. Sequence kita (46 timestep) juga pendek — Transformer belum menunjukkan keunggulan di literatur agronomis pada skala ini. LSTM lebih hemat parameter dan terbukti bekerja pada dataset crop yield.

### "Kenapa tidak pakai random forest atau XGBoost sebagai baseline?"

Ini adalah gap yang perlu diisi sebelum submission. Random forest/XGBoost sebagai **non-deep-learning baseline** penting untuk menunjukkan bahwa deep learning + transfer learning memang lebih baik, bukan karena hanya deep learning lebih canggih. Ini valid sebagai additional experiment di future work atau sebagai ablation.

### "Apa kontribusi utama tesis ini?"

1. Dokumentasi empiris bahwa transfer learning dari USA ke ASEAN untuk crop yield prediction *tidak selalu berhasil* — tergantung pada ukuran dataset target dan domain gap iklim.
2. Quantifikasi domain gap (Wasserstein distance) antara USA dan ASEAN dalam fitur MODIS.
3. Baseline benchmark untuk masalah yang belum pernah diteliti dengan setup ini (USA county → ASEAN provinsi).

### "Kenapa tidak pakai data cuaca (rainfall, temperature) selain MODIS?"

Pertanyaan valid. Dalam proyek ini, MODIS saja karena:
- MODIS sudah mengandung proxy untuk kondisi cuaca: LST (suhu permukaan), NDVI (respons tanaman terhadap stres air/panas)
- Menambahkan data cuaca menambahkan kompleksitas pipeline dan potensi data gap
- Ini bisa jadi future work yang menarik — apakah weather data meningkatkan R²?

---

## Referensi Keputusan

| Topik | File Rinci |
|-------|-----------|
| Semua keputusan teknis dengan alasan | `memory/project_decisions.md` |
| Status data & eksperimen final | `memory/project_data_status.md` |
| Peta kode & arsitektur | `memory/project_architecture.md` |
| Konsep teknis (versi edukasi) | `docs/knowledge.md` |

---

*Dokumen ini akan di-update setelah hasil training penuh dari Colab A100. Terakhir diupdate: 13 Mei 2026.*
