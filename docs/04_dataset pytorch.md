
---

## Tahap 6: Dataset PyTorch & Split

### 6.1. Apa yang dikerjakan?

`src/data/dataset.py` — kelas `MaizeDataset` yang me-wrap tensor .npz ke format PyTorch.

```
Input:  usa_modis.npz (atau idn_modis.npz)
        country_name, split (train/val/test)

Proses: 1. Load .npz
        2. Filter tahun sesuai split (temporal, bukan random!)
        3. Z-score normalization: (x - mean_train) / std_train
        4. Drop sampel yield = 0.0

Output: PyTorch DataLoader → siap training
```

### 6.2. Train/Val/Test Split

**Split-nya temporal (berdasarkan tahun), BUKAN random.**

```
USA: Train 2003–2020 | Val 2021–2022 | Test 2023
IDN: Train 2020–2022 | Val (kosong)  | Test 2023–2024
```

**Kenapa bukan random?** Karena kita prediksi **masa depan**, bukan mengisi data yang hilang. Kalau random, model bisa "curang" — lihat data 2023 saat training → overfit → evaluasi tidak valid.

### 6.3. Z-score normalization — kenapa cuma dari training set?

```
mean = rata-rata fitur di training set
std  = standar deviasi fitur di training set

x_norm = (x - mean) / std   ← diterapkan ke train, val, DAN test
```

**Kenapa nggak dihitung dari seluruh data?** Karena val/test adalah "masa depan". Kalau kita hitung mean/std dari masa depan, itu **data leakage** — model mendapat informasi tentang data yang seharusnya belum dilihat.

---

## Ringkasan: Dari Mentah ke Siap Training

```
TAHAP 1-2         TAHAP 3            TAHAP 4           TAHAP 5           TAHAP 6
(download)        (download)         (GEE)             (merge)           (dataset)

USDA NASS ──► yield parquet ──────────────────────┐
             (41.349 × 9)                          │
                                                    ├──► usa_modis.npz ──► DataLoader
BDSP Kmtn ──► yield CSV ──────────────────────┐    │    (32296,46,10)      (batch,46,10)
             (671 × 6)                         │    │
                                               ├────┤
                   MODIS CSV (21 file) ────────┘    │
                   (~130k baris/file)               │
                                                     │
                   MODIS CSV (5 file) ──────────────┘
                   (~2.000 baris/file)

Output akhir: DataLoader siap masuk model LSTM
```

---

## Checklist Validasi Final

Sebelum training, pastikan semua ini terpenuhi:

- [ ] Yield USA: 41.349 baris, 2.280 county, 2003–2023
- [ ] Yield USA: konversi bu/acre → ton/ha sudah benar (×0.06277)
- [ ] Yield USA: 206 sampel yield=0 sudah ditandai (akan dihapus di dataset.py)
- [ ] Yield IDN: 671 baris, 38 provinsi, 2003–2023
- [ ] Yield IDN: provinsi <5 tahun data di-exclude dari training
- [ ] MODIS CSV: tiap county/provinsi tepat 46 tanggal per tahun
- [ ] MODIS CSV: tidak ada NaN di tensor final
- [ ] Tensor NPZ: X shape (N, 46, 10), y shape (N,)
- [ ] Tensor NPZ: 10 fitur = b01–b07, ndvi, LST_Day, LST_Night (EVI dibuang)
- [ ] Split temporal: tidak ada tahun test di training set
- [ ] Normalisasi: mean/std dihitung dari training set saja

---

*Panduan ini fokus ke logika penelitian — bukan sekadar output script. Kalau ada tahap yang kurang jelas, diskusikan sebelum lanjut.*







## LANGKAH 5: PyTorch Dataset — Siap Training

**Script:** [src/data/dataset.py](../src/data/dataset.py)

**Yang terjadi:**
1. `MaizeDataset` load `usa_modis.npz` → filter baris berdasarkan tahun
2. Split **temporal** (bukan random):
   - Train: 2003–2020 (~27.000 sampel)
   - Val: 2021–2022 (~3.000 sampel)
   - Test: 2023 (~1.500 sampel)
3. Z-score normalisasi per fitur: `(x − mean) / std` — mean & std **hanya dari training set**
4. Kembalikan `(X_tensor, y_tensor)` per `__getitem__` call

**Jalankan di notebook:** Cell 5 — load MaizeDataset, cek shape, cek normalisasi

**Pertanyaan:**
- [ ] Kenapa split temporal, bukan random shuffle?
- [ ] Apa yang terjadi kalau normalisasi dihitung dari seluruh data (termasuk test)?
- [ ] Setelah normalisasi, mean dan std tiap fitur seharusnya berapa?
- [ ] `__getitem__` return shape apa? Kenapa `y` punya shape `(1,)` bukan scalar?
