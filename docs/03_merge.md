## Tahap 3: Merge — Gabung Label + Fitur

### 3.1. Apa yang dikerjakan?

Jalankan `src/data/merge_modis.py`. Ini file paling kritis — menjembatani yield data dan MODIS data.

```
Input:  21 CSV MODIS USA (data/raw/modis/modis_usa_*.csv)
        yield_usa_2003_2023.parquet (data/processed/usa/)

Proses: 1. Baca semua CSV MODIS per negara
        2. Per county per tahun: pivot 46 tanggal → array (46, 10)
        3. Buang EVI (overflow bug → pakai 10 fitur)
        4. Isi NaN: forward-fill → backward-fill → isi 0
        5. Clip NDVI ke [-1, 1]
        6. Join dengan yield parquet via region_id + year
        7. Simpan sebagai .npz

Output: data/processed/modis/usa_modis.npz
        data/processed/modis/idn_modis.npz
```

### 5.2. Output: `usa_modis.npz`

Format NPZ = seperti ZIP tapi isinya array NumPy. Buka di Python (`00_coba.ipynb` cell 2C).

```
X: (32296, 46, 10)  ← 32.296 sampel × 46 timestep × 10 fitur
y: (32296,)          ← yield dalam ton/ha
```

| Apa yang dicek | Ekspektasi |
|----------------|------------|
| **Sampel** | 32.296 (41.349 yield - yang disensor/NaN) |
| **Timestep** | 46 (8-hari komposit) |
| **Fitur** | 10 (b01–b07, ndvi, LST_Day, LST_Night) |
| **y range** | 0.0 – 16.96 t/ha |

### 5.3. Validasi: kenapa 41.349 → 32.296?

Hilang 9.053 sampel. Kenapa?
- County-year yang yield-nya NaN (disensor) = tidak bisa jadi sampel (nggak ada label)
- County-year yang tidak ada di MODIS CSV (GEE gagal export)

### 5.4. Validasi tensor shape

Di notebook:
```python
X.shape  # harus (32296, 46, 10)
y.shape  # harus (32296,)

# Tidak boleh ada NaN di tensor final:
np.isnan(X).sum()  # harus 0
np.isnan(y).sum()  # harus 0
```

### 5.5. Tensor Indonesia

```
idn_modis.npz: TBD — menunggu re-ekstrak GEE di level kabupaten (ADM2)
```

**Status:** BPS data 2003–2025 sudah tersedia di level kabupaten (~514 kabupaten). MODIS masih perlu di-re-ekstrak menggunakan GAUL ADM2 boundaries. File `idn_modis.npz` yang ada sekarang menggunakan provinsi level (162 sampel) dan akan diganti.







## LANGKAH 3: Merge — Gabungkan Yield + MODIS → Tensor

**Script:** [src/data/merge_modis.py](../src/data/merge_modis.py)

**Yang terjadi step by step:**
1. Baca semua 21 CSV MODIS → concat jadi 1 DataFrame besar (`~139k × 21 = ~2.9M baris`)
2. Pivot per county: reshape dari `(county × 46_tanggal)` menjadi `(county, 46, 10_fitur)` — hasilnya array 3D
3. Buang kolom EVI (export overflow saat GEE run)
4. Clip NDVI ke `[-1, 1]` — menghapus noise artefak
5. Handle NaN: forward-fill → backward-fill → isi 0 (urutan ini penting)
6. Join dengan yield pada key `(GEOID, year)` → pasangkan X (fitur) dengan y (label)
7. Drop sampel dengan `yield = 0.0` (206 baris — bukan nol sungguhan, data kosong)
8. Simpan: `X=(N, 46, 10)` float32, `y=(N,)` float32, `region_ids`, `years` → `.npz`

**Output tensor:**
```
usa_modis.npz
  X          : (32296, 46, 10)   — N sampel × 46 waktu × 10 fitur
  y          : (32296,)          — yield ton/ha per sampel
  region_ids : (32296,)          — FIPS county per sampel
  years      : (32296,)          — tahun per sampel
```

**10 Fitur (urutan di axis terakhir):**

| Idx | Nama | Sumber | Arti |
|-----|------|--------|------|
| 0 | b01 | MOD09A1 | Reflektansi merah (620–670 nm) |
| 1 | b02 | MOD09A1 | Reflektansi NIR (841–876 nm) |
| 2 | b03 | MOD09A1 | Reflektansi biru (459–479 nm) |
| 3 | b04 | MOD09A1 | Reflektansi hijau (545–565 nm) |
| 4 | b05 | MOD09A1 | Reflektansi SWIR-1 (1230–1250 nm) |
| 5 | b06 | MOD09A1 | Reflektansi MIR (1628–1652 nm) |
| 6 | b07 | MOD09A1 | Reflektansi SWIR-2 (2105–2155 nm) |
| 7 | ndvi | hitung | `(b02−b01)/(b02+b01)` — kehijauan |
| 8 | LST_Day | MYD11A2 | Suhu permukaan siang (°C) |
| 9 | LST_Night | MYD11A2 | Suhu permukaan malam (°C) |

**Jalankan di notebook:** Cell 4 — load npz, cek shape, plot time series satu county

**Pertanyaan:**
- [ ] Setelah pivot, kenapa dimensinya `(N, 46, 10)` bukan `(N, 10, 46)`?
- [ ] Kenapa NaN diisi 0 sebagai last resort, bukan dibuang barisnya?
- [ ] Kenapa 206 sampel yield=0 dibuang? Apa bedanya yield=0 genuine vs data kosong?
- [ ] NDVI di-clip ke [-1,1] — nilai NDVI di luar range itu berarti apa?

---