## LANGKAH 6: Model (Sekilas — Detail Nanti)

**Script:** [src/models/cnn_lstm.py](../src/models/cnn_lstm.py)

```
Input  : (batch, T=46, F=10)
  → LSTM(hidden=256, layers=2, dropout=0.3)
  → output terakhir → (batch, 256)
  → Linear(256→64) → ReLU
  → Linear(64→1)
Output : (batch, 1)  — prediksi yield ton/ha
```

**Pertanyaan (untuk nanti):**
- [ ] Kenapa LSTM, bukan Transformer?
- [ ] Kenapa pakai output timestep terakhir, bukan semua timestep?

---

## LANGKAH 7: Training (Sekilas — Detail Nanti)

**Script:** [src/training/train.py](../src/training/train.py)

Hasil smoke test di Mac (bukan training penuh):
- USA LSTM: **R² = 0.39**, RMSE = 1.73 t/ha
- USA CNN-LSTM: R² = 0.13 (lebih buruk)

R² 0.39 rendah karena belum ada cropland masking — sinyal jagung terdilusi oleh hutan & kota.

