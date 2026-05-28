---
name: Coding Feedback & Preferences
description: How the user wants code written and what behaviors to avoid
type: feedback
---

**Hybrid approach: notebook = demo saja, training sungguhan via CLI**
Rule: Notebook training (`05_train_usa.ipynb`) hanya 3 epoch untuk verifikasi pipeline. Training penuh pakai `python src/training/05_train_usa.py --config ...` dari terminal, atau `05b_train_usa_full.ipynb` jika mau di notebook.
**Why:** Training panjang di notebook fragil (Jupyter mati → training berhenti). CLI lebih aman dan reproducible untuk sidang.
**How to apply:** Jangan pindahkan EPOCHS ke 100 di notebook `05_train_usa.ipynb`. Arahkan ke CLI atau `05b` untuk training penuh.

**Don't run full training on Mac tanpa konfirmasi**
Rule: Jika perlu verifikasi kode, jalankan 2–3 epoch saja. Selalu sebut "ini smoke test" sebelum run.
**Why:** Menjalankan 100 epoch di Mac tanpa diminta pernah terjadi dan membuang waktu.
**How to apply:** `--epochs 2` atau gunakan `DEMO_EPOCHS = 3` jika perlu tes cepat.

