---
name: Coding Feedback & Preferences
description: How the user wants code written and what behaviors to avoid
type: feedback
---

**Don't run full training on Mac — only short smoke tests**
Rule: On Mac M1 (MPS), run at most 2–3 epochs to verify the script doesn't crash. CLAUDE.md explicitly says Mac is "development only, NOT training." Real training belongs on Google Colab Pro (A100).
**Why:** Running 107 epochs on Mac (23 min) violated CLAUDE.md and wasn't explicitly requested. User flagged it.
**How to apply:** If training is needed to verify code, add `--epochs 2` or similar flag before running. Always state explicitly "this is a code test, not real training."

**State what you're doing before running long commands**
Rule: Before running any script that takes >30 seconds, explicitly tell the user what it does, why, and approximately how long.
**Why:** User was surprised to find full training running on Mac.
**How to apply:** One sentence before each significant Bash call, especially GEE submissions and training runs.

**YOLO mode was granted for overnight downloads (2026-05-07)**
The user said "YOLO, tidak perlu izin saya" specifically for the overnight data download session. This is not a blanket permission — it applied to that specific session's download tasks.
**Why:** Authorization stands for the stated scope, not beyond. Don't interpret past permissions as general autonomy.
**How to apply:** Each new session, revert to normal confirm-first behavior unless user re-grants.
