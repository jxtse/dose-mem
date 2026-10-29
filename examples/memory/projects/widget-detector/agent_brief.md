# Widget Detector — agent brief

Detects widgets in factory camera frames. Training code lives in the repo; evaluation uses the frozen `val-2026q1` split.

## Decisions
- 2026-04-02 — Freeze `val-2026q1` for all comparisons; never tune on it.
- 2026-05-21 — Unverified entries in the dashboard are teal.
