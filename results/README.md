# Frozen experiment summaries

`aggregate.csv` and `per_run.csv` were copied from the completed, protocol-checked
local experiment suite on 2026-09-30. They contain main-comparison and ablation
metrics, not model weights or raw maps. They are provided for auditability; they
are not inputs to model training.

The primary error domain is `raw / 127.5 - 1`. Metrics are macro-averaged per map.
Three-seed standard deviations are sample standard deviations. The fitted
log-distance row uses the fixed data-selection seed rather than stochastic
neural-network initializations.

Regenerate these files after executing the suite with
`python scripts/summarize_icc_suite.py --require-complete`, then review the new
`outputs/icc_suite_summary/` against this snapshot before updating it.

The CSVs do not by themselves contain the complete checkpoints or per-pixel
predictions necessary to independently reproduce all scores.
