# Reproducibility protocol

This document defines the publication-facing execution order for GeoRadioTwin.
It is designed to prevent metric cherry-picking, split leakage, and accidental
mixing of normalized errors with physical dB units.

## One-command pipeline

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_paper_pipeline.ps1
```

The runner reuses completed runs, executes missing three-seed experiments,
validates digital-twin capabilities, regenerates the protocol-checked aggregate,
rebuilds paper figures/source data, and runs the full unit-test suite. If all
training and evaluation outputs already exist:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_paper_pipeline.ps1 `
  -SkipTraining -SkipTwinValidation -SkipTwinCompletion
```

To use another interpreter, pass its absolute path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_paper_pipeline.ps1 `
  -PythonExecutable D:\Anaconda3\envs\rmdm5070\python.exe
```

## Fixed protocol

- Data-selection seed: 42.
- Training seeds: 42, 43, and 44.
- Split: scene-disjoint train/validation/test.
- Test set: the same 128 balanced anchor maps for the main comparison.
- Support: valid receiver pixels only.
- Primary target domain: `x / 127.5 - 1.0`.
- Aggregation: per-map metric followed by macro mean; seed variability is the
  sample standard deviation across independent training seeds.
- Selection: best validation NMSE, never best test performance.
- Sparse measurements: none in the active paper protocol.

## Six publication deliverables

1. Main results table: `paper_artifacts/tables/table1_main_results.tex`.
2. Ablation table: `paper_artifacts/tables/table2_ablation.tex`.
3. Digital-twin validation table: `paper_artifacts/tables/table3_twin_validation.tex`.
4. Submission figures and CSV source data: `paper_artifacts/figures/` and
   `paper_artifacts/source_data/`.
5. English Results/Ablation text and captions: `paper_artifacts/manuscript/`.
6. Code, environment requirements, tests, and runners: repository root,
   `requirements.txt`, `tests/`, and `scripts/`.

## Mandatory reporting boundaries

- Do not label normalized or raw-code RMSE/NMSE/MAE as dB. The official
  PNG-to-dB mapping has not been independently verified.
- State that UniRM, PEFNet-inspired, RMTransformer-style, RadioDiff-3D-style,
  and RadioUNet-style entries are adaptations under the common protocol, not
  official reproductions.
- Do not turn one-seed diagnostic ablations into significance claims.
- The twin validation covers transmitter position, receiver height, planning,
  and epistemic uncertainty. It does not validate temporal fading or building
  edits without paired independent simulator labels.
