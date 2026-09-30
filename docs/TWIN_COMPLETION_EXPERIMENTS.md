# Twin-completion experiments

These experiments extend the sampling-free HybridGeo study without treating
model-generated values as ground truth. All accuracy metrics remain in the
encoded normalized domain until an official PNG-to-dB mapping is documented.

## 1. Unseen receiver heights

`evaluate_unseen_heights.py` evaluates the nearby unseen receiver heights h5
through h8 directly from the raw UrbanRadio3D folders. It retains only the
original held-out test scenes and requires every selected `(scene,
transmitter)` anchor to exist at all four requested heights. The model was
trained only on h1--h4. This experiment therefore supports a claim about
nearby unseen-height generalization, not arbitrary-height extrapolation.

```powershell
python scripts\evaluate_unseen_heights.py `
  --heights 5 6 7 8 --anchors 64 --build-missing-caches
```

Outputs are written under `outputs/unseen_height_generalization/`.

## 2. Three-seed uncertainty

`evaluate_ensemble_uncertainty.py` uses the existing seed 42/43/44 HybridGeo
checkpoints. Ensemble standard deviation estimates model/epistemic uncertainty.
The 90% and 95% scale factors are fitted on additional maps from the
scene-disjoint validation partition, after skipping the first 128 anchors used
for checkpoint selection, and are frozen before testing. Because calibration
shares validation scenes with checkpoint selection, the paper should report
empirical coverage rather than claim an exact scene-level conformal guarantee.

```powershell
python scripts\evaluate_ensemble_uncertainty.py --build-missing-caches
```

The output reports ensemble-mean accuracy, interval coverage and width,
error--uncertainty correlation, and risk--coverage curves. With only three
members, the result must be described as model uncertainty, not total aleatoric
uncertainty.

## 3. Building counterfactuals

Building experiments are deliberately two-stage because the repository does
not contain a ray tracer capable of producing scientifically valid labels.

First create deterministic held-out edits and simulator jobs:

```powershell
python scripts\prepare_building_counterfactuals.py
```

This creates removal, height-increase, and addition rasters, plus
`data/counterfactual/generated/simulator_jobs.csv`. Run every job with exactly
the same propagation engine, carrier frequency, material model, antenna model,
resolution, and output encoding used for the base maps. Put each generated
label at its specified path. Replace the three `FILL_REQUIRED` provenance
fields in `data/counterfactual/manifest.csv`, including the SHA-256 of the exact
simulator configuration.

Then evaluate:

```powershell
python scripts\evaluate_geometry_counterfactuals.py `
  --manifest data\counterfactual\manifest.csv
```

The evaluator rejects missing provenance, unchanged building rasters, identical
base/edited label files, missing labels, and interventions with no affected
valid pixels. It reports both full-map change metrics and metrics restricted to
pixels whose ground-truth encoded value changes by at least one raw unit.

## One-command model-side run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1
```

Short verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1 -SmokeTest
```

Prepare counterfactual jobs separately:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1 `
  -SkipUnseenHeights -SkipUncertainty -PrepareCounterfactuals
```
