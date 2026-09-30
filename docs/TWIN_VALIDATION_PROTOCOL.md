# Digital-twin-facing validation protocol

This suite tests the trained HybridGeo model as a propagation surrogate. It
does not claim that the repository is a complete, real-time bidirectional
digital twin.

## 1. Stratified scene-disjoint generalization

`evaluate_twin_capabilities.py` evaluates the fixed scene-disjoint test set and
reports macro map-level metrics by receiver height, LoS/NLoS status, Tx-Rx
distance, and building-density quartile. The data-selection seed remains 42.

## 2. State updates

The same script constructs held-out paired states for:

- a transmitter-position change at fixed scene and receiver height;
- a receiver-height change at fixed scene and transmitter position.

It evaluates the predicted change, not just each endpoint, using delta RMSE,
delta NMSE, correlation, and sign agreement. Missing LoS/depth caches are built
from the unchanged source building raster and recorded with provenance.

## 3. Downstream planning utility

For each held-out scene, the model ranks transmitter candidates by mean encoded
propagation strength over their common valid receiver support. The report gives
top-1 agreement, normalized regret, and rank correlation against the ground
truth ordering. This remains an encoded-domain experiment until an official,
cited PNG-to-dB mapping is configured.

## 4. Efficiency

`benchmark_efficiency.py` compares HybridGeo, adapted RadioUNet, adapted UniRM,
the PEFNet-inspired adaptation, RMTransformer-style, and RadioDiff-3D-style when
their seed-42 checkpoints exist. Deterministic models use one 256x256 map per
inference. RadioDiff-3D uses one four-height 256x256 volume and reports both
volume latency and latency divided by four generated height slices; the main
accuracy protocol still scores only the canonical anchor slice from each
volume. Its formal fixed DDIM step count is stored with the result. The benchmark
reports model parameters, checkpoint size, median/P10/P90 model-only latency,
throughput, and incremental peak CUDA memory.
Host-to-device transfer is excluded. FLOPs are fvcore traces; unsupported
operators are preserved. For RadioDiff-3D, the trace is explicitly one denoiser
forward and the step-multiplied value is marked a denoiser-only estimate rather
than an exact end-to-end FLOP count.

## 5. Building counterfactuals

Building changes are optional and must use independently generated paired
labels. Copy `data/counterfactual/manifest_template.csv` to `manifest.csv` only
after the base and edited geometry have both been simulated with identical ray
tracing settings. The evaluator refuses missing edited labels. Neural model
predictions are never accepted as ground truth.

`prepare_building_counterfactuals.py` creates deterministic held-out removal,
height-increase, and addition rasters plus external-simulator jobs. It does not
create PathLoss labels. Formal evaluation requires simulator identity, version,
and the SHA-256 of the exact simulator configuration in every manifest row.

## 6. Unseen receiver heights

`evaluate_unseen_heights.py` evaluates the nearby unseen heights h5 through h8,
which are disjoint from the h1--h4 training heights. Selection remains within
the original held-out test scenes and uses paired scene/transmitter anchors at
all four heights. The protocol measures nearby unseen-height generalization
and does not claim arbitrary-height extrapolation.

## 7. Predictive uncertainty

`evaluate_ensemble_uncertainty.py` treats the independently trained seed
42/43/44 checkpoints as a deep ensemble. Interval multipliers are fitted on
additional maps from the validation partition after excluding the first 128
checkpoint-selection anchors, and then frozen for the test split. Empirical
coverage is reported without claiming strict scene-level conformal validity.
The uncertainty is epistemic/model uncertainty in the encoded domain, not
calibrated physical-dB uncertainty.

## Commands

Short end-to-end validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1 -SmokeTest
```

Formal experiments:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1
```

After paired building-intervention labels are available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1 `
  -SkipCapabilities -SkipEfficiency -IncludeCounterfactual
```

Run unseen-height and uncertainty experiments:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1
```

Primary outputs:

- `outputs/twin_validation_seed42/generalization_summary.csv`
- `outputs/twin_validation_seed42/state_transitions.csv`
- `outputs/twin_validation_seed42/planning_utility.csv`
- `outputs/twin_validation_seed42/physical_scale_audit.json`
- `outputs/twin_efficiency_seed42/efficiency.csv`
- `outputs/geometry_counterfactuals_seed42/counterfactual_pairs.csv`
