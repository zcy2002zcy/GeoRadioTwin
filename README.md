# GeoRadioTwin

**Geometry-Guided Continuous Radio-Field Learning**

GeoRadioTwin predicts radio fields from building-height rasters, transmitter states, and receiver coordinates without measured radio samples at inference. It combines a continuous coordinate-based representation with explicit propagation geometry and learned spatial context.

A trained model supports repeated queries under transmitter-position and receiver-height changes without retraining, providing a propagation-field component for wireless digital-twin applications.

## Overview

GeoRadioTwin combines two complementary prediction branches:

- **Continuous base field:** Fourier-encoded coordinates and analytic line-of-sight (LoS) and first-obstruction depth features are processed by a coordinate MLP.
- **Raster-conditioned residual:** A convolutional encoder extracts environmental features, which are aggregated at transmitter and receiver locations and along their connecting path to refine the base prediction.

Predictions at individual receiver coordinates can be assembled into dense radio maps.

```text
Building-height raster + transmitter state + receiver queries
                         |
           +-------------+-------------+
           |                           |
   Coordinates and geometry      Raster context
           |                           |
   Continuous base field      Path-feature aggregation
           |                           |
           +------ Residual fusion ----+
                         |
               Predicted radio field
                         |
      State updates, height queries, and site planning
```

The model represents a state-conditioned propagation field, not a complete synchronized network digital twin. Online synchronization, temporal fading, and validated building interventions are outside the current scope.

## Key Features

- **Measurement-free inference:** No measured radio samples are required at prediction time.
- **Continuous spatial queries:** Receiver coordinates are queried directly rather than being restricted to a fixed output grid.
- **Geometry-guided prediction:** LoS and first-obstruction depth provide explicit transmitter–receiver geometry.
- **Compact architecture:** The main model contains 1,093,154 trainable parameters.
- **State-conditioned reuse:** One trained model supports transmitter-position and receiver-height updates.
- **Evaluation tools:** The repository includes baseline comparisons, ablations, state-update tests, site-selection evaluation, unseen-height tests, and ensemble uncertainty analysis.

## Repository Structure

| Directory | Description |
|---|---|
| `configs/` | Main-model, baseline, and ablation configurations |
| `datasets/` | Raw-data parsing, coordinate queries, and dense-baseline data loaders |
| `geometry/` | Ray-based geometry queries and caching |
| `models/` | Continuous-field models, HybridGeo, and baseline adapters |
| `physics/`, `calibration/` | Physical regularization and legacy calibration modules |
| `scripts/` | Training, evaluation, experiment runners, and figure generation |
| `utils/` | Configuration, metrics, random seeds, and checkpoint utilities |
| `tests/` | Unit tests using synthetic or temporary data |
| `docs/` | Experimental protocols, adaptation notes, and audit records |
| `external/` | Upstream source registry; no third-party source checkout |
| `results/` | Recorded numerical summaries |
| `paper_artifacts/` | Numerical source tables and LaTeX tables |

## Installation

Clone the repository:

```bash
git clone https://github.com/zcy2002zcy/GeoRadioTwin.git
cd GeoRadioTwin
```

Python 3.10 or later is recommended. Full training requires CUDA-enabled PyTorch.

Create an environment:

```powershell
conda create -n georadiotwin python=3.10 -y
conda activate georadiotwin
```

Install a PyTorch build compatible with your GPU and driver by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

The experiment-runner examples below use PowerShell.

## Dataset

Experiments use **UrbanRadio3D**. Obtain the raw data separately from the [upstream project](https://github.com/UNIC-Lab/UrbanRadio3D) and follow the data provider's terms.

This repository does not distribute raw datasets, pretrained weights, or geometry caches.

Set the dataset paths before running experiments. Replace the example paths with your local directories:

```powershell
$env:GEORADIOTWIN_PATHLOSS_ROOT = 'D:\data\UrbanRadio3D\PathLoss'
$env:GEORADIOTWIN_BUILDING_ROOT = 'D:\data\UrbanRadio3D\buildingsWHeight'
```

Example file locations include:

```text
<PathLoss root>/PathLoss_h1_train/h1/0_X101_Y75.png
<Building root>/0.png
```

The coordinate convention is:

```text
row = H - 1 - X
col = Y
```

Here, X increases upward and Y increases to the right.

Training uses receiver heights **h1–h4**. Unseen-height evaluation uses **h5–h8**.

## Data Preparation and Initial Checks

Run the unit tests and inspect the dataset:

```powershell
python -m unittest discover -s tests -v
python scripts/prepare_raw_dataset.py --config configs/base.yaml --inspect-only --inspect-count 8
```

Prepare the manifests and scene-disjoint splits:

```powershell
python scripts/prepare_raw_dataset.py --config configs/base.yaml
```

Generated manifests are written to `data/processed/`. The original PNG files are not modified.

Optional tests that require unavailable external model source code are skipped.

## Train and Evaluate GeoRadioTwin

### 1. Build geometry caches

```powershell
python scripts/build_geometry_cache.py --config configs/base.yaml --split train --max-maps 4096 --selection-strategy balanced_scene
python scripts/build_geometry_cache.py --config configs/base.yaml --split val --max-maps 128 --selection-strategy balanced_scene
```

### 2. Train the continuous base field

```powershell
python scripts/train.py --config configs/coord_los_depth.yaml
```

### 3. Train the hybrid model

```powershell
python scripts/train_hybrid.py --config configs/hybrid_geo.yaml
```

HybridGeo is initialized from the best LoS+depth-stage checkpoint.

### 4. Evaluate

```powershell
python scripts/evaluate_hybrid.py --config configs/hybrid_geo.yaml
```

Checkpoint selection uses validation NMSE, not test performance. Intermediate files are written to `outputs/` and `cache/`.

## Baselines and Ablations

The experiment suite includes:

- GeoRadioTwin
- UniRM adaptation
- PEFNet-inspired adaptation
- RadioUNet-style adaptation
- RMTransformer-style adaptation
- RadioDiff-3D-style adaptation
- Fitted log-distance regression
- Structural ablations

The UniRM baseline requires a separately obtained, pinned upstream checkout. Do not commit that checkout to this repository.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_external_baselines.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_icc_repro_suite.ps1 -PythonExecutable (Get-Command python).Source
```

The existing runner filename is retained for compatibility.

The data-selection seed is fixed at 42, while training uses seeds 42, 43, and 44. Completed runs are reused by default. Full experiments can require substantial GPU time; the runner supports `-SmokeTest` for a short workflow check.

PMNet and sparse-sampling modules are retained as historical code and are excluded from the current main comparison.

### Comparison boundaries

These baselines are independently adapted implementations, not official reproductions of the original papers. Shared evaluation data and metrics do not imply identical architectures, training objectives, or optimization budgets.

For implementation sources and adaptation assumptions, see:

- [Upstream source registry](external/BASELINE_SOURCES.json)
- [Baseline documentation](docs/PUBLIC_BASELINES.md)
- [PEFNet adaptation notes](docs/PEFNET_ADAPTATION.md)
- [Reproducibility protocol](docs/REPRODUCIBILITY.md)

## State-Conditioned Evaluation

The validation tools examine:

- Responses to transmitter-position changes
- Receiver-height queries
- Candidate-site selection
- Generalization to unseen heights
- Ensemble uncertainty

Run the validation workflows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1 -PythonExe (Get-Command python).Source
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1 -PythonExecutable (Get-Command python).Source
```

Building-intervention validation requires independently generated reference fields and is not included in the default workflow.

## Full Experiment Pipeline

After preparing the dataset manifests and external baseline dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_paper_pipeline.ps1 -PythonExecutable (Get-Command python).Source
```

This entry point coordinates training, result aggregation, state-conditioned validation, figure generation, and unit tests.

Review the configurations before launching the full pipeline. Existing results may be reused; running the command does not necessarily retrain every model.

## Evaluation Protocol

The main evaluation uses:

- Scene-disjoint training, validation, and test splits
- A fixed set of 128 test maps per training seed
- Shared valid-receiver masks
- Per-map metrics followed by macro-averaging
- Mean and sample standard deviation across three training seeds
- Validation-based checkpoint selection

Single-seed diagnostic ablations should be interpreted separately from three-seed comparisons.

### Metric units

Training targets use raw pixel values divided by 255. Main reported metrics use the normalized encoded domain corresponding to:

```text
normalized value = raw pixel value / 127.5 - 1
```

RMSE and MAE are normalized encoding errors. NMSE is a linear ratio, and PSNR is expressed in dB.

The mapping from source PNG values to physical path-loss units has not been independently verified
