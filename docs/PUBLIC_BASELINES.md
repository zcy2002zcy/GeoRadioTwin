# Public baseline protocol

## Non-negotiable comparison rules

All reported numbers must use the project scene-disjoint test split, the same
128 canonical anchor maps selected by `balanced_scene` with data seed 42, the
valid-Rx mask, raw target files, and per-map averaging. For each selected
RadioDiff-3D anchor, the other h1--h4 slices sharing its scene and transmitter
are loaded only to form the required four-height conditioning volume; only the
original anchor slice is scored. Thus RadioDiff-3D uses 128 conditioning volumes
and returns 128 scored maps, with exactly the same scored map identities as the
2D methods. Sparse methods must use the same seeded random masks and must be
scored only on unobserved valid pixels. Published numbers from a different
dataset or split are not copied into the comparison table.

## Runnable baselines

### RMTransformer-style (height-adapted)

RMTransformer describes a multi-scale Transformer encoder with a convolutional
decoder for radio-map construction. No author-maintained training repository was
verified for the cited paper. The local model is therefore an independent,
paper-guided implementation and is reported only as **RMTransformer-style
(height-adapted)**, never as an official reproduction. It replaces the original
environment representation with the common three-channel `[building height, Tx
source, Rx height]` tensor, uses no external pretraining, and is trained from
scratch on the fixed scene-disjoint split.

### RadioDiff-3D-style conditional diffusion (paper-aligned adapted v2)

The RadioDiff-3D paper and official UrbanRadio3D project establish the source
dataset and diffusion benchmark, but the project repository does not provide a
directly runnable official 3D training package. The local implementation is
therefore reported as **RadioDiff-3D-style conditional diffusion (paper-aligned
adapted v2)**. The corrected condition tensor follows the paper's no-sampling
inputs exactly: building segmentation, building height, and transmitter
location, each repeated over the h1--h4 depth axis. The diffusion objective uses
full-volume x0 prediction with L1 loss because the released manuscript permits
either epsilon or denoised-x0 output and reports L1 in its experimental setup;
this also avoids the high-noise x0 amplification diagnosed in the failed legacy
epsilon implementation. Conditions are injected at multiple U-Net scales.
Training uses 10,400 volumes (10% of the project training volumes), effective
batch size 2 (micro-batch 1 with two-step gradient accumulation), learning rate
5e-5, EMA 0.995 updated every 10 optimizer iterations, and no sparse
target measurements. Formal evaluation uses fixed-seed 200-step DDIM and one
generated volume per anchor; only the canonical anchor slice is scored and
best-of-K selection is prohibited. This remains an independent adaptation, not
a faithful reproduction of unavailable author code.

### Spatial interpolation

`evaluate_interpolation_baselines.py` implements nearest-neighbour, 8-neighbour
inverse-distance weighting (power 2), and piecewise-linear interpolation with a
nearest fallback outside the convex hull. These methods receive only sparse
measurements; they do not receive geometry or transmitter metadata.

### RadioUNet-style (adapted)

The official RadioUNet repository is pinned in `external/BASELINE_SOURCES.json`.
Its released implementation is notebook-based and was designed for RadioMapSeer.
The local PyTorch implementation is therefore reported as **RadioUNet-style
(adapted)**, never as an unmodified official reproduction. Adaptations are:

- UrbanRadio3D targets and scene-disjoint splits;
- building-height, transmitter-position, and receiver-height input channels;
- sparse-value and sparse-mask channels for 1/5/10 percent experiments;
- loss and metrics on unobserved valid receiver pixels.

Each output contains `provenance.json` with the upstream URL, commit, license,
and adaptations.

## Audited but not currently active

- **RadioDiff:** source is public, but the authors request contact for pretrained
  weights and no UrbanRadio3D-trained checkpoint is included. A new result would
  require a documented two-stage retraining on the common split.
- **RadioDiff-Inverse:** the pinned repository has no license file and provides
  FFHQ/ImageNet priors rather than an UrbanRadio3D-trained radio-map score model.
  Using those weights as if they were the published radio baseline would be
  methodologically invalid.
- **PMNet-v1 (adapted):** the pinned MIT-licensed source, local adapter, and
  historical outputs are retained for auditability. By experiment-design choice,
  PMNet is excluded from active runners and from the default ICC aggregate; its
  old results must not be mixed into the new main table.

The first two methods must remain marked `N/A (official artifact unavailable)`
unless a faithful checkpoint or complete training recipe is obtained from the
authors. PMNet is simply out of scope for the active comparison.

## Citations

- R. Levie et al., "RadioUNet: Fast Radio Map Estimation With Convolutional
  Neural Networks," IEEE TWC, 2021. Official code:
  https://github.com/RonLevie/RadioUNet
- J.-H. Lee et al., "PMNet: Robust Pathloss Map Prediction via Supervised
  Learning," IEEE GLOBECOM, 2023. Official code:
  https://github.com/abman23/PMNet
- Y. Li, C. Zhang, W. Wang, and Y. Huang, "RMTransformer: Accurate Radio
  Map Construction and Coverage Prediction," IEEE VTC2025-Spring, 2025,
  doi:10.1109/VTC2025-SPRING65109.2025.11174709.
- X. Wang et al., "RadioDiff: An Effective Generative Diffusion Model for
  Sampling-Free Dynamic Radio Map Construction," IEEE TCCN, 2025,
  doi:10.1109/TCCN.2024.3504489.
- X. Wang et al., "RadioDiff-3D: A 3D x 3D Radio Map Dataset and Generative
  Diffusion Based Benchmark for 6G Environment-Aware Communication," IEEE TNSE,
  early access 2025; vol. 13, pp. 3773--3789, 2026,
  doi:10.1109/TNSE.2025.3590545.

Machine-readable records are stored in `docs/baselines.bib`. Verify final issue
metadata against IEEE Xplore immediately before manuscript submission.
