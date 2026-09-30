# RadioDiff-3D reproduction audit and v2 repair

The first local RadioDiff-3D-style run is retained as an audit artifact, but it
is not a successful reproduction and must not be used in the paper table.  Its
three-seed test result was normalized RMSE `0.505675 +/- 0.017121` and NMSE
`1.161480 +/- 0.072273`; all three runs selected epoch 1 and stopped at epoch 16.

## Confirmed causes

1. **The conditional inputs did not match the paper.**  The paper specifies
   building segmentation, building height, and transmitter location tensors.
   The legacy code instead used building height, transmitter location, and an
   receiver-height ramp, omitting building segmentation.
2. **The diffusion objective was masked.**  Noise loss was evaluated only at
   valid receiver voxels.  Reverse diffusion, however, evolves the complete
   volume, so the score was unconstrained in building/zero-valued voxels and
   could contaminate neighbouring valid voxels through 3-D convolutions.
3. **The effective training subset was too small.**  The project training split
   contains 104,000 complete h1--h4 volumes.  The legacy configuration used only
   1,024 (0.985%), whereas the paper's preliminary experiment used 10%.
4. **The sampler was evaluated far from the reported setting.**  The legacy
   test used 50 DDIM steps.  The paper reports no-sampling results for 200-step
   DDIM and for 1,000 full diffusion steps.
5. **Uniform epsilon error did not control reconstruction error at high noise.**
   In the checked-in one-volume diagnostic, epsilon-L2 reached a last-20-step
   training loss of `0.02136`, but the clean-map RMSE reconstructed from the
   final noisy timestep was `111.49`.  The resulting DDIM overfit RMSE was
   `0.48029`.  This explains the old observation that training loss improved
   while validation reconstruction deteriorated.

## Repair

The v2 configuration uses the paper's three condition maps, injects them at
multiple U-Net resolutions, supervises the complete volume, and uses direct x0
prediction with L1 loss.  The manuscript explicitly allows the U-Net to output
either epsilon or denoised x0 and reports L1 in its experimental settings.  In
the same one-volume, 1,000-step diagnostic, v2 achieved:

- valid-voxel DDIM RMSE: `0.006106`;
- full-volume DDIM RMSE: `0.006610`;
- clean-map RMSE at low/mid/high diffusion timesteps: `0.006127`, `0.006105`,
  and `0.006073`.

This passes the implementation-level overfit check.  It does not by itself
establish test-set accuracy; the formal three-seed training is still required.

## Commands

Smoke-test the complete path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_radiodiff3d_paper_aligned_v2.ps1 -SmokeTest
```

Repeat the one-volume diagnostic:

```powershell
python scripts\diagnose_radiodiff3d_pipeline.py --steps 1000 --size 32 --diffusion-steps 50 --loss-type l1 --parameterization x0 --output outputs\diagnostics\radiodiff3d_pipeline_overfit_x0_l1.json
```

Train and evaluate seed 42 first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_radiodiff3d_paper_aligned_v2.ps1 -Seeds 42
```

After seed 42 is confirmed, run seeds 43 and 44:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_radiodiff3d_paper_aligned_v2.ps1 -Seeds 43,44
```

The output directories are
`outputs/radiodiff3d_paper_aligned_v2_random0_seed{42,43,44}`.  Report the
method only as **RadioDiff-3D-style conditional diffusion (paper-aligned adapted
v2)**, never as an official reproduction.
