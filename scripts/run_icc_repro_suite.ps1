param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [bool]$SkipCompleted = $true,
    [switch]$SmokeTest,
    [switch]$SkipGeometryBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Invoke-CheckedPython {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

function Invoke-IfMissing {
    param([string]$Label, [string]$Marker, [string[]]$Arguments)
    if ($SkipCompleted -and (Test-Path -LiteralPath $Marker)) {
        Write-Host "SKIP completed: $Label" -ForegroundColor DarkGray
        return
    }
    Invoke-CheckedPython -Label $Label -Arguments $Arguments
}

function Invoke-TrainingIfNeeded {
    param(
        [string]$Label,
        [string]$CompleteMarker,
        [string]$EvaluationMarker,
        [string[]]$Arguments
    )
    if ($SkipCompleted -and (
        (Test-Path -LiteralPath $CompleteMarker) -or
        (Test-Path -LiteralPath $EvaluationMarker)
    )) {
        Write-Host "SKIP completed: $Label" -ForegroundColor DarkGray
        return
    }
    Invoke-CheckedPython -Label $Label -Arguments $Arguments
}

if ($SmokeTest) {
    Invoke-CheckedPython "Unit tests" @("-m", "unittest", "discover", "-s", "tests", "-v")
    Invoke-CheckedPython "UniRM two-stage smoke train" @(
        "scripts\train_unirm_baseline.py", "--seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "UniRM smoke evaluate" @(
        "scripts\evaluate_unirm_baseline.py", "--seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "PEFNet-inspired smoke train" @(
        "scripts\train_pefnet_inspired_baseline.py",
        "--config", "configs\pefnet_inspired_adapted.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "PEFNet-inspired smoke evaluate" @(
        "scripts\evaluate_pefnet_inspired_baseline.py",
        "--config", "configs\pefnet_inspired_adapted.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "RMTransformer-style smoke train" @(
        "scripts\train_rmtransformer_baseline.py",
        "--config", "configs\rmtransformer_adapted.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "RMTransformer-style smoke evaluate" @(
        "scripts\evaluate_rmtransformer_baseline.py",
        "--config", "configs\rmtransformer_adapted.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "RadioDiff-3D-style paper-aligned v2 smoke train" @(
        "scripts\train_radiodiff3d_adapted.py",
        "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "RadioDiff-3D-style paper-aligned v2 smoke evaluate" @(
        "scripts\evaluate_radiodiff3d_adapted.py",
        "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "Log-distance smoke evaluate" @(
        "scripts\evaluate_log_distance_baseline.py", "--seed", "42", "--smoke-test"
    )
    foreach ($config in @(
        "configs\hybrid_no_ray.yaml",
        "configs\hybrid_no_los.yaml",
        "configs\hybrid_no_depth.yaml",
        "configs\hybrid_no_geometry.yaml"
    )) {
        Invoke-CheckedPython "Hybrid smoke: $config" @(
            "scripts\train_hybrid.py", "--config", $config, "--smoke-test"
        )
    }
    Write-Host "`nSmoke suite passed." -ForegroundColor Green
    exit 0
}

# Full-pipeline seeds: retrain the LOS+Depth initializer for every seed when absent.
foreach ($seed in @(42, 43, 44)) {
    $baseConfig = if ($seed -eq 42) {
        "configs\coord_los_depth.yaml"
    } else {
        "configs\coord_los_depth_seed$seed.yaml"
    }
    if (-not $SkipGeometryBuild -and $seed -eq 42) {
        foreach ($splitSpec in @(
            @("train", "4096"), @("val", "128"), @("test", "128")
        )) {
            Invoke-CheckedPython "Geometry seed $seed $($splitSpec[0])" @(
                "scripts\build_geometry_cache.py", "--config", $baseConfig,
                "--split", $splitSpec[0], "--max-maps", $splitSpec[1],
                "--selection-strategy", "balanced_scene"
            )
        }
    }
    $baseDirectory = if ($seed -eq 42) { "coord_los_depth_valid_rx_v1" } else { "coord_los_depth_valid_rx_seed$seed" }
    $baseComplete = "outputs\$baseDirectory\run_complete.json"
    $baseMetrics = "outputs\$baseDirectory\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "LOS+Depth initializer seed $seed" $baseComplete $baseMetrics @(
        "scripts\train.py", "--config", $baseConfig
    )
    Invoke-IfMissing "LOS+Depth no-raster evaluate seed $seed" $baseMetrics @(
        "scripts\evaluate.py", "--config", $baseConfig,
        "--no-save-arrays", "--no-save-figures"
    )
    $hybridConfig = if ($seed -eq 42) { "configs\hybrid_geo.yaml" } else { "configs\hybrid_geo_seed$seed.yaml" }
    $hybridDirectory = if ($seed -eq 42) { "hybrid_geo_valid_rx_v1" } else { "hybrid_geo_valid_rx_seed$seed" }
    $hybridComplete = "outputs\$hybridDirectory\run_complete.json"
    $hybridMetrics = "outputs\$hybridDirectory\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "HybridGeo train seed $seed" $hybridComplete $hybridMetrics @(
        "scripts\train_hybrid.py", "--config", $hybridConfig
    )
    Invoke-IfMissing "HybridGeo evaluate seed $seed" $hybridMetrics @(
        "scripts\evaluate_hybrid.py", "--config", $hybridConfig
    )
}

# Classical neural and component baselines required by the main/ablation tables.
$referenceRuns = @(
    @("coord_only", "configs\coord_only.yaml"),
    @("coord_los", "configs\coord_los.yaml"),
    @("coord_depth", "configs\coord_depth.yaml"),
    @("coord_los_depth", "configs\coord_los_depth.yaml")
)
foreach ($item in $referenceRuns) {
    $name = $item[0]
    $config = $item[1]
    $complete = "outputs\${name}_valid_rx_v1\run_complete.json"
    $metrics = "outputs\${name}_valid_rx_v1\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "$name train" $complete $metrics @("scripts\train.py", "--config", $config)
    Invoke-IfMissing "$name evaluate" $metrics @(
        "scripts\evaluate.py", "--config", $config, "--no-save-arrays", "--no-save-figures"
    )
}

foreach ($seed in @(42, 43, 44)) {
    $radioComplete = "outputs\radiounet_adapted_random0_seed$seed\run_complete.json"
    $radioMetrics = "outputs\radiounet_adapted_random0_seed$seed\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "RadioUNet train seed $seed" $radioComplete $radioMetrics @(
        "scripts\train_radiounet_baseline.py", "--architecture", "radiounet",
        "--sample-rate", "0.0", "--epochs", "100", "--seed", "$seed", "--data-seed", "42"
    )
    Invoke-IfMissing "RadioUNet evaluate seed $seed" $radioMetrics @(
        "scripts\evaluate_radiounet_baseline.py", "--architecture", "radiounet",
        "--sample-rate", "0.0", "--seed", "$seed", "--data-seed", "42"
    )
}

# Strong sampling-free baselines use exactly the same three seeds.
$logDistanceMetrics = "outputs\log_distance_fitted_seed42\evaluation\test\metrics_summary.json"
Invoke-IfMissing "Fitted log-distance baseline" $logDistanceMetrics @(
    "scripts\evaluate_log_distance_baseline.py", "--seed", "42"
)

foreach ($seed in @(42, 43, 44)) {
    $uniComplete = "outputs\unirm_adapted_random0_seed$seed\run_complete.json"
    $uniMetrics = "outputs\unirm_adapted_random0_seed$seed\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "UniRM train seed $seed" $uniComplete $uniMetrics @(
        "scripts\train_unirm_baseline.py", "--seed", "$seed",
        "--data-seed", "42",
        "--stage0-epochs", "100", "--stage1-epochs", "100",
        "--batch-size", "2", "--skip-completed-stage0"
    )
    Invoke-IfMissing "UniRM evaluate seed $seed" $uniMetrics @(
        "scripts\evaluate_unirm_baseline.py", "--seed", "$seed", "--data-seed", "42"
    )

    $pefDirectory = "pefnet_inspired_adapted_stable_v2_random0_seed$seed"
    $pefComplete = "outputs\$pefDirectory\run_complete.json"
    $pefMetrics = "outputs\$pefDirectory\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "PEFNet-inspired train seed $seed" $pefComplete $pefMetrics @(
        "scripts\train_pefnet_inspired_baseline.py",
        "--config", "configs\pefnet_inspired_adapted.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )
    Invoke-IfMissing "PEFNet-inspired evaluate seed $seed" $pefMetrics @(
        "scripts\evaluate_pefnet_inspired_baseline.py",
        "--config", "configs\pefnet_inspired_adapted.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )

    # Both paper-guided adaptations keep the data split fixed at seed 42 and
    # derive training randomness from the loop seed for paired comparison.
    $rmDirectory = "rmtransformer_adapted_random0_seed$seed"
    $rmComplete = "outputs\$rmDirectory\run_complete.json"
    $rmMetrics = "outputs\$rmDirectory\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "RMTransformer-style train seed $seed" $rmComplete $rmMetrics @(
        "scripts\train_rmtransformer_baseline.py",
        "--config", "configs\rmtransformer_adapted.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )
    Invoke-IfMissing "RMTransformer-style evaluate seed $seed" $rmMetrics @(
        "scripts\evaluate_rmtransformer_baseline.py",
        "--config", "configs\rmtransformer_adapted.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )

    $diffusionDirectory = "radiodiff3d_paper_aligned_v2_random0_seed$seed"
    $diffusionComplete = "outputs\$diffusionDirectory\run_complete.json"
    $diffusionMetrics = "outputs\$diffusionDirectory\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "RadioDiff-3D-style paper-aligned v2 train seed $seed" $diffusionComplete $diffusionMetrics @(
        "scripts\train_radiodiff3d_adapted.py",
        "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )
    Invoke-IfMissing "RadioDiff-3D-style paper-aligned v2 evaluate seed $seed" $diffusionMetrics @(
        "scripts\evaluate_radiodiff3d_adapted.py",
        "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
        "--seed", "$seed", "--data-seed", "42"
    )
}

# Structural ablations use the fixed seed 42; only the named component changes.
$ablations = @(
    @("hybrid_no_ray", "configs\hybrid_no_ray.yaml"),
    @("hybrid_no_los", "configs\hybrid_no_los.yaml"),
    @("hybrid_no_depth", "configs\hybrid_no_depth.yaml"),
    @("hybrid_no_geometry", "configs\hybrid_no_geometry.yaml")
)
foreach ($item in $ablations) {
    $name = $item[0]
    $config = $item[1]
    $complete = "outputs\${name}_valid_rx_v1\run_complete.json"
    $metrics = "outputs\${name}_valid_rx_v1\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "$name train" $complete $metrics @("scripts\train_hybrid.py", "--config", $config)
    Invoke-IfMissing "$name evaluate" $metrics @("scripts\evaluate_hybrid.py", "--config", $config)
}

# The key ray-aggregation ablation is repeated across the same three model
# seeds as the main method. Seed 42 is covered by the structural loop above.
foreach ($seed in @(43, 44)) {
    $config = "configs\hybrid_no_ray_seed$seed.yaml"
    $complete = "outputs\hybrid_no_ray_valid_rx_seed$seed\run_complete.json"
    $metrics = "outputs\hybrid_no_ray_valid_rx_seed$seed\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "hybrid_no_ray train seed $seed" $complete $metrics @(
        "scripts\train_hybrid.py", "--config", $config
    )
    Invoke-IfMissing "hybrid_no_ray evaluate seed $seed" $metrics @(
        "scripts\evaluate_hybrid.py", "--config", $config
    )
}

# Repeat the coordinate-only initializer and no-analytic-geometry hybrid
# ablation for the remaining paired model seeds.
foreach ($seed in @(43, 44)) {
    $coordConfig = "configs\coord_only_seed$seed.yaml"
    $coordComplete = "outputs\coord_only_valid_rx_seed$seed\run_complete.json"
    $coordMetrics = "outputs\coord_only_valid_rx_seed$seed\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "coord_only train seed $seed" $coordComplete $coordMetrics @(
        "scripts\train.py", "--config", $coordConfig
    )
    Invoke-IfMissing "coord_only evaluate seed $seed" $coordMetrics @(
        "scripts\evaluate.py", "--config", $coordConfig,
        "--no-save-arrays", "--no-save-figures"
    )

    $hybridConfig = "configs\hybrid_no_geometry_seed$seed.yaml"
    $hybridComplete = "outputs\hybrid_no_geometry_valid_rx_seed$seed\run_complete.json"
    $hybridMetrics = "outputs\hybrid_no_geometry_valid_rx_seed$seed\evaluation\test\metrics_summary.json"
    Invoke-TrainingIfNeeded "hybrid_no_geometry train seed $seed" $hybridComplete $hybridMetrics @(
        "scripts\train_hybrid.py", "--config", $hybridConfig
    )
    Invoke-IfMissing "hybrid_no_geometry evaluate seed $seed" $hybridMetrics @(
        "scripts\evaluate_hybrid.py", "--config", $hybridConfig
    )
}

Invoke-CheckedPython "Validate and summarize suite" @(
    "scripts\summarize_icc_suite.py", "--require-complete"
)
Write-Host "`nAll requested ICC experiments completed." -ForegroundColor Green
Write-Host "Summary: outputs\icc_suite_summary\aggregate.csv"
