param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [switch]$SkipRadioUNet,
    [switch]$SkipRMTransformer,
    [switch]$SkipRadioDiff3D,
    [bool]$SkipCompleted = $true
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

function Invoke-TrainAndEvaluate {
    param(
        [string]$Label,
        [string]$Directory,
        [string[]]$TrainArguments,
        [string[]]$EvaluateArguments
    )
    $complete = "outputs\$Directory\run_complete.json"
    $metrics = "outputs\$Directory\evaluation\test\metrics_summary.json"
    if (-not ($SkipCompleted -and ((Test-Path -LiteralPath $complete) -or (Test-Path -LiteralPath $metrics)))) {
        Invoke-CheckedPython "$Label train" $TrainArguments
    } else {
        Write-Host "SKIP completed training: $Label" -ForegroundColor DarkGray
    }
    if (-not ($SkipCompleted -and (Test-Path -LiteralPath $metrics))) {
        Invoke-CheckedPython "$Label evaluate" $EvaluateArguments
    } else {
        Write-Host "SKIP completed evaluation: $Label" -ForegroundColor DarkGray
    }
}

if (-not $SkipRadioUNet) {
    Invoke-TrainAndEvaluate "RadioUNet seed 42" "radiounet_adapted_random0_seed42" @(
        "scripts\train_radiounet_baseline.py", "--architecture", "radiounet",
        "--sample-rate", "0.0", "--seed", "42", "--data-seed", "42"
    ) @(
        "scripts\evaluate_radiounet_baseline.py", "--architecture", "radiounet",
        "--sample-rate", "0.0", "--seed", "42", "--data-seed", "42"
    )
}

foreach ($seed in @(42, 43, 44)) {
    if (-not $SkipRMTransformer) {
        Invoke-TrainAndEvaluate "RMTransformer-style seed $seed" "rmtransformer_adapted_random0_seed$seed" @(
            "scripts\train_rmtransformer_baseline.py",
            "--config", "configs\rmtransformer_adapted.yaml",
            "--seed", "$seed", "--data-seed", "42"
        ) @(
            "scripts\evaluate_rmtransformer_baseline.py",
            "--config", "configs\rmtransformer_adapted.yaml",
            "--seed", "$seed", "--data-seed", "42"
        )
    }
    if (-not $SkipRadioDiff3D) {
        Invoke-TrainAndEvaluate "RadioDiff-3D-style paper-aligned v2 seed $seed" "radiodiff3d_paper_aligned_v2_random0_seed$seed" @(
            "scripts\train_radiodiff3d_adapted.py",
            "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
            "--seed", "$seed", "--data-seed", "42"
        ) @(
            "scripts\evaluate_radiodiff3d_adapted.py",
            "--config", "configs\radiodiff3d_paper_aligned_v2.yaml",
            "--seed", "$seed", "--data-seed", "42"
        )
    }
}
