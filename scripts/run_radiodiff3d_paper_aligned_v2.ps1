param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [int[]]$Seeds = @(42, 43, 44),
    [int]$SamplingSteps = 200,
    [bool]$SkipCompleted = $true,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$Config = "configs\radiodiff3d_paper_aligned_v2.yaml"

function Invoke-CheckedPython {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

if ($SmokeTest) {
    Invoke-CheckedPython "RadioDiff-3D v2 smoke train" @(
        "scripts\train_radiodiff3d_adapted.py", "--config", $Config,
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    Invoke-CheckedPython "RadioDiff-3D v2 smoke evaluate" @(
        "scripts\evaluate_radiodiff3d_adapted.py", "--config", $Config,
        "--seed", "42", "--data-seed", "42", "--smoke-test"
    )
    exit 0
}

foreach ($seed in $Seeds) {
    $directory = "radiodiff3d_paper_aligned_v2_random0_seed$seed"
    $complete = "outputs\$directory\run_complete.json"
    $metrics = "outputs\$directory\evaluation\test\metrics_summary.json"
    if (-not ($SkipCompleted -and (Test-Path -LiteralPath $complete))) {
        Invoke-CheckedPython "RadioDiff-3D v2 train seed $seed" @(
            "scripts\train_radiodiff3d_adapted.py", "--config", $Config,
            "--seed", "$seed", "--data-seed", "42"
        )
    } else {
        Write-Host "SKIP completed training: RadioDiff-3D v2 seed $seed" -ForegroundColor DarkGray
    }
    if (-not ($SkipCompleted -and (Test-Path -LiteralPath $metrics))) {
        Invoke-CheckedPython "RadioDiff-3D v2 evaluate seed $seed" @(
            "scripts\evaluate_radiodiff3d_adapted.py", "--config", $Config,
            "--seed", "$seed", "--data-seed", "42",
            "--sampling-algorithm", "DDIM", "--sampling-steps", "$SamplingSteps"
        )
    } else {
        Write-Host "SKIP completed evaluation: RadioDiff-3D v2 seed $seed" -ForegroundColor DarkGray
    }
}
