param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [switch]$SmokeTest,
    [switch]$SkipUnseenHeights,
    [switch]$SkipUncertainty,
    [switch]$PrepareCounterfactuals,
    [switch]$EvaluateCounterfactuals,
    [switch]$ForceCounterfactualPreparation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
& $PythonExecutable -c "import numpy, torch, PIL; print('Python ready; CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "The selected Python environment is missing required packages" }

function Invoke-CheckedPython {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $script:PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

if (-not $SkipUnseenHeights) {
    $Arguments = @(
        "scripts\evaluate_unseen_heights.py",
        "--heights", "5", "6", "7", "8",
        "--anchors", "64",
        "--build-missing-caches"
    )
    if ($SmokeTest) { $Arguments += "--smoke-test" }
    Invoke-CheckedPython "Unseen receiver-height generalization" $Arguments
}

if (-not $SkipUncertainty) {
    $Arguments = @(
        "scripts\evaluate_ensemble_uncertainty.py",
        "--build-missing-caches"
    )
    if ($SmokeTest) { $Arguments += "--smoke-test" }
    Invoke-CheckedPython "Three-seed ensemble uncertainty" $Arguments
}

if ($PrepareCounterfactuals) {
    $Arguments = @("scripts\prepare_building_counterfactuals.py")
    if ($ForceCounterfactualPreparation) { $Arguments += "--force" }
    if ($SmokeTest) { $Arguments += "--smoke-test" }
    Invoke-CheckedPython "Prepare building counterfactual simulator jobs" $Arguments
}

if ($EvaluateCounterfactuals) {
    if ($SmokeTest) {
        throw "Counterfactual evaluation has no smoke labels. Generate independent simulator labels first."
    }
    Invoke-CheckedPython "Evaluate building counterfactuals" @(
        "scripts\evaluate_geometry_counterfactuals.py",
        "--manifest", "data\counterfactual\manifest.csv"
    )
}

Write-Host "`nRequested twin-completion experiments finished." -ForegroundColor Green
