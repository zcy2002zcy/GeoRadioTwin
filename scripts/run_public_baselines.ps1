param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

foreach ($rate in @(0.01, 0.05, 0.10)) {
    & $PythonExecutable scripts\evaluate_interpolation_baselines.py `
        --sample-rate $rate --methods nearest idw linear
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

foreach ($rate in @(0.0, 0.01, 0.05, 0.10)) {
    if (-not $SkipTraining) {
        & $PythonExecutable scripts\train_radiounet_baseline.py --sample-rate $rate
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $PythonExecutable scripts\evaluate_radiounet_baseline.py --sample-rate $rate
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# The current ICC paper is sampling-free. Its active neural baselines are run
# by run_sampling_free_baselines.ps1 or the complete run_icc_repro_suite.ps1.
