param(
    [string]$PythonExecutable = (Get-Command python -ErrorAction Stop).Source,
    [switch]$SkipTraining,
    [switch]$SkipTwinValidation,
    [switch]$SkipTwinCompletion,
    [switch]$SkipGeometryBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked "Environment audit" {
    & $PythonExecutable -c "import sys,numpy,scipy,torch,matplotlib; print(sys.executable); print('CUDA:',torch.cuda.is_available())"
}

if (-not $SkipTraining) {
    $TrainingArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", "scripts\run_icc_repro_suite.ps1",
        "-PythonExecutable", $PythonExecutable
    )
    if ($SkipGeometryBuild) { $TrainingArgs += "-SkipGeometryBuild" }
    Invoke-Checked "Three-seed ICC experiment suite" { & powershell @TrainingArgs }
}

if (-not $SkipTwinValidation) {
    Invoke-Checked "Digital-twin capability and efficiency validation" {
        & powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1 `
            -PythonExe $PythonExecutable
    }
}

if (-not $SkipTwinCompletion) {
    Invoke-Checked "Unseen-height and uncertainty validation" {
        & powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1 `
            -PythonExecutable $PythonExecutable
    }
}

Invoke-Checked "Protocol-checked result aggregation" {
    & $PythonExecutable scripts\summarize_icc_suite.py --require-complete
}
Invoke-Checked "ICC manuscript artifacts" {
    & $PythonExecutable scripts\generate_paper_artifacts.py
}
Invoke-Checked "GeoRadioTwin framework figure" {
    & $PythonExecutable scripts\export_framework_scene.py
    & $PythonExecutable scripts\generate_framework_figure.py
    & $PythonExecutable scripts\export_framework_panels.py
}
Invoke-Checked "Unit tests" {
    & $PythonExecutable -m unittest discover -s tests -v
}

Write-Host "`nFull paper pipeline completed." -ForegroundColor Green
Write-Host "Results: outputs\icc_suite_summary\aggregate.csv"
Write-Host "Paper bundle: paper_artifacts\README.md"
