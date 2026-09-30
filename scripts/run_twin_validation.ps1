param(
    [switch]$SmokeTest,
    [switch]$SkipCapabilities,
    [switch]$SkipEfficiency,
    [switch]$IncludeCounterfactual,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Candidates = [System.Collections.Generic.List[string]]::new()
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $Candidates.Add([System.IO.Path]::GetFullPath($PythonExe))
} else {
    if (-not [string]::IsNullOrWhiteSpace($env:GEORADIOTWIN_PYTHON)) {
        $Candidates.Add($env:GEORADIOTWIN_PYTHON)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $Candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    # A child PowerShell can reset CONDA_PREFIX to base while displaying the
    # parent's activated prompt. Keep the project's documented environment as
    # an explicit candidate and verify it by importing the required packages.
    # Interpreter is selected from the active environment or PATH.
    $Candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
    $ResolvedPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $ResolvedPython) {
        $Candidates.Add($ResolvedPython.Source)
    }
}

$PythonCommand = $null
foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $Candidate -c "import numpy, torch" 2>$null
    $CandidateExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($CandidateExitCode -eq 0) {
        $PythonCommand = [System.IO.Path]::GetFullPath($Candidate)
        break
    }
}
if ([string]::IsNullOrWhiteSpace($PythonCommand)) {
    throw "No Python with NumPy and PyTorch found. Pass -PythonExe D:\Anaconda3\envs\rmdm5070\python.exe"
}

Write-Host "Using Python: $PythonCommand"
& $PythonCommand -c "import sys, numpy, torch; print('Interpreter:', sys.executable); print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $script:PythonCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $script:PythonCommand $($Arguments -join ' ')"
    }
}

if (-not $SkipCapabilities) {
    $Arguments = @(
        "scripts/evaluate_twin_capabilities.py",
        "--config", "configs/hybrid_geo.yaml",
        "--build-missing-caches"
    )
    if ($SmokeTest) { $Arguments += "--smoke-test" }
    Invoke-PythonChecked $Arguments
}

if (-not $SkipEfficiency) {
    $Arguments = @(
        "scripts/benchmark_efficiency.py",
        "--config", "configs/hybrid_geo.yaml"
    )
    if ($SmokeTest) { $Arguments += "--smoke-test" }
    Invoke-PythonChecked $Arguments
}

if ($IncludeCounterfactual) {
    $Manifest = Join-Path $ProjectRoot "data/counterfactual/manifest.csv"
    if (-not (Test-Path -LiteralPath $Manifest)) {
        throw "Create data/counterfactual/manifest.csv from manifest_template.csv after ray tracing both states."
    }
    Invoke-PythonChecked @(
        "scripts/evaluate_geometry_counterfactuals.py",
        "--manifest", $Manifest,
        "--config", "configs/hybrid_geo.yaml"
    )
}

Write-Host "Twin validation finished."
