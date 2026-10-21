$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source
$steps = @(
    "scripts/03_make_extraction_figures.py",
    "scripts/05_make_change_figures.py"
)
Push-Location $projectRoot
try {
    foreach ($step in $steps) {
        Write-Host "Running $step"
        & $python $step
        if ($LASTEXITCODE -ne 0) { throw "Step failed: $step" }
    }
} finally {
    Pop-Location
}
