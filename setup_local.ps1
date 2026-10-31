param(
    [ValidateSet("cuda", "cpu")]
    [string]$Torch = "cuda",
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv-local\Scripts\python.exe")) {
    python -m venv .venv-local
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed." }
}
$python = Join-Path $PSScriptRoot ".venv-local\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Pip upgrade failed." }
if ($Torch -eq "cuda") {
    & $python -m pip install "torch==2.11.0" --index-url "https://download.pytorch.org/whl/cu128"
} else {
    & $python -m pip install "torch==2.11.0" --index-url "https://download.pytorch.org/whl/cpu"
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed. Try -Torch cpu." }
& $python -m pip install -r requirements_local.txt
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
if (-not $SkipModels) {
    & $python download_models.py
    if ($LASTEXITCODE -ne 0) { throw "Model download failed. Run the setup script again to resume." }
}
Write-Host ""
Write-Host "Setup completed. Start using run_local.bat" -ForegroundColor Green
if ($SkipModels) { Write-Host "Models skipped. Run: .\.venv-local\Scripts\python.exe download_models.py" }
