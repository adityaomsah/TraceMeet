# Mention in the README that if PowerShell blocks the script, the command is powershell -ExecutionPolicy Bypass -File scripts\run.ps1.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --quiet

if (-not (Test-Path ".env")) {
    Write-Warning "No .env file found. Copy .env.example to .env and add your API key(s)."
}

streamlit run app.py