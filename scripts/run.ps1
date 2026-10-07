# Launch after following README installation. Does not reinstall packages.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$TraceMeetPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $TraceMeetPython)) {
    throw "Virtual environment missing. Follow the README setup before running this launcher."
}
if (-not (Test-Path ".env")) {
    Write-Warning "No .env file found. New model stages require GEMINI_API_KEY and GROQ_API_KEY; environment variables may also provide them."
}
& $TraceMeetPython -m streamlit run app.py
exit $LASTEXITCODE
