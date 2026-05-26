# Build ensemble_forecast.pbix from ensemble_forecast.pbip (Windows + pbi-tools).
# Usage: .\build_pbix.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Pbip = Join-Path $Root "ensemble_forecast.pbip"
$Out = Join-Path $Root "ensemble_forecast.pbix"
$ToolsZip = Join-Path $env:TEMP "pbi-tools.1.2.0.zip"
$ToolsDir = Join-Path $env:TEMP "pbi-tools-1.2.0"

if (-not (Test-Path $Pbip)) {
    throw "PBIP not found: $Pbip"
}

if (-not (Test-Path (Join-Path $ToolsDir "pbi-tools.exe"))) {
    Write-Host "Downloading pbi-tools 1.2.0..."
    Invoke-WebRequest -Uri "https://github.com/pbi-tools/pbi-tools/releases/download/1.2.0/pbi-tools.1.2.0.zip" -OutFile $ToolsZip
    Expand-Archive $ToolsZip -DestinationPath $ToolsDir -Force
}

$Exe = Join-Path $ToolsDir "pbi-tools.exe"
& $Exe compile $Pbip -format Pbix -outPath $Out
Write-Host "Created: $Out"
