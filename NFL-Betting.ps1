Param(
  [string]$Port = "8502",
  [switch]$Headless
)

# Resolve paths relative to this script
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$NflCompareDir = Join-Path $Root "nfl_compare"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$AppPath = Join-Path $NflCompareDir "ui\app.py"

if (-not (Test-Path $AppPath)) {
  Write-Error "Streamlit app not found: $AppPath"
  exit 1
}

$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

# Build the command that will run inside the new PowerShell window
$HeadlessFlag = if ($Headless) { "--server.headless true" } else { "" }
$Inner = @"
Set-Location -Path `"$NflCompareDir`"
# Calibration and position multipliers for consistency with daily_update
$env:PROPS_CALIB_ALPHA = '0.35'
$env:PROPS_CALIB_BETA  = '0.30'
$env:PROPS_CALIB_GAMMA = '0.25'
$env:PROPS_CALIB_QB    = '0.40'
$env:PROPS_POS_WR_REC_YDS   = '1.02'
$env:PROPS_POS_TE_REC_YDS   = '0.98'
$env:PROPS_POS_TE_REC       = '0.98'
$env:PROPS_POS_QB_PASS_YDS  = '0.97'
$env:PROPS_POS_QB_PASS_TDS  = '0.95'
& `"$Python`" -m streamlit run `"$AppPath`" --server.port $Port $HeadlessFlag
"@

# Launch in a new PowerShell window and keep it open
Start-Process -FilePath "powershell.exe" -ArgumentList @(
  '-NoExit',
  '-ExecutionPolicy','Bypass',
  '-Command',
  $Inner
) -WindowStyle Normal
