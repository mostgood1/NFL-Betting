Param(
  [switch]$Quiet,
  [string]$LogDir = "logs",
  [switch]$NoTrain
)

$ErrorActionPreference = 'Stop'

# Root = folder containing this script
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root

$PythonVenv = Join-Path $Root '.venv/Scripts/python.exe'
$Python = if (Test-Path $PythonVenv) { $PythonVenv } else { 'python' }

# Ensure log directory
$LogPath = Join-Path $Root $LogDir
if (-not (Test-Path $LogPath)) { New-Item -ItemType Directory -Path $LogPath | Out-Null }
$Stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$LogFile = Join-Path $LogPath "daily_update_$Stamp.log"

function Write-Log {
  param([string]$Msg)
  $ts = (Get-Date).ToString('u')
  $line = "[$ts] $Msg"
  $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
  if (-not $Quiet) { Write-Host $line }
}

Write-Log "Starting daily update run"
Write-Log "Python: $Python"

# Optional model training (skip if NoTrain)
if (-not $NoTrain) {
  try {
    Write-Log 'Training models (python -m nfl_compare.src.train)' 
    & $Python -m nfl_compare.src.train | Tee-Object -FilePath $LogFile -Append
  } catch {
    Write-Log "Training failed: $($_.Exception.Message)"
  }
} else {
  Write-Log 'Skipping training per --NoTrain'
}

# Run the daily updater
$ExitCode = 0
try {
  Write-Log 'Running daily updater (python -m nfl_compare.src.daily_updater)'
  $env:PROPS_JITTER_SCALE = '0.3'
  $env:PROPS_OBS_BLEND = '0.55'
  $env:PROPS_EMA_BLEND = '0.6'
  # Reconciliation-driven calibration strengths (defaults tuned in code; reinforce here)
  $env:PROPS_CALIB_ALPHA = '0.35'
  $env:PROPS_CALIB_BETA  = '0.30'
  $env:PROPS_CALIB_GAMMA = '0.25'
  $env:PROPS_CALIB_QB    = '0.40'
  # Position-level multipliers to reduce residual biases
  $env:PROPS_POS_WR_REC_YDS   = '1.02'
  $env:PROPS_POS_TE_REC_YDS   = '0.98'
  $env:PROPS_POS_TE_REC       = '0.98'
  $env:PROPS_POS_QB_PASS_YDS  = '0.97'
  $env:PROPS_POS_QB_PASS_TDS  = '0.95'
  & $Python -m nfl_compare.src.daily_updater | Tee-Object -FilePath $LogFile -Append
  $ExitCode = $LASTEXITCODE
} catch {
  $ExitCode = 1
  Write-Log "daily_updater failed: $($_.Exception.Message)"
}

Write-Log "daily_updater exit code: $ExitCode"

# Simple retention: keep last 14 logs
Get-ChildItem -Path $LogPath -Filter 'daily_update_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 14 | ForEach-Object { Remove-Item $_.FullName -ErrorAction SilentlyContinue }

if ($ExitCode -ne 0) {
  Write-Log 'Completed with errors'
  exit $ExitCode
}

# Run props pipeline (fetch Bovada -> edges -> ladders) using current_week.json
try {
  Write-Log 'Running props pipeline (scripts/run_props_pipeline.py)'
  & $Python scripts/run_props_pipeline.py | Tee-Object -FilePath $LogFile -Append
  $PipelineExit = $LASTEXITCODE
  Write-Log "props_pipeline exit code: $PipelineExit"
  if ($PipelineExit -ne 0) {
    Write-Log 'Props pipeline completed with errors'
    exit $PipelineExit
  }
} catch {
  Write-Log "props_pipeline failed: $($_.Exception.Message)"
  exit 1
}

Write-Log 'Completed successfully'
