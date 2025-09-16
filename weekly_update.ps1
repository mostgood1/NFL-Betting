# Weekly Update Automation Script
# Purpose: Fetch latest schedules and team stats, retrain models including new week,
# generate player props for target week, reconcile props vs actuals for prior week,
# and commit/push updated data artifacts.
#
# Usage examples (PowerShell):
#   .\weekly_update.ps1 -Season 2025 -PriorWeek 2 -TargetWeek 3 -Commit -Push
#   .\weekly_update.ps1 -Season 2025 -PriorWeek 2 -TargetWeek 3 -NoRetrain
#
# Parameters
param(
  # If not provided (<=0), Season/PriorWeek/TargetWeek will be derived from app-level
  # current week inference: env CURRENT_SEASON/CURRENT_WEEK (or DEFAULT_*),
  # else nfl_compare/data/current_week.json
  [int]$Season = 0,
  [int]$PriorWeek = 0,
  [int]$TargetWeek = 0,
  [switch]$Commit,
  [switch]$Push,
  [switch]$NoRetrain
)

$ErrorActionPreference = 'Stop'

function Invoke-Step($Name, [scriptblock]$Block) {
  Write-Host "[STEP] $Name" -ForegroundColor Cyan
  try {
    & $Block
    Write-Host "[OK]   $Name" -ForegroundColor Green
  } catch {
    Write-Host "[FAIL] $Name -> $($_.Exception.Message)" -ForegroundColor Red
    throw
  }
}

$script:DataDir = $null
function Get-DataDir {
  if ($script:DataDir) { return $script:DataDir }
  if ($env:NFL_DATA_DIR -and (Test-Path $env:NFL_DATA_DIR)) {
    $script:DataDir = (Resolve-Path $env:NFL_DATA_DIR).Path
  } else {
    $script:DataDir = (Join-Path $PSScriptRoot 'nfl_compare\data')
  }
  return $script:DataDir
}

function Resolve-CurrentWeek {
  # Try env first
  $s = $null; $w = $null
  if ($env:CURRENT_SEASON) { [int]::TryParse($env:CURRENT_SEASON, [ref]$s) | Out-Null }
  if (-not $s -and $env:DEFAULT_SEASON) { [int]::TryParse($env:DEFAULT_SEASON, [ref]$s) | Out-Null }
  if ($env:CURRENT_WEEK) { [int]::TryParse($env:CURRENT_WEEK, [ref]$w) | Out-Null }
  if (-not $w -and $env:DEFAULT_WEEK) { [int]::TryParse($env:DEFAULT_WEEK, [ref]$w) | Out-Null }
  if ($s -and $w) { return @{ Season = $s; Week = $w } }
  # Fallback: data/current_week.json
  $fp = Join-Path (Get-DataDir) 'current_week.json'
  if (Test-Path $fp) {
    try {
      $obj = Get-Content -Raw -Path $fp | ConvertFrom-Json
      $ss = 0; $ww = 0
      if ($obj.season) { [int]::TryParse([string]$obj.season, [ref]$ss) | Out-Null }
      if ($obj.week) { [int]::TryParse([string]$obj.week, [ref]$ww) | Out-Null }
      if ($ss -and $ww) { return @{ Season = $ss; Week = $ww } }
    } catch { }
  }
  return $null
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { $venvPy = "python" }

# Derive defaults for Season/Weeks if not provided
if ($Season -le 0 -or $PriorWeek -le 0 -or $TargetWeek -le 0) {
  $cur = Resolve-CurrentWeek
  if (-not $cur) { throw "Unable to resolve current week via env or current_week.json; please pass -Season/-PriorWeek/-TargetWeek explicitly." }
  if ($Season -le 0) { $Season = [int]$cur.Season }
  if ($TargetWeek -le 0) { $TargetWeek = [int]$cur.Week }
  if ($PriorWeek -le 0) { $PriorWeek = [Math]::Max(1, [int]$TargetWeek - 1) }
}

Write-Host "Resolved Season=$Season PriorWeek=$PriorWeek TargetWeek=$TargetWeek (DataDir=$(Get-DataDir))" -ForegroundColor DarkGray

# 1) Fetch schedules (games.csv) for current season
Invoke-Step "Fetch schedules ($Season)" {
  & $venvPy -m nfl_compare.src.fetch_nflfastr --seasons $Season --only-schedules | Write-Host
}

# 2) Fetch team-week stats (team_stats.csv) for current season
Invoke-Step "Fetch team-week stats ($Season)" {
  & $venvPy -m nfl_compare.src.fetch_nflfastr --seasons $Season --only-stats | Write-Host
}

# 3) Retrain models (unless disabled)
if (-not $NoRetrain.IsPresent) {
  Invoke-Step "Retrain models" {
    & $venvPy -m nfl_compare.src.train | Write-Host
  }
} else {
  Write-Host "[SKIP] Retrain models (NoRetrain flag)" -ForegroundColor Yellow
}

# 4) Generate props for target week
Invoke-Step "Generate props (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/gen_props.py $Season $TargetWeek | Write-Host
}

# 5) Reconcile prior week props vs actuals (writes player_props_vs_actuals_*.csv)
Invoke-Step "Reconcile props vs actuals (Season=$Season Week=$PriorWeek)" {
  # Prefer in-package reconciliation (local PBP fallback)
  & $venvPy -c "from nfl_compare.src.reconciliation import reconcile_props, summarize_errors; import pandas as pd; m=reconcile_props($Season,$PriorWeek); s=summarize_errors(m); fp=f'nfl_compare/data/player_props_vs_actuals_{Season}_wk{PriorWeek}.csv'; m.to_csv(fp, index=False); print(f'Wrote {fp} ({len(m)} rows)'); print(s.to_string(index=False))" | Write-Host
}

# 6) Stage and optionally commit/push
Invoke-Step "Stage updated data" {
  git add nfl_compare/data/games.csv nfl_compare/data/team_stats.csv nfl_compare/data/player_props_${Season}_wk${TargetWeek}.csv nfl_compare/data/player_props_vs_actuals_${Season}_wk${PriorWeek}.csv | Write-Host
}

if ($Commit.IsPresent) {
  $msg = "weekly: season $Season, prior wk $PriorWeek reconciled; wk $TargetWeek props generated"
  Invoke-Step "Commit" {
    git commit -m $msg | Write-Host
  }
}
if ($Push.IsPresent) {
  Invoke-Step "Push" {
    git push | Write-Host
  }
}

Write-Host "Weekly update complete." -ForegroundColor Cyan
