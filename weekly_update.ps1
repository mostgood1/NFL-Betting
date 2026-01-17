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
  [string]$GitRemote = 'origin',
  [string]$GitBranch = '' ,
  [switch]$NoRetrain,
  [int]$SimNSims = 2000
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

# 0) Auto-advance current_week.json to the upcoming slate (runs safely even if already up-to-date)
Invoke-Step "Auto-advance current_week.json" {
  & $venvPy scripts/update_current_week.py | Write-Host
  # Re-resolve after potential update to keep TargetWeek aligned
  $cur2 = Resolve-CurrentWeek
  if ($cur2) {
    if ($Season -le 0) { $Season = [int]$cur2.Season }
    # If caller didn't explicitly provide a TargetWeek, follow the marker
    if ($PSBoundParameters.ContainsKey('TargetWeek') -eq $false -and $TargetWeek -gt 0) {
      # TargetWeek already resolved above; keep as-is
    } elseif ($TargetWeek -le 0 -or -not $PSBoundParameters.ContainsKey('TargetWeek')) {
      $TargetWeek = [int]$cur2.Week
    }
    if ($PriorWeek -le 0) { $PriorWeek = [Math]::Max(1, [int]$TargetWeek - 1) }
    Write-Host "Re-aligned to Season=$Season PriorWeek=$PriorWeek TargetWeek=$TargetWeek based on marker" -ForegroundColor DarkGray
  }
}

# 1) Fetch schedules (games.csv) for current season
Invoke-Step "Fetch schedules ($Season)" {
  & $venvPy -m nfl_compare.src.fetch_nflfastr --seasons $Season --only-schedules | Write-Host
}

# 2) Fetch team-week stats (team_stats.csv) for current season
Invoke-Step "Fetch team-week stats ($Season)" {
  & $venvPy -m nfl_compare.src.fetch_nflfastr --seasons $Season --only-stats | Write-Host
}

# 2b) Build Phase-A team-week feeds from pbp up through TargetWeek (incl. playoffs)
Invoke-Step "Build Phase-A features (Season=$Season EndWeek=$TargetWeek)" {
  & $venvPy scripts/build_phase_a_features.py --season $Season --end-week $TargetWeek | Write-Host
}

# 2c) Backfill per-date weather snapshots (Open-Meteo) and materialize NOAA-like game weather features
Invoke-Step "Backfill weather + build NOAA features (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/backfill_weather_open_meteo.py --season $Season --weeks $TargetWeek | Write-Host
  & $venvPy scripts/build_weather_noaa.py --season $Season --week $TargetWeek | Write-Host
}

# 2d) Materialize officiating crew feature placeholders for the target week
Invoke-Step "Build officiating crews (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/build_officiating_crews.py --season $Season --week $TargetWeek | Write-Host
}

# 3) Retrain models (unless disabled)
if (-not $NoRetrain.IsPresent) {
  Invoke-Step "Retrain models" {
    & $venvPy -m nfl_compare.src.train | Write-Host
  }
} else {
  Write-Host "[SKIP] Retrain models (NoRetrain flag)" -ForegroundColor Yellow
}

# 3a) Build team ratings artifacts for TargetWeek (EMA priors)
Invoke-Step "Build team ratings (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/build_team_ratings.py --season $Season --week $TargetWeek | Write-Host
}

# 3b) Fetch latest odds snapshot and seed/enrich lines for the target week
Invoke-Step "Fetch odds + seed lines (Season=$Season Week=$TargetWeek)" {
  # Ensure region default; ODDS_API_KEY should be in env or .env
  if (-not $env:ODDS_API_REGION) { $env:ODDS_API_REGION = 'us' }
  & $venvPy -m nfl_compare.src.odds_api_client | Write-Host
  & $venvPy scripts/seed_lines_for_week.py --season $Season --week $TargetWeek | Write-Host
}

# 3c) Build weekly depth chart BEFORE generating props (ensures latest starters/actives)
Invoke-Step "Build depth chart (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/build_depth_chart.py $Season $TargetWeek | Write-Host
}

# 4) Generate props for target week
Invoke-Step "Generate props (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/gen_props.py $Season $TargetWeek | Write-Host
}

# 4b) Also recompute props EV/edges/ladders for the target week
Invoke-Step "Props pipeline (Season=$Season Week=$TargetWeek)" {
  & $venvPy scripts/run_props_pipeline.py | Write-Host
}

# 4c) Compute shipped sim artifacts for the target week (Render should be shipped-only)
Invoke-Step "Simulate games (Season=$Season Week=$TargetWeek)" {
  $outDir = "nfl_compare/data/backtests/${Season}_wk${TargetWeek}"
  & $venvPy scripts/simulate_games.py --season $Season --start-week $TargetWeek --end-week $TargetWeek --n-sims $SimNSims --out-dir $outDir --quarters --drives | Write-Host
}

# 5) Reconcile prior week props vs actuals (writes player_props_vs_actuals_*.csv)
Invoke-Step "Reconcile props vs actuals (Season=$Season Week=$PriorWeek)" {
  # Prefer in-package reconciliation (local PBP fallback)
  & $venvPy -c "from nfl_compare.src.reconciliation import reconcile_props, summarize_errors; import pandas as pd; m=reconcile_props($Season,$PriorWeek); s=summarize_errors(m); fp=f'nfl_compare/data/player_props_vs_actuals_{Season}_wk{PriorWeek}.csv'; m.to_csv(fp, index=False); print(f'Wrote {fp} ({len(m)} rows)'); print(s.to_string(index=False))" | Write-Host
}

# 5b) Evaluate calibration uplift on recent weeks (quick summary for reports)
Invoke-Step "Calibration uplift eval (Season=$Season end=$PriorWeek lookback=8)" {
  & $venvPy scripts/eval_calibration_uplift.py --season $Season --end-week $PriorWeek --lookback 8 | Write-Host
}

# 6) Stage and optionally commit/pull --rebase/push with safety
Invoke-Step "Stage updated data" {
  git add -- nfl_compare/data/games.csv nfl_compare/data/team_stats.csv nfl_compare/data/player_props_${Season}_wk${TargetWeek}.csv nfl_compare/data/player_props_vs_actuals_${Season}_wk${PriorWeek}.csv nfl_compare/data/team_ratings_${Season}_wk${TargetWeek}.csv | Write-Host
  if (Test-Path 'nfl_compare/data/lines.csv') { git add -- nfl_compare/data/lines.csv | Write-Host }
  # Shipped sim artifacts for the target week
  $simDir = "nfl_compare/data/backtests/${Season}_wk${TargetWeek}"
  if (Test-Path (Join-Path $simDir 'sim_probs.csv')) { git add -- (Join-Path $simDir 'sim_probs.csv') | Write-Host }
  if (Test-Path (Join-Path $simDir 'sim_quarters.csv')) { git add -- (Join-Path $simDir 'sim_quarters.csv') | Write-Host }
  if (Test-Path (Join-Path $simDir 'sim_drives.csv')) { git add -- (Join-Path $simDir 'sim_drives.csv') | Write-Host }
  if (Test-Path (Join-Path $simDir 'sim_summary.json')) { git add -- (Join-Path $simDir 'sim_summary.json') | Write-Host }
  # Phase-A team-week feature feeds
  if (Test-Path 'nfl_compare/data/pfr_drive_stats.csv') { git add -- nfl_compare/data/pfr_drive_stats.csv | Write-Host }
  if (Test-Path 'nfl_compare/data/redzone_splits.csv') { git add -- nfl_compare/data/redzone_splits.csv | Write-Host }
  if (Test-Path 'nfl_compare/data/explosive_rates.csv') { git add -- nfl_compare/data/explosive_rates.csv | Write-Host }
  if (Test-Path 'nfl_compare/data/penalties_stats.csv') { git add -- nfl_compare/data/penalties_stats.csv | Write-Host }
  if (Test-Path 'nfl_compare/data/special_teams.csv') { git add -- nfl_compare/data/special_teams.csv | Write-Host }
  # Optional: crew + NOAA augmentations
  if (Test-Path 'nfl_compare/data/officiating_crews.csv') { git add -- nfl_compare/data/officiating_crews.csv | Write-Host }
  if (Test-Path 'nfl_compare/data/weather_noaa.csv') { git add -- nfl_compare/data/weather_noaa.csv | Write-Host }
  # Optional: calibration artifacts (small JSON; used by runtime if present)
  if (Test-Path 'nfl_compare/data/prob_calibration.json') { git add -- nfl_compare/data/prob_calibration.json | Write-Host }
  if (Test-Path 'nfl_compare/data/sigma_calibration.json') { git add -- nfl_compare/data/sigma_calibration.json | Write-Host }
  if (Test-Path 'nfl_compare/data/totals_calibration.json') { git add -- nfl_compare/data/totals_calibration.json | Write-Host }
  $todayJson = (Get-ChildItem -Path (Join-Path (Join-Path $PSScriptRoot 'nfl_compare') 'data') -Filter "real_betting_lines_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
  if ($todayJson) { git add -- $todayJson.FullName | Write-Host }
  if (Test-Path 'nfl_compare/data/predictions.csv') { git add -- nfl_compare/data/predictions.csv | Write-Host }
  if (Test-Path 'nfl_compare/models/nfl_models.joblib') { git add -- nfl_compare/models/nfl_models.joblib | Write-Host }
}

if ($Commit.IsPresent -or $Push.IsPresent) {
  $msg = "weekly: season $Season; built team ratings wk $TargetWeek; reconciled wk $PriorWeek; generated wk $TargetWeek props; calibration uplift eval"
  # Detect if anything is staged to commit
  $cached = git diff --cached --name-only
  if ($cached) {
    Invoke-Step "Commit" { git commit -m $msg | Write-Host }
  } else {
    Write-Host "[SKIP] Commit (no staged changes)" -ForegroundColor Yellow
  }
}

if ($Push.IsPresent) {
  $branch = $GitBranch
  if (-not $branch) { $branch = (git rev-parse --abbrev-ref HEAD).Trim(); if (-not $branch) { $branch = 'master' } }
  # Stash any remaining working changes to avoid pull conflicts
  $stashOut = git stash push -u --keep-index -m 'weekly_update_autostash'
  if ($stashOut -and ($stashOut -notmatch 'No local changes to save')) { Write-Host "[INFO] Stashed local changes: $stashOut" -ForegroundColor DarkGray }
  Invoke-Step "Pull --rebase $GitRemote/$branch" { git pull --rebase $GitRemote $branch | Write-Host }
  Invoke-Step "Push to $GitRemote/$branch" { git push $GitRemote $branch | Write-Host }
  if ($stashOut -and ($stashOut -notmatch 'No local changes to save')) { git stash pop | Out-Null }
}

Write-Host "Weekly update complete." -ForegroundColor Cyan

# Write last update marker for UI footer (merge: update last_weekly)
try {
  $dataDir = Get-DataDir
  $marker = Join-Path $dataDir 'last_update.json'
  $rootObj = @{}
  if (Test-Path $marker) {
    try { $rootObj = Get-Content -Raw -Path $marker | ConvertFrom-Json | ConvertTo-Json -Compress | ConvertFrom-Json } catch { $rootObj = @{} }
  }
  # Build optional retrain skip note (PowerShell 5.1-safe)
  $skipNote = ''
  if ($NoRetrain.IsPresent) { $skipNote = ' (skipped)' }
  $lastWeekly = [ordered]@{
    ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    season = [int]$Season
    week = [int]$TargetWeek
    prior_week = [int]$PriorWeek
    retrain = (-not $NoRetrain.IsPresent)
    note = "Weekly: schedules, team stats, retrain$skipNote, odds+seed lines, props gen, props pipeline, prior-week recon"
    tasks = @(
      @{ name = 'odds_seed_lines'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'props_generated'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'props_pipeline'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
    )
  }
  # Build new root with last_weekly
  $out = [ordered]@{}
  foreach ($k in $rootObj.PSObject.Properties.Name) { $out[$k] = $rootObj.$k }
  $out['last_weekly'] = $lastWeekly
  ($out | ConvertTo-Json -Compress) | Out-File -FilePath $marker -Encoding UTF8
  Write-Host "Wrote last_update marker: $marker" -ForegroundColor DarkGray
} catch {
  Write-Host "Failed to write last_update.json: $($_.Exception.Message)" -ForegroundColor Yellow
}
