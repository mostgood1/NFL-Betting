Param(
  [switch]$Quiet,
  [string]$LogDir = "logs",
  [switch]$NoTrain,
  # If set, stage/commit/pull --rebase/push repo changes (data/ and nfl_compare/data)
  [switch]$GitPush,
  # If set, do a 'git pull --rebase' before running to reduce conflicts
  [switch]$GitSyncFirst,
  # Include model artifact even if ignored (force add)
  [switch]$IncludeModel,
  # Remote/branch for push; if branch empty, current branch is used
  [string]$GitRemote = "origin",
  [string]$GitBranch = "",
  # Optional reconciliation steps
  [switch]$ReconcileProps,
  [switch]$ReconcileGames,
  [switch]$NoReconcileProps,
  [switch]$NoReconcileGames,
  # Control failure behavior when props pipeline errors occur (default: continue)
  [switch]$FailOnPipeline
)

$ErrorActionPreference = 'Stop'

# Root = folder containing this script
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root

$PythonVenv = Join-Path $Root '.venv/Scripts/python.exe'
$Python = if (Test-Path $PythonVenv) { $PythonVenv } else { 'python' }

# Determine default push behavior:
# - If -GitPush not explicitly provided, default to $true.
# - Env DAILY_UPDATE_GITPUSH can override default: accepts 1/true/on or 0/false/off.
$explicitGitPushProvided = $PSBoundParameters.ContainsKey('GitPush')
if (-not $explicitGitPushProvided) {
  $GitPush = $true
  $envPush = $env:DAILY_UPDATE_GITPUSH
  if ($envPush) {
    if ($envPush -match '^(0|false|no|off)$') { $GitPush = $false }
    elseif ($envPush -match '^(1|true|yes|on)$') { $GitPush = $true }
  }
}

# Env toggles for optional features
if (-not $GitSyncFirst) {
  $envSync = $env:DAILY_UPDATE_GITSYNC
  if ($envSync -and ($envSync -match '^(1|true|yes|on)$')) { $GitSyncFirst = $true }
}
if (-not $IncludeModel) {
  $envIncModel = $env:DAILY_UPDATE_INCLUDE_MODEL
  if ($envIncModel -and ($envIncModel -match '^(1|true|yes|on)$')) { $IncludeModel = $true }
}
$envFailPipe = $env:DAILY_UPDATE_FAIL_ON_PIPELINE
if ($envFailPipe) {
  if ($envFailPipe -match '^(1|true|yes|on)$') { $FailOnPipeline = $true }
  elseif ($envFailPipe -match '^(0|false|no|off)$') { $FailOnPipeline = $false }
}
# Read env overrides (on/off) for reconciliation
$envReconProps = $env:DAILY_UPDATE_RECON_PROPS
if ($envReconProps) {
  if ($envReconProps -match '^(0|false|no|off)$') { $ReconcileProps = $false }
  elseif ($envReconProps -match '^(1|true|yes|on)$') { $ReconcileProps = $true }
}
$envReconGames = $env:DAILY_UPDATE_RECON_GAMES
if ($envReconGames) {
  if ($envReconGames -match '^(0|false|no|off)$') { $ReconcileGames = $false }
  elseif ($envReconGames -match '^(1|true|yes|on)$') { $ReconcileGames = $true }
}

# Odds/prediction pipeline toggles (defaults ON)
$FullOdds = $true
if ($env:DAILY_UPDATE_FULL_ODDS) {
  if ($env:DAILY_UPDATE_FULL_ODDS -match '^(0|false|no|off)$') { $FullOdds = $false }
}
$FullPredict = $true
if ($env:DAILY_UPDATE_FULL_PREDICT) {
  if ($env:DAILY_UPDATE_FULL_PREDICT -match '^(0|false|no|off)$') { $FullPredict = $false }
}
$PredictNext = $true
if ($env:DAILY_UPDATE_PREDICT_NEXT) {
  if ($env:DAILY_UPDATE_PREDICT_NEXT -match '^(0|false|no|off)$') { $PredictNext = $false }
}

# Totals calibration/retune toggles
$FitTotalsCal = $true
if ($env:DAILY_UPDATE_FIT_TOTALS_CAL) {
  if ($env:DAILY_UPDATE_FIT_TOTALS_CAL -match '^(0|false|no|off)$') { $FitTotalsCal = $false }
}
$RunBacktests = $false
if ($env:DAILY_UPDATE_RUN_BACKTESTS) {
  if ($env:DAILY_UPDATE_RUN_BACKTESTS -match '^(1|true|yes|on)$') { $RunBacktests = $true }
}

# Defaults: reconciliation ON unless explicitly disabled
$ReconcilePropsFinal = $true
$ReconcileGamesFinal = $true
if ($NoReconcileProps) { $ReconcilePropsFinal = $false }
if ($NoReconcileGames) { $ReconcileGamesFinal = $false }
if ($ReconcileProps -eq $false) { $ReconcilePropsFinal = $false }
if ($ReconcileGames -eq $false) { $ReconcileGamesFinal = $false }
if ($ReconcileProps -eq $true) { $ReconcilePropsFinal = $true }
if ($ReconcileGames -eq $true) { $ReconcileGamesFinal = $true }

# Helper: resolve current season/week from env or nfl_compare/data/current_week.json
function Resolve-CurrentWeek {
  $s = $null; $w = $null
  if ($env:CURRENT_SEASON) { [int]::TryParse($env:CURRENT_SEASON, [ref]$s) | Out-Null }
  if (-not $s -and $env:DEFAULT_SEASON) { [int]::TryParse($env:DEFAULT_SEASON, [ref]$s) | Out-Null }
  if ($env:CURRENT_WEEK) { [int]::TryParse($env:CURRENT_WEEK, [ref]$w) | Out-Null }
  if (-not $w -and $env:DEFAULT_WEEK) { [int]::TryParse($env:DEFAULT_WEEK, [ref]$w) | Out-Null }
  if ($s -and $w) { return @{ Season = [int]$s; Week = [int]$w } }
  $fp = Join-Path (Join-Path $Root 'nfl_compare') 'data/current_week.json'
  if (Test-Path $fp) {
    try {
      $obj = Get-Content -Raw -Path $fp | ConvertFrom-Json
      $ss = 0; $ww = 0
      if ($obj.season) { [int]::TryParse([string]$obj.season, [ref]$ss) | Out-Null }
      if ($obj.week) { [int]::TryParse([string]$obj.week, [ref]$ww) | Out-Null }
      if ($ss -and $ww) { return @{ Season = [int]$ss; Week = [int]$ww } }
    } catch { }
  }
  return $null
}

## (env toggles applied above)

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

# Auto-advance current_week.json to the upcoming slate (idempotent)
try {
  Write-Log 'Auto-advance current_week.json (scripts/update_current_week.py)'
  & $Python scripts/update_current_week.py | Tee-Object -FilePath $LogFile -Append | Out-Null
} catch {
  Write-Log "Auto-advance failed: $($_.Exception.Message)"
}

# Optional pre-sync (reduce conflicts before artifact generation)
if ($GitSyncFirst) {
  try {
    Write-Log 'Git: pre-sync (pull --rebase)'
    try { git --version | Out-Null } catch { throw }
    & git rev-parse --is-inside-work-tree 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
      $Branch0 = $GitBranch
      if (-not $Branch0) { $Branch0 = (& git rev-parse --abbrev-ref HEAD).Trim(); if (-not $Branch0) { $Branch0 = 'master' } }
      & git pull --rebase $GitRemote $Branch0 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null
    } else {
      Write-Log 'Git: pre-sync skipped (not a git repository)'
    }
  } catch {
    Write-Log "Git: pre-sync failed: $($_.Exception.Message)"
  }
}

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

# Full odds fetch + seed lines (ensures freshest markets regardless of snapshot state)
if ($FullOdds) {
  try {
    $cur = Resolve-CurrentWeek
    if ($null -eq $cur) { throw 'Unable to resolve current week for odds/seed lines' }
    $Season = [int]$cur.Season
    $Week = [int]$cur.Week
    Write-Log "Full odds fetch (python -m nfl_compare.src.odds_api_client)"
    if (-not $env:ODDS_API_REGION) { $env:ODDS_API_REGION = 'us' }
    & $Python -m nfl_compare.src.odds_api_client | Tee-Object -FilePath $LogFile -Append
    Write-Log "Seed/enrich lines for current week (scripts/seed_lines_for_week.py --season $Season --week $Week)"
    & $Python scripts/seed_lines_for_week.py --season $Season --week $Week | Tee-Object -FilePath $LogFile -Append
    if ($PredictNext) {
      $NextWeek = [int]($Week + 1)
      Write-Log "Seed/enrich lines for next week (scripts/seed_lines_for_week.py --season $Season --week $NextWeek)"
      & $Python scripts/seed_lines_for_week.py --season $Season --week $NextWeek | Tee-Object -FilePath $LogFile -Append
    }
  } catch {
    Write-Log "Full odds/seed lines step failed: $($_.Exception.Message)"
  }
} else {
  Write-Log 'Full odds fetch disabled via DAILY_UPDATE_FULL_ODDS; using snapshot check later'
}

# Post-odds smoke: verify odds freshness and lines seeding for current week
try {
  Write-Log 'Smoke: odds freshness and lines seeding (scripts/smoke_odds_update.py)'
  & $Python scripts/smoke_odds_update.py | Tee-Object -FilePath $LogFile -Append
  $SmokeOddsExit = $LASTEXITCODE
  Write-Log "smoke_odds_update exit code: $SmokeOddsExit"
  if ($SmokeOddsExit -ne 0) {
    Write-Log 'Smoke odds update failed'
    if ($FailOnPipeline) {
      Write-Log 'FailOnPipeline is set; exiting due to smoke failure'
      exit $SmokeOddsExit
    } else {
      Write-Log 'Continuing despite smoke failure'
    }
  }
} catch {
  Write-Log "smoke_odds_update exception: $($_.Exception.Message)"
  if ($FailOnPipeline) { exit 1 }
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
  # Enforce per-team usage scaling so player targets sum to team attempts
  $env:PROPS_ENFORCE_TEAM_USAGE = '1'
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

# Full prediction backfill for uncompleted games (current and optionally next week)
if ($FullPredict) {
  try {
    $cur = Resolve-CurrentWeek
    if ($null -eq $cur) { throw 'Unable to resolve current week for predictions' }
    $Season = [int]$cur.Season
    $Week = [int]$cur.Week
    Write-Log "Backfill predictions for current week (scripts/backfill_missing_week_predictions.py --season $Season --week $Week)"
    & $Python scripts/backfill_missing_week_predictions.py --season $Season --week $Week | Tee-Object -FilePath $LogFile -Append | Out-Null
    if ($PredictNext) {
      $NextWeek = [int]($Week + 1)
      Write-Log "Backfill predictions for next week (scripts/backfill_missing_week_predictions.py --season $Season --week $NextWeek)"
      & $Python scripts/backfill_missing_week_predictions.py --season $Season --week $NextWeek | Tee-Object -FilePath $LogFile -Append | Out-Null
    }
    # Build team ratings artifacts for current and next week (optional, fast)
    try {
      Write-Log ("Build team ratings (scripts/build_team_ratings.py --season {0} --week {1})" -f $Season, $Week)
      & $Python scripts/build_team_ratings.py --season $Season --week $Week | Tee-Object -FilePath $LogFile -Append | Out-Null
      if ($PredictNext) {
        Write-Log ("Build team ratings (scripts/build_team_ratings.py --season {0} --week {1})" -f $Season, $NextWeek)
        & $Python scripts/build_team_ratings.py --season $Season --week $NextWeek | Tee-Object -FilePath $LogFile -Append | Out-Null
      }
    } catch {
      Write-Log ("Build team ratings failed: {0}" -f $_.Exception.Message)
    }
  } catch {
    Write-Log "Predictions backfill failed: $($_.Exception.Message)"
    if ($FailOnPipeline) { exit 1 }
  }
} else {
  Write-Log 'Full prediction backfill disabled via DAILY_UPDATE_FULL_PREDICT'
}

# Fit totals calibration (and optional backtests) using midseason retune orchestrator
try {
  if ($FitTotalsCal) {
    $cur = Resolve-CurrentWeek
    if ($null -eq $cur) { throw 'Unable to resolve current week for calibration' }
    $Season = [int]$cur.Season
    $Week = [int]$cur.Week
    $CalibWeeks = 4
    $btFlag = if ($RunBacktests) { '--run-backtests' } else { '' }
    $PropsEndWeek = [int]([Math]::Max(1, $Week - 1))
    Write-Log ("Totals calibration (scripts/retune_midseason.py --season {0} --week {1} --calib-weeks {2} {3} --props-end-week {4})" -f $Season, $Week, $CalibWeeks, $btFlag, $PropsEndWeek)
    & $Python scripts/retune_midseason.py --season $Season --week $Week --calib-weeks $CalibWeeks $btFlag --props-end-week $PropsEndWeek | Tee-Object -FilePath $LogFile -Append
    $CalibExit = $LASTEXITCODE
    Write-Log "totals_calibration exit code: $CalibExit"
    if ($CalibExit -ne 0 -and $FailOnPipeline) {
      Write-Log 'FailOnPipeline is set; exiting due to calibration failure'
      exit $CalibExit
    }
  } else {
    Write-Log 'Totals calibration step disabled via DAILY_UPDATE_FIT_TOTALS_CAL'
  }
} catch {
  Write-Log ("Totals calibration step failed: {0}" -f $_.Exception.Message)
  if ($FailOnPipeline) { exit 1 }
}

# Fit sigma calibration (ATS/TOTAL) from recent completed weeks
try {
  $cur = Resolve-CurrentWeek
  if ($null -eq $cur) { throw 'Unable to resolve current week for sigma calibration' }
  $Season = [int]$cur.Season
  $Week = [int]$cur.Week
  $Lookback = 4
  Write-Log ("Sigma calibration (scripts/fit_sigma_calibration.py --season {0} --week {1} --lookback {2})" -f $Season, $Week, $Lookback)
  & $Python scripts/fit_sigma_calibration.py --season $Season --week $Week --lookback $Lookback | Tee-Object -FilePath $LogFile -Append
  $SigExit = $LASTEXITCODE
  Write-Log "sigma_calibration exit code: $SigExit"
  if ($SigExit -ne 0 -and $FailOnPipeline) {
    Write-Log 'FailOnPipeline is set; exiting due to sigma calibration failure'
    exit $SigExit
  }
} catch {
  Write-Log ("Sigma calibration step failed: {0}" -f $_.Exception.Message)
  if ($FailOnPipeline) { exit 1 }
}

# Fit probability calibration (moneyline, ATS, totals) from recent finalized weeks
try {
  $cur = Resolve-CurrentWeek
  if ($null -eq $cur) { throw 'Unable to resolve current week for probability calibration' }
  $Season = [int]$cur.Season
  $Week = [int]$cur.Week
  $Lookback = 6
  Write-Log ("Probability calibration (scripts/fit_prob_calibration.py --season {0} --end-week {1} --lookback {2})" -f $Season, $Week, $Lookback)
  & $Python scripts/fit_prob_calibration.py --season $Season --end-week $Week --lookback $Lookback | Tee-Object -FilePath $LogFile -Append
  $ProbExit = $LASTEXITCODE
  Write-Log "prob_calibration exit code: $ProbExit"
  if ($ProbExit -ne 0 -and $FailOnPipeline) {
    Write-Log 'FailOnPipeline is set; exiting due to prob calibration failure'
    exit $ProbExit
  }
  # Evaluate calibration uplift and write artifacts for weekly report
  $EvalLookback = 8
  Write-Log ("Calibration uplift eval (scripts/eval_calibration_uplift.py --season {0} --end-week {1} --lookback {2})" -f $Season, $Week, $EvalLookback)
  & $Python scripts/eval_calibration_uplift.py --season $Season --end-week $Week --lookback $EvalLookback | Tee-Object -FilePath $LogFile -Append
  $CalEvalExit = $LASTEXITCODE
  Write-Log "calibration_eval exit code: $CalEvalExit"
  if ($CalEvalExit -ne 0 -and $FailOnPipeline) {
    Write-Log 'FailOnPipeline is set; exiting due to calibration eval failure'
    exit $CalEvalExit
  }
} catch {
  Write-Log ("Probability calibration step failed: {0}" -f $_.Exception.Message)
  if ($FailOnPipeline) { exit 1 }
}

# Optional: Standardized backtests + weekly report
try {
  if ($RunBacktests) {
    $cur = Resolve-CurrentWeek
    if ($null -eq $cur) { throw 'Unable to resolve current week for backtests' }
    $Season = [int]$cur.Season
    $Week = [int]$cur.Week
    $BtOutDir = Join-Path (Join-Path $Root 'nfl_compare/data/backtests') ("{0}_wk{1}" -f $Season, $Week)
    Write-Log ("Backtest games (scripts/backtest_games.py --season {0} --start-week 1 --end-week {1} --include-same-season --out-dir {2})" -f $Season, ($Week - 1), $BtOutDir)
    & $Python scripts/backtest_games.py --season $Season --start-week 1 --end-week ($Week - 1) --include-same-season --out-dir $BtOutDir | Tee-Object -FilePath $LogFile -Append
    $BtGamesExit = $LASTEXITCODE
    Write-Log "backtest_games exit code: $BtGamesExit"
    Write-Log ("Backtest props (scripts/backtest_props.py --season {0} --start-week 1 --end-week {1} --out-dir {2})" -f $Season, ($Week - 1), $BtOutDir)
    & $Python scripts/backtest_props.py --season $Season --start-week 1 --end-week ($Week - 1) --out-dir $BtOutDir | Tee-Object -FilePath $LogFile -Append
    $BtPropsExit = $LASTEXITCODE
    Write-Log "backtest_props exit code: $BtPropsExit"
    # Generate markdown report
    Write-Log ("Generate weekly report (scripts/generate_weekly_report.py --season {0} --week {1})" -f $Season, $Week)
    & $Python scripts/generate_weekly_report.py --season $Season --week $Week | Tee-Object -FilePath $LogFile -Append
    $BtRptExit = $LASTEXITCODE
    Write-Log "weekly_report exit code: $BtRptExit"
  } else {
    Write-Log 'Standardized backtests/report disabled (enable with DAILY_UPDATE_RUN_BACKTESTS=1)'
  }
} catch {
  Write-Log ("Standardized backtests/report failed: {0}" -f $_.Exception.Message)
  if ($FailOnPipeline) { exit 1 }
}

# Verify today's odds snapshot exists; if missing or stale, attempt direct fetch
try {
  $OddsDir = Join-Path $Root 'nfl_compare/data'
  $latestOdds = Get-ChildItem -Path $OddsDir -Filter 'real_betting_lines_*.json' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  $today = (Get-Date).Date
  $needsFetch = $true
  if ($null -ne $latestOdds) {
    if ($latestOdds.LastWriteTime.Date -eq $today) { $needsFetch = $false }
  }
  if ($needsFetch) {
    Write-Log 'Odds snapshot missing or not from today; attempting direct odds fetch (python -m nfl_compare.src.odds_api_client)'
    # Ensure default region if not set; books can be configured via .env
    if (-not $env:ODDS_API_REGION) { $env:ODDS_API_REGION = 'us' }
    & $Python -m nfl_compare.src.odds_api_client | Tee-Object -FilePath $LogFile -Append
    # Re-check after fetch
    $latestOdds = Get-ChildItem -Path $OddsDir -Filter 'real_betting_lines_*.json' -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($null -ne $latestOdds -and $latestOdds.LastWriteTime.Date -eq $today) {
      Write-Log ("Odds snapshot ready: {0}" -f $latestOdds.Name)
    } else {
      Write-Log 'Warning: Direct odds fetch did not produce a snapshot for today.'
    }
  } else {
    Write-Log ("Found today's odds snapshot: {0}" -f $latestOdds.Name)
  }
} catch {
  Write-Log ("Odds snapshot check/fetch step failed: {0}" -f $_.Exception.Message)
}

# Run props pipeline (fetch Bovada -> edges -> ladders) using current_week.json
try {
  Write-Log 'Running props pipeline (scripts/run_props_pipeline.py)'
  & $Python scripts/run_props_pipeline.py | Tee-Object -FilePath $LogFile -Append
  $PipelineExit = $LASTEXITCODE
  Write-Log "props_pipeline exit code: $PipelineExit"
    if ($PipelineExit -ne 0) {
      Write-Log 'Props pipeline completed with errors'
      if ($FailOnPipeline) {
        Write-Log 'FailOnPipeline is set; exiting with pipeline error code'
        exit $PipelineExit
      } else {
        Write-Log 'Continuing despite props pipeline errors'
      }
    }
} catch {
  Write-Log "props_pipeline failed: $($_.Exception.Message)"
    if ($FailOnPipeline) {
      exit 1
    } else {
      Write-Log 'Continuing despite props pipeline exception'
    }
}

# Optional: Smoke check that ESPN WR/TE starters appear in weekly props
try {
  $runSmoke = $true
  $envSmoke = $env:DAILY_UPDATE_SMOKE_STARTERS
  if ($envSmoke) {
    if ($envSmoke -match '^(0|false|no|off)$') { $runSmoke = $false }
    elseif ($envSmoke -match '^(1|true|yes|on)$') { $runSmoke = $true }
  }
  if ($runSmoke) {
    Write-Log 'Smoke: WR/TE starters present in props (scripts/smoke_depth_starters_in_props.py)'
    & $Python scripts/smoke_depth_starters_in_props.py | Tee-Object -FilePath $LogFile -Append
    $SmokeExit = $LASTEXITCODE
    Write-Log "smoke_depth_starters_in_props exit code: $SmokeExit"
    if ($SmokeExit -ne 0) {
      Write-Log 'Smoke starters check failed'
      if ($FailOnPipeline) {
        Write-Log 'FailOnPipeline is set; exiting due to smoke failure'
        exit $SmokeExit
      } else {
        Write-Log 'Continuing despite smoke failure'
      }
    }
  } else {
    Write-Log 'Smoke starters check disabled via DAILY_UPDATE_SMOKE_STARTERS'
  }
} catch {
  Write-Log "Smoke starters check exception: $($_.Exception.Message)"
  if ($FailOnPipeline) { exit 1 }
}

  # Optional: Reconcile props vs actuals for prior week
  if ($ReconcilePropsFinal) {
    try {
      $cur = Resolve-CurrentWeek
      if ($null -eq $cur) { throw 'Unable to resolve current week for reconciliation' }
      $Season = [int]$cur.Season
      $Week = [int]$cur.Week
      $PriorWeek = [Math]::Max(1, $Week - 1)
      Write-Log "Recon: props vs actuals (Season=$Season Week=$PriorWeek)"
      & $Python scripts/reconcile_props_vs_actuals.py --season $Season --week $PriorWeek | Tee-Object -FilePath $LogFile -Append | Out-Null
    } catch {
      Write-Log "Recon props failed: $($_.Exception.Message)"
    }
  }

  # Optional: Reconcile games schedule vs predictions for current week
  if ($ReconcileGamesFinal) {
    try {
      $cur = Resolve-CurrentWeek
      if ($null -eq $cur) { throw 'Unable to resolve current week for game reconciliation' }
      $Season = [int]$cur.Season
      $Week = [int]$cur.Week
      Write-Log "Recon: schedule vs predictions (Season=$Season Week=$Week)"
      & $Python scripts/reconcile_schedule_vs_predictions.py --season $Season --week $Week | Tee-Object -FilePath $LogFile -Append | Out-Null
    } catch {
      Write-Log "Recon games failed: $($_.Exception.Message)"
    }
  }

Write-Log 'Completed successfully'

# Optionally commit and push updated artifacts to git
if ($GitPush) {
  Write-Log 'Git: starting commit/push of updated data files'
  # Temporarily relax native command error handling to ignore git warnings on stderr
  $PrevErrPref = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  # Verify git is available and we are inside a repo
  try { git --version | Out-Null } catch { Write-Log 'Git: git not available on PATH; skipping push'; $ErrorActionPreference = $PrevErrPref; return }
  & git rev-parse --is-inside-work-tree 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Log 'Git: not inside a git repository; skipping push'; $ErrorActionPreference = $PrevErrPref; return }
  # Determine branch
  $Branch = $GitBranch
  if (-not $Branch) {
    $Branch = (& git rev-parse --abbrev-ref HEAD).Trim()
    if (-not $Branch) { $Branch = 'master' }
  }
  # Stage typical data directories and updated models, but avoid logs to reduce noise
  & git add -- nfl_compare/data data 2>$null | Tee-Object -FilePath $LogFile -Append
  if ($IncludeModel -and (Test-Path 'nfl_compare/models/nfl_models.joblib')) {
    & git add -f -- 'nfl_compare/models/nfl_models.joblib' 2>$null | Tee-Object -FilePath $LogFile -Append
  }

  # If there are remaining unstaged changes (e.g., logs), stash them while keeping index intact
  $Stashed = $false
  Write-Log 'Git: stashing unstaged/untracked changes with --keep-index (if any)'
  $stashOut = & git stash push -u --keep-index -m 'daily_update_autostash' 2>$null
  $stashOut | Tee-Object -FilePath $LogFile -Append | Out-Null
  if ($stashOut -and ($stashOut -notmatch 'No local changes to save')) { $Stashed = $true }

  # Check if anything is staged to commit
  $cachedList = & git diff --cached --name-only 2>$null
  $HasStaged = [bool]$cachedList
  if (-not $HasStaged) {
    Write-Log 'Git: no staged changes to commit; skipping push'
    # If we stashed, restore the stash to keep working tree unchanged
    if ($Stashed) { & git stash pop 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null }
    $ErrorActionPreference = $PrevErrPref
    return
  }

  # Commit
  $CommitMsg = "Daily update: " + (Get-Date).ToString('u')
  & git commit -m $CommitMsg 2>$null | Tee-Object -FilePath $LogFile -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log 'Git: commit failed; skipping pull/push'
    if ($Stashed) { & git stash pop 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null }
    $ErrorActionPreference = $PrevErrPref
    return
  }

  # Pull rebase to reduce conflicts (index is clean after commit; stash keeps local noise aside)
  & git pull --rebase $GitRemote $Branch 2>$null | Tee-Object -FilePath $LogFile -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log 'Git: pull --rebase failed; resolve manually and push later'
    if ($Stashed) { & git stash pop 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null }
    $ErrorActionPreference = $PrevErrPref
    return
  }
  
  # Push
  & git push $GitRemote $Branch 2>$null | Tee-Object -FilePath $LogFile -Append
  if ($LASTEXITCODE -eq 0) {
    Write-Log "Git: push completed to $GitRemote/$Branch"
  } else {
    Write-Log 'Git: push failed; check credentials/remote and try again'
  }

  # Restore stashed local changes (e.g., logs) without affecting commit
  if ($Stashed) { & git stash pop 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null }

  # Restore error handling
  $ErrorActionPreference = $PrevErrPref
}
else {
  Write-Log 'Git: push disabled (use -GitPush:$true or set DAILY_UPDATE_GITPUSH=1)'
}

# Post-push verification: if data dirs still have changes, attempt a secondary commit/push
try {
  $dirtyData = & git status --porcelain -- nfl_compare/data data 2>$null
  if ($dirtyData) {
    Write-Log 'Git: detected remaining changes in data dirs after run; attempting secondary commit/push'
    # Stage again
    & git add -- nfl_compare/data data 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null
    # Check staged
    $cachedAgain = & git diff --cached --name-only 2>$null
    if ($cachedAgain) {
      $CommitMsg2 = "Daily update (post-check): " + (Get-Date).ToString('u')
      & git commit -m $CommitMsg2 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null
      # Determine branch again
      $Branch2 = $GitBranch
      if (-not $Branch2) { $Branch2 = (& git rev-parse --abbrev-ref HEAD).Trim(); if (-not $Branch2) { $Branch2 = 'master' } }
      # Pull --rebase and push if GitPush enabled, otherwise just log
      if ($GitPush) {
        & git pull --rebase $GitRemote $Branch2 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null
        & git push $GitRemote $Branch2 2>$null | Tee-Object -FilePath $LogFile -Append | Out-Null
        Write-Log "Git: secondary push attempted to $GitRemote/$Branch2"
      } else {
        Write-Log 'Git: secondary commit created but push disabled; run with -GitPush to publish'
      }
    } else {
      Write-Log 'Git: no staged changes found on secondary attempt'
    }
  }
} catch {
  Write-Log "Git: post-push verification error: $($_.Exception.Message)"
}

# Write last update marker for UI footer (merge: update last_daily)
try {
  $dataDir = Join-Path $Root 'nfl_compare/data'
  $marker = Join-Path $dataDir 'last_update.json'
  $cur = Resolve-CurrentWeek
  $rootObj = @{}
  if (Test-Path $marker) {
    try { $rootObj = Get-Content -Raw -Path $marker | ConvertFrom-Json | ConvertTo-Json -Compress | ConvertFrom-Json } catch { $rootObj = @{} }
  }
  # PowerShell 5.1 compatible: compute season/week without ternary
  $seasonVal = $null
  $weekVal = $null
  if ($null -ne $cur) {
    try { $seasonVal = [int]$cur.Season } catch { $seasonVal = $null }
    try { $weekVal = [int]$cur.Week } catch { $weekVal = $null }
  }
  $lastDaily = [ordered]@{
    ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    season = $seasonVal
    week = $weekVal
    note = 'Daily: odds fetch, seed/enrich lines, predictions, weather, props, props pipeline, reconciliations'
    tasks = @(
      @{ name = 'odds_fetch'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'lines_seed_enrich'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'predictions'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'weather'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'player_props'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'props_pipeline'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') },
      @{ name = 'reconciliations'; ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
    )
  }
  $out = [ordered]@{}
  foreach ($k in $rootObj.PSObject.Properties.Name) { $out[$k] = $rootObj.$k }
  $out['last_daily'] = $lastDaily
  ($out | ConvertTo-Json -Compress) | Out-File -FilePath $marker -Encoding UTF8
  Write-Log ("Wrote last_update marker: {0}" -f $marker)
} catch {
  Write-Log ("Failed to write last_update.json: {0}" -f $_.Exception.Message)
}
