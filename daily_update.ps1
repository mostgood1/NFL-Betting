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
  [switch]$NoReconcileGames
)

$ErrorActionPreference = 'Stop'

# Root = folder containing this script
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $Root

$PythonVenv = Join-Path $Root '.venv/Scripts/python.exe'
$Python = if (Test-Path $PythonVenv) { $PythonVenv } else { 'python' }

# Allow env var to force git push without passing -GitPush explicitly
if (-not $GitPush) {
  $envPush = $env:DAILY_UPDATE_GITPUSH
  if ($envPush -and ($envPush -match '^(1|true|yes|on)$')) {
    $GitPush = $true
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

# Allow env vars to control pre-sync and model inclusion
if (-not $GitSyncFirst) {
  $envSync = $env:DAILY_UPDATE_GITSYNC
  if ($envSync -and ($envSync -match '^(1|true|yes|on)$')) { $GitSyncFirst = $true }
}
if (-not $IncludeModel) {
  $envIncModel = $env:DAILY_UPDATE_INCLUDE_MODEL
  if ($envIncModel -and ($envIncModel -match '^(1|true|yes|on)$')) { $IncludeModel = $true }
}

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

# Optional pre-sync (pull --rebase) to reduce conflicts before generating artifacts
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
  Write-Log 'Git: push disabled (use -GitPush or set DAILY_UPDATE_GITPUSH=1)'
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
