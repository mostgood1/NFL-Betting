param(
  [int]$Season = 2025,
  [int]$EndWeek = 17,
  [int]$Lookback = 6
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
Set-Location -Path $Repo

$PyVenv = Join-Path $Repo '.venv/Scripts/python.exe'
$Python = if (Test-Path $PyVenv) { $PyVenv } else { 'python' }

# Strict recommendation gates to prioritize precision
$env:RECS_ALLOWED_MARKETS   = 'MONEYLINE,SPREAD,TOTAL'
$env:RECS_ONE_PER_GAME      = 'true'
$env:RECS_MIN_EV_PCT        = '8.0'
$env:RECS_WP_SHRINK         = '0.50'
$env:RECS_WP_MARKET_BAND    = '0.08'
$env:RECS_ATS_BAND          = '0.18'
$env:RECS_TOTAL_BAND        = '0.25'
$env:RECS_PROB_SHRINK       = '0.50'
# Per-market gates
$env:RECS_MIN_WP_DELTA      = '0.10'  # ML prob delta from 0.5
$env:RECS_MIN_EV_PCT_ML     = '8.0'
$env:RECS_MIN_ATS_DELTA     = '0.12'
$env:RECS_MIN_EV_PCT_ATS    = '10.0'
$env:RECS_MIN_TOTAL_DELTA   = '0.12'
$env:RECS_MIN_EV_PCT_TOTAL  = '10.0'

Write-Host ("Running backtest_recommendations for Season={0} EndWeek={1} Lookback={2}" -f $Season, $EndWeek, $Lookback)
& $Python (Join-Path $Repo 'scripts/backtest_recommendations.py') --season $Season --end-week $EndWeek --lookback $Lookback
$btExit = $LASTEXITCODE
Write-Host ("backtest_recommendations exit code: {0}" -f $btExit)

Write-Host ("Running recs_sweep for Season={0} EndWeek={1} Lookback={2}" -f $Season, $EndWeek, $Lookback)
& $Python (Join-Path $Repo 'scripts/recs_sweep.py') --season $Season --end-week $EndWeek --lookback $Lookback
$sweepExit = $LASTEXITCODE
Write-Host ("recs_sweep exit code: {0}" -f $sweepExit)

if ($btExit -ne 0 -or $sweepExit -ne 0) { exit 1 } else { exit 0 }
