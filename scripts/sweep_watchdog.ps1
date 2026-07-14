# Watchdog + auto-checkpointer for the tissue-damage benchmark sweep.
#
# Durable / reboot-safe:
#   * Lives in the repo (survives scratchpad/temp cleanup).
#   * Self-locates the running sweep via results/benchmark.lock (no hard-coded PID),
#     so it reattaches across sweep restarts and reboots.
#   * Single-instance guarded (results/.watchdog.lock) so it can never double-commit.
#   * Never self-exits on sweep death -- does one final checkpoint, then keeps watching
#     for the sweep to come back. A logon scheduled task relaunches it after a reboot.
#
# Read-only w.r.t. the sweep: never writes benchmark_results.csv, never touches the sweep
# lock, never signals the sweep process. It only reads the CSV and pushes it to origin/main.
#
# Push safety (never put a truncated CSV on main):
#   Stage the CSV, then validate the EXACT staged blob -- data rows a multiple of 9
#   (whole METRIC_KEYS cells) AND the final row a complete 's_used' row with a full ISO+Z
#   timestamp (the last field; a complete tail proves the whole row is intact), cross-checked
#   against the atomic heartbeat. If not clean: unstage and skip. Commit author = sweep-bot.
#
# ASCII only (PS 5.1 safe).

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $repo) { $repo = 'C:\Users\karti\spatial-foundation-benchmark' }
$results   = Join-Path $repo 'results'
$csv       = Join-Path $results 'benchmark_results.csv'
$log       = Join-Path $results 'autocheckpoint.log'
$status    = Join-Path $results 'benchmark.status.json'
$sweepLock = Join-Path $results 'benchmark.lock'
$selfLock  = Join-Path $results '.watchdog.lock'
$rowsPerCell   = 9
$cellThreshold = 50
$authorName = 'sweep-bot'
$authorMail = 'noreply@localhost'
$sleepSec   = 300

function LogLine($m) {
  $ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
  try { Add-Content -Path $log -Value ("{0} {1}" -f $ts, $m) -Encoding ascii } catch {}
}

# --- single-instance guard ---
if (Test-Path $selfLock) {
  $other = (Get-Content $selfLock -Raw -ErrorAction SilentlyContinue)
  if ($other) { $other = $other.Trim() }
  if ($other -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
    LogLine "another watchdog (pid $other) already running; exiting."
    return
  }
}
Set-Content -Path $selfLock -Value $PID -Encoding ascii
LogLine ("watchdog start (repo/durable): pid=$PID author=$authorName repo=$repo")

function SweepPid {
  if (-not (Test-Path $sweepLock)) { return -1 }
  try { return [int]((Get-Content $sweepLock -Raw).Trim()) } catch { return -1 }
}
function SweepAlive {
  $p = SweepPid
  if ($p -le 0) { return $false }
  return ($null -ne (Get-Process -Id $p -ErrorAction SilentlyContinue))
}
function CsvCells {
  if (-not (Test-Path $csv)) { return 0 }
  [int](((Get-Content $csv | Measure-Object -Line).Lines - 1) / $rowsPerCell)
}
function HeartbeatCells {
  try { return [int]((Get-Content $status -Raw | ConvertFrom-Json).cells_done) } catch { return -1 }
}

# Stage, validate the staged blob, commit+push only if it is a clean cell boundary.
# Returns committed cell count on success, -1 if skipped.
function TryCheckpoint($tag) {
  & git -C $repo add results/benchmark_results.csv 2>&1 | Out-Null
  $staged = & git -C $repo show ":results/benchmark_results.csv" 2>$null
  if (-not $staged -or $staged.Count -lt 2) {
    & git -C $repo reset -q -- results/benchmark_results.csv 2>&1 | Out-Null
    LogLine "skip: staged CSV unreadable"; return -1
  }
  $objs = $staged | ConvertFrom-Csv
  $dataRows   = $objs.Count
  $onBoundary = ($dataRows % $rowsPerCell) -eq 0
  $lastObj    = $objs[$objs.Count - 1]
  # timestamp is the LAST field; a full ISO+Z at row end proves the whole row is intact.
  $lastComplete = ($lastObj.metric -eq 's_used') -and ($lastObj.timestamp -match '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
  $csvCells = [int]($dataRows / $rowsPerCell)
  $hbCells  = HeartbeatCells
  $hbOk = ($hbCells -lt 0) -or ($csvCells -eq $hbCells) -or ($csvCells -eq ($hbCells + 1))

  if ($onBoundary -and $lastComplete -and $hbOk) {
    & git -C $repo -c user.name="$authorName" -c user.email="$authorMail" commit -m "${tag}: $csvCells cells" 2>&1 | Out-Null
    & git -C $repo push origin main 2>&1 | Out-Null
    LogLine ("pushed {0}: {1} cells (rows={2} hb={3} exit={4})" -f $tag, $csvCells, $dataRows, $hbCells, $LASTEXITCODE)
    return $csvCells
  } else {
    & git -C $repo reset -q -- results/benchmark_results.csv 2>&1 | Out-Null
    LogLine ("skip: not clean boundary (rows={0} mult9={1} lastMetric={2} hbCells={3} csvCells={4})" -f `
      $dataRows, $onBoundary, $lastObj.metric, $hbCells, $csvCells)
    return -1
  }
}

$headText  = & git -C $repo show HEAD:results/benchmark_results.csv 2>$null
$lastCells = [int]((($headText | Measure-Object -Line).Lines - 1) / $rowsPerCell)
if ($lastCells -lt 0) { $lastCells = 0 }
LogLine ("baseline committed cells=$lastCells")

$finalDone = $false
while ($true) {
  $alive = SweepAlive
  $cur   = CsvCells
  $staleMin = -1
  if (Test-Path $status) { $staleMin = [int]((Get-Date) - (Get-Item $status).LastWriteTime).TotalMinutes }

  if (($cur - $lastCells) -ge $cellThreshold) {
    $done = TryCheckpoint 'auto-checkpoint'
    if ($done -ge 0) { $lastCells = $done }
  } else {
    LogLine ("sweepAlive={0} sweepPid={1} cells={2} (+{3} since ckpt) stale_min={4}" -f $alive, (SweepPid), $cur, ($cur - $lastCells), $staleMin)
  }

  if ($alive -and $staleMin -ge 20) {
    LogLine ("WARN: status.json stale {0} min while sweep pid alive" -f $staleMin)
  }

  if (-not $alive) {
    if (-not $finalDone) {
      $done = TryCheckpoint 'auto-checkpoint(final)'
      if ($done -ge 0) { $lastCells = $done }
      LogLine "sweep not running; did final checkpoint; will keep watching for a restart."
      $finalDone = $true
    }
  } else { $finalDone = $false }

  Start-Sleep -Seconds $sleepSec
}
