$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"
$logsDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logsDirectory "resolution_ledger_update.log"

if (-not (Test-Path -LiteralPath $weatherAlpha)) {
    throw "weather-alpha executable not found: $weatherAlpha"
}
New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null

$startedAt = [DateTimeOffset]::UtcNow.ToString("o")
"[$startedAt] Resolution ledger update starting" |
    Out-File -LiteralPath $logPath -Append -Encoding utf8

try {
    $cycleOutput = & $weatherAlpha backfill-resolutions --root $projectRoot --refresh 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "backfill-resolutions failed with exit code $LASTEXITCODE"
    }
    $cycleOutput | Out-File -LiteralPath $logPath -Append -Encoding utf8
    $cycleOutput | Write-Output
    "[$([DateTimeOffset]::UtcNow.ToString('o'))] Resolution ledger update completed" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8
}
catch {
    "[$([DateTimeOffset]::UtcNow.ToString('o'))] ERROR: $($_.Exception.Message)" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8
    throw
}
