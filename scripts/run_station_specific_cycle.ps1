$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"
$logsDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logsDirectory "station_specific_paper_cycle.log"

if (-not (Test-Path -LiteralPath $weatherAlpha)) {
    throw "weather-alpha executable not found: $weatherAlpha"
}
New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null

try {
    $cycleOutput = & $weatherAlpha paper-station-strategies-cycle `
    if ($LASTEXITCODE -ne 0) {
        throw "paper-station-strategies-cycle failed with exit code $LASTEXITCODE"
    }
    "[$([DateTimeOffset]::UtcNow.ToString('o'))] cycle completed" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8
    $cycleOutput | Out-File -LiteralPath $logPath -Append -Encoding utf8
}
catch {
    "[$([DateTimeOffset]::UtcNow.ToString('o'))] ERROR: $($_.Exception.Message)" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8
    throw
}
