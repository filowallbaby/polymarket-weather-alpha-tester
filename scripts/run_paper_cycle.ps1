param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"
$runsDirectory = Join-Path $projectRoot "data\runs"
$logsDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logsDirectory "paper_trading_cycle.log"
$mutexName = "Local\WeatherAlphaPaperCycle"

if (-not (Test-Path -LiteralPath $weatherAlpha)) {
    throw "weather-alpha executable not found: $weatherAlpha"
}
New-Item -ItemType Directory -Force -Path $logsDirectory | Out-Null

$mutex = [Threading.Mutex]::new($false, $mutexName)
$acquired = $false
try {
    $acquired = $mutex.WaitOne(0)
    if (-not $acquired) {
        exit 0
    }
    $runningRun = Get-ChildItem -LiteralPath $runsDirectory -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object {
            $manifestPath = Join-Path $_.FullName "manifest.json"
            if (-not (Test-Path -LiteralPath $manifestPath)) {
                return
            }
            try {
                $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
                if ($manifest.status -eq "running") {
                    $_
                }
            }
            catch {
                return
            }
        } |
        Select-Object -First 1
    if ($null -eq $runningRun) {
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
            "$([DateTimeOffset]::UtcNow.ToString('o')) no running collector run; cycle skipped"
        )
        exit 0
    }

    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "$([DateTimeOffset]::UtcNow.ToString('o')) cycle starting run=$($runningRun.Name)"
    )
    $cycleOutput = & $weatherAlpha paper-cycle --run-id $runningRun.Name --root $projectRoot 2>&1 |
        Out-String
    $cycleExitCode = $LASTEXITCODE
    if (-not [string]::IsNullOrWhiteSpace($cycleOutput)) {
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value $cycleOutput.TrimEnd()
    }
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "$([DateTimeOffset]::UtcNow.ToString('o')) cycle completed run=$($runningRun.Name) " +
        "exit_code=$cycleExitCode"
    )
    exit $cycleExitCode
}
finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
