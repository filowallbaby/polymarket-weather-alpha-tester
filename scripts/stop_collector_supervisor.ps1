param(
    [ValidateRange(1, 300)]
    [int]$WaitSeconds = 120
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"
$stopMarker = Join-Path $projectRoot "data\collector_supervisor.stop"

New-Item -ItemType File -Force -Path $stopMarker | Out-Null
& $weatherAlpha stop-collection --wait-seconds $WaitSeconds --root $projectRoot
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Warning "No collector stopped cleanly; stop-collection exit code: $exitCode"
}
Write-Output "Supervisor stop marker: $stopMarker"
