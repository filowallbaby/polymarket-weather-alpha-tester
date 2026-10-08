param(
    [ValidateRange(600, 604800)]
    [int]$DurationSeconds = 172800
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"

if (-not (Test-Path -LiteralPath $weatherAlpha)) {
    throw "weather-alpha executable not found: $weatherAlpha"
}

Write-Output "Research cycle starting at $([DateTimeOffset]::UtcNow.ToString('o'))"
Write-Output "Project root: $projectRoot"
Write-Output "Collection duration: $DurationSeconds seconds"

& $weatherAlpha collect --duration $DurationSeconds --root $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Collection failed with exit code $LASTEXITCODE"
}

$latestRun = Get-ChildItem -LiteralPath (Join-Path $projectRoot "data\runs") -Directory |
    Sort-Object Name |
    Select-Object -Last 1
if ($null -eq $latestRun) {
    throw "Collection completed but no run directory was found"
}

$runId = $latestRun.Name
Write-Output "Post-collection pipeline for run $runId"

$postCollectionError = $null
$auditExit = $null
$archiveExit = $null
try {
    & $weatherAlpha audit-run --run-id $runId --root $projectRoot
    $auditExit = $LASTEXITCODE
    Write-Output "audit-run exit code: $auditExit"

    & $weatherAlpha build-dataset --run-id $runId --root $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "build-dataset failed with exit code $LASTEXITCODE"
    }

    & $weatherAlpha discover-alpha --root $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "discover-alpha failed with exit code $LASTEXITCODE"
    }
}
catch {
    $postCollectionError = $_
}
finally {
    & $weatherAlpha archive-run --run-id $runId --root $projectRoot
    $archiveExit = $LASTEXITCODE
    Write-Output "archive-run exit code: $archiveExit"
}

if ($null -ne $postCollectionError) {
    throw $postCollectionError
}
if ($archiveExit -ne 0) {
    throw "archive-run failed with exit code $archiveExit"
}

Write-Output "Research cycle completed at $([DateTimeOffset]::UtcNow.ToString('o'))"
Write-Output "Run: $runId"
Write-Output "Audit exit code: $auditExit"
Write-Output "Archive exit code: $archiveExit"
