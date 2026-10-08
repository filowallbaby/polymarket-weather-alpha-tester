param(
    [ValidatePattern("^([01]\d|2[0-3]):[0-5]\d$")]
    [string]$DailyAt = "13:30",

    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cycleScript = Join-Path $PSScriptRoot "run_resolution_ledger_update.ps1"
$taskName = "WeatherAlphaDailyResolutionLedger"
$powerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$userId = [Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $cycleScript)) {
    throw "Resolution ledger update script not found: $cycleScript"
}

$runAt = [datetime]::Today.Add([timespan]::Parse($DailyAt))
$arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$cycleScript`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $runAt
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -Priority 6 `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description (
        "Refreshes official Polymarket weather resolutions and point-in-time " +
        "model lead diagnostics. Never sends live orders."
    ) `
    -Force | Out-Null

if ($StartNow) {
    Start-ScheduledTask -TaskName $taskName
}

$task = Get-ScheduledTask -TaskName $taskName
[pscustomobject]@{
    TaskName = $task.TaskName
    State = $task.State
    User = $userId
    DailyAtLocal = $DailyAt
    StartedNow = [bool]$StartNow
    LiveOrdersEnabled = $false
    Script = $cycleScript
}
