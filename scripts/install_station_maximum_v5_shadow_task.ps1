param(
    [ValidateRange(5, 180)]
    [int]$IntervalMinutes = 15,

    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cycleScript = Join-Path $PSScriptRoot `
    "run_station_maximum_v5_shadow_cycle.ps1"
$taskName = "WeatherAlphaStationMaximumV5ShadowCycle"
$powerShell = Join-Path $env:SystemRoot `
    "System32\WindowsPowerShell\v1.0\powershell.exe"
$userId = [Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $cycleScript)) {
    throw "Station maximum V5 shadow cycle script not found: $cycleScript"
}

$arguments = (
    "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden " +
    "-File `"$cycleScript`""
)
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments `
    -WorkingDirectory $projectRoot
$firstRun = (Get-Date).AddMinutes(6)
try {
    $referenceInfo = Get-ScheduledTaskInfo `
        -TaskName "WeatherAlphaProspectivePaperChallengerCycle" `
        -ErrorAction Stop
    if ($referenceInfo.NextRunTime -gt (Get-Date)) {
        $firstRun = $referenceInfo.NextRunTime.AddMinutes(5)
    }
}
catch {
}
$periodicTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $firstRun `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -Priority 6 `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger @($periodicTrigger, $logonTrigger) `
    -Principal $principal `
    -Settings $settings `
    -Description (
        "Scores the immutable V5 multiyear meteorological model as a " +
        "probability-only prospective shadow; never creates positions."
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
    IntervalMinutes = $IntervalMinutes
    StartedNow = [bool]$StartNow
    PaperTradesEnabled = $false
    LiveOrdersEnabled = $false
    Script = $cycleScript
}
