param(
    [switch]$StartNow,

    [ValidateRange(5, 180)]
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$cycleScript = Join-Path $PSScriptRoot "run_paper_challenger_cycle.ps1"
$taskName = "WeatherAlphaProspectivePaperChallengerCycle"
$powerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$userId = [Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path -LiteralPath $cycleScript)) {
    throw "Paper challenger cycle script not found: $cycleScript"
}

$arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$cycleScript`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $projectRoot
$periodicTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(3) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -Priority 6 `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger @($periodicTrigger, $logonTrigger) `
    -Principal $principal `
    -Settings $settings `
    -Description "Scores the forecast-anchored weather calibration in parallel; never sends live orders." `
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
    LiveOrdersEnabled = $false
    Script = $cycleScript
}
