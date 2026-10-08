param(
    [ValidateRange(5, 3600)]
    [int]$RestartDelaySeconds = 30,

    [ValidateRange(512, 262144)]
    [int]$HardMemoryLimitMB = 6144,

    [ValidateRange(1, 60)]
    [int]$WatchdogPollSeconds = 5,

    [ValidateRange(10, 600)]
    [int]$GracefulStopSeconds = 90,

    [ValidateRange(10, 10080)]
    [int]$HardRuntimeLimitMinutes = 480
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$weatherAlpha = Join-Path $projectRoot ".venv\Scripts\weather-alpha.exe"
$runsDirectory = Join-Path $projectRoot "data\runs"
$logsDirectory = Join-Path $projectRoot "logs"
$stopMarker = Join-Path $projectRoot "data\collector_supervisor.stop"
$logPath = Join-Path $logsDirectory "collector_supervisor.log"
$mutexName = "Local\WeatherAlphaCollectorSupervisor"
$hardMemoryLimitBytes = [int64]$HardMemoryLimitMB * 1MB

if (-not (Test-Path -LiteralPath $weatherAlpha)) {
    throw "weather-alpha executable not found: $weatherAlpha"
}
New-Item -ItemType Directory -Force -Path $runsDirectory, $logsDirectory | Out-Null

function Write-SupervisorLog {
    param([string]$Message)
    $line = "$([DateTimeOffset]::UtcNow.ToString('o')) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
}

function Get-FrozenExperimentCollectionDeadline {
    $activePath = Join-Path $projectRoot `
        "reports\paper_v6_residual_experiment\active_experiment.json"
    if (-not (Test-Path -LiteralPath $activePath)) {
        return $null
    }
    try {
        $active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
        $protocolPath = Join-Path $active.deployment_directory "protocol.json"
        $protocol = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
        return [DateTimeOffset]::Parse([string]$protocol.collection_end_utc)
    }
    catch {
        Write-SupervisorLog (
            "experiment deadline unavailable path=$activePath " +
            "error=$($_.Exception.Message)"
        )
        return $null
    }
}

function Set-ExperimentStopMarker {
    param([DateTimeOffset]$Deadline)
    @{
        requested_at = [DateTimeOffset]::UtcNow.ToString("o")
        requested_by = "collector_supervisor"
        reason = "preregistered_experiment_collection_window_complete"
        collection_deadline_utc = $Deadline.ToString("o")
    } |
        ConvertTo-Json |
        Set-Content -LiteralPath $stopMarker -Encoding utf8
}

function Get-RunningManifest {
    Get-ChildItem -LiteralPath $runsDirectory -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object {
            $manifestPath = Join-Path $_.FullName "manifest.json"
            if (-not (Test-Path -LiteralPath $manifestPath)) {
                return
            }
            try {
                $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
                if ($manifest.status -eq "running") {
                    [pscustomobject]@{
                        RunId = $_.Name
                        Pid = [int]$manifest.pid
                        Path = $manifestPath
                        RunDirectory = $_.FullName
                    }
                }
            }
            catch {
                Write-SupervisorLog "invalid manifest ignored path=$manifestPath error=$($_.Exception.Message)"
            }
        } |
        Select-Object -First 1
}

function Test-CollectorProcess {
    param([int]$ProcessId)
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $candidate) {
        return $false
    }
    $commandLine = [string]$candidate.CommandLine
    return (
        $commandLine -like "*weather-alpha*" -and
        $commandLine -like "* collect *" -and
        $commandLine -like "*$projectRoot*"
    )
}

function Get-CollectorProcessTreeUsage {
    param([int]$RootProcessId)

    $processTable = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Select-Object ProcessId, ParentProcessId
    )
    if (-not ($processTable | Where-Object { $_.ProcessId -eq $RootProcessId })) {
        return $null
    }

    $pending = [System.Collections.Generic.Queue[int]]::new()
    $seen = [System.Collections.Generic.HashSet[int]]::new()
    $pending.Enqueue($RootProcessId)
    [int64]$privateBytes = 0
    [int64]$workingSetBytes = 0
    [int]$processCount = 0

    while ($pending.Count -gt 0) {
        $currentId = $pending.Dequeue()
        if (-not $seen.Add($currentId)) {
            continue
        }
        $current = Get-Process -Id $currentId -ErrorAction SilentlyContinue
        if ($null -ne $current) {
            $privateBytes += [int64]$current.PrivateMemorySize64
            $workingSetBytes += [int64]$current.WorkingSet64
            $processCount += 1
        }
        foreach ($child in $processTable | Where-Object { $_.ParentProcessId -eq $currentId }) {
            $pending.Enqueue([int]$child.ProcessId)
        }
    }

    [pscustomobject]@{
        PrivateBytes = $privateBytes
        WorkingSetBytes = $workingSetBytes
        ProcessCount = $processCount
    }
}

function Stop-CollectorProcessTree {
    param([int]$RootProcessId)

    $processTable = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Select-Object ProcessId, ParentProcessId
    )
    $pending = [System.Collections.Generic.Queue[int]]::new()
    $orderedIds = [System.Collections.Generic.List[int]]::new()
    $seen = [System.Collections.Generic.HashSet[int]]::new()
    $pending.Enqueue($RootProcessId)
    while ($pending.Count -gt 0) {
        $currentId = $pending.Dequeue()
        if (-not $seen.Add($currentId)) {
            continue
        }
        $orderedIds.Add($currentId)
        foreach ($child in $processTable | Where-Object { $_.ParentProcessId -eq $currentId }) {
            $pending.Enqueue([int]$child.ProcessId)
        }
    }

    for ($index = $orderedIds.Count - 1; $index -ge 0; $index--) {
        Stop-Process -Id $orderedIds[$index] -Force -ErrorAction SilentlyContinue
    }
}

function Request-CollectorStop {
    param(
        [Parameter(Mandatory = $true)]$RunningManifest,
        [Parameter(Mandatory = $true)][string]$Reason
    )
    $requestPath = Join-Path $RunningManifest.RunDirectory "stop.requested"
    $temporary = "$requestPath.tmp"
    @{
        requested_at = [DateTimeOffset]::UtcNow.ToString("o")
        requested_by = "collector_supervisor"
        reason = $Reason
    } |
        ConvertTo-Json |
        Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $requestPath -Force
    Write-SupervisorLog "graceful stop requested run=$($RunningManifest.RunId) reason=$Reason"
}

function Wait-NewCollectorManifest {
    param([datetime]$StartedAfter)
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(60)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $candidate = Get-RunningManifest
        if (
            $null -ne $candidate -and
            (Test-CollectorProcess -ProcessId $candidate.Pid) -and
            ((Get-Item -LiteralPath $candidate.Path).LastWriteTimeUtc -ge $StartedAfter)
        ) {
            return $candidate
        }
        Start-Sleep -Milliseconds 250
    }
    return $null
}

function Watch-Collector {
    param(
        [Parameter(Mandatory = $true)]$RunningManifest,
        [System.Diagnostics.Process]$Launcher = $null
    )

    $collectorPid = $RunningManifest.Pid
    $watchStarted = [DateTimeOffset]::UtcNow
    $stopRequestedAt = $null
    $watchMode = if ($null -eq $Launcher) { "attached" } else { "launched" }
    Write-SupervisorLog "collector monitored run=$($RunningManifest.RunId) pid=$collectorPid mode=$watchMode"

    while ($true) {
        $process = Get-Process -Id $collectorPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            break
        }
        $treeUsage = Get-CollectorProcessTreeUsage -RootProcessId $collectorPid
        if ($null -eq $treeUsage) {
            break
        }
        $runtimeMinutes = ([DateTimeOffset]::UtcNow - $watchStarted).TotalMinutes

        $experimentDeadline = Get-FrozenExperimentCollectionDeadline

        if (
            $null -eq $stopRequestedAt -and
            $null -ne $experimentDeadline -and
            [DateTimeOffset]::UtcNow -ge $experimentDeadline
        ) {
            Write-SupervisorLog (
                "frozen experiment window complete deadline=$($experimentDeadline.ToString('o'))"
            )
            Set-ExperimentStopMarker -Deadline $experimentDeadline
            Request-CollectorStop `
                -RunningManifest $RunningManifest `
                -Reason "preregistered_experiment_collection_window_complete"
            $stopRequestedAt = [DateTimeOffset]::UtcNow
        }
        elseif ($null -eq $stopRequestedAt -and $treeUsage.PrivateBytes -ge $hardMemoryLimitBytes) {
            Write-SupervisorLog (
                "hard memory guard run=$($RunningManifest.RunId) pid=$collectorPid " +
                "tree_private_mb=$([math]::Round($treeUsage.PrivateBytes / 1MB, 1)) " +
                "tree_working_set_mb=$([math]::Round($treeUsage.WorkingSetBytes / 1MB, 1)) " +
                "tree_processes=$($treeUsage.ProcessCount) limit_mb=$HardMemoryLimitMB"
            )
            Request-CollectorStop -RunningManifest $RunningManifest -Reason "supervisor_hard_tree_memory_limit"
            $stopRequestedAt = [DateTimeOffset]::UtcNow
        }
        elseif ($null -eq $stopRequestedAt -and $runtimeMinutes -ge $HardRuntimeLimitMinutes) {
            Write-SupervisorLog "hard runtime guard run=$($RunningManifest.RunId) minutes=$([math]::Round($runtimeMinutes, 1))"
            Request-CollectorStop -RunningManifest $RunningManifest -Reason "supervisor_hard_runtime_limit"
            $stopRequestedAt = [DateTimeOffset]::UtcNow
        }

        if (
            $null -ne $stopRequestedAt -and
            ([DateTimeOffset]::UtcNow - $stopRequestedAt).TotalSeconds -ge $GracefulStopSeconds
        ) {
            Write-SupervisorLog "collector ignored graceful stop; terminating process tree root_pid=$collectorPid"
            Stop-CollectorProcessTree -RootProcessId $collectorPid
            break
        }
        Start-Sleep -Seconds $WatchdogPollSeconds
    }

    if ($null -eq $Launcher) {
        Write-SupervisorLog "attached collector exited run=$($RunningManifest.RunId) pid=$collectorPid"
        return
    }

    try {
        $finished = $Launcher.WaitForExit(10000)
        if ($finished) {
            $Launcher.WaitForExit()
            $Launcher.Refresh()
        }
    }
    catch {
        Write-SupervisorLog "launcher wait failed pid=$($Launcher.Id) error=$($_.Exception.Message)"
    }
    if (-not $Launcher.HasExited) {
        Stop-Process -Id $Launcher.Id -Force -ErrorAction SilentlyContinue
        try {
            $Launcher.WaitForExit(5000) | Out-Null
            $Launcher.Refresh()
        }
        catch {
            Write-SupervisorLog "launcher termination wait failed pid=$($Launcher.Id) error=$($_.Exception.Message)"
        }
    }
    $exitCode = "unknown"
    if ($Launcher.HasExited) {
        try {
            $exitCode = [int]$Launcher.ExitCode
        }
        catch {
            Write-SupervisorLog "launcher exit code unavailable pid=$($Launcher.Id) error=$($_.Exception.Message)"
        }
    }
    Write-SupervisorLog "collector exited run=$($RunningManifest.RunId) pid=$collectorPid exit_code=$exitCode"
}

function Invoke-Collector {
    $stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $stdoutPath = Join-Path $logsDirectory "collector_$stamp.stdout.log"
    $stderrPath = Join-Path $logsDirectory "collector_$stamp.stderr.log"
    $startedAt = [DateTime]::UtcNow
    $arguments = @("collect", "--no-archive", "--root", "`"$projectRoot`"")
    $launcher = Start-Process `
        -FilePath $weatherAlpha `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    Write-SupervisorLog "collector launcher started pid=$($launcher.Id) stdout=$stdoutPath stderr=$stderrPath"

    $runningManifest = Wait-NewCollectorManifest -StartedAfter $startedAt
    if ($null -eq $runningManifest) {
        if (-not $launcher.HasExited) {
            Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue
        }
        Write-SupervisorLog "collector did not publish an active manifest within 60 seconds"
        return
    }

    Watch-Collector -RunningManifest $runningManifest -Launcher $launcher
}

$mutex = [Threading.Mutex]::new($false, $mutexName)
$ownsMutex = $false
try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        Write-SupervisorLog "another supervisor already owns mutex=$mutexName; exiting"
        exit 0
    }

    Write-SupervisorLog (
        "supervisor started root=$projectRoot restart_delay=$RestartDelaySeconds " +
        "hard_memory_mb=$HardMemoryLimitMB hard_runtime_minutes=$HardRuntimeLimitMinutes"
    )
    while (-not (Test-Path -LiteralPath $stopMarker)) {
        $experimentDeadline = Get-FrozenExperimentCollectionDeadline
        if (
            $null -ne $experimentDeadline -and
            [DateTimeOffset]::UtcNow -ge $experimentDeadline
        ) {
            Set-ExperimentStopMarker -Deadline $experimentDeadline
            Write-SupervisorLog (
                "collector not started because frozen experiment ended " +
                "deadline=$($experimentDeadline.ToString('o'))"
            )
            break
        }
        $runningManifest = Get-RunningManifest
        if ($null -ne $runningManifest) {
            if (Test-CollectorProcess -ProcessId $runningManifest.Pid) {
                Write-SupervisorLog "attaching watchdog to active collector run=$($runningManifest.RunId) pid=$($runningManifest.Pid)"
                Watch-Collector -RunningManifest $runningManifest
                if (-not (Test-Path -LiteralPath $stopMarker)) {
                    Start-Sleep -Seconds $RestartDelaySeconds
                }
                continue
            }
            Write-SupervisorLog "recovering stale run=$($runningManifest.RunId)"
            & $weatherAlpha recover-run --run-id $runningManifest.RunId --root $projectRoot
            Write-SupervisorLog "recovery exit_code=$LASTEXITCODE run=$($runningManifest.RunId)"
        }

        Invoke-Collector
        if (-not (Test-Path -LiteralPath $stopMarker)) {
            Start-Sleep -Seconds $RestartDelaySeconds
        }
    }
    Write-SupervisorLog "stop marker detected; supervisor exiting"
}
catch {
    Write-SupervisorLog "supervisor fatal error=$($_.Exception.Message)"
    throw
}
finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
