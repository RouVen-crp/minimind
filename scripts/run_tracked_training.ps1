param(
    [Parameter(Mandatory = $true)][string]$Stage,
    [Parameter(Mandatory = $true)][string]$TrainingScript,
    [Parameter(Mandatory = $true)][string[]]$TrainingArguments,
    [int]$GpuSampleSeconds = 60
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$trainerDir = Join-Path $repoRoot "trainer"
$logDir = Join-Path $repoRoot "experiments\logs"
$metricsDir = Join-Path $repoRoot "experiments\metrics"
$runtimeDir = Join-Path $repoRoot "experiments\runtime"
$condaExe = "D:\Anaconda3\Scripts\conda.exe"

New-Item -ItemType Directory -Force -Path $logDir, $metricsDir, $runtimeDir | Out-Null

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and $_.CommandLine -like "*$TrainingScript*"
}
if ($active) {
    throw "Training process already exists: $($active.ProcessId -join ', ')"
}

$startedAt = Get-Date
$runId = $startedAt.ToString("yyyyMMdd-HHmmss")
$stdoutPath = Join-Path $logDir "$Stage-$runId.stdout.log"
$stderrPath = Join-Path $logDir "$Stage-$runId.stderr.log"
$gpuPath = Join-Path $metricsDir "$Stage-$runId.gpu.csv"
$runtimePath = Join-Path $runtimeDir "$Stage-$runId.json"
$latestPath = Join-Path $runtimeDir "$Stage-latest.json"

$condaArgs = @("run", "-n", "minimind", "--no-capture-output", "python", "-u", $TrainingScript) + $TrainingArguments
$process = Start-Process -FilePath $condaExe -ArgumentList $condaArgs -WorkingDirectory $trainerDir -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru

$runtime = [ordered]@{
    stage = $Stage
    run_id = $runId
    status = "running"
    pid = $process.Id
    started_at = $startedAt.ToString("o")
    command = "$condaExe $($condaArgs -join ' ')"
    working_directory = $trainerDir
    stdout = $stdoutPath.Substring($repoRoot.Length + 1)
    stderr = $stderrPath.Substring($repoRoot.Length + 1)
    gpu_metrics = $gpuPath.Substring($repoRoot.Length + 1)
}
$runtime | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $runtimePath
$runtime | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $latestPath

"timestamp,index,name,temperature_c,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w" | Set-Content -Encoding UTF8 $gpuPath
while (-not $process.HasExited) {
    $sample = & nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    if ($LASTEXITCODE -eq 0 -and $sample) {
        $sample | Add-Content -Encoding UTF8 $gpuPath
    }
    Start-Sleep -Seconds $GpuSampleSeconds
    $process.Refresh()
}

$endedAt = Get-Date
$runtime.status = if ($process.ExitCode -eq 0) { "completed" } else { "failed" }
$runtime.exit_code = $process.ExitCode
$runtime.ended_at = $endedAt.ToString("o")
$runtime.duration_seconds = [math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
$runtime | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $runtimePath
$runtime | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $latestPath

