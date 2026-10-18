# Run prebuilt probes serially. Never build or run other GPU work during a sweep.
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$ReferenceAudio,
    [Parameter(Mandatory)][string]$ReferenceText,
    [Parameter(Mandatory)][string]$Output,
    [string]$Corpus = "tools/tts/acceptance-corpus.txt",
    [string]$Probe = "target/release/examples/voxcpm_candle_probe.exe",
    [ValidateRange(1, 30)][int]$Rounds = 5,
    [switch]$Screen
)
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $Output) { throw "Output already exists: $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null
$names = @("VOXCPM_CUDA_GRAPH", "VOXCPM_DECODER_GRAPH", "VOXCPM_BENCH_ROUNDS", "VOXCPM_BENCH_WARMUP", "VOXCPM_BENCH_PROFILE", "VOXCPM_BENCH_PRECISION")
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
$power = powercfg /getactivescheme
if ($LASTEXITCODE -ne 0 -or "$power" -notmatch "[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}") { throw "Cannot read active power plan" }
$previousPower = $Matches[0]
$runs = @()
$sequence = if ($Screen) { @("baseline", "candidate") } else { @("baseline", "candidate", "candidate", "baseline", "baseline", "candidate", "candidate", "baseline") }
try {
    powercfg /setactive SCHEME_MIN
    if ($LASTEXITCODE -ne 0) { throw "Cannot select high performance power plan" }
    powercfg /getactivescheme | Set-Content "$Output/power-active.txt"
    nvidia-smi -lgc 2700
    if ($LASTEXITCODE -ne 0) { throw "Cannot request GPU clock lock" }
    $env:VOXCPM_BENCH_ROUNDS = "$Rounds"
    $env:VOXCPM_BENCH_WARMUP = "1"
    $env:VOXCPM_CUDA_GRAPH = "1"
    Remove-Item Env:VOXCPM_BENCH_PROFILE -ErrorAction SilentlyContinue
    $env:VOXCPM_BENCH_PRECISION = "q8"
    foreach ($variant in $sequence) {
        $env:VOXCPM_DECODER_GRAPH = if ($variant -eq "candidate") { "1" } else { "0" }
        $directory = Join-Path $Output "$variant-$($runs.Count)"
        nvidia-smi --query-gpu=pstate,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv | Set-Content "$Output/gpu-before-$($runs.Count).csv"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | Set-Content "$Output/processes-$($runs.Count).csv"
        & $Probe $Model cuda $directory $Corpus $ReferenceAudio $ReferenceText > "$Output/run-$($runs.Count).log" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Probe failed: $directory" }
        $summary = Get-Content "$directory/summary.json" -Raw | ConvertFrom-Json
        if ($summary.regressions -ne "passed" -or !$summary.rounds[-1].cuda_graph) { throw "Regressions failed or CFM graph unavailable" }
        if ($summary.rounds[-1].cuda_decoder_graph -ne ($variant -eq "candidate")) {
            throw "Requested decoder graph fell back or was not disabled: $directory"
        }
        $runs += [pscustomobject]@{ index = $runs.Count; variant = $variant; rtf_mean = $summary.measured.rtf_mean; cuda_graph = $summary.rounds[-1].cuda_graph }
        $runs | ConvertTo-Json | Set-Content "$Output/runs.json"
        $runs[-1] | ConvertTo-Json -Compress | Write-Output
    }
    $statistics = @{}
    foreach ($variant in @("baseline", "candidate")) {
        $values = @($runs | Where-Object variant -EQ $variant | ForEach-Object rtf_mean)
        $statistics[$variant] = @{ mean = ($values | Measure-Object -Average).Average; min = ($values | Measure-Object -Minimum).Minimum; max = ($values | Measure-Object -Maximum).Maximum }
    }
    $statistics["improvement"] = 1 - $statistics.candidate.mean / $statistics.baseline.mean
    $statistics["screen_only"] = [bool]$Screen
    $statistics["performance_gate"] = !$Screen -and $statistics.improvement -ge 0.10 -and $statistics.candidate.max -lt $statistics.baseline.min
    $statistics | ConvertTo-Json | Set-Content "$Output/statistics.json"
    $statistics | ConvertTo-Json | Write-Output
}
finally {
    nvidia-smi -rgc
    powercfg /setactive $previousPower
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process") }
}
