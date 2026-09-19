param(
    [string]$TrainUntil = "",
    [int]$WaitForPid = 0,
    [switch]$SkipPreEvaluation,
    [switch]$TrainingOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "runs\overnight_day2"
$StageLog = Join-Path $RunRoot "stages.log"
$Python = (Get-Command python -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
Set-Location $ProjectRoot
Set-Content -Path (Join-Path $RunRoot "controller.pid") -Value $PID

function Write-Stage([string]$Message) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Add-Content -Path $StageLog
}

function Invoke-PythonStage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$Optional
    )

    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        $Stdout = Join-Path $RunRoot "$Name-attempt-$Attempt.stdout.log"
        $Stderr = Join-Path $RunRoot "$Name-attempt-$Attempt.stderr.log"
        Write-Stage "$Name attempt $Attempt started"
        $Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
        Write-Stage "$Name attempt $Attempt exited with code $($Process.ExitCode)"
        if ($Process.ExitCode -eq 0) {
            return $true
        }
    }

    if ($Optional) {
        Write-Stage "$Name failed after 3 attempts; continuing because the stage is optional"
        return $false
    }
    throw "$Name failed after 3 attempts"
}

# Keep the display state unchanged while preventing automatic system sleep for the
# lifetime of this controller. Python stages also make their own keep-awake call.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class FlytrisPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
[void][FlytrisPower]::SetThreadExecutionState(2147483649)

try {
    $DeadlineDescription = if ($TrainUntil) { $TrainUntil } else { "none (manual stop)" }
    Write-Stage "Day 2 controller started (PID $PID); training deadline $DeadlineDescription"

    if ($WaitForPid -gt 0) {
        Write-Stage "waiting for existing evaluation PID $WaitForPid"
        Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
        Write-Stage "existing evaluation PID $WaitForPid finished"
    }

    # The offboard identified this as the first missing measurement. A separate output
    # directory prevents the old generation-29 rows from suppressing this fresh run.
    if (-not $SkipPreEvaluation) {
        Invoke-PythonStage -Name "pre-evaluate-gen69" -Optional -Arguments @(
            "-X", "utf8", "scripts\evaluate.py",
            "--train-dir", "runs\train_after_v2",
            "--nobrain-dir", "runs\nobrain_after",
            "--out", "runs\eval_day2_pre",
            "--seeds", "20", "--cap", "150", "--record", "20",
            "--max-batch", "384"
        ) | Out-Null
    }

    $TrainArguments = @(
        "-X", "utf8", "scripts\train.py",
        "--player", "brain",
        "--out", "runs\train_after_v2",
        "--head", "afterstate",
        "--pop", "64", "--elites", "8", "--cap", "150",
        "--gens", "10000",
        "--validate-every", "5", "--max-batch", "384",
        "--init", "runs\init_comparison.npz",
        "--init-std", "0.4", "--smooth", "0.7"
    )
    if ($TrainUntil) {
        $TrainArguments += @("--until", $TrainUntil)
    }
    Invoke-PythonStage -Name "train" -Arguments $TrainArguments | Out-Null

    if ($TrainingOnly) {
        Write-Stage "training-only controller finished"
        return
    }

    Invoke-PythonStage -Name "final-evaluate" -Arguments @(
        "-X", "utf8", "scripts\evaluate.py",
        "--train-dir", "runs\train_after_v2",
        "--nobrain-dir", "runs\nobrain_after",
        "--out", "runs\eval_day2_final",
        "--seeds", "20", "--cap", "150", "--record", "50",
        "--max-batch", "384"
    ) | Out-Null

    Invoke-PythonStage -Name "tournament" -Arguments @(
        "-X", "utf8", "scripts\tournament.py",
        "--train-dir", "runs\train_after_v2",
        "--out", "runs\tournament_day2",
        "--n", "1000", "--cap", "300", "--chunk", "250",
        "--max-batch", "384"
    ) | Out-Null

    Write-Stage "Day 2 controller finished successfully"
}
catch {
    Write-Stage "Day 2 controller stopped: $($_.Exception.Message)"
    throw
}
finally {
    [void][FlytrisPower]::SetThreadExecutionState(2147483648)
}
