param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "runs\overnight_v2"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$StageLog = Join-Path $RunRoot "stages.log"

function Write-Stage([string]$Message) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" | Add-Content -Path $StageLog
}

Set-Location $ProjectRoot
Write-Stage "waiting for training PID $TrainingPid"
Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
Write-Stage "training exited; starting held-out evaluation"

& python -X utf8 scripts/evaluate.py `
    --train-dir runs/train_after_v2 `
    --out runs/eval_v2 `
    *> (Join-Path $RunRoot "evaluate.log")
$EvalCode = $LASTEXITCODE
Write-Stage "evaluation exited with code $EvalCode"

if ($EvalCode -eq 0) {
    Write-Stage "starting 1,000-fly tournament"
    & python -X utf8 scripts/tournament.py `
        --train-dir runs/train_after_v2 `
        --out runs/tournament_v2 `
        *> (Join-Path $RunRoot "tournament.log")
    Write-Stage "tournament exited with code $LASTEXITCODE"
}
