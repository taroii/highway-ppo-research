<#
.SYNOPSIS
    Architectures experiment - SAC vs Uniform vs Zooming on racetrack-v0.

.DESCRIPTION
    Runs all three across multiple seeds, then plots.

    Outputs:
      checkpoints/highway/architectures/<arm>_seed<S>.pt
      plots/highway/architectures.png

    This is a different experiment from src/highway/run_scarcity_sweep.py.
    The pipeline tests whether *adaptive zooming* beats *uniform discretization*
    at a single matched action budget; the scarcity sweep tests how that gap
    scales with action budget N.

    Per-run stdout/stderr is tee'd to logs/highway/architectures/<label>.log.
    A failing run is logged and the pipeline continues.

.PARAMETER Seeds
    Array of seeds. Default 42 43 44 45 46.

.PARAMETER SacTimesteps
    SAC training timesteps. Default 150000.

.PARAMETER DqnTimesteps
    DQN training timesteps for uniform + zooming. Default 150000.

.PARAMETER NActions
    Uniform grid size. Default 16.

.PARAMETER MaxDepth
    Zooming max depth (2^MaxDepth = max cubes). Default 4.

.PARAMETER Python
    Python interpreter. Default "python".

.PARAMETER Force
    Re-run every arm even if its checkpoint file already exists.
    Default: skip arms whose output .pt is already on disk.

.EXAMPLE
    .\run_highway_pipeline.ps1

.EXAMPLE
    .\run_highway_pipeline.ps1 -Seeds 42,43 -SacTimesteps 50000 `
        -Python "C:\Users\Polar\miniconda3\envs\highway\python.exe"

.EXAMPLE
    # retrain everything from scratch even if checkpoints exist
    .\run_highway_pipeline.ps1 -Force
#>
[CmdletBinding()]
param(
    [int[]]  $Seeds        = @(42, 43, 44, 45, 46),
    [int]    $SacTimesteps = 150000,
    [int]    $DqnTimesteps = 150000,
    [int]    $NActions     = 16,
    [int]    $MaxDepth     = 4,
    [string] $Python       = "python",
    [switch] $Force
)

# Mirror bash 'set -uo pipefail' as best PS allows: strict variable use,
# but keep going when a child process fails (we check $LASTEXITCODE).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$CkptDir = Join-Path $Root "checkpoints/highway/architectures"
$LogDir  = Join-Path $Root "logs/highway/architectures"
$PlotOut = Join-Path $Root "plots/highway/architectures.png"

New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir  | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PlotOut) | Out-Null

$script:Failed  = New-Object System.Collections.Generic.List[string]
$script:Skipped = New-Object System.Collections.Generic.List[string]
$script:Total   = 0
$script:Ok      = 0

function Get-Stamp { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

function Invoke-Run {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string]   $Label,
        [Parameter(Mandatory)] [string]   $Output,
        [Parameter(Mandatory)] [string]   $Exe,
        [Parameter(Mandatory)] [string[]] $Arguments
    )
    $log = Join-Path $LogDir "$Label.log"
    $script:Total++

    if ((-not $Force) -and (Test-Path -LiteralPath $Output)) {
        [void]$script:Skipped.Add($Label)
        Write-Host ""
        Write-Host "=== [$(Get-Stamp)] SKIP   $Label  (checkpoint exists: $Output) ==="
        Write-Host "    use -Force to retrain"
        return
    }

    Write-Host ""
    Write-Host "=== [$(Get-Stamp)] START  $Label  ==="
    Write-Host "    cmd: $Exe $($Arguments -join ' ')"
    Write-Host "    log: $log"

    # Pipe both streams to the log AND the console. 2>&1 on native exes in PS 5.1
    # wraps stderr lines in ErrorRecords (cosmetic), but $LASTEXITCODE is reliable,
    # so we trust that for the success check.
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        $script:Ok++
        Write-Host "=== [$(Get-Stamp)] OK     $Label  ==="
    } else {
        [void]$script:Failed.Add($Label)
        Write-Host "=== [$(Get-Stamp)] FAIL   $Label (exit $code, continuing) ===" -ForegroundColor Red
    }
}

Write-Host "highway architectures pipeline"
Write-Host "  seeds:          $($Seeds -join ' ')"
Write-Host "  sac timesteps:  $SacTimesteps"
Write-Host "  dqn timesteps:  $DqnTimesteps"
Write-Host "  uniform N:      $NActions"
Write-Host "  zooming depth:  $MaxDepth"
Write-Host "  python:         $Python"
Write-Host "  ckpts -> $CkptDir"
Write-Host "  plot  -> $PlotOut"

foreach ($seed in $Seeds) {
    $sacOut     = "$CkptDir/sac_seed$seed.pt"
    $uniformOut = "$CkptDir/uniform_n${NActions}_seed${seed}.pt"
    $zoomingOut = "$CkptDir/zooming_d${MaxDepth}_seed${seed}.pt"

    Invoke-Run -Label "sac_seed$seed" -Output $sacOut -Exe $Python -Arguments @(
        "src/highway/run_sac.py",
        "--seed",            "$seed",
        "--total_timesteps", "$SacTimesteps",
        "--output",          $sacOut
    )

    Invoke-Run -Label "uniform_n${NActions}_seed${seed}" -Output $uniformOut -Exe $Python -Arguments @(
        "src/highway/run_uniform.py",
        "--seed",            "$seed",
        "--n_actions",       "$NActions",
        "--total_timesteps", "$DqnTimesteps",
        "--output",          $uniformOut
    )

    Invoke-Run -Label "zooming_d${MaxDepth}_seed${seed}" -Output $zoomingOut -Exe $Python -Arguments @(
        "src/highway/run_zooming.py",
        "--seed",            "$seed",
        "--max_depth",       "$MaxDepth",
        "--total_timesteps", "$DqnTimesteps",
        "--output",          $zoomingOut
    )
}

Write-Host ""
Write-Host "=== runs complete: $($script:Ok) ran, $($script:Skipped.Count) skipped, $($script:Failed.Count) failed (of $($script:Total) total) ==="
if ($script:Skipped.Count -gt 0) {
    Write-Host "skipped (checkpoint already existed):"
    foreach ($s in $script:Skipped) { Write-Host "  $s" }
}
if ($script:Failed.Count -gt 0) {
    Write-Host "failed runs:"
    foreach ($f in $script:Failed) { Write-Host "  $f" }
}

Write-Host ""
Write-Host "=== running compare.py ==="
& $Python "src/highway/compare.py" `
    --checkpoints-dir $CkptDir `
    --output          $PlotOut `
    --title           "Architectures - racetrack-v0"

Write-Host ""
Write-Host "=== pipeline done ==="
