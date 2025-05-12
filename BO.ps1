#===============================================================================
#  CRT-STYLE COMMAND LAUNCHER ⬢  PLA/Django War-Game Simulation Suite (PowerShell)
#===============================================================================
#  Requires: PowerShell ≥7, Python ≥3.10. All actions are idempotent & cross-platform.
#-------------------------------------------------------------------------------

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

#--- GLOBALS -------------------------------------------------------------------
$VENV_DIR  = ".venv"            # project-local virtual-env
$IsWin     = $IsWindows
function Get-PythonPath {
    if ($IsWin) { Join-Path $VENV_DIR 'Scripts\python.exe' }
    else        { Join-Path $VENV_DIR 'bin/python' }
}

#--- HELPERS -------------------------------------------------------------------
function Ensure-Venv {
    if (-not (Test-Path $VENV_DIR)) {
        Write-Host "[CRT] BOOT  | Spawning Python virtual environment…" -ForegroundColor Cyan
        python -m venv $VENV_DIR
    }
    $script:PYTHON_BIN = Get-PythonPath
    # prepend venv Scripts to PATH for child processes
    $env:Path = "$(Split-Path $script:PYTHON_BIN -Parent)$([IO.Path]::PathSeparator)$env:Path"
}

function Upgrade-Pip {
    Write-Host "[CRT] PIP   | Upgrading pip → latest stable…" -ForegroundColor Cyan
    & $script:PYTHON_BIN -m pip install --quiet --upgrade pip
}

function Install-Deps {
    Write-Host "[CRT] DEPS  | Installing simulation + Django requirements…" -ForegroundColor Cyan
    try {
        & $script:PYTHON_BIN -m pip install --quiet --upgrade `
            "django~=5.0" numpy packaging cupy-cuda12x plotly websockets d3graph flask numba
    } catch {
        Write-Warning "[CRT] WARN  | GPU-accelerated CuPy unavailable; continuing with NumPy fallback."
        & $script:PYTHON_BIN -m pip install --quiet --upgrade numpy
    }
}

function Migrate-DB {
    Write-Host "[CRT] MIGRT | Aligning database schema (no-input)…" -ForegroundColor Cyan
    & $script:PYTHON_BIN manage.py migrate --noinput
}

#--- SCENARIO RUNNERS ----------------------------------------------------------
function Run-Admin {
    Write-Host "[CRT] ADMIN | Dispatching Django dev-server ⇢ http://127.0.0.1:8000/" -ForegroundColor Cyan
    & $script:PYTHON_BIN manage.py runserver 0.0.0.0:8000
}

function Sim-Live {
    Write-Host "[CRT] SIM-L | Taiwan 1-v-1 live visual feed firing now…" -ForegroundColor Cyan
    $env:RUN_LIVE_VIS = "1"
    & $script:PYTHON_BIN manage.py
}

function Sim-Mass([string]$Port = "5000") {
    Write-Host "[CRT] SIM-M | 100-v-100 engagement; React/D3 UI on :$Port…" -ForegroundColor Cyan
    $env:RUN_100v100 = "1"
    & $script:PYTHON_BIN manage.py
}

function Sim-AsyncGPU {
    Write-Host "[CRT] SIM-G | GPU batch async websocket demo on :8888…" -ForegroundColor Cyan
    $env:RUN_ASYNC_GPU_DEMO = "1"
    & $script:PYTHON_BIN manage.py
}

function Sim-Carrier {
    Write-Host "[CRT] SIM-C | DF-21D vs Carrier; exporting Plotly HTML…" -ForegroundColor Cyan
    $env:RUN_PLA_VS_CARRIER = "1"
    & $script:PYTHON_BIN manage.py
}

function Sound-Announce {
    Write-Host "[CRT] AUDIO | Pete Hegseth ↔ Gen Zhang; sound server on :5050…" -ForegroundColor Cyan
    $env:RUN_SOUND_ANNOUNCEMENT = "1"
    & $script:PYTHON_BIN manage.py
}

#--- ENTRYPOINT AND FULL DJANGO MANAGEMENT SUPPORT ------------------------------
function Show-Usage {
@"
[CRT] USAGE | ./wargame_cmd.ps1 <command> [args]

  bootstrap-env        → make venv + install modern deps
  migrate-db           → apply Django migrations (sqlite default)
  run-admin            → start Django dev-server
  sim-live             → 1 v 1 live visualisation (SSE)
  sim-mass  [port]     → 100 v 100 modern UI (default port 5000)
  sim-async-gpu        → 1000-entity GPU async websocket demo
  sim-carrier          → DF-21D vs Carrier Plotly export
  sound-announce       → serve commencement-of-wargames audio page
  <any manage.py cmd>  → transparently forwarded to Django manage.py
"@ | Write-Host
    exit 1
}

function Main([string[]]$argv) {
    $cmd  = if ($argv.Count) { $argv[0] } else { "" }
    $rest = if ($argv.Count -gt 1) { $argv[1..($argv.Count-1)] } else { @() }

    switch ($cmd) {
        'bootstrap-env'   { Ensure-Venv; Upgrade-Pip; Install-Deps }
        'migrate-db'      { Ensure-Venv; Migrate-DB }
        'run-admin'       { Ensure-Venv; Run-Admin }
        'sim-live'        { Ensure-Venv; Sim-Live }
        'sim-mass'        { Ensure-Venv; Sim-Mass @rest }
        'sim-async-gpu'   { Ensure-Venv; Sim-AsyncGPU }
        'sim-carrier'     { Ensure-Venv; Sim-Carrier }
        'sound-announce'  { Ensure-Venv; Sound-Announce }
        ''                { Show-Usage }
        default {
            # Full Django manage.py pass-through
            Ensure-Venv
            Write-Host "[CRT] DJANG | manage.py $cmd $($rest -join ' ')" -ForegroundColor DarkCyan
            & $script:PYTHON_BIN manage.py $cmd @rest
        }
    }
}

# Script invocation
Main $args
