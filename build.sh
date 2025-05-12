#!/usr/bin/env bash
#===============================================================================
#  CRT-STYLE COMMAND LAUNCHER  ⬢  PLA/Django War-Game Simulation Suite
#===============================================================================
#  Every line prefixed with “[CRT]” mirrors Combat-Results-Table battle chatter.
#  Requires: Bash ≥5, Python ≥3.10. All actions are idempotent.
#-------------------------------------------------------------------------------

set -euo pipefail
shopt -s inherit_errexit

#––– GLOBALS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
VENV_DIR=".venv"                     # project-local virtual-env
PYTHON_BIN="${VENV_DIR}/bin/python"  # python inside venv (auto-resolved)

#––– HELPERS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
command_exists() { command -v "$1" &>/dev/null; }

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[CRT] BOOT  | Spawning Python virtual environment…"
    python -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
}

upgrade_pip() {
  echo "[CRT] PIP   | Upgrading pip → latest stable…"
  "${PYTHON_BIN}" -m pip install --quiet --upgrade pip
}

install_deps() {
  echo "[CRT] DEPS  | Installing simulation + Django requirements…"
  "${PYTHON_BIN}" -m pip install --quiet --upgrade \
      "django~=5.0" \
      numpy \
      packaging \
      cupy-cuda12x \
      plotly \
      websockets \
      d3graph \
      flask \
      numba || true   # cupy optional; numpy fallback ok
}

migrate_db() {
  echo "[CRT] MIGRT | Aligning database schema (no-input)…"
  "${PYTHON_BIN}" manage.py migrate --noinput
}

#––– SCENARIO RUNNERS –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
run_admin() {
  echo "[CRT] ADMIN | Dispatching Django dev-server ⇢ http://127.0.0.1:8000/"
  "${PYTHON_BIN}" manage.py runserver 0.0.0.0:8000
}

sim_live() {
  echo "[CRT] SIM-L | Taiwan 1-v-1 live visual feed firing now…"
  RUN_LIVE_VIS=1 "${PYTHON_BIN}" manage.py
}

sim_mass() {
  local port="${1:-5000}"
  echo "[CRT] SIM-M | 100-v-100 engagement; React/D3 UI on :${port}…"
  RUN_100v100=1 "${PYTHON_BIN}" manage.py
}

sim_async_gpu() {
  echo "[CRT] SIM-G | GPU batch async websocket demo on :8888…"
  RUN_ASYNC_GPU_DEMO=1 "${PYTHON_BIN}" manage.py
}

sim_carrier() {
  echo "[CRT] SIM-C | DF-21D vs Carrier; exporting Plotly HTML…"
  RUN_PLA_VS_CARRIER=1 "${PYTHON_BIN}" manage.py
}

sound_announce() {
  echo "[CRT] AUDIO | Pete Hegseth ↔ Gen Zhang; sound server on :5050…"
  RUN_SOUND_ANNOUNCEMENT=1 "${PYTHON_BIN}" manage.py
}

#––– ENTRYPOINT ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
usage() {
cat <<EOF
[CRT] USAGE | ./wargame_cmd.sh <command> [args]

  bootstrap-env        → make venv + install modern deps
  migrate-db           → apply Django migrations (sqlite default)
  run-admin            → start Django dev-server
  sim-live             → 1 v 1 live visualisation (SSE)
  sim-mass  [port]     → 100 v 100 modern UI (default port 5000)
  sim-async-gpu        → 1000-entity GPU async websocket demo
  sim-carrier          → DF-21D vs Carrier Plotly export
  sound-announce       → serve commencement-of-wargames audio page
EOF
exit 1
}

main() {
  local cmd="${1:-}"; shift || true

  case "${cmd}" in
    bootstrap-env)
      ensure_venv
      upgrade_pip
      install_deps
      ;;
    migrate-db)
      ensure_venv
      migrate_db
      ;;
    run-admin)
      ensure_venv
      run_admin
      ;;
    sim-live)
      ensure_venv
      sim_live
      ;;
    sim-mass)
      ensure_venv
      sim_mass "$@"
      ;;
    sim-async-gpu)
      ensure_venv
      sim_async_gpu
      ;;
    sim-carrier)
      ensure_venv
      sim_carrier
      ;;
    sound-announce)
      ensure_venv
      sound_announce
      ;;
    *)
      usage
      ;;
  esac
}

# script invocation
main "$@"
