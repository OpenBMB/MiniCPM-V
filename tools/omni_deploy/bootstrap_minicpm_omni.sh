#!/usr/bin/env bash
set -Eeuo pipefail

# MiniCPM-o one-click bootstrap script
# - single-file deploy helper
# - source switch: cn/global
# - optional sudo password reuse
# - prepare dependencies, docker, models, images, services

#######################################
# Defaults
#######################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$REPO_ROOT"
OMNI_DIR="$WORKSPACE/omni_docker"
COMPOSE_BASE="$OMNI_DIR/docker-compose.yml"
COMPOSE_ACTIVE="$OMNI_DIR/docker-compose.active.yml"
MODEL_DIR="$WORKSPACE/models/gguf"

SOURCE=""
OPEN_BROWSER=1
SKIP_MODEL=0
FORCE_DOWNLOAD=0
ASK_SUDO=1

# If omni_docker directory is absent, script can download a bundle zip.
BUNDLE_FILE_ID="1i7HrGBZE3E-6lsrHjQgaEQK0Qxdi6tSN"
BUNDLE_URL=""
BUNDLE_ZIP="$WORKSPACE/omni_docker.zip"

SUDO_ENABLED=0
SUDO_PASSWORD=""
SUDO_KEEPALIVE_PID=""
SHOW_FAIL_SUMMARY=0

TOTAL_STEPS=12
CURRENT_STEP=0

#######################################
# Colors
#######################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()  { echo -e "${RED}[ERR ]${NC} $*"; }

draw_progress() {
  local current="$1"
  local total="$2"
  local message="$3"
  local width=36
  local done=$(( current * width / total ))
  local left=$(( width - done ))
  local p=$(( current * 100 / total ))
  printf "\r${BOLD}Progress ["
  printf "%0.s#" $(seq 1 "$done")
  printf "%0.s-" $(seq 1 "$left")
  printf "] %3d%%${NC} %s\n" "$p" "$message"
}

progress_step() {
  local message="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  draw_progress "$CURRENT_STEP" "$TOTAL_STEPS" "$message"
}

print_banner() {
  clear || true
  cat <<'BANNER'
=========================================================
   MiniCPM-o One-Click Bootstrap (Single File)
=========================================================
BANNER
}

usage() {
  cat <<'EOF_USAGE'
Usage:
  ./bootstrap_minicpm_omni.sh [options]

Options:
  --source <cn|global>      Download source
  --workspace <path>        Runtime workspace (default: repo root)
  --omni-dir <path>         omni_docker dir (default: <workspace>/omni_docker)
  --bundle-url <url>        Direct URL of omni_docker.zip
  --bundle-file-id <id>     Google Drive file id for bundle download
  --ask-sudo                Ask sudo password once and reuse (default on)
  --no-ask-sudo             Disable sudo auto-auth flow
  --skip-model              Skip model download
  --force-download          Force redownload of model files
  --no-browser              Do not auto-open browser on success
  -h, --help                Show help

Notes:
  1) sudo password is used locally only, in memory only.
  2) password is never written to disk and never uploaded.
  3) this script prefers downloading into the current workspace.
EOF_USAGE
}

cleanup() {
  local code="$1"

  if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
  SUDO_PASSWORD=""

  if [[ "$code" -eq 0 || "$SHOW_FAIL_SUMMARY" -ne 1 ]]; then
    return
  fi

  echo
  log_err "Script failed (exit code=$code)"
  if [[ -f "$COMPOSE_ACTIVE" ]]; then
    log_warn "Generated compose file: $COMPOSE_ACTIVE"
  fi
  log_warn "Check docker compose logs for details"
}
trap 'cleanup $?' EXIT

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --source)
        if [[ $# -lt 2 || -z "${2:-}" || "${2:0:1}" == "-" ]]; then
          log_err "--source requires: cn or global"
          exit 1
        fi
        SOURCE="$2"
        shift 2
        ;;
      --workspace)
        WORKSPACE="$2"
        shift 2
        ;;
      --omni-dir)
        OMNI_DIR="$2"
        shift 2
        ;;
      --bundle-url)
        BUNDLE_URL="$2"
        shift 2
        ;;
      --bundle-file-id)
        BUNDLE_FILE_ID="$2"
        shift 2
        ;;
      --ask-sudo)
        ASK_SUDO=1
        shift
        ;;
      --no-ask-sudo)
        ASK_SUDO=0
        shift
        ;;
      --skip-model)
        SKIP_MODEL=1
        shift
        ;;
      --force-download)
        FORCE_DOWNLOAD=1
        shift
        ;;
      --no-browser)
        OPEN_BROWSER=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log_err "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done

  if [[ -n "$SOURCE" && "$SOURCE" != "cn" && "$SOURCE" != "global" ]]; then
    log_err "--source only supports cn/global"
    exit 1
  fi

  COMPOSE_BASE="$OMNI_DIR/docker-compose.yml"
  COMPOSE_ACTIVE="$OMNI_DIR/docker-compose.active.yml"
  MODEL_DIR="$WORKSPACE/models/gguf"
  BUNDLE_ZIP="$WORKSPACE/omni_docker.zip"
}

select_source_if_needed() {
  if [[ -n "$SOURCE" ]]; then
    return
  fi

  echo
  echo "Select source:"
  echo "  1) China mirror (cn)"
  echo "  2) Global source (global)"
  read -r -p "Input [1/2, default 1]: " source_choice
  case "${source_choice:-1}" in
    1) SOURCE="cn" ;;
    2) SOURCE="global" ;;
    *) SOURCE="cn" ;;
  esac
}

setup_sudo_if_needed() {
  if ! command -v sudo >/dev/null 2>&1; then
    log_warn "sudo not found, skip sudo pre-auth"
    return
  fi

  if [[ "$ASK_SUDO" -ne 1 ]]; then
    return
  fi

  if [[ ! -t 0 ]]; then
    log_warn "non-interactive terminal, skip sudo pre-auth"
    return
  fi

  echo
  echo -e "${BOLD}Security Notice${NC}"
  echo "  - sudo password is used locally only"
  echo "  - stored in process memory only"
  echo "  - never written to disk, never uploaded"

  local attempts=0
  while [[ "$attempts" -lt 3 ]]; do
    read -r -s -p "Enter sudo password: " SUDO_PASSWORD
    echo
    attempts=$((attempts + 1))

    if printf '%s\n' "$SUDO_PASSWORD" | sudo -S -k -p '' true >/dev/null 2>&1; then
      SUDO_ENABLED=1
      log_ok "sudo auth enabled"
      start_sudo_keepalive
      return
    fi

    log_warn "sudo verify failed ($attempts/3)"
  done

  log_err "sudo verify failed too many times"
  exit 1
}

start_sudo_keepalive() {
  if [[ "$SUDO_ENABLED" -ne 1 ]]; then
    return
  fi

  (
    while true; do
      printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' -v >/dev/null 2>&1 || true
      sleep 45
    done
  ) &
  SUDO_KEEPALIVE_PID="$!"
}

run_sudo() {
  if [[ "$SUDO_ENABLED" -eq 1 ]]; then
    printf '%s\n' "$SUDO_PASSWORD" | sudo -S -p '' "$@"
  else
    if sudo -n true >/dev/null 2>&1; then
      sudo "$@"
    else
      log_err "sudo needed but auto-auth is off; rerun with --ask-sudo"
      exit 1
    fi
  fi
}

install_base_dependencies() {
  local packages=()

  add_pkg() {
    local cmd="$1"
    local pkg="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
      packages+=("$pkg")
    fi
  }

  add_pkg curl curl
  add_pkg sed sed
  add_pkg awk gawk
  add_pkg grep grep
  add_pkg tar tar
  add_pkg unzip unzip
  add_pkg file file
  add_pkg ss iproute2

  if command -v rg >/dev/null 2>&1; then
    :
  else
    packages+=("ripgrep")
  fi

  if [[ "${#packages[@]}" -eq 0 ]]; then
    log_ok "base dependencies ready"
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    run_sudo apt-get update -y
    run_sudo apt-get install -y "${packages[@]}"
    log_ok "base dependencies installed"
  else
    log_err "unsupported package manager; install manually: ${packages[*]}"
    exit 1
  fi
}

install_docker_if_needed() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log_ok "docker and compose are installed"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log_err "docker auto install supports apt-get only"
    exit 1
  fi

  run_sudo apt-get update -y
  run_sudo apt-get install -y ca-certificates curl gnupg lsb-release
  run_sudo apt-get install -y docker.io docker-compose-v2 || run_sudo apt-get install -y docker.io docker-compose-plugin
  log_ok "docker installed"
}

ensure_docker_running() {
  if docker info >/dev/null 2>&1; then
    log_ok "docker daemon ready"
    return
  fi

  run_sudo systemctl enable --now docker >/dev/null 2>&1 || run_sudo service docker start >/dev/null 2>&1 || true
  if docker info >/dev/null 2>&1; then
    log_ok "docker daemon started"
    return
  fi

  log_warn "docker daemon not directly accessible; sudo mode will be used"
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    run_sudo docker "$@"
  fi
}

docker_compose() {
  if docker info >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_ACTIVE" "$@"
  else
    run_sudo docker compose -f "$COMPOSE_ACTIVE" "$@"
  fi
}

download_gdrive_file() {
  local file_id="$1"
  local output_path="$2"
  local cookie_file
  cookie_file="$(mktemp)"
  trap 'rm -f "$cookie_file"' RETURN

  local page
  page="$(curl -s -L -c "$cookie_file" "https://drive.google.com/uc?export=download&id=${file_id}")"
  local confirm
  confirm="$(printf '%s' "$page" | grep -oE 'confirm=[0-9A-Za-z_]+' | head -n 1 | cut -d'=' -f2 || true)"
  local uuid
  uuid="$(printf '%s' "$page" | sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' | head -n 1)"

  if [[ -n "$uuid" ]]; then
    local confirm_value="${confirm:-t}"
    curl -fL -b "$cookie_file" \
      "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=${confirm_value}&uuid=${uuid}" \
      -o "$output_path"
  elif [[ -n "$confirm" ]]; then
    curl -fL -b "$cookie_file" \
      "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" \
      -o "$output_path"
  else
    curl -fL -b "$cookie_file" \
      "https://drive.google.com/uc?export=download&id=${file_id}" \
      -o "$output_path"
  fi
}

prepare_omni_bundle() {
  mkdir -p "$WORKSPACE"

  if [[ -d "$OMNI_DIR" && -f "$COMPOSE_BASE" ]]; then
    log_ok "using existing omni_docker: $OMNI_DIR"
    return
  fi

  if [[ ! -f "$BUNDLE_ZIP" ]]; then
    if [[ -n "$BUNDLE_URL" ]]; then
      log_info "downloading bundle from URL"
      curl -fL --retry 3 --retry-delay 2 --retry-all-errors -o "$BUNDLE_ZIP" "$BUNDLE_URL"
    else
      log_info "downloading bundle from Google Drive"
      download_gdrive_file "$BUNDLE_FILE_ID" "$BUNDLE_ZIP"
    fi
  else
    log_ok "found local bundle zip: $BUNDLE_ZIP"
  fi

  if [[ ! -s "$BUNDLE_ZIP" ]]; then
    log_err "bundle zip is empty: $BUNDLE_ZIP"
    exit 1
  fi

  local mime
  mime="$(file --mime-type -b "$BUNDLE_ZIP" || true)"
  if [[ "$mime" == "text/html" || "$mime" == "text/plain" ]]; then
    log_err "bundle download appears invalid mime=$mime"
    exit 1
  fi

  unzip -o "$BUNDLE_ZIP" -d "$WORKSPACE" >/dev/null

  if [[ ! -d "$OMNI_DIR" || ! -f "$COMPOSE_BASE" ]]; then
    log_err "omni_docker not found after unzip"
    exit 1
  fi

  log_ok "omni bundle ready"
}

prepare_compose_by_source() {
  cp "$COMPOSE_BASE" "$COMPOSE_ACTIVE"

  if [[ "$SOURCE" == "global" ]]; then
    sed -i 's#docker\.m\.daocloud\.io/##g' "$COMPOSE_ACTIVE"
  fi

  log_ok "active compose prepared: $COMPOSE_ACTIVE"
}

load_local_tar_images() {
  local frontend_tar="$OMNI_DIR/o45-frontend.tar"
  local backend_tar="$OMNI_DIR/omini_backend_code/omni_backend.tar"

  if [[ -f "$frontend_tar" ]]; then
    log_info "loading frontend tar"
    docker_cmd load -i "$frontend_tar" >/dev/null
    log_ok "frontend image loaded"
  fi

  if [[ -f "$backend_tar" ]]; then
    log_info "loading backend tar"
    docker_cmd load -i "$backend_tar" >/dev/null
    log_ok "backend image loaded"
  fi
}

pull_service_images() {
  docker_compose pull --ignore-pull-failures || true
  log_ok "compose image pull finished"
}

download_file_with_resume() {
  local url="$1"
  local dest="$2"
  local part="${dest}.part"

  mkdir -p "$(dirname "$dest")"

  if [[ "$FORCE_DOWNLOAD" -eq 1 && -f "$dest" ]]; then
    rm -f "$dest"
  fi

  if [[ -s "$dest" ]]; then
    log_ok "exists, skip: $dest"
    return 0
  fi

  curl -fL --retry 4 --retry-delay 2 --retry-all-errors -C - --progress-bar "$url" -o "$part"
  mv "$part" "$dest"
}

download_models_if_needed() {
  if [[ "$SKIP_MODEL" -eq 1 ]]; then
    log_warn "skip model download by argument"
    return
  fi

  local -a model_files=(
    "MiniCPM-o-4_5-Q4_K_M.gguf"
    "audio/MiniCPM-o-4_5-audio-F16.gguf"
    "vision/MiniCPM-o-4_5-vision-F16.gguf"
    "tts/MiniCPM-o-4_5-projector-F16.gguf"
    "tts/MiniCPM-o-4_5-tts-F16.gguf"
    "token2wav-gguf/prompt_cache.gguf"
    "token2wav-gguf/flow_extra.gguf"
    "token2wav-gguf/encoder.gguf"
    "token2wav-gguf/flow_matching.gguf"
    "token2wav-gguf/hifigan2.gguf"
  )

  local -a bases=()
  if [[ "$SOURCE" == "cn" ]]; then
    bases=(
      "https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master"
      "https://hf-mirror.com/openbmb/MiniCPM-o-4_5-gguf/resolve/main"
      "https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/resolve/main"
    )
  else
    bases=(
      "https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/resolve/main"
      "https://hf-mirror.com/openbmb/MiniCPM-o-4_5-gguf/resolve/main"
      "https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master"
    )
  fi

  local rel dest url ok base
  for rel in "${model_files[@]}"; do
    dest="$MODEL_DIR/$rel"
    ok=0

    for base in "${bases[@]}"; do
      url="$base/$rel"
      if download_file_with_resume "$url" "$dest"; then
        ok=1
        break
      fi
      rm -f "${dest}.part" || true
      log_warn "retrying with next source for $rel"
    done

    if [[ "$ok" -ne 1 ]]; then
      log_err "model download failed: $rel"
      exit 1
    fi
  done

  log_ok "models ready under: $MODEL_DIR"
}

wait_http_ok() {
  local url="$1"
  local timeout_sec="${2:-240}"
  local interval=3
  local elapsed=0

  while [[ "$elapsed" -lt "$timeout_sec" ]]; do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done

  return 1
}

start_services() {
  (cd "$OMNI_DIR" && docker_compose up -d --build)
  log_ok "services started"
}

health_check() {
  local failed=0

  wait_http_ok "http://127.0.0.1:3000" 180 || failed=1
  wait_http_ok "http://127.0.0.1:9060/health" 360 || failed=1
  wait_http_ok "http://127.0.0.1:19060/health" 360 || failed=1

  if ! ss -ltnp | rg -q '(:3000|:8021|:7880|:9060|:9061|:19060)'; then
    failed=1
  fi

  if [[ "$failed" -ne 0 ]]; then
    log_err "health check failed"
    (cd "$OMNI_DIR" && docker_compose ps) || true
    (cd "$OMNI_DIR" && docker_compose logs --tail 120) || true
    exit 1
  fi

  log_ok "health check passed"
}

open_browser_if_needed() {
  if [[ "$OPEN_BROWSER" -ne 1 ]]; then
    return
  fi

  local url="http://127.0.0.1:3000"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  fi

  log_ok "browser open attempted: $url"
}

main() {
  parse_args "$@"
  SHOW_FAIL_SUMMARY=1

  print_banner

  progress_step "Initialize"
  select_source_if_needed
  log_info "source = $SOURCE"
  log_info "workspace = $WORKSPACE"

  progress_step "Prepare sudo"
  setup_sudo_if_needed

  progress_step "Prepare dependencies"
  install_base_dependencies

  progress_step "Install Docker if needed"
  install_docker_if_needed

  progress_step "Ensure Docker daemon"
  ensure_docker_running

  progress_step "Prepare omni bundle"
  prepare_omni_bundle

  progress_step "Prepare compose for source"
  prepare_compose_by_source

  progress_step "Prepare images"
  load_local_tar_images
  pull_service_images

  progress_step "Prepare models"
  download_models_if_needed

  progress_step "Start services"
  start_services

  progress_step "Health check"
  health_check

  progress_step "Open browser"
  open_browser_if_needed

  echo
  log_ok "All done"
  echo "----------------------------------------"
  echo "Frontend:   http://127.0.0.1:3000"
  echo "Backend:    http://127.0.0.1:8021"
  echo "LiveKit:    http://127.0.0.1:7880"
  echo "Inference:  http://127.0.0.1:9060"
  echo "llama:      http://127.0.0.1:19060"
  echo "----------------------------------------"
}

main "$@"
