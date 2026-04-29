#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-ai-chatbot}"
IMAGE_NAME="${IMAGE_NAME:-ai-chatbot:dev}"
HOST_PORT="${HOST_PORT:-9090}"
CONTAINER_PORT="${CONTAINER_PORT:-8090}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-12}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-5}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"
RUNTIME_DIR="${PROJECT_DIR}/runtime"
LOG_CAPTURE_FILE="${RUNTIME_DIR}/deploy-dev.log"
GIT_REMOTE="${GIT_REMOTE:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-dev}"
APT_MIRROR_HOST="${APT_MIRROR_HOST:-mirrors.aliyun.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
SERVER_IP=""

log() {
  echo "$1"
}

fail() {
  echo "$1" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -f "$path" ] || fail "缺少文件: $path"
}

require_dir() {
  local path="$1"
  [ -d "$path" ] || fail "缺少目录: $path"
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "未找到命令: $cmd"
}

require_port_free() {
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:|])${HOST_PORT}$"; then
    fail "端口 ${HOST_PORT} 已被占用，立即停止部署"
  fi
}

require_project_layout() {
  require_file "${PROJECT_DIR}/Dockerfile"
  require_file "${PROJECT_DIR}/requirements.txt"
  require_file "${PROJECT_DIR}/api/main.py"
  require_file "${PROJECT_DIR}/templates/index.html"
  require_dir "${PROJECT_DIR}/static"
}

require_env_file() {
  require_file "${ENV_FILE}"
}

require_env_value() {
  local key="$1"
  grep -Eq "^${key}=.+" "${ENV_FILE}" || fail ".env 缺少必要配置: ${key}"
}

require_git_repo() {
  git -C "${PROJECT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "当前目录不是 git 仓库"
  git -C "${PROJECT_DIR}" remote get-url "${GIT_REMOTE}" >/dev/null 2>&1 || fail "git remote 不存在: ${GIT_REMOTE}"
}

require_target_branch_exists() {
  git -C "${PROJECT_DIR}" ls-remote --exit-code --heads "${GIT_REMOTE}" "${TARGET_BRANCH}" >/dev/null 2>&1 \
    || fail "远端分支不存在: ${GIT_REMOTE}/${TARGET_BRANCH}"
}

require_runtime_dirs() {
  mkdir -p "${RUNTIME_DIR}"
  [ -w "${RUNTIME_DIR}" ] || fail "目录不可写: ${RUNTIME_DIR}"
}

require_docker_ready() {
  require_command docker
  docker info >/dev/null 2>&1 || fail "Docker daemon 不可用"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${APP_NAME}"
}

require_container_running() {
  local status
  status="$(docker inspect -f '{{.State.Status}}' "${APP_NAME}" 2>/dev/null || true)"
  [ "${status}" = "running" ] || fail "容器未处于 running 状态: ${APP_NAME}"
}

print_container_logs() {
  docker logs "${APP_NAME}" 2>&1 | tee -a "${LOG_CAPTURE_FILE}" || true
}

health_check() {
  curl -fsS "http://127.0.0.1:${HOST_PORT}/" >/dev/null
}

wait_for_health_check() {
  local attempt
  for attempt in $(seq 1 "${HEALTH_CHECK_RETRIES}"); do
    if health_check; then
      return 0
    fi
    sleep "${HEALTH_CHECK_INTERVAL}"
  done
  return 1
}

resolve_server_ip() {
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -z "${ip}" ]; then
    ip="$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i = 1; i <= NF; i++) if ($i == "src") {print $(i+1); exit}}')"
  fi
  SERVER_IP="${ip:-127.0.0.1}"
}

force_sync_dev_branch() {
  log "警告: 即将强制覆盖本地改动并同步 ${GIT_REMOTE}/${TARGET_BRANCH}"
  cd "${PROJECT_DIR}"
  git fetch "${GIT_REMOTE}" "${TARGET_BRANCH}"
  git checkout "${TARGET_BRANCH}"
  git reset --hard "${GIT_REMOTE}/${TARGET_BRANCH}"
  git clean -fd
}

log "[1/10] 检查项目结构、git 仓库和环境文件"
require_project_layout
require_git_repo
require_target_branch_exists
require_env_file
require_env_value "OPENROUTER_API_KEY"
require_env_value "OPENROUTER_API_BASE"
require_env_value "MEM0_API_KEY"
require_env_value "MODEL_NAME"
require_env_value "MEM0_EMBED_MODEL"
require_env_value "MEM0_EMBEDDING_DIMS"
require_env_value "WINDOW_SIZE"
require_env_value "MAX_WORKERS"

log "[2/10] 检查 Docker、curl、ss 和运行目录"
require_docker_ready
require_command git
require_command curl
require_command ss
require_runtime_dirs

log "[3/10] 强制同步 GitHub ${GIT_REMOTE}/${TARGET_BRANCH} 分支"
force_sync_dev_branch

log "[4/10] 清理旧容器"
if container_exists; then
  docker rm -f "${APP_NAME}" >/dev/null 2>&1 || fail "删除旧容器失败: ${APP_NAME}"
fi

log "[5/10] 检查宿主机端口 ${HOST_PORT} 是否可用"
require_port_free

log "[6/10] 使用国内镜像参数构建 Docker 镜像"
docker build \
  --pull \
  --build-arg APT_MIRROR_HOST=mirrors.aliyun.com \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  -t "${IMAGE_NAME}" "${PROJECT_DIR}"

log "[7/10] 启动 Docker 容器"
docker run -d \
  --name "${APP_NAME}" \
  --restart unless-stopped \
  --env-file "${ENV_FILE}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${RUNTIME_DIR}:/app/runtime" \
  "${IMAGE_NAME}" >/dev/null

log "[8/10] 检查容器运行状态"
require_container_running

log "[9/10] 等待健康检查通过"
if ! wait_for_health_check; then
  echo "服务启动失败，打印容器日志" >&2
  print_container_logs
  exit 1
fi

log "[10/10] 输出部署结果"
resolve_server_ip
log "部署成功"
log "已强制同步分支: ${GIT_REMOTE}/${TARGET_BRANCH}"
log "访问地址: http://${SERVER_IP}:${HOST_PORT}/"
log "查看日志: docker logs -f ${APP_NAME}"
