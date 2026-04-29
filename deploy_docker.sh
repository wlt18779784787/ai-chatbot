#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-ai-chatbot}"
IMAGE_NAME="${IMAGE_NAME:-ai-chatbot:latest}"
HOST_PORT="${HOST_PORT:-8090}"
CONTAINER_PORT="${CONTAINER_PORT:-8090}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"
DATA_DIR="${PROJECT_DIR}/data"
RUNTIME_DIR="${PROJECT_DIR}/runtime"
LOG_CAPTURE_FILE="${RUNTIME_DIR}/deploy.log"

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
  if ss -ltn | awk '{print $4}' | grep -q ":${HOST_PORT}$"; then
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

require_runtime_dirs() {
  mkdir -p "${DATA_DIR}" "${RUNTIME_DIR}"
  [ -w "${DATA_DIR}" ] || fail "目录不可写: ${DATA_DIR}"
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

log "[1/8] 检查端口 ${HOST_PORT} 是否被占用"
require_port_free

log "[2/8] 检查项目结构和环境文件"
require_project_layout
require_env_file
require_env_value "OPENROUTER_API_KEY"
require_env_value "OPENROUTER_API_BASE"
require_env_value "MODEL_NAME"
require_env_value "MEM0_EMBED_MODEL"
require_env_value "MEM0_EMBEDDING_DIMS"
require_env_value "WINDOW_SIZE"
require_env_value "MAX_WORKERS"

log "[3/8] 检查 Docker 和宿主机运行目录"
require_docker_ready
require_runtime_dirs

log "[4/8] 构建镜像前复检"
require_docker_ready
require_project_layout
docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"

log "[5/8] 清理旧容器前复检"
require_docker_ready
if container_exists; then
  docker rm -f "${APP_NAME}" >/dev/null 2>&1 || fail "删除旧容器失败: ${APP_NAME}"
fi

log "[6/8] 启动容器前复检"
require_port_free
require_env_file
require_runtime_dirs
docker run -d \
  --name "${APP_NAME}" \
  --restart unless-stopped \
  --env-file "${ENV_FILE}" \
  -e MEM0_HISTORY_DB_PATH=/app/data/history.db \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${DATA_DIR}:/app/data" \
  -v "${RUNTIME_DIR}:/app/runtime" \
  "${IMAGE_NAME}" >/dev/null

log "[7/8] 健康检查前复检"
require_container_running
require_command curl
sleep 5

log "[8/8] 执行健康检查"
if ! health_check; then
  echo "服务启动失败，打印容器日志" >&2
  print_container_logs
  exit 1
fi

log "部署成功"
log "访问地址: http://127.0.0.1:${HOST_PORT}/"
log "查看日志: docker logs -f ${APP_NAME}"
