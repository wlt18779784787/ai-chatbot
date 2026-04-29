#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-ai-chatbot}"
IMAGE_NAME="${IMAGE_NAME:-ai-chatbot:latest}"
HOST_PORT="${HOST_PORT:-8090}"
CONTAINER_PORT="${CONTAINER_PORT:-8090}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-12}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-5}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"
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

require_milvus_target() {
  local milvus_host
  local milvus_url
  milvus_host="$(grep -E '^MILVUS_HOST=' "${ENV_FILE}" | cut -d= -f2- || true)"
  milvus_url="$(grep -E '^MILVUS_URL=' "${ENV_FILE}" | cut -d= -f2- || true)"

  if [ -n "${milvus_host}" ] && [ "${milvus_host}" != "localhost" ] && [ "${milvus_host}" != "127.0.0.1" ]; then
    return 0
  fi

  if [ -n "${milvus_url}" ] && [[ "${milvus_url}" != *"localhost"* ]] && [[ "${milvus_url}" != *"127.0.0.1"* ]]; then
    return 0
  fi

  fail "Milvus 地址不能使用 localhost/127.0.0.1；阿里云 ECS 容器内无法访问宿主机本地 Milvus"
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

log "[1/9] 检查项目结构和环境文件"
require_project_layout
require_env_file
require_env_value "OPENROUTER_API_KEY"
require_env_value "OPENROUTER_API_BASE"
require_env_value "MEM0_API_KEY"
require_env_value "MODEL_NAME"
require_env_value "MEM0_EMBED_MODEL"
require_env_value "MEM0_EMBEDDING_DIMS"
require_env_value "WINDOW_SIZE"
require_env_value "MAX_WORKERS"

log "[2/9] 检查 Docker、curl 和宿主机运行目录"
require_docker_ready
require_command curl
require_command ss
require_runtime_dirs

log "[3/9] 清理旧容器"
if container_exists; then
  docker rm -f "${APP_NAME}" >/dev/null 2>&1 || fail "删除旧容器失败: ${APP_NAME}"
fi

log "[4/9] 检查宿主机端口 ${HOST_PORT} 是否可用"
require_port_free

log "[5/9] 构建镜像"
docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"

log "[6/9] 启动容器"
docker run -d \
  --name "${APP_NAME}" \
  --restart unless-stopped \
  --env-file "${ENV_FILE}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${RUNTIME_DIR}:/app/runtime" \
  "${IMAGE_NAME}" >/dev/null

log "[7/9] 检查容器运行状态"
require_container_running

log "[8/9] 等待健康检查通过"
if ! wait_for_health_check; then
  echo "服务启动失败，打印容器日志" >&2
  print_container_logs
  exit 1
fi

log "[9/9] 输出部署结果"
log "部署成功"
log "访问地址: http://127.0.0.1:${HOST_PORT}/"
log "查看日志: docker logs -f ${APP_NAME}"
