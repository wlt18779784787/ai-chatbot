#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-ai-chatbot}"
HOST_PORT="${HOST_PORT:-9090}"

if ! command -v docker >/dev/null 2>&1; then
  echo "未找到 docker 命令" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${APP_NAME}"; then
  docker rm -f "${APP_NAME}"
  echo "已停止并删除容器 ${APP_NAME} (port ${HOST_PORT})"
else
  echo "容器 ${APP_NAME} 不存在 (port ${HOST_PORT})"
fi
