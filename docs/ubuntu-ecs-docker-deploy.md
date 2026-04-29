# 阿里云 ECS Ubuntu Docker 部署说明

本文档面向阿里云 ECS Ubuntu 服务器，目标是让本项目在国内网络环境下尽量只使用国内镜像完成安装、构建和启动。

## 1. 服务器准备

- Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS
- 已开放安全组入站端口 `8090`
- 服务器需要能访问 OpenRouter 和你配置的 Milvus 地址

## 2. 使用国内源安装 Docker

先更新 Ubuntu `apt` 软件源。阿里云 ECS 常用国内镜像如下：

```bash
sudo sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g; s|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list
sudo apt-get update
```

安装基础依赖：

```bash
sudo apt-get install -y ca-certificates curl gnupg
```

添加 Docker 官方仓库密钥并安装 Docker：

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
```

## 3. 配置 Docker 镜像加速

`FROM python:3.12-slim` 这类基础镜像仍然依赖 Docker Hub 路径，所以阿里云 ECS 上建议先配置阿里云镜像加速器。

编辑 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://<你的阿里云镜像加速器地址>"
  ]
}
```

重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

阿里云镜像加速器地址在阿里云控制台可申请，关键词就是“阿里云镜像加速器”。

## 4. 拉取项目并准备环境变量

将项目放到服务器，例如 `/opt/ai-chatbot`：

```bash
mkdir -p /opt/ai-chatbot
cd /opt/ai-chatbot
```

准备 `.env`：

```bash
cp .env.example .env
```

重点配置：

- `OPENROUTER_API_KEY`
- `OPENROUTER_API_BASE`
- `MODEL_NAME`
- `REASONING_ENABLED`
- `MEM0_EMBED_MODEL`
- `MEM0_EMBEDDING_DIMS`
- `MILVUS_HOST` 或 `MILVUS_URL`

注意：

- 不要把 `MILVUS_HOST` 写成 `localhost` 或 `127.0.0.1`
- 不要把 `MILVUS_URL` 写成 `http://localhost:19530`
- Docker 容器内的 `localhost` 指向容器自己，不是阿里云 ECS 宿主机

如果你使用的是远端 Milvus，应该填写 ECS 能直接访问的公网或 VPC 内网地址。

## 5. Python 依赖镜像说明

项目 `Dockerfile` 已固定使用国内镜像：

- Debian `apt`：`mirrors.aliyun.com`
- `pip`：`https://mirrors.aliyun.com/pypi/simple/`

如果你在容器外手工安装 Python 工具，建议也走国内源，例如清华：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
```

## 6. 启动项目

给脚本执行权限并启动：

```bash
chmod +x deploy_docker.sh stop_docker.sh
./deploy_docker.sh
```

部署脚本会做这些检查：

- 校验 `.env` 必填项
- 校验 Milvus 地址不是 `localhost`
- 清理同名旧容器
- 检查宿主机端口是否可用
- 用国内镜像源构建 Docker 镜像
- 重试健康检查，避免 ECS 首次冷启动误判

## 7. 常用命令

查看容器：

```bash
docker ps -a
```

查看日志：

```bash
docker logs -f ai-chatbot
```

停止并删除容器：

```bash
./stop_docker.sh
```

## 8. 常见问题

### 1. `docker build` 拉基础镜像很慢

先检查 Docker daemon 是否已经配置阿里云镜像加速器。

### 2. 服务启动后健康检查失败

先看日志：

```bash
docker logs ai-chatbot
```

重点排查：

- `OPENROUTER_API_KEY` 是否正确
- ECS 是否能访问 OpenRouter
- `MILVUS_HOST` / `MILVUS_URL` 是否可从 ECS 访问

### 3. Milvus 在宿主机本地运行

当前脚本默认不接受 `localhost`，因为 Ubuntu ECS 上 Docker 容器无法直接把容器内 `localhost` 当宿主机使用。更稳的做法是：

- 把 Milvus 部署到独立地址
- 或改成容器编排后使用明确的容器网络名称
