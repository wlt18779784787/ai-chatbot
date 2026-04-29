import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class DockerDeployAssetsTests(unittest.TestCase):
    def test_dockerfile_uses_python_312_slim_and_domestic_mirrors(self):
        dockerfile = (BASE_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn('CMD ["uvicorn", "api.main:app"', dockerfile)
        self.assertIn("mirrors.aliyun.com", dockerfile)
        self.assertIn("mirrors.aliyun.com/pypi/simple", dockerfile)

    def test_deploy_script_removes_old_container_before_port_check_and_build(self):
        script = (BASE_DIR / "deploy_docker.sh").read_text(encoding="utf-8")
        self.assertIn('HOST_PORT="${HOST_PORT:-8090}"', script)
        self.assertIn('log "[1/9] 检查项目结构和环境文件"', script)
        self.assertIn('if container_exists; then', script)
        self.assertIn('docker rm -f "${APP_NAME}"', script)
        self.assertIn('log "[4/9] 检查宿主机端口 ${HOST_PORT} 是否可用"', script)
        self.assertIn('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"', script)
        self.assertLess(
            script.index('docker rm -f "${APP_NAME}"'),
            script.index('log "[4/9] 检查宿主机端口 ${HOST_PORT} 是否可用"'),
        )
        self.assertLess(
            script.index('log "[4/9] 检查宿主机端口 ${HOST_PORT} 是否可用"'),
            script.index('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"'),
        )

    def test_deploy_script_uses_retry_health_check(self):
        script = (BASE_DIR / "deploy_docker.sh").read_text(encoding="utf-8")
        self.assertIn('HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-12}"', script)
        self.assertIn('HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-5}"', script)
        self.assertIn('for attempt in $(seq 1 "${HEALTH_CHECK_RETRIES}")', script)
        self.assertIn('sleep "${HEALTH_CHECK_INTERVAL}"', script)

    def test_dockerignore_excludes_env_file(self):
        dockerignore = (BASE_DIR / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".env", dockerignore.splitlines())

    def test_ubuntu_ecs_deploy_doc_mentions_domestic_mirrors(self):
        deploy_doc = (BASE_DIR / "docs" / "ubuntu-ecs-docker-deploy.md").read_text(encoding="utf-8")
        self.assertIn("阿里云 ECS Ubuntu", deploy_doc)
        self.assertIn("阿里云镜像加速器", deploy_doc)
        self.assertIn("mirrors.aliyun.com", deploy_doc)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", deploy_doc)


if __name__ == "__main__":
    unittest.main()
