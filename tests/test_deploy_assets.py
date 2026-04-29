import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class DockerDeployAssetsTests(unittest.TestCase):
    def test_dockerfile_uses_python_312_slim_domestic_mirrors_and_9090(self):
        dockerfile = (BASE_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn("mirrors.aliyun.com", dockerfile)
        self.assertIn("mirrors.aliyun.com/pypi/simple", dockerfile)
        self.assertIn("EXPOSE 9090", dockerfile)
        self.assertIn('CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9090"]', dockerfile)

    def test_api_main_uses_9090_for_direct_uvicorn_start(self):
        api_main = (BASE_DIR / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('uvicorn.run(app, host="0.0.0.0", port=9090)', api_main)

    def test_deploy_script_removes_old_container_before_port_check_and_build(self):
        script = (BASE_DIR / "deploy_docker.sh").read_text(encoding="utf-8")
        self.assertIn('HOST_PORT="${HOST_PORT:-9090}"', script)
        self.assertIn('CONTAINER_PORT="${CONTAINER_PORT:-9090}"', script)
        self.assertIn('if container_exists; then', script)
        self.assertIn('docker rm -f "${APP_NAME}"', script)
        self.assertIn('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"', script)
        self.assertLess(
            script.index('docker rm -f "${APP_NAME}"'),
            script.index('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"'),
        )

    def test_deploy_script_uses_retry_health_check(self):
        script = (BASE_DIR / "deploy_docker.sh").read_text(encoding="utf-8")
        self.assertIn('HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-12}"', script)
        self.assertIn('HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-5}"', script)
        self.assertIn('for attempt in $(seq 1 "${HEALTH_CHECK_RETRIES}")', script)
        self.assertIn('sleep "${HEALTH_CHECK_INTERVAL}"', script)

    def test_dev_deploy_script_force_syncs_origin_dev_uses_9090_and_public_ip(self):
        script = (BASE_DIR / "deploy_github_dev_docker_9090.sh").read_text(encoding="utf-8")
        self.assertIn('HOST_PORT="${HOST_PORT:-9090}"', script)
        self.assertIn('CONTAINER_PORT="${CONTAINER_PORT:-9090}"', script)
        self.assertIn('TARGET_BRANCH="${TARGET_BRANCH:-dev}"', script)
        self.assertIn('git fetch "${GIT_REMOTE}" "${TARGET_BRANCH}"', script)
        self.assertIn('git checkout "${TARGET_BRANCH}"', script)
        self.assertIn('git reset --hard "${GIT_REMOTE}/${TARGET_BRANCH}"', script)
        self.assertIn('git clean -fd', script)
        self.assertIn('--build-arg APT_MIRROR_HOST=mirrors.aliyun.com', script)
        self.assertIn('--build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/', script)
        self.assertIn('PUBLIC_IP="${PUBLIC_IP:-}"', script)
        self.assertIn('curl -fsS https://api.ipify.org', script)
        self.assertIn('curl -fsS https://ifconfig.me', script)
        self.assertIn('fail "无法解析公网 IP', script)
        self.assertIn('http://${SERVER_IP}:${HOST_PORT}/', script)
        self.assertLess(script.index('git reset --hard "${GIT_REMOTE}/${TARGET_BRANCH}"'), script.index('docker build \\'))

    def test_dev_stop_script_stops_matching_container(self):
        script = (BASE_DIR / "stop_github_dev_docker_9090.sh").read_text(encoding="utf-8")
        self.assertIn('APP_NAME="${APP_NAME:-ai-chatbot}"', script)
        self.assertIn('HOST_PORT="${HOST_PORT:-9090}"', script)
        self.assertIn("docker rm -f", script)
        self.assertIn("docker ps -a --format '{{.Names}}'", script)

    def test_dockerignore_excludes_env_file(self):
        dockerignore = (BASE_DIR / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".env", dockerignore.splitlines())

    def test_ubuntu_ecs_deploy_doc_mentions_domestic_mirrors(self):
        deploy_doc = (BASE_DIR / "docs" / "ubuntu-ecs-docker-deploy.md").read_text(encoding="utf-8")
        self.assertIn("mirrors.aliyun.com", deploy_doc)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", deploy_doc)


if __name__ == "__main__":
    unittest.main()
