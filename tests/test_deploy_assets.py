import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class DockerDeployAssetsTests(unittest.TestCase):
    def test_dockerfile_uses_python_312_slim(self):
        dockerfile = (BASE_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn('CMD ["uvicorn", "api.main:app"', dockerfile)

    def test_deploy_script_checks_port_before_build(self):
        script = (BASE_DIR / "deploy_docker.sh").read_text(encoding="utf-8")
        self.assertIn('HOST_PORT="${HOST_PORT:-8090}"', script)
        self.assertIn('log "[1/8] 检查端口 ${HOST_PORT} 是否被占用"', script)
        self.assertIn('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"', script)
        self.assertLess(
            script.index('log "[1/8] 检查端口 ${HOST_PORT} 是否被占用"'),
            script.index('docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"'),
        )


if __name__ == "__main__":
    unittest.main()
