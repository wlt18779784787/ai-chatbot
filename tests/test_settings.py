import importlib
import os
import unittest
from unittest.mock import patch


class Mem0SettingsTests(unittest.TestCase):
    def test_default_model_name_is_kimi_k2(self):
        import sys

        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            import config.settings as settings

            importlib.reload(settings)

        self.assertEqual(settings.Config.MODEL_NAME, "moonshotai/kimi-k2")
        self.assertFalse(settings.Config.REASONING_ENABLED)

    def test_openrouter_reasoning_settings_disable_reasoning_when_switch_off(self):
        import sys

        sys.modules.pop("config", None)
        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {"REASONING_ENABLED": "false"}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            import config.settings as settings

            importlib.reload(settings)
            self.assertEqual(
                settings.get_openrouter_reasoning_config(),
                {"effort": "none", "exclude": True},
            )

    def test_openrouter_reasoning_settings_omit_reasoning_when_switch_on(self):
        import sys

        sys.modules.pop("config", None)
        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {"REASONING_ENABLED": "true"}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            import config.settings as settings

            importlib.reload(settings)
            self.assertIsNone(settings.get_openrouter_reasoning_config())

    def test_get_mem0_client_config_builds_platform_config(self):
        env = {
            "MEM0_API_KEY": "m0-test-key",
            "OPENROUTER_API_KEY": "or-key",
        }

        with patch.dict(os.environ, env, clear=True):
            import config.settings as settings

            importlib.reload(settings)
            config = settings.get_mem0_client_config()

        self.assertEqual(config, {"api_key": "m0-test-key"})

    def test_get_mem0_client_config_ignores_legacy_milvus_settings(self):
        env = {
            "MEM0_API_KEY": "m0-test-key",
            "MILVUS_HOST": "8.155.168.98",
            "MILVUS_PORT": "19530",
            "MILVUS_USER": "root",
            "MILVUS_PASSWORD": "secret",
            "MILVUS_URL": "http://legacy.example.com",
            "MILVUS_TOKEN": "legacy-token",
            "MEM0_HISTORY_DB_PATH": "history.db",
        }

        with patch.dict(os.environ, env, clear=True):
            import config.settings as settings

            importlib.reload(settings)
            config = settings.get_mem0_client_config()

        self.assertEqual(config, {"api_key": "m0-test-key"})


if __name__ == "__main__":
    unittest.main()
