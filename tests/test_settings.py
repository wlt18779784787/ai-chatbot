import importlib
import os
import unittest
from unittest.mock import patch


class Mem0SettingsTests(unittest.TestCase):
    def test_default_model_name_is_kimi_2_5(self):
        import sys

        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            import config.settings as settings

            importlib.reload(settings)

        self.assertEqual(settings.Config.MODEL_NAME, "moonshot/kimi2.5")
        self.assertFalse(settings.Config.REASONING_ENABLED)

    def test_openrouter_reasoning_settings_disable_reasoning_when_switch_off(self):
        import sys

        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {"REASONING_ENABLED": "false"}, clear=True):
            import config.settings as settings

            importlib.reload(settings)

        self.assertEqual(
            settings.get_openrouter_reasoning_config(),
            {"effort": "none", "exclude": True},
        )

    def test_openrouter_reasoning_settings_omit_reasoning_when_switch_on(self):
        import sys

        sys.modules.pop("config.settings", None)
        with patch.dict(os.environ, {"REASONING_ENABLED": "true"}, clear=True):
            import config.settings as settings

            importlib.reload(settings)

        self.assertIsNone(settings.get_openrouter_reasoning_config())

    def test_get_mem0_oss_config_builds_openrouter_milvus_config(self):
        env = {
            "OPENROUTER_API_KEY": "or-key",
            "OPENROUTER_API_BASE": "https://openrouter.ai/api/v1",
            "MODEL_NAME": "moonshot/kimi2.5",
            "REASONING_ENABLED": "false",
            "MEM0_EMBED_MODEL": "qwen/qwen3-embedding-4b",
            "MEM0_EMBEDDING_DIMS": "2560",
            "MILVUS_HOST": "",
            "MILVUS_PORT": "",
            "MILVUS_USER": "",
            "MILVUS_PASSWORD": "",
            "MILVUS_URL": "https://milvus.example.com",
            "MILVUS_TOKEN": "milvus-token",
            "MILVUS_DB_NAME": "chatbot",
            "MILVUS_COLLECTION_NAME": "memories",
            "MILVUS_METRIC_TYPE": "COSINE",
            "MEM0_HISTORY_DB_PATH": "history.db",
        }

        with patch.dict(os.environ, env, clear=True):
            import config.settings as settings

            importlib.reload(settings)
            config = settings.get_mem0_oss_config()

        self.assertEqual(config["llm"]["provider"], "openai")
        self.assertEqual(config["llm"]["config"]["model"], "moonshot/kimi2.5")
        self.assertEqual(config["llm"]["config"]["openai_base_url"], "https://openrouter.ai/api/v1")
        self.assertFalse(config["reasoning_enabled"])
        self.assertEqual(config["embedder"]["provider"], "openai")
        self.assertEqual(config["embedder"]["config"]["model"], "qwen/qwen3-embedding-4b")
        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 2560)
        self.assertEqual(config["vector_store"]["provider"], "milvus")
        self.assertEqual(config["vector_store"]["config"]["url"], "https://milvus.example.com")
        self.assertEqual(config["vector_store"]["config"]["token"], "milvus-token")
        self.assertEqual(config["vector_store"]["config"]["db_name"], "chatbot")
        self.assertEqual(config["vector_store"]["config"]["collection_name"], "memories")
        self.assertEqual(config["vector_store"]["config"]["embedding_model_dims"], 2560)
        self.assertEqual(config["vector_store"]["config"]["metric_type"], "COSINE")
        self.assertEqual(config["history_db_path"], "history.db")

    def test_get_mem0_oss_config_builds_milvus_url_and_token_from_host_credentials(self):
        env = {
            "OPENROUTER_API_KEY": "or-key",
            "MODEL_NAME": "moonshot/kimi2.5",
            "REASONING_ENABLED": "true",
            "MEM0_EMBED_MODEL": "qwen/qwen3-embedding-4b",
            "MEM0_EMBEDDING_DIMS": "2560",
            "MILVUS_HOST": "8.155.168.98",
            "MILVUS_PORT": "19530",
            "MILVUS_USER": "root",
            "MILVUS_PASSWORD": "secret",
            "MILVUS_URL": "",
            "MILVUS_TOKEN": "",
        }

        with patch.dict(os.environ, env, clear=True):
            import config.settings as settings

            importlib.reload(settings)
            config = settings.get_mem0_oss_config()

        self.assertEqual(config["vector_store"]["config"]["url"], "http://8.155.168.98:19530")
        self.assertEqual(config["vector_store"]["config"]["token"], "root:secret")
        self.assertTrue(config["reasoning_enabled"])


if __name__ == "__main__":
    unittest.main()
