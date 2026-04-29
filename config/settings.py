"""
Unified configuration management.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # === API ===
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    OPENROUTER_CHAT_URL: str = f"{OPENROUTER_API_BASE}/chat/completions"

    # === Models ===
    MODEL_NAME: str = os.getenv("MODEL_NAME", "moonshot/kimi2.5")
    REASONING_ENABLED: bool = _get_bool_env("REASONING_ENABLED", False)
    MEM0_EMBED_MODEL: str = os.getenv("MEM0_EMBED_MODEL", "qwen/qwen3-embedding-4b")
    MEM0_EMBEDDING_DIMS: int = int(os.getenv("MEM0_EMBEDDING_DIMS", "2560"))

    # === Mem0 OSS / Milvus ===
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER: str = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_URL: str = os.getenv("MILVUS_URL", "http://localhost:19530")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "mem0_qwen4b")
    MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "IP")
    MEM0_HISTORY_DB_PATH: str = os.getenv("MEM0_HISTORY_DB_PATH", str(BASE_DIR / "history.db"))

    # === Chat ===
    WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", "20"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))


def _build_milvus_url() -> str:
    if Config.MILVUS_HOST:
        return f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
    return Config.MILVUS_URL


def _build_milvus_token():
    if Config.MILVUS_TOKEN:
        return Config.MILVUS_TOKEN
    if Config.MILVUS_USER or Config.MILVUS_PASSWORD:
        return f"{Config.MILVUS_USER}:{Config.MILVUS_PASSWORD}"
    return None


def get_openrouter_reasoning_config():
    reasoning_enabled = getattr(globals().get("config"), "REASONING_ENABLED", Config.REASONING_ENABLED)
    if reasoning_enabled:
        return None
    return {"effort": "none", "exclude": True}


def get_mem0_oss_config() -> dict:
    """Build Mem0 OSS configuration for OpenRouter + Milvus."""
    return {
        "reasoning_enabled": Config.REASONING_ENABLED,
        "llm": {
            "provider": "openai",
            "config": {
                "model": Config.MODEL_NAME,
                "api_key": Config.OPENROUTER_API_KEY,
                "openai_base_url": Config.OPENROUTER_API_BASE,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": Config.MEM0_EMBED_MODEL,
                "api_key": Config.OPENROUTER_API_KEY,
                "openai_base_url": Config.OPENROUTER_API_BASE,
                "embedding_dims": Config.MEM0_EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "milvus",
            "config": {
                "url": _build_milvus_url(),
                "token": _build_milvus_token(),
                "db_name": Config.MILVUS_DB_NAME,
                "collection_name": Config.MILVUS_COLLECTION_NAME,
                "embedding_model_dims": Config.MEM0_EMBEDDING_DIMS,
                "metric_type": Config.MILVUS_METRIC_TYPE,
            },
        },
        "history_db_path": Config.MEM0_HISTORY_DB_PATH,
    }


config = Config()
