"""
Compatibility patches for Mem0 integrations used by this project.
"""

import os

from config import get_openrouter_reasoning_config


def apply_mem0_milvus_dense_only_patch():
    """
    Patch Mem0's Milvus adapter to use dense-only collections.

    Mem0's upstream adapter creates BM25 sparse indexes by default. That fails
    on Milvus deployments that support sparse vectors but only accept `IP`
    metrics for sparse indexes. This patch keeps the dense vector index and
    skips BM25-specific fields and indexes.
    """
    try:
        from mem0.vector_stores import milvus as milvus_module
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(milvus_module, "_dense_only_patch_applied", False):
        return

    def create_col(self, collection_name, vector_size, metric_type=milvus_module.MetricType.COSINE):
        if self.client.has_collection(collection_name):
            self._has_bm25_schema = False
            return

        fields = [
            milvus_module.FieldSchema(name="id", dtype=milvus_module.DataType.VARCHAR, is_primary=True, max_length=512),
            milvus_module.FieldSchema(name="vectors", dtype=milvus_module.DataType.FLOAT_VECTOR, dim=vector_size),
            milvus_module.FieldSchema(name="metadata", dtype=milvus_module.DataType.JSON),
        ]
        schema = milvus_module.CollectionSchema(fields, enable_dynamic_field=True)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vectors",
            metric_type=metric_type,
            index_type="AUTOINDEX",
            index_name="vector_index",
        )
        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        self._has_bm25_schema = False

    def update(self, vector_id=None, vector=None, payload=None):
        if vector is None or payload is None:
            existing = self.client.get(collection_name=self.collection_name, ids=vector_id)
            if not existing:
                raise ValueError(f"Vector with id {vector_id} not found in collection {self.collection_name}")
            if vector is None:
                vector = existing[0].get("vectors")
                if vector is None:
                    raise ValueError(f"Existing record {vector_id} has no vector data")
            if payload is None:
                payload = existing[0].get("metadata")

        schema = {"id": vector_id, "vectors": vector, "metadata": payload}
        self.client.upsert(collection_name=self.collection_name, data=schema)

    milvus_module.MilvusDB.create_col = create_col
    milvus_module.MilvusDB.update = update
    milvus_module._dense_only_patch_applied = True


def apply_mem0_openrouter_reasoning_config_patch():
    """
    Patch Mem0's OpenRouter LLM calls to respect the project's reasoning switch.

    Mem0 uses the OpenAI SDK for OpenRouter traffic. We wrap the SDK call site
    so every chat completion request sent through Mem0 uses the same reasoning
    configuration as the primary chat request.
    """
    try:
        from mem0.llms import openai as openai_module
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(openai_module, "_openrouter_reasoning_config_patch_applied", False):
        return

    original_init = openai_module.OpenAILLM.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not os.getenv("OPENROUTER_API_KEY"):
            return

        completions = getattr(getattr(self.client, "chat", None), "completions", None)
        create = getattr(completions, "create", None)
        if create is None or getattr(completions, "_no_reasoning_wrapper_applied", False):
            return

        def create_with_reasoning_config(*create_args, **create_kwargs):
            reasoning = get_openrouter_reasoning_config()
            if reasoning is not None:
                extra_body = dict(create_kwargs.get("extra_body") or {})
                extra_body["reasoning"] = reasoning
                create_kwargs["extra_body"] = extra_body
            return create(*create_args, **create_kwargs)

        completions.create = create_with_reasoning_config
        completions._no_reasoning_wrapper_applied = True

    openai_module.OpenAILLM.__init__ = patched_init
    openai_module._openrouter_reasoning_config_patch_applied = True
