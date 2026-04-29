import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class FakeMemoryStore:
    def __init__(self):
        self.search_calls = []
        self.add_calls = []

    def search(self, query, filters=None, top_k=3):
        self.search_calls.append((query, filters, top_k))
        return {"results": []}

    def add(self, messages, user_id=None, agent_id=None, run_id=None):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
            }
        )
        return {"results": []}


class ChatCoreMem0IntegrationTests(unittest.TestCase):
    def _load_chat_core(self, fake_memory):
        fake_mem0 = types.ModuleType("mem0")
        fake_mem0.Memory = type("FakeMemory", (), {"from_config": staticmethod(lambda config: fake_memory)})

        sys.modules.pop("core.chat_core", None)
        sys.modules.pop("core", None)

        with patch.dict(sys.modules, {"mem0": fake_mem0}):
            import core.chat_core as chat_core_module

            importlib.reload(chat_core_module)
            return chat_core_module, chat_core_module.ChatCore()

    def test_chat_core_uses_memory_from_config(self):
        fake_memory = FakeMemoryStore()
        captured = {}

        def fake_from_config(config):
            captured["config"] = config
            return fake_memory

        fake_mem0 = types.ModuleType("mem0")
        fake_mem0.Memory = type("FakeMemory", (), {"from_config": staticmethod(fake_from_config)})

        sys.modules.pop("core.chat_core", None)
        sys.modules.pop("core", None)

        with patch.dict(sys.modules, {"mem0": fake_mem0}):
            import core.chat_core as chat_core_module

            importlib.reload(chat_core_module)
            chat_core = chat_core_module.ChatCore()

        self.assertIs(chat_core.memory_client, fake_memory)
        self.assertEqual(captured["config"]["vector_store"]["provider"], "milvus")

    def test_get_or_create_session_uses_full_scope(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)

        scope_a = chat_core_module.SessionScope("alice", "agent-1")
        scope_b = chat_core_module.SessionScope("alice", "agent-2")

        session_a = chat_core.get_or_create_session(scope_a)
        session_b = chat_core.get_or_create_session(scope_b)

        self.assertIsNot(session_a, session_b)

    def test_search_memories_uses_flat_scope_filters(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")

        chat_core.search_memories("hello", scope)

        self.assertEqual(len(fake_memory.search_calls), 1)
        _, filters, _ = fake_memory.search_calls[0]
        self.assertEqual(
            filters,
            {
                "user_id": "alice",
                "agent_id": "agent-1",
            },
        )

    def test_follow_up_query_is_rewritten_with_last_round_context(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")
        session = chat_core.get_or_create_session(scope)
        session.conversation_window = [
            {"role": "user", "content": "你记得我喜欢喝什么吗"},
            {"role": "assistant", "content": "你之前提过喜欢无糖可乐"},
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            chat_core._build_messages_with_timings("还有呢", session)

        self.assertEqual(len(fake_memory.search_calls), 1)
        query, _, _ = fake_memory.search_calls[0]
        self.assertIn("上一轮用户问题：你记得我喜欢喝什么吗", query)
        self.assertIn("上一轮助手回复：你之前提过喜欢无糖可乐", query)
        self.assertIn("当前用户追问：还有呢", query)
        self.assertIn("记忆检索续问改写", output.getvalue())

    def test_follow_up_query_without_context_falls_back_to_original_input(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")
        session = chat_core.get_or_create_session(scope)

        chat_core._build_messages_with_timings("还有呢", session)

        self.assertEqual(len(fake_memory.search_calls), 1)
        query, _, _ = fake_memory.search_calls[0]
        self.assertEqual(query, "还有呢")

    def test_first_memory_batch_flushes_after_five_rounds(self):
        fake_memory = FakeMemoryStore()

        class FakeExecutor:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args, **kwargs):
                self.submissions.append((fn, args, kwargs))

        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        chat_core.executor = FakeExecutor()
        scope = chat_core_module.SessionScope("alice", "agent-1")
        session = chat_core.get_or_create_session(scope)

        for index in range(5):
            chat_core.update_window(f"u{index}", f"a{index}", session)

        self.assertEqual(len(chat_core.executor.submissions), 1)
        fn, args, kwargs = chat_core.executor.submissions[0]
        self.assertEqual(fn.__name__, "_persist_memory_batch")
        self.assertEqual(args[1], scope)
        self.assertEqual(args[2], 5)
        self.assertEqual(len(args[0]), 10)
        self.assertEqual(len(session.memory_round_buffer), 2)
        self.assertEqual(session.memory_flush_count, 1)
        self.assertEqual(session.memory_round_buffer[0][0]["content"], "u3")
        self.assertEqual(kwargs, {})

    def test_second_memory_batch_flushes_after_three_new_rounds_with_overlap(self):
        fake_memory = FakeMemoryStore()

        class FakeExecutor:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args, **kwargs):
                self.submissions.append((fn, args, kwargs))

        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        chat_core.executor = FakeExecutor()
        scope = chat_core_module.SessionScope("alice", "agent-1")
        session = chat_core.get_or_create_session(scope)

        for index in range(8):
            chat_core.update_window(f"u{index}", f"a{index}", session)

        self.assertEqual(len(chat_core.executor.submissions), 2)
        _, args, _ = chat_core.executor.submissions[1]
        self.assertEqual(args[2], 5)
        self.assertEqual(len(args[0]), 10)
        self.assertEqual(args[0][0]["content"], "u3")
        self.assertEqual(args[0][-1]["content"], "a7")
        self.assertEqual(len(session.memory_round_buffer), 2)
        self.assertEqual(session.memory_round_buffer[0][0]["content"], "u6")
        self.assertEqual(session.memory_flush_count, 2)

    def test_persist_memory_batch_passes_scope_ids_to_mem0(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")
        messages = [{"role": "user", "content": "hello"}]

        chat_core._persist_memory_batch(messages, scope, 10)

        self.assertEqual(len(fake_memory.add_calls), 1)
        add_call = fake_memory.add_calls[0]
        self.assertEqual(add_call["messages"], messages)
        self.assertEqual(add_call["user_id"], "alice")
        self.assertEqual(add_call["agent_id"], "agent-1")
        self.assertIsNone(add_call["run_id"])

    def test_chat_prints_prompt_messages_before_model_call(self):
        fake_memory = FakeMemoryStore()
        _, chat_core = self._load_chat_core(fake_memory)

        output = io.StringIO()
        with patch.object(chat_core, "call_model", return_value="ok"), redirect_stdout(output):
            result = chat_core.chat("你好", "alice", "agent-1")

        self.assertTrue(result["success"])
        self.assertIn("本轮模型请求提示词", output.getvalue())
        self.assertIn("agent_id=agent-1", output.getvalue())
        self.assertNotIn("run_id=", output.getvalue())
        self.assertIn("role=system", output.getvalue())
        self.assertIn("role=user", output.getvalue())
        self.assertIn("你好", output.getvalue())

    def test_call_model_disables_reasoning_when_switch_off(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        fake_response = type(
            "ResponseStub",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"choices": [{"message": {"content": "ok"}}]},
            },
        )()

        with (
            patch.object(chat_core_module.requests, "post", return_value=fake_response) as post_mock,
            patch.object(chat_core_module.config, "MODEL_NAME", "moonshot/kimi2.5"),
            patch.object(chat_core_module.config, "REASONING_ENABLED", False),
        ):
            content = chat_core.call_model([{"role": "user", "content": "hello"}])

        self.assertEqual(content, "ok")
        self.assertEqual(post_mock.call_count, 1)
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["json"]["model"], "moonshot/kimi2.5")
        self.assertEqual(kwargs["json"]["reasoning"], {"effort": "none", "exclude": True})

    def test_call_model_omits_reasoning_when_switch_on(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        fake_response = type(
            "ResponseStub",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"choices": [{"message": {"content": "ok"}}]},
            },
        )()

        with (
            patch.object(chat_core_module.requests, "post", return_value=fake_response) as post_mock,
            patch.object(chat_core_module.config, "MODEL_NAME", "moonshot/kimi2.5"),
            patch.object(chat_core_module.config, "REASONING_ENABLED", True),
        ):
            content = chat_core.call_model([{"role": "user", "content": "hello"}])

        self.assertEqual(content, "ok")
        self.assertEqual(post_mock.call_count, 1)
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["json"]["model"], "moonshot/kimi2.5")
        self.assertNotIn("reasoning", kwargs["json"])

    def test_chat_duration_covers_full_request_not_just_model_call(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)

        perf_counter_values = iter([100.0, 104.0, 110.0, 112.0])

        def fake_build_messages(user_input, session):
            chat_core_module.time.perf_counter()
            chat_core_module.time.perf_counter()
            return [
                {"role": "system", "content": "system"},
                {"role": "user", "content": user_input},
            ], {}

        with (
            patch.object(chat_core_module.time, "perf_counter", side_effect=lambda: next(perf_counter_values)),
            patch.object(chat_core, "_build_messages_with_timings", side_effect=fake_build_messages),
            patch.object(chat_core, "_print_prompt_messages"),
            patch.object(chat_core, "call_model", return_value="ok"),
            patch.object(chat_core, "update_window"),
        ):
            result = chat_core.chat("你好", "alice", "agent-1")

        self.assertTrue(result["success"])
        self.assertEqual(result["duration"], 12.0)

    def test_search_memories_returns_empty_when_mem0_search_fails(self):
        class FailingMemoryStore(FakeMemoryStore):
            def search(self, query, filters=None, top_k=3):
                raise RuntimeError("embedding failed")

        chat_core_module, chat_core = self._load_chat_core(FailingMemoryStore())
        scope = chat_core_module.SessionScope("alice", "agent-1")

        output = io.StringIO()
        with redirect_stdout(output):
            memories = chat_core.search_memories("hi", scope)

        self.assertEqual(memories, [])
        self.assertIn("记忆检索失败", output.getvalue())

    def test_persist_memory_batch_logs_error_without_raising(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")

        with patch.object(chat_core, "save_to_memory", side_effect=RuntimeError("save failed")):
            output = io.StringIO()
            with redirect_stdout(output):
                chat_core._persist_memory_batch([{"role": "user", "content": "hi"}], scope, 10)

        self.assertIn("记忆保存失败", output.getvalue())

    def test_build_messages_includes_current_time_in_system_prompt(self):
        fake_memory = FakeMemoryStore()
        chat_core_module, chat_core = self._load_chat_core(fake_memory)
        scope = chat_core_module.SessionScope("alice", "agent-1")

        with patch.object(chat_core, "_get_current_time_text", return_value="2026-04-28 09:30:45 CST"):
            session = chat_core.get_or_create_session(scope)
            messages = chat_core.build_messages("你好", session)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("当前时间", messages[0]["content"])
        self.assertIn("2026-04-28 09:30:45 CST", messages[0]["content"])


class Mem0CompatPatchTests(unittest.TestCase):
    def test_dense_only_patch_creates_collection_without_sparse_index(self):
        fake_milvus = types.ModuleType("mem0.vector_stores.milvus")
        fake_milvus.MetricType = type("MetricType", (), {"COSINE": "COSINE"})
        fake_milvus.DataType = type("DataType", (), {"VARCHAR": "VARCHAR", "FLOAT_VECTOR": "FLOAT_VECTOR", "JSON": "JSON"})

        def field_schema(**kwargs):
            return kwargs

        def collection_schema(fields, enable_dynamic_field=False):
            return {"fields": fields, "enable_dynamic_field": enable_dynamic_field}

        fake_milvus.FieldSchema = field_schema
        fake_milvus.CollectionSchema = collection_schema
        fake_milvus.MilvusDB = type("MilvusDB", (), {})

        fake_pkg = types.ModuleType("mem0.vector_stores")
        fake_pkg.milvus = fake_milvus

        sys.modules.pop("core.mem0_compat", None)

        with patch.dict(sys.modules, {"mem0.vector_stores": fake_pkg, "mem0.vector_stores.milvus": fake_milvus}):
            import core.mem0_compat as compat

            importlib.reload(compat)
            compat.apply_mem0_milvus_dense_only_patch()

        class FakeIndexParams:
            def __init__(self):
                self.calls = []

            def add_index(self, **kwargs):
                self.calls.append(kwargs)

        class FakeClient:
            def __init__(self):
                self.index_params = FakeIndexParams()
                self.created = None

            def has_collection(self, collection_name):
                return False

            def prepare_index_params(self):
                return self.index_params

            def create_collection(self, collection_name, schema, index_params):
                self.created = {
                    "collection_name": collection_name,
                    "schema": schema,
                    "index_calls": index_params.calls,
                }

        store = fake_milvus.MilvusDB()
        store.client = FakeClient()
        store._has_bm25_schema = True

        store.create_col("memories", 2560)

        self.assertFalse(store._has_bm25_schema)
        self.assertEqual(len(store.client.created["schema"]["fields"]), 3)
        self.assertEqual(len(store.client.created["index_calls"]), 1)
        self.assertEqual(store.client.created["index_calls"][0]["field_name"], "vectors")

    def test_openrouter_patch_adds_no_reasoning_extra_body_when_switch_off(self):
        fake_openai_module = types.ModuleType("mem0.llms.openai")

        class FakeOpenAILLM:
            def __init__(self, config=None):
                self.config = config
                self.calls = []

                class FakeCompletions:
                    def __init__(inner_self, recorder):
                        inner_self._recorder = recorder

                    def create(inner_self, **kwargs):
                        inner_self._recorder.append(kwargs)
                        return {"ok": True}

                self.client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(
                        completions=FakeCompletions(self.calls)
                    )
                )

            def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
                params = {
                    "model": "moonshot/kimi2.5",
                    "messages": messages,
                }
                params.update(kwargs)
                return self.client.chat.completions.create(**params)

        fake_openai_module.OpenAILLM = FakeOpenAILLM

        sys.modules.pop("core.mem0_compat", None)

        with patch.dict(sys.modules, {"mem0.llms.openai": fake_openai_module}), patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "or-key", "REASONING_ENABLED": "false"},
            clear=False,
        ):
            import core.mem0_compat as compat

            importlib.reload(compat)
            compat.apply_mem0_openrouter_reasoning_config_patch()
            llm = fake_openai_module.OpenAILLM()
            llm.generate_response([{"role": "user", "content": "hello"}])

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(
            llm.calls[0]["extra_body"]["reasoning"],
            {"effort": "none", "exclude": True},
        )

    def test_openrouter_patch_omits_reasoning_extra_body_when_switch_on(self):
        fake_openai_module = types.ModuleType("mem0.llms.openai")

        class FakeOpenAILLM:
            def __init__(self, config=None):
                self.config = config
                self.calls = []

                class FakeCompletions:
                    def __init__(inner_self, recorder):
                        inner_self._recorder = recorder

                    def create(inner_self, **kwargs):
                        inner_self._recorder.append(kwargs)
                        return {"ok": True}

                self.client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(
                        completions=FakeCompletions(self.calls)
                    )
                )

            def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
                params = {
                    "model": "moonshot/kimi2.5",
                    "messages": messages,
                }
                params.update(kwargs)
                return self.client.chat.completions.create(**params)

        fake_openai_module.OpenAILLM = FakeOpenAILLM

        sys.modules.pop("core.mem0_compat", None)

        with patch.dict(sys.modules, {"mem0.llms.openai": fake_openai_module}), patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "or-key", "REASONING_ENABLED": "true"},
            clear=False,
        ):
            import core.mem0_compat as compat

            importlib.reload(compat)
            compat.apply_mem0_openrouter_reasoning_config_patch()
            llm = fake_openai_module.OpenAILLM()
            llm.generate_response([{"role": "user", "content": "hello"}])

        self.assertEqual(len(llm.calls), 1)
        self.assertNotIn("extra_body", llm.calls[0])


if __name__ == "__main__":
    unittest.main()
