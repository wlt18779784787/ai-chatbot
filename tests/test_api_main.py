import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ApiMainPathTests(unittest.TestCase):
    def test_api_main_uses_project_root_for_static_and_templates(self):
        captured = {}

        class FakeStaticFiles:
            def __init__(self, directory):
                captured["static_dir"] = directory

        class FakeFastAPI:
            def __init__(self, *args, **kwargs):
                self.routes = []

            def mount(self, path, app, name=None):
                captured["mount_path"] = path

            def include_router(self, router):
                captured["router"] = router

            def get(self, path, response_class=None):
                def decorator(func):
                    captured["root_handler"] = func
                    return func

                return decorator

        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = FakeFastAPI

        fake_staticfiles = types.ModuleType("fastapi.staticfiles")
        fake_staticfiles.StaticFiles = FakeStaticFiles

        fake_responses = types.ModuleType("fastapi.responses")
        fake_responses.HTMLResponse = object

        fake_core = types.ModuleType("core.chat_core")
        fake_core.chat_core = type("ChatCoreStub", (), {"shutdown": staticmethod(lambda: None)})()

        fake_routes = types.ModuleType("api.routes")
        fake_routes.router = object()

        sys.modules.pop("api.main", None)

        with patch.dict(
            sys.modules,
            {
                "fastapi": fake_fastapi,
                "fastapi.staticfiles": fake_staticfiles,
                "fastapi.responses": fake_responses,
                "core.chat_core": fake_core,
                "api.routes": fake_routes,
            },
        ):
            import api.main as main_module

            importlib.reload(main_module)

        self.assertEqual(captured["mount_path"], "/static")
        self.assertEqual(captured["static_dir"], Path(main_module.BASE_DIR) / "static")
        self.assertEqual(main_module.TEMPLATES_DIR, Path(main_module.BASE_DIR) / "templates")

    def test_chat_route_uses_asyncio_to_thread_with_scope_ids(self):
        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = type("FakeFastAPI", (), {})
        fake_fastapi.APIRouter = lambda: type(
            "Router",
            (),
            {
                "post": staticmethod(lambda *args, **kwargs: (lambda fn: fn)),
                "get": staticmethod(lambda *args, **kwargs: (lambda fn: fn)),
            },
        )()
        fake_fastapi.HTTPException = Exception
        fake_fastapi.Form = lambda default=...: default
        fake_fastapi.Query = lambda default=...: default

        fake_core = types.ModuleType("core.chat_core")
        fake_core.SessionScope = type("SessionScope", (), {})
        fake_core.chat_core = type(
            "ChatCoreStub",
            (),
            {"chat": staticmethod(lambda message, user_id, agent_id: {"success": True})},
        )()

        sys.modules.pop("api.routes", None)

        with patch.dict(sys.modules, {"fastapi": fake_fastapi, "core.chat_core": fake_core}):
            import api.routes as routes_module

            importlib.reload(routes_module)

        calls = {}

        async def run_test():
            async def fake_to_thread(fn, *args, **kwargs):
                calls["fn"] = fn
                calls["args"] = args
                calls["kwargs"] = kwargs
                return {"success": True, "response": "ok"}

            with patch.object(routes_module.asyncio, "to_thread", fake_to_thread):
                result = await routes_module.chat("hello", "alice", "agent-1")
            return result

        result = asyncio.run(run_test())

        self.assertEqual(result["response"], "ok")
        self.assertEqual(calls["args"], ("hello", "alice", "agent-1"))


if __name__ == "__main__":
    unittest.main()
