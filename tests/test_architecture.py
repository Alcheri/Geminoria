import importlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..config.config_runtime import RuntimeConfig
from ..core.core import GeminoriaCore, gemversion_reply_text
from ..core.services import AsyncGeminiService
from ..state.cache import CacheRepository, cache_key, normalize_query
from ..state.memory import MemoryStore


class ConfigRuntimeTestCase(unittest.TestCase):
    def test_runtime_config_coerces_defaults(self):
        cfg = RuntimeConfig(
            progress_indicator_style="unknown", history_tools_channel_allowlist=None
        )
        self.assertEqual(cfg.progress_indicator_style, "dots")
        self.assertIn("progress_indicator_enabled", cfg)
        self.assertEqual(cfg["model"], "gemini-3-flash-preview")


class MemoryStoreTestCase(unittest.TestCase):
    def test_request_slot_cooldown_and_release(self):
        store = MemoryStore()
        err = store.acquire_request_slot(
            prefix="nick!user@host",
            channel="#ops",
            cooldown_seconds=10,
            max_concurrent_per_channel=1,
        )
        self.assertIsNone(err)
        err2 = store.acquire_request_slot(
            prefix="nick!user@host",
            channel="#ops",
            cooldown_seconds=10,
            max_concurrent_per_channel=1,
        )
        self.assertIn("Please wait", err2)
        store.release_request_slot("#ops")


class CacheRepositoryTestCase(unittest.TestCase):
    def test_cache_context_isolated_by_model(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            repo = CacheRepository(tmp.name)
            cfg = RuntimeConfig(cache_min_query_length=1)
            repo.store(
                cfg,
                network="DALnet",
                channel="#ops",
                model="m1",
                allow_search_last=True,
                allow_search_urls=True,
                query="flood settings",
                response="A",
            )
            miss = repo.lookup(
                cfg,
                network="DALnet",
                channel="#ops",
                model="m2",
                allow_search_last=True,
                allow_search_urls=True,
                query="flood settings",
            )
            hit = repo.lookup(
                cfg,
                network="DALnet",
                channel="#ops",
                model="m1",
                allow_search_last=True,
                allow_search_urls=True,
                query="flood settings",
            )
            self.assertIsNone(miss)
            self.assertEqual(hit, "A")
            self.assertEqual(
                cache_key(
                    normalize_query("flood settings"),
                    network="DALnet",
                    channel="#ops",
                    model="m1",
                    allow_search_last=True,
                    allow_search_urls=True,
                ),
                cache_key(
                    normalize_query("flood settings"),
                    network="DALnet",
                    channel="#ops",
                    model="m1",
                    allow_search_last=True,
                    allow_search_urls=True,
                ),
            )


class AsyncServiceTestCase(unittest.TestCase):
    def test_async_service_sync_facade(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                return {"ok": True, "model": kwargs["model"]}

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        with patch("Geminoria.core.services._build_client", return_value=FakeClient()):
            svc = AsyncGeminiService()
            try:
                out = svc.generate_content(
                    api_key="k",
                    model="gemini-test",
                    contents=[],
                    config=None,
                    timeout_s=5,
                )
                self.assertEqual(out["model"], "gemini-test")
            finally:
                svc.close()


class CoreCompatibilityTestCase(unittest.TestCase):
    def test_core_handle_query_uses_cache_prefix(self):
        class FakeService:
            def generate_content(self, **kwargs):
                return SimpleNamespace(candidates=[], text="unused")

            def close(self):
                return None

        class FakeIrc:
            network = "DALnet"

            @staticmethod
            def isChannel(value):
                return bool(value and value.startswith("#"))

        msg = SimpleNamespace(prefix="nick!u@h", args=["#ops", "hello"])

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            core = GeminoriaCore(
                cache_db_path=tmp.name,
                service=FakeService(),
                channel_flag_getter=lambda key, channel, network: True,
            )
            cfg = RuntimeConfig(cache_min_query_length=1, cache_prefix_hits=True)
            core.load_cfg = lambda: cfg
            core._cache.store(
                cfg,
                network="DALnet",
                channel="#ops",
                model=cfg.model,
                allow_search_last=True,
                allow_search_urls=True,
                query="hello",
                response="cached response",
            )
            answer = core.handle_query(
                FakeIrc(), msg, "hello", emit_progress=lambda: None
            )
            self.assertTrue(answer.startswith("[cached]"))

    def test_gemversion_text_format(self):
        text = gemversion_reply_text()
        self.assertIn("Geminoria version:", text)
        self.assertIn("| model:", text)


class PackageStructureTestCase(unittest.TestCase):
    def test_expected_layout_files_exist(self):
        root = Path(__file__).resolve().parents[1]
        required = [
            "core/core.py",
            "core/system.py",
            "core/services.py",
            "core/textutils.py",
            "state/cache.py",
            "state/memory.py",
            "config/config.py",
            "config/config_runtime.py",
            "tests/test.py",
            "tests/test_architecture.py",
        ]
        for rel in required:
            self.assertTrue((root / rel).exists(), rel)

    def test_new_package_layout_imports(self):
        self.assertIsNotNone(importlib.import_module("Geminoria.core.core"))
        self.assertIsNotNone(importlib.import_module("Geminoria.state.cache"))
        self.assertIsNotNone(importlib.import_module("Geminoria.config.config_runtime"))
        self.assertIsNotNone(importlib.import_module("Geminoria.tests.test"))

    def test_legacy_import_paths_are_removed_in_phase_two(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.cache")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.memory")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.services")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.system")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.textutils")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("Geminoria.config_runtime")


# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
