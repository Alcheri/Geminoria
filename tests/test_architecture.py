import importlib
from pathlib import Path
import re
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .. import plugin
from ..config.config_runtime import RuntimeConfig
from ..core.core import (
    _OUT_OF_SCOPE_REPLY,
    GeminoriaCore,
    gemversion_reply_text,
)
from ..core.services import AsyncGeminiService
from ..state.cache import CacheRepository, cache_key, normalize_query
from ..state.memory import MemoryStore


class ConfigRuntimeTestCase(unittest.TestCase):
    def test_runtime_config_coerces_defaults(self):
        cfg = RuntimeConfig(
            progress_indicator_style="unknown",
            history_tools_channel_allowlist=None,
        )
        self.assertEqual(cfg.progress_indicator_style, "dots")
        self.assertIn("progress_indicator_enabled", cfg)
        self.assertEqual(cfg["model"], "gemini-3-flash-preview")

    def test_runtime_config_clamps_numeric_bounds(self):
        cfg = RuntimeConfig(
            max_results=0,
            buffer_size=0,
            max_rounds=0,
            cooldown_seconds=-5,
            max_concurrent_per_channel=0,
            cache_ttl_seconds=-1,
            cache_max_entries=0,
            cache_min_query_length=0,
            cache_fuzzy_min_score=101,
        )

        self.assertEqual(cfg.max_results, 1)
        self.assertEqual(cfg.buffer_size, 1)
        self.assertEqual(cfg.max_rounds, 1)
        self.assertEqual(cfg.cooldown_seconds, 0)
        self.assertEqual(cfg.max_concurrent_per_channel, 1)
        self.assertEqual(cfg.cache_ttl_seconds, 0)
        self.assertEqual(cfg.cache_max_entries, 1)
        self.assertEqual(cfg.cache_min_query_length, 1)
        self.assertEqual(cfg.cache_fuzzy_min_score, 100)


class MemoryStoreTestCase(unittest.TestCase):
    def test_history_buffers_are_isolated_by_network_and_channel(self):
        store = MemoryStore()
        url_re = re.compile(r"https?://\S+", re.IGNORECASE)
        store.add_message(
            "#Borg",
            "alice",
            "limnoria dalnet note https://dalnet.example/item",
            10,
            url_re,
            network="DALnet",
        )
        store.add_message(
            "#Borg",
            "bob",
            "limnoria libera note https://libera.example/item",
            10,
            url_re,
            network="Libera",
        )

        dalnet = store.search_last("#Borg", "limnoria", 5, network="DALnet")
        libera = store.search_last("#Borg", "limnoria", 5, network="Libera")
        dalnet_urls = store.search_urls(
            "#Borg", "example", 5, network="DALnet"
        )
        libera_urls = store.search_urls(
            "#Borg", "example", 5, network="Libera"
        )

        self.assertIn("dalnet note", dalnet)
        self.assertNotIn("libera note", dalnet)
        self.assertIn("libera note", libera)
        self.assertNotIn("dalnet note", libera)
        self.assertIn("dalnet.example", dalnet_urls)
        self.assertNotIn("libera.example", dalnet_urls)
        self.assertIn("libera.example", libera_urls)
        self.assertNotIn("dalnet.example", libera_urls)

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

    def test_request_slot_inflight_limit_is_network_scoped(self):
        store = MemoryStore()
        first = store.acquire_request_slot(
            prefix="nick1!user@host",
            channel="#Borg",
            network="DALnet",
            cooldown_seconds=0,
            max_concurrent_per_channel=1,
        )
        second_same_network = store.acquire_request_slot(
            prefix="nick2!user@host",
            channel="#Borg",
            network="DALnet",
            cooldown_seconds=0,
            max_concurrent_per_channel=1,
        )
        second_other_network = store.acquire_request_slot(
            prefix="nick3!user@host",
            channel="#Borg",
            network="Libera",
            cooldown_seconds=0,
            max_concurrent_per_channel=1,
        )

        self.assertIsNone(first)
        self.assertIn("busy", second_same_network)
        self.assertIsNone(second_other_network)
        store.release_request_slot("#Borg", network="DALnet")
        store.release_request_slot("#Borg", network="Libera")


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

        with patch(
            "Geminoria.core.services._build_client", return_value=FakeClient()
        ):
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

    def test_async_service_timeout_does_not_sync_fallback(self):
        class FakeModels:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                time.sleep(0.05)
                return {"ok": True, "model": kwargs["model"]}

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        fake_client = FakeClient()
        with patch(
            "Geminoria.core.services._build_client",
            return_value=fake_client,
        ):
            svc = AsyncGeminiService()
            try:
                with self.assertRaises(TimeoutError):
                    svc.generate_content(
                        api_key="k",
                        model="gemini-test",
                        contents=[],
                        config=None,
                        timeout_s=0.001,
                    )
                self.assertEqual(fake_client.models.calls, 1)
            finally:
                svc.close()

    def test_async_service_recovers_after_timeout(self):
        class FakeModels:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    time.sleep(0.05)
                return {"ok": True, "model": kwargs["model"]}

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        fake_client = FakeClient()
        with patch(
            "Geminoria.core.services._build_client",
            return_value=fake_client,
        ):
            svc = AsyncGeminiService()
            try:
                with self.assertRaises(TimeoutError):
                    svc.generate_content(
                        api_key="k",
                        model="gemini-test",
                        contents=[],
                        config=None,
                        timeout_s=0.001,
                    )
                out = svc.generate_content(
                    api_key="k",
                    model="gemini-test",
                    contents=[],
                    config=None,
                    timeout_s=5,
                )
                self.assertEqual(out["model"], "gemini-test")
                self.assertGreaterEqual(fake_client.models.calls, 2)
            finally:
                svc.close()


class CoreCompatibilityTestCase(unittest.TestCase):
    def test_plugin_check_owner_falls_back_when_core_helper_is_missing(self):
        geminoria = plugin.Geminoria.__new__(plugin.Geminoria)
        geminoria._core = object()
        msg = SimpleNamespace(prefix="owner!ident@transient.example")
        user = MagicMock()
        user._checkCapability.return_value = True

        with patch.object(
            plugin.ircdb,
            "users",
            SimpleNamespace(getUser=MagicMock(return_value=user)),
        ):
            with patch.object(
                plugin.ircdb, "checkCapability", return_value=False
            ) as check_capability:
                self.assertTrue(geminoria._check_owner(msg))

        user._checkCapability.assert_called_once_with("owner")
        check_capability.assert_not_called()

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
            cfg = RuntimeConfig(
                cache_min_query_length=1, cache_prefix_hits=True
            )
            core.load_cfg = lambda: cfg
            core._cache.store(
                cfg,
                network="DALnet",
                channel="#ops",
                model=cfg.model,
                allow_search_last=True,
                allow_search_urls=True,
                query="limnoria hello",
                response="cached response",
            )
            answer = core.handle_query(
                FakeIrc(), msg, "limnoria hello", emit_progress=lambda: None
            )
            self.assertTrue(answer.startswith("[cached]"))

    def test_core_history_tools_use_irc_network(self):
        class FakeService:
            def close(self):
                return None

        class FakeIrc:
            callbacks = []

            def __init__(self, network):
                self.network = network

            @staticmethod
            def isChannel(value):
                return bool(value and value.startswith("#"))

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            core = GeminoriaCore(
                cache_db_path=tmp.name,
                service=FakeService(),
                channel_flag_getter=lambda key, channel, network: True,
            )
            cfg = RuntimeConfig(buffer_size=10)
            core.on_privmsg(
                FakeIrc("DALnet"),
                SimpleNamespace(
                    nick="alice",
                    args=[
                        "#Borg",
                        "limnoria dalnet note https://dalnet.example/item",
                    ],
                ),
                cfg,
            )
            core.on_privmsg(
                FakeIrc("Libera"),
                SimpleNamespace(
                    nick="bob",
                    args=[
                        "#Borg",
                        "limnoria libera note https://libera.example/item",
                    ],
                ),
                cfg,
            )

            dalnet = core._execute_tool(
                irc=FakeIrc("DALnet"),
                channel="#Borg",
                fn="search_last",
                tool_args={"text": "limnoria"},
                limit=5,
                allow_search_last=True,
                allow_search_urls=True,
            )
            libera_urls = core._execute_tool(
                irc=FakeIrc("Libera"),
                channel="#Borg",
                fn="search_urls",
                tool_args={"word": "example"},
                limit=5,
                allow_search_last=True,
                allow_search_urls=True,
            )

        self.assertIn("dalnet note", dalnet)
        self.assertNotIn("libera note", dalnet)
        self.assertIn("libera.example", libera_urls)
        self.assertNotIn("dalnet.example", libera_urls)

    def test_gemversion_text_format(self):
        text = gemversion_reply_text()
        self.assertIn("Geminoria version:", text)
        self.assertIn("| model:", text)

    def test_core_rejects_out_of_scope_query_without_calling_service(self):
        class FakeService:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(candidates=[], text="should not run")

            def close(self):
                return None

        class FakeIrc:
            network = "DALnet"
            callbacks = []

            @staticmethod
            def isChannel(value):
                return bool(value and value.startswith("#"))

        msg = SimpleNamespace(prefix="nick!u@h", args=["#ops", "hello"])
        service = FakeService()

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            core = GeminoriaCore(
                cache_db_path=tmp.name,
                service=service,
                channel_flag_getter=lambda key, channel, network: True,
            )
            answer = core.handle_query(
                FakeIrc(), msg, "explain tsunamis", emit_progress=lambda: None
            )

        self.assertEqual(answer, _OUT_OF_SCOPE_REPLY)
        self.assertEqual(service.calls, 0)

    def test_core_allows_limnoria_scoped_query(self):
        class FakeService:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    candidates=[], text="Use @config search flood"
                )

            def close(self):
                return None

        class FakeIrc:
            network = "DALnet"
            callbacks = []

            @staticmethod
            def isChannel(value):
                return bool(value and value.startswith("#"))

        msg = SimpleNamespace(prefix="nick!u@h", args=["#ops", "hello"])
        service = FakeService()

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            core = GeminoriaCore(
                cache_db_path=tmp.name,
                service=service,
                channel_flag_getter=lambda key, channel, network: True,
            )
            cfg = RuntimeConfig(cache_enabled=False, api_key="k")
            core.load_cfg = lambda: cfg
            answer = core.handle_query(
                FakeIrc(),
                msg,
                "what limnoria config controls flood protection?",
                emit_progress=lambda: None,
            )

        self.assertEqual(answer, "Use @config search flood")
        self.assertEqual(service.calls, 1)

    def test_max_reply_chars_zero_disables_truncation(self):
        class FakeService:
            def close(self):
                return None

        class FakeIrc:
            network = "DALnet"
            callbacks = []

            @staticmethod
            def isChannel(value):
                return bool(value and value.startswith("#"))

        msg = SimpleNamespace(prefix="nick!u@h", args=["#ops", "hello"])
        long_answer = "x" * 500

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            core = GeminoriaCore(
                cache_db_path=tmp.name,
                service=FakeService(),
                channel_flag_getter=lambda key, channel, network: True,
            )
            cfg = RuntimeConfig(
                cache_min_query_length=1,
                cache_prefix_hits=False,
                max_reply_chars=0,
            )
            core.load_cfg = lambda: cfg
            core._cache.store(
                cfg,
                network="DALnet",
                channel="#ops",
                model=cfg.model,
                allow_search_last=True,
                allow_search_urls=True,
                query="limnoria long response",
                response=long_answer,
            )
            answer = core.handle_query(
                FakeIrc(),
                msg,
                "limnoria long response",
                emit_progress=lambda: None,
            )

        self.assertEqual(answer, long_answer)


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
        self.assertIsNotNone(
            importlib.import_module("Geminoria.config.config_runtime")
        )
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
