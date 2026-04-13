###
# Copyright © MMXXVI, Barry Suridge
# All rights reserved.
###

import unittest
import sqlite3
import time
from unittest.mock import patch

from .. import plugin


class GeminoriaSmokeTestCase(unittest.TestCase):
    def test_plugin_module_exports_class(self):
        self.assertTrue(hasattr(plugin, "Class"))

    def test_gemversion_reply_text_includes_configured_model(self):
        with patch.object(
            plugin, "_get_cfg", return_value={"model": "gemini-test-model"}
        ):
            text = plugin._gemversion_reply_text()
        self.assertIn(f"Geminoria version: {plugin.PLUGIN_VERSION}", text)
        self.assertIn("model: gemini-test-model", text)


class GeminoriaCacheHelperTestCase(unittest.TestCase):
    def test_normalize_query(self):
        self.assertEqual(
            plugin._normalize_query("  What are Flood-Protection options?!  "),
            "what are flood protection options",
        )

    def test_similarity_score(self):
        left = plugin._normalize_query("show me flood command settings")
        right = plugin._normalize_query("flood command settings please")
        self.assertGreaterEqual(plugin._similarity_score(left, right), 50)

    def test_cache_key_changes_with_context(self):
        query_norm = plugin._normalize_query("how do i set max flood")
        key_a = plugin._cache_key(
            query_norm,
            network="DALnet",
            channel="#ops",
            model="gemini-3-flash-preview",
            allow_search_last=True,
            allow_search_urls=True,
        )
        key_b = plugin._cache_key(
            query_norm,
            network="DALnet",
            channel="#ops",
            model="gemini-3-flash-preview",
            allow_search_last=False,
            allow_search_urls=True,
        )
        self.assertNotEqual(key_a, key_b)

    def test_fts_bm25_requires_table_name_not_alias(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            conn.execute("INSERT INTO t(x) VALUES ('hello world')")

            with self.assertRaises(sqlite3.OperationalError) as alias_ctx:
                conn.execute("""
                    SELECT rowid
                    FROM t AS f
                    WHERE t MATCH 'hello'
                    ORDER BY bm25(f)
                    """).fetchall()
            self.assertIn("no such column: f", str(alias_ctx.exception))

            rows = conn.execute("""
                SELECT rowid
                FROM t AS f
                WHERE t MATCH 'hello'
                ORDER BY bm25(t)
                """).fetchall()
            self.assertEqual(rows, [(1,)])
        finally:
            conn.close()


class GeminoriaProgressConfigTestCase(unittest.TestCase):
    def test_get_cfg_includes_progress_keys(self):
        cfg = plugin._get_cfg()
        self.assertIn("progress_indicator_enabled", cfg)
        self.assertIn("progress_indicator_delay_ms", cfg)
        self.assertIn("progress_indicator_style", cfg)
        self.assertIn("progress_indicator_message", cfg)
        self.assertGreaterEqual(int(cfg["progress_indicator_delay_ms"]), 0)
        self.assertIn(
            cfg["progress_indicator_style"],
            ("dots", "plain"),
        )

    def test_progress_style_fallback(self):
        self.assertEqual(plugin._normalized_progress_style("plain"), "plain")
        self.assertEqual(plugin._normalized_progress_style("dots"), "dots")
        self.assertEqual(plugin._normalized_progress_style("unknown-style"), "dots")


class GeminoriaProgressIndicatorTestCase(unittest.TestCase):
    def test_fast_run_does_not_emit_indicator(self):
        events = []
        value = plugin._run_with_delayed_indicator(
            run_fn=lambda: "ok",
            indicator_fn=lambda: events.append("indicator"),
            delay_ms=50,
        )
        self.assertEqual(value, "ok")
        self.assertEqual(events, [])

    def test_slow_run_emits_indicator_once(self):
        events = []

        def slow_run():
            time.sleep(0.06)
            return "done"

        value = plugin._run_with_delayed_indicator(
            run_fn=slow_run,
            indicator_fn=lambda: events.append("indicator"),
            delay_ms=10,
        )
        self.assertEqual(value, "done")
        self.assertEqual(events, ["indicator"])

    def test_error_before_delay_does_not_emit_indicator(self):
        events = []

        def fail_fast():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            plugin._run_with_delayed_indicator(
                run_fn=fail_fast,
                indicator_fn=lambda: events.append("indicator"),
                delay_ms=50,
            )
        self.assertEqual(events, [])

    def test_disable_ansi_progress_text_has_no_irc_formatting(self):
        text = plugin._progress_indicator_text(
            {
                "disable_ansi": True,
                "progress_indicator_style": "dots",
                "progress_indicator_message": "",
            }
        )
        self.assertEqual(text, "Geminoria is thinking ...")
        self.assertNotIn("\x03", text)
        self.assertNotIn("\x02", text)


# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
