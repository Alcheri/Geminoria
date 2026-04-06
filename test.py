###
# Copyright © MMXXVI, Barry Suridge
# All rights reserved.
###

import unittest
import sqlite3

from . import plugin


class GeminoriaSmokeTestCase(unittest.TestCase):
    def test_plugin_module_exports_class(self):
        self.assertTrue(hasattr(plugin, "Class"))


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
                conn.execute(
                    """
                    SELECT rowid
                    FROM t AS f
                    WHERE t MATCH 'hello'
                    ORDER BY bm25(f)
                    """
                ).fetchall()
            self.assertIn("no such column: f", str(alias_ctx.exception))

            rows = conn.execute(
                """
                SELECT rowid
                FROM t AS f
                WHERE t MATCH 'hello'
                ORDER BY bm25(t)
                """
            ).fetchall()
            self.assertEqual(rows, [(1,)])
        finally:
            conn.close()


# vim:set shiftwidth=4 tabstop=4 expandtab textwidth=79:
