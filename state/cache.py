"""SQLite-backed cache repository and query similarity helpers."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from typing import Optional

import supybot.log as log

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def normalize_query(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_tokens(text: str) -> set[str]:
    return set(_QUERY_TOKEN_RE.findall(normalize_query(text)))


def similarity_score(left: str, right: str) -> int:
    left_tokens = query_tokens(left)
    right_tokens = query_tokens(right)
    if not left_tokens and not right_tokens:
        return 100
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return int((overlap / max(1, union)) * 100)


def cache_key(
    query_norm: str,
    *,
    network: str,
    channel: Optional[str],
    model: str,
    allow_search_last: bool,
    allow_search_urls: bool,
) -> str:
    material = "|".join(
        [
            query_norm,
            str(network or ""),
            str(channel or ""),
            str(model or ""),
            "1" if allow_search_last else "0",
            "1" if allow_search_urls else "0",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class CacheRepository:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._has_fts = False
        self._ready = self._init_db()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def has_fts(self) -> bool:
        return self._has_fts

    def _init_db(self) -> bool:
        try:
            with self._lock:
                conn = sqlite3.connect(self._db_path, timeout=2.0)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS geminoria_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at INTEGER NOT NULL,
                            updated_at INTEGER NOT NULL,
                            last_hit_at INTEGER NOT NULL,
                            hit_count INTEGER NOT NULL DEFAULT 0,
                            network TEXT NOT NULL,
                            channel TEXT NOT NULL,
                            model TEXT NOT NULL,
                            allow_search_last INTEGER NOT NULL,
                            allow_search_urls INTEGER NOT NULL,
                            query_original TEXT NOT NULL,
                            query_norm TEXT NOT NULL,
                            query_hash TEXT NOT NULL UNIQUE,
                            response TEXT NOT NULL
                        )
                        """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_geminoria_cache_updated_at ON geminoria_cache(updated_at)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_geminoria_cache_context ON geminoria_cache(network, channel, model)"
                    )

                    self._has_fts = False
                    fts_tokenizers: list[Optional[str]] = [
                        None,
                        "unicode61",
                        "porter unicode61",
                        "ascii",
                        "unicode61 porter",
                    ]
                    last_fts_exc: Optional[Exception] = None
                    for tokenizer in fts_tokenizers:
                        try:
                            conn.execute("DROP TABLE IF EXISTS geminoria_cache_fts")
                            if tokenizer is None:
                                conn.execute("""
                                    CREATE VIRTUAL TABLE geminoria_cache_fts
                                    USING fts5(
                                        entry_id UNINDEXED,
                                        query_norm,
                                        response
                                    )
                                    """)
                            else:
                                conn.execute(f"""
                                    CREATE VIRTUAL TABLE geminoria_cache_fts
                                    USING fts5(
                                        entry_id UNINDEXED,
                                        query_norm,
                                        response,
                                        tokenize = '{tokenizer}'
                                    )
                                    """)
                            self._has_fts = True
                            log.debug(
                                "Geminoria: FTS5 enabled with tokenizer=%s",
                                tokenizer or "<default>",
                            )
                            break
                        except sqlite3.Error as exc:
                            last_fts_exc = exc

                    if not self._has_fts:
                        log.warning(
                            "Geminoria: FTS5 unavailable; fuzzy cache lookups disabled: %s",
                            last_fts_exc,
                        )
                    conn.commit()
                finally:
                    conn.close()
            return True
        except Exception as exc:
            log.error("Geminoria: cache initialisation failed: %s", exc)
            return False

    def lookup(
        self,
        cfg,
        *,
        network: str,
        channel: Optional[str],
        model: str,
        allow_search_last: bool,
        allow_search_urls: bool,
        query: str,
    ) -> Optional[str]:
        if not self._ready or not cfg.get("cache_enabled", True):
            return None
        if len((query or "").strip()) < max(1, int(cfg["cache_min_query_length"])):
            return None

        query_norm = normalize_query(query)
        if not query_norm:
            return None

        query_hash = cache_key(
            query_norm,
            network=network,
            channel=channel,
            model=model,
            allow_search_last=allow_search_last,
            allow_search_urls=allow_search_urls,
        )
        now = int(time.time())
        ttl_cutoff = now - max(0, int(cfg["cache_ttl_seconds"]))

        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            try:
                exact = conn.execute(
                    """
                    SELECT id, response
                    FROM geminoria_cache
                    WHERE query_hash = ? AND updated_at >= ?
                    LIMIT 1
                    """,
                    (query_hash, ttl_cutoff),
                ).fetchone()
                if exact:
                    conn.execute(
                        """
                        UPDATE geminoria_cache
                        SET hit_count = hit_count + 1, last_hit_at = ?
                        WHERE id = ?
                        """,
                        (now, int(exact[0])),
                    )
                    conn.commit()
                    log.debug("Geminoria: cache hit type=exact")
                    return str(exact[1] or "")

                if not (cfg.get("cache_allow_fuzzy", True) and self._has_fts):
                    return None

                tokens = sorted(query_tokens(query_norm))
                if not tokens:
                    return None
                fts_query = " OR ".join(tokens[:8])

                fuzzy_rows = conn.execute(
                    """
                    SELECT c.id, c.response, c.query_norm
                    FROM geminoria_cache_fts f
                    JOIN geminoria_cache c ON c.id = CAST(f.entry_id AS INTEGER)
                    WHERE f.query_norm MATCH ?
                      AND c.network = ?
                      AND c.channel = ?
                      AND c.model = ?
                      AND c.allow_search_last = ?
                      AND c.allow_search_urls = ?
                      AND c.updated_at >= ?
                    ORDER BY bm25(geminoria_cache_fts)
                    LIMIT 12
                    """,
                    (
                        fts_query,
                        str(network or ""),
                        str(channel or ""),
                        str(model or ""),
                        1 if allow_search_last else 0,
                        1 if allow_search_urls else 0,
                        ttl_cutoff,
                    ),
                ).fetchall()

                best = None
                best_score = -1
                for row_id, response, cached_query_norm in fuzzy_rows:
                    score = similarity_score(query_norm, str(cached_query_norm or ""))
                    if score > best_score:
                        best = (int(row_id), str(response or ""))
                        best_score = score

                if not best:
                    return None
                min_score = max(0, min(100, int(cfg["cache_fuzzy_min_score"])))
                if best_score < min_score:
                    return None

                conn.execute(
                    """
                    UPDATE geminoria_cache
                    SET hit_count = hit_count + 1, last_hit_at = ?
                    WHERE id = ?
                    """,
                    (now, best[0]),
                )
                conn.commit()
                log.debug(
                    "Geminoria: cache hit type=fuzzy score=%s threshold=%s",
                    best_score,
                    min_score,
                )
                return best[1]
            except Exception as exc:
                log.warning("Geminoria: cache lookup failed: %s", exc)
                return None
            finally:
                conn.close()

    def store(
        self,
        cfg,
        *,
        network: str,
        channel: Optional[str],
        model: str,
        allow_search_last: bool,
        allow_search_urls: bool,
        query: str,
        response: str,
    ) -> None:
        if not self._ready or not cfg.get("cache_enabled", True):
            return
        if len((query or "").strip()) < max(1, int(cfg["cache_min_query_length"])):
            return
        if not response:
            return
        if response.startswith("Gemini error:") or response.startswith("Geminoria:"):
            return
        if response.startswith("No answer produced"):
            return

        query_norm = normalize_query(query)
        if not query_norm:
            return

        query_hash = cache_key(
            query_norm,
            network=network,
            channel=channel,
            model=model,
            allow_search_last=allow_search_last,
            allow_search_urls=allow_search_urls,
        )
        now = int(time.time())

        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            try:
                row = conn.execute(
                    "SELECT id FROM geminoria_cache WHERE query_hash = ? LIMIT 1",
                    (query_hash,),
                ).fetchone()
                if row:
                    row_id = int(row[0])
                    conn.execute(
                        """
                        UPDATE geminoria_cache
                        SET updated_at = ?,
                            network = ?,
                            channel = ?,
                            model = ?,
                            allow_search_last = ?,
                            allow_search_urls = ?,
                            query_original = ?,
                            query_norm = ?,
                            response = ?
                        WHERE id = ?
                        """,
                        (
                            now,
                            str(network or ""),
                            str(channel or ""),
                            str(model or ""),
                            1 if allow_search_last else 0,
                            1 if allow_search_urls else 0,
                            query,
                            query_norm,
                            response,
                            row_id,
                        ),
                    )
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO geminoria_cache (
                            created_at, updated_at, last_hit_at, hit_count,
                            network, channel, model, allow_search_last, allow_search_urls,
                            query_original, query_norm, query_hash, response
                        )
                        VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now,
                            now,
                            now,
                            str(network or ""),
                            str(channel or ""),
                            str(model or ""),
                            1 if allow_search_last else 0,
                            1 if allow_search_urls else 0,
                            query,
                            query_norm,
                            query_hash,
                            response,
                        ),
                    )
                    lastrowid = cur.lastrowid
                    if lastrowid is None:
                        raise RuntimeError("cache insert did not return lastrowid")
                    row_id = lastrowid

                if self._has_fts:
                    conn.execute(
                        "DELETE FROM geminoria_cache_fts WHERE entry_id = ?",
                        (str(row_id),),
                    )
                    conn.execute(
                        """
                        INSERT INTO geminoria_cache_fts (entry_id, query_norm, response)
                        VALUES (?, ?, ?)
                        """,
                        (str(row_id), query_norm, response),
                    )
                conn.commit()
                self._prune(conn, max(1, int(cfg["cache_max_entries"])))
            except Exception as exc:
                log.warning("Geminoria: cache store failed: %s", exc)
            finally:
                conn.close()

    def _prune(self, conn: sqlite3.Connection, max_entries: int) -> None:
        try:
            row = conn.execute("SELECT COUNT(*) FROM geminoria_cache").fetchone()
            total = int(row[0]) if row else 0
            if total <= max_entries:
                return
            overflow = total - max_entries
            stale_ids = [
                int(r[0])
                for r in conn.execute(
                    """
                    SELECT id
                    FROM geminoria_cache
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    (overflow,),
                ).fetchall()
            ]
            if not stale_ids:
                return
            placeholders = ",".join("?" for _ in stale_ids)
            conn.execute(
                f"DELETE FROM geminoria_cache WHERE id IN ({placeholders})", stale_ids
            )
            if self._has_fts:
                conn.execute(
                    f"DELETE FROM geminoria_cache_fts WHERE entry_id IN ({placeholders})",
                    [str(i) for i in stale_ids],
                )
            conn.commit()
        except Exception as exc:
            log.warning("Geminoria: cache prune failed: %s", exc)

    def stats(self, cfg) -> str:
        if not self._ready:
            return "Geminoria cache is unavailable."

        ttl_seconds = max(0, int(cfg.get("cache_ttl_seconds", 0)))
        now = int(time.time())
        ttl_cutoff = now - ttl_seconds
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            try:
                total_row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM geminoria_cache"
                ).fetchone()
                recent_row = conn.execute(
                    "SELECT COUNT(*) FROM geminoria_cache WHERE updated_at >= ?",
                    (ttl_cutoff,),
                ).fetchone()
                time_row = conn.execute("""
                    SELECT MIN(updated_at), MAX(updated_at), MAX(last_hit_at)
                    FROM geminoria_cache
                    """).fetchone()
                total = int(total_row[0]) if total_row else 0
                total_hits = int(total_row[1]) if total_row else 0
                active = int(recent_row[0]) if recent_row else 0
                oldest = int(time_row[0]) if time_row and time_row[0] else 0
                newest = int(time_row[1]) if time_row and time_row[1] else 0
                last_hit = int(time_row[2]) if time_row and time_row[2] else 0
                oldest_age = max(0, now - oldest) if oldest else 0
                newest_age = max(0, now - newest) if newest else 0
                hit_age = max(0, now - last_hit) if last_hit else 0
                return (
                    "gemcache stats | "
                    f"rows={total} active_ttl={active} hits={total_hits} "
                    f"fts={'on' if self._has_fts else 'off'} "
                    f"oldest_age_s={oldest_age} newest_age_s={newest_age} "
                    f"last_hit_age_s={hit_age}"
                )
            except Exception as exc:
                log.warning("Geminoria: cache stats failed: %s", exc)
                return "Unable to read gemcache stats."
            finally:
                conn.close()

    def clear(self) -> tuple[bool, int]:
        if not self._ready:
            return False, 0

        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            try:
                row = conn.execute("SELECT COUNT(*) FROM geminoria_cache").fetchone()
                before = int(row[0]) if row else 0
                conn.execute("DELETE FROM geminoria_cache")
                if self._has_fts:
                    conn.execute("DELETE FROM geminoria_cache_fts")
                conn.commit()
                return True, before
            except Exception as exc:
                log.warning("Geminoria: cache clear failed: %s", exc)
                return False, 0
            finally:
                conn.close()
