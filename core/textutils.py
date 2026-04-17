"""Text, logging, and progress helper utilities for Geminoria."""

from __future__ import annotations

import re
import threading
from typing import Any

_CFG_KEY_RE = re.compile(r"\bsupybot(?:\.[A-Za-z0-9_-]+)+\b")
_IRC_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|bearer)\b\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)\b(authorization)\s*:\s*bearer\s+\S+"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]


def summarize_for_log(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def redact_sensitive(text: str) -> str:
    if not text:
        return ""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def loggable_text(text: str, cfg: Any, limit: int = 120) -> str:
    if cfg.get("log_sensitive", False):
        return summarize_for_log(text, limit=limit)
    return f"<redacted len={len(text or '')}>"


def loggable_args(args: dict[str, Any], cfg: Any) -> Any:
    if cfg.get("log_sensitive", False):
        return args
    return {"keys": sorted(args.keys())}


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def sanitize_irc_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _IRC_CTRL_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", " | ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def highlight_config_keys(text: str) -> str:
    if not text:
        return ""
    return _CFG_KEY_RE.sub(lambda m: f"\x02{m.group(0)}\x02", text)


def normalized_progress_style(style: str) -> str:
    normalized = str(style or "").strip().lower()
    return normalized if normalized in ("dots", "plain") else "dots"


def progress_indicator_text(cfg: Any) -> str:
    custom = str(cfg.get("progress_indicator_message", "") or "").strip()
    if custom:
        return sanitize_irc_text(custom)

    style = normalized_progress_style(str(cfg.get("progress_indicator_style", "dots")))
    if cfg.get("disable_ansi", False):
        return "Geminoria is thinking ..."

    if style == "plain":
        return "\x0312Geminoria is thinking ...\x0f"
    return "\x0312■\x0306■\x0310■\x0f Geminoria is thinking ..."


def run_with_delayed_indicator(run_fn, indicator_fn, delay_ms: int):
    done = threading.Event()
    result: dict[str, Any] = {"value": None, "error": None}

    def worker() -> None:
        try:
            result["value"] = run_fn()
        except Exception as exc:  # pragma: no cover - passthrough behavior
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="GeminoriaRunWorker", daemon=True)
    thread.start()

    wait_seconds = max(0, int(delay_ms)) / 1000.0
    if not done.wait(wait_seconds):
        indicator_fn()
        done.wait()

    if result["error"] is not None:
        raise result["error"]
    return result["value"]
