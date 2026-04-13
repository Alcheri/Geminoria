# -*- coding: utf-8 -*-
"""Gemini API services for Geminoria."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Optional

import supybot.log as log

try:
    from google import genai
except ImportError as ie:  # pragma: no cover
    raise ImportError(f"Cannot import google-genai: {ie}")


class GeminiService(ABC):
    @abstractmethod
    def generate_content(
        self,
        *,
        api_key: str,
        model: str,
        contents: list[Any],
        config: Any,
        timeout_s: int = 120,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def _build_client(api_key: str) -> Optional[genai.Client]:
    if not api_key:
        log.error("Geminoria: Gemini API key is not configured.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        log.error("Geminoria: failed to initialise Gemini client: %s", exc)
        return None


class AsyncGeminiService(GeminiService):
    """Runs blocking Gemini SDK calls on a dedicated asyncio loop thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="GeminoriaAsyncServiceLoop",
            daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait(timeout=2)
        self._client: Optional[genai.Client] = None
        self._client_api_key: Optional[str] = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _run_coro_threadsafe(self, coro, timeout_s: int):
        if self._loop.is_closed():
            raise RuntimeError("Geminoria async service loop is closed.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError("Gemini request timed out.") from exc

    async def _generate_content_async(
        self, *, model: str, contents: list[Any], config: Any
    ) -> Any:
        if self._client is None:
            raise RuntimeError("Gemini client is unavailable.")
        return await asyncio.to_thread(
            self._client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

    def generate_content(
        self,
        *,
        api_key: str,
        model: str,
        contents: list[Any],
        config: Any,
        timeout_s: int = 120,
    ) -> Any:
        if self._client is None or self._client_api_key != api_key:
            log.debug("Geminoria: refreshing Gemini client from config.")
            self._client = _build_client(api_key)
            self._client_api_key = api_key if self._client is not None else None
        if self._client is None:
            raise RuntimeError(
                "Geminoria: API client unavailable - check supybot.plugins.Geminoria.apiKey."
            )

        try:
            return self._run_coro_threadsafe(
                self._generate_content_async(
                    model=model, contents=contents, config=config
                ),
                timeout_s=max(1, int(timeout_s)),
            )
        except RuntimeError as exc:
            # Compatibility fallback: keep responses flowing even if the async loop stalls.
            log.warning("Geminoria: async service fallback to sync call: %s", exc)
            return self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        if self._loop.is_running():
            log.warning("Geminoria: async service loop did not stop cleanly.")
            return
        self._loop.close()
