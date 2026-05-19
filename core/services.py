"""Gemini API services for Geminoria."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

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
        timeout_s: float = 120.0,
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


def _coerce_timeout_seconds(timeout_s: float) -> float:
    try:
        return max(0.001, float(timeout_s))
    except (TypeError, ValueError):
        return 120.0


class AsyncGeminiService(GeminiService):
    """Runs blocking Gemini SDK calls away from Limnoria's IRC thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = self._new_executor()
        self._client: Optional[genai.Client] = None
        self._client_api_key: Optional[str] = None
        self._closed = False

    @staticmethod
    def _new_executor() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="GeminoriaAsyncService",
        )

    def _run_blocking(self, func: Callable[[], Any], timeout_s: float) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("Geminoria async service is closed.")
            executor = self._executor
            future = executor.submit(func)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            self._recover_after_timeout(executor)
            raise TimeoutError("Gemini request timed out.") from exc

    def _generate_content(
        self,
        *,
        client: genai.Client,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> Any:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    def _recover_after_timeout(
        self, timed_out_executor: ThreadPoolExecutor
    ) -> None:
        with self._lock:
            if self._executor is timed_out_executor and not self._closed:
                self._executor = self._new_executor()
                self._client = None
                self._client_api_key = None
        timed_out_executor.shutdown(wait=False, cancel_futures=True)

    def generate_content(
        self,
        *,
        api_key: str,
        model: str,
        contents: list[Any],
        config: Any,
        timeout_s: float = 120.0,
    ) -> Any:
        with self._lock:
            if self._client is None or self._client_api_key != api_key:
                log.debug("Geminoria: refreshing Gemini client from config.")
                self._client = _build_client(api_key)
                self._client_api_key = (
                    api_key if self._client is not None else None
                )
            client = self._client
        if client is None:
            raise RuntimeError(
                "Geminoria: API client unavailable - check supybot.plugins.Geminoria.apiKey."
            )

        return self._run_blocking(
            lambda: self._generate_content(
                client=client,
                model=model,
                contents=contents,
                config=config,
            ),
            timeout_s=_coerce_timeout_seconds(timeout_s),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        executor.shutdown(wait=False, cancel_futures=True)
