"""Back-end for any OpenAI-compatible chat endpoint (Assignment 4 addition).

Used here for Google's Gemini models through their OpenAI-compatible endpoint
(https://generativelanguage.googleapis.com/v1beta/openai/), but it works with any
server that speaks ``/chat/completions`` (vLLM, Ollama, a gateway, the offline
fake server in ``a4/fake_server.py``).

Differences from the upstream ``OpenAILLM``:

* **API failures are never returned as model text.** Upstream returns strings such
  as "Error generating response: ..." which the loop then treats as the model's
  answer, so a rate-limit error is scored as a robustness failure. Here a call that
  still fails after all retries returns a string starting with
  :data:`API_ERROR_PREFIX`, and the analysis scripts count those cycles separately.
  Authentication / permission / not-found errors abort the run instead.
* ``batch_generate`` runs requests concurrently, which the loop runner uses
  automatically (``BatchEvaluationCycle`` detects the method).
* An optional on-disk response cache (JSONL, keyed by every generation parameter)
  makes interrupted runs resumable: with greedy decoding a re-run replays the
  cached answers and only calls the API for the missing ones.
* Token usage is accumulated and written into the results file.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base import BaseLLM

API_ERROR_PREFIX = "<<EVALOOP_API_ERROR>>"

#: Error types that are worth retrying (looked up by class name so that the module
#: can be imported without the ``openai`` package, e.g. in the offline tests).
_RETRYABLE = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ConflictError",
    "SSLError",
    "ConnectError",
    "ReadError",
    "WriteError",
    "RemoteProtocolError",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "ConnectionResetError",
}
_FATAL = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}


def is_api_error(text: Optional[str]) -> bool:
    """True if ``text`` is the sentinel produced for a failed API call."""
    return isinstance(text, str) and text.startswith(API_ERROR_PREFIX)


class FatalAPIError(RuntimeError):
    """Raised for errors that retrying cannot fix (bad key, unknown model, ...)."""


class OpenAICompatibleLLM(BaseLLM):
    """Chat-completions client with concurrency, retries, caching and usage stats."""

    def __init__(self, model_config: Dict[str, Any], client: Any = None):
        super().__init__(model_config)
        self.model_id = model_config["model_id"]
        self.base_url = model_config.get("base_url")
        self.temperature = float(model_config.get("temperature", 0.0))
        self.top_p = float(model_config.get("top_p", 1.0))
        self.max_tokens = int(model_config.get("max_tokens", 4096))
        self.system_prompt = model_config.get("system_prompt") or None
        self.reasoning_effort = model_config.get("reasoning_effort") or None
        self.max_concurrency = max(1, int(model_config.get("max_concurrency", 8)))
        self.max_retries = int(model_config.get("max_retries", 8))
        self.empty_retries = int(model_config.get("empty_retries", 2))
        self.request_timeout = float(model_config.get("request_timeout", 120))
        self.logger = logging.getLogger(__name__)

        self._lock = threading.Lock()
        self.usage = {
            "api_calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "retries": 0,
            "api_errors": 0,
            "empty_responses": 0,
            "finish_reasons": {},
            "reasoning_effort_dropped": False,
        }

        self.cache_path = model_config.get("cache_path")
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry
                    except (json.JSONDecodeError, KeyError):
                        continue  # a torn last line from an interrupted run
            self.logger.info(f"Loaded {len(self._cache)} cached responses from {self.cache_path}")

        if client is None:
            from openai import OpenAI

            api_key = model_config.get("api_key") or os.environ.get(
                model_config.get("api_key_env", "OPENAI_API_KEY"), ""
            )
            client = OpenAI(
                api_key=api_key or "missing-key",
                base_url=self.base_url,
                max_retries=0,
                timeout=self.request_timeout,
            )
        self.client = client

    # ------------------------------------------------------------------ public API
    def generate(self, prompt: str, **kwargs) -> str:
        if self.usage.get("fatal_error"):  # e.g. unknown model: do not send 378 more requests
            raise FatalAPIError(self.usage["fatal_error"])
        key = self._cache_key(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            with self._lock:
                self.usage["cache_hits"] += 1
            return cached["response"]

        text, finish_reason = self._call_with_retries(prompt)
        if not is_api_error(text):
            self._store(key, prompt, text, finish_reason)
        return text

    def batch_generate(self, prompts: List[str]) -> List[str]:
        if self.max_concurrency == 1 or len(prompts) <= 1:
            return [self.generate(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(self.generate, prompts))

    def usage_summary(self) -> Dict[str, Any]:
        with self._lock:
            summary = json.loads(json.dumps(self.usage))
        summary.update(
            model_id=self.model_id,
            base_url=self.base_url,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            system_prompt=self.system_prompt,
        )
        return summary

    # ------------------------------------------------------------------ internals
    def _cache_key(self, prompt: str) -> str:
        material = json.dumps(
            [self.model_id, self.system_prompt, prompt, self.temperature, self.top_p,
             self.max_tokens, self.reasoning_effort],
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _store(self, key: str, prompt: str, response: str, finish_reason: Optional[str]) -> None:
        entry = {"key": key, "response": response, "finish_reason": finish_reason}
        with self._lock:
            self._cache[key] = entry
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request(self, prompt: str):
        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            messages=self._messages(prompt),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self.client.chat.completions.create(**kwargs)

    def _record_usage(self, response: Any, finish_reason: Optional[str]) -> None:
        usage = getattr(response, "usage", None)
        with self._lock:
            self.usage["api_calls"] += 1
            reasons = self.usage["finish_reasons"]
            reasons[str(finish_reason)] = reasons.get(str(finish_reason), 0) + 1
            if usage is not None:
                self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0

    def _call_with_retries(self, prompt: str):
        attempt = 0
        empty_attempts = 0
        last_error: Optional[BaseException] = None
        while attempt <= self.max_retries:
            try:
                response = self._request(prompt)
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                self._record_usage(response, finish_reason)
                if content.strip() or empty_attempts >= self.empty_retries:
                    if not content.strip():
                        with self._lock:
                            self.usage["empty_responses"] += 1
                    return content, finish_reason
                empty_attempts += 1
                continue
            except Exception as error:  # noqa: BLE001 - classified below
                name = type(error).__name__
                message = str(error)
                if name in _FATAL:
                    with self._lock:
                        self.usage["fatal_error"] = f"{name}: {message}"
                    raise FatalAPIError(f"{name}: {message}") from error
                if (name == "BadRequestError" and self.reasoning_effort
                        and any(t in message.lower() for t in ("reasoning", "thinking"))):
                    self.logger.warning(
                        f"{self.model_id} rejected reasoning_effort={self.reasoning_effort}; "
                        "continuing without it"
                    )
                    with self._lock:
                        self.reasoning_effort = None
                        self.usage["reasoning_effort_dropped"] = True
                    continue
                if name not in _RETRYABLE and not _looks_transient(message):
                    last_error = error
                    break
                last_error = error
                attempt += 1
                with self._lock:
                    self.usage["retries"] += 1
                delay = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
                log = self.logger.warning if attempt >= 3 else self.logger.info
                log(f"{name} (attempt {attempt}/{self.max_retries}); sleeping {delay:.1f}s")
                time.sleep(delay)
        with self._lock:
            self.usage["api_errors"] += 1
        self.logger.error(f"API call failed permanently: {last_error}")
        return f"{API_ERROR_PREFIX} {type(last_error).__name__}: {last_error}", None


def _looks_transient(message: str) -> bool:
    message = message.lower()
    return any(token in message for token in (
        "overloaded", "unavailable", "try again", "please retry", "429", "500", "502", "503",
        "504", "resource_exhausted", "deadline", "timeout",
        # network-level failures (TLS resets, dropped connections) are transient too
        "ssl", "tls", "connection", "reset by peer", "eof occurred", "remote end closed",
        "temporarily", "broken pipe",
    ))
