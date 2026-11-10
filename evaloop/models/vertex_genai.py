"""Gemini on Vertex AI through the google-genai SDK (Assignment 4 addition).

Same behaviour as :class:`OpenAICompatibleLLM` (concurrency, retries, API-error
sentinel, resumable cache, usage accounting); only the transport differs. Used when
the course's Google Cloud credit lives on a university account for which AI Studio
(and therefore the Gemini API's OpenAI-compatible endpoint) is disabled.

Authentication, in this order:
  * an API key in ``api_key`` / ``$<api_key_env>`` (Vertex AI API key, "express mode"), or
  * Application Default Credentials (e.g. ``google.colab.auth.authenticate_user()``)
    with ``project`` / ``$GOOGLE_CLOUD_PROJECT`` and ``location`` (default "global").
"""

import os
from types import SimpleNamespace
from typing import Any, Dict

from .openai_compatible import OpenAICompatibleLLM


def _renamed(error: Exception, name: str) -> Exception:
    """Re-raise SDK errors under the class names the retry logic understands."""
    return type(name, (Exception,), {})(str(error))


class VertexGenAILLM(OpenAICompatibleLLM):
    def __init__(self, model_config: Dict[str, Any], client: Any = None):
        if client is None:
            from google import genai

            key = model_config.get("api_key") or os.environ.get(model_config.get("api_key_env") or "", "")
            if key:
                client = genai.Client(vertexai=True, api_key=key)
            else:
                client = genai.Client(
                    vertexai=True,
                    project=model_config.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    location=model_config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                )
        super().__init__(model_config, client=client)

    def _request(self, prompt: str):
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            system_instruction=self.system_prompt,
            thinking_config=(types.ThinkingConfig(thinking_level=self.reasoning_effort.upper())
                             if self.reasoning_effort else None),
        )
        try:
            response = self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)
        except errors.APIError as error:
            code = getattr(error, "code", None)
            if code == 429:
                raise _renamed(error, "RateLimitError") from error
            if code in (401,):
                raise _renamed(error, "AuthenticationError") from error
            if code == 403:
                raise _renamed(error, "PermissionDeniedError") from error
            if code == 404:
                raise _renamed(error, "NotFoundError") from error
            if code == 400:
                raise _renamed(error, "BadRequestError") from error
            if code and code >= 500:
                raise _renamed(error, "InternalServerError") from error
            raise
        return _as_chat_completion(response)


def _as_chat_completion(response: Any):
    """Shape a GenerateContentResponse like an OpenAI chat completion."""
    candidates = getattr(response, "candidates", None) or []
    finish = None
    text = ""
    if candidates:
        candidate = candidates[0]
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None) or str(
            getattr(candidate, "finish_reason", None))
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    meta = getattr(response, "usage_metadata", None)
    usage = SimpleNamespace(
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        completion_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
        + (getattr(meta, "thoughts_token_count", 0) or 0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=getattr(meta, "thoughts_token_count", 0) or 0),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=usage,
    )
