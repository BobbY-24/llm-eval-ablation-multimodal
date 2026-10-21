"""Provider adapters: offline mock, OpenAI-compatible chat completions, Gemini.

API keys are read from a named environment variable at request time. They are
never stored on the provider, written to records, placed in URLs, or echoed in
errors. HTTP goes through an injectable ``transport`` so tests run offline.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote, urlsplit

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif"}
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# transport(url, headers, body, timeout) -> (status, response_bytes)
Transport = Callable[[str, dict, bytes, float], tuple[int, bytes]]


@dataclass(frozen=True)
class Generation:
    text: str
    usage: dict | None = None


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never forward credentials elsewhere
        return None


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_RESPONSE_BYTES + 1)


def encode_image(path: str) -> tuple[str, str]:
    """Return (mime_type, base64) for a local image within the size limit."""
    file = Path(path)
    mime = MIME_TYPES.get(file.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image type: {file.suffix!r}")
    size = file.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds {MAX_IMAGE_BYTES} bytes: {file.name}")
    return mime, base64.b64encode(file.read_bytes()).decode("ascii")


def _check_base_url(url: str) -> str:
    parts = urlsplit(url)
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise ValueError("base_url must use https (plain http only for localhost)")
    if parts.username or parts.password or parts.query:
        raise ValueError("base_url must not contain credentials or a query string")
    return url.rstrip("/")


class _HTTPProvider:
    live = True

    def __init__(self, model: str, api_key_env: str, timeout: float = 120.0,
                 transport: Transport | None = None, max_output_tokens: int = 512):
        if not _ENV_NAME.match(api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.transport = transport or urllib_transport
        self.max_output_tokens = max_output_tokens
        self.name = f"{self.kind}:{model}"

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ProviderError(f"Environment variable {self.api_key_env} is not set")
        return key

    def _post(self, url: str, payload: dict) -> dict:
        key = self._key()
        headers = {"Content-Type": "application/json", **self._auth(key)}
        status, body = self.transport(url, headers, json.dumps(payload).encode("utf-8"), self.timeout)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ProviderError("Response exceeded size limit")
        text = body.decode("utf-8", errors="replace")
        if status != 200:
            raise ProviderError(f"HTTP {status}: {text[:300].replace(key, '[redacted]')}", status=status)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Provider returned non-JSON body: {exc}") from None


class OpenAICompatibleProvider(_HTTPProvider):
    """POST {base_url}/chat/completions; works for OpenAI and compatible hosts."""
    kind = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", json_mode: bool = True, **kwargs):
        super().__init__(model, api_key_env, **kwargs)
        self.base_url = _check_base_url(base_url)
        self.json_mode = json_mode

    @staticmethod
    def _auth(key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    def request_payload(self, prompt: str, image_paths: Sequence[str]) -> dict:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            mime, data = encode_image(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": self.max_output_tokens}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(self, prompt: str, image_paths: Sequence[str]) -> Generation:
        payload = self.request_payload(prompt, image_paths)
        data = self._post(f"{self.base_url}/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Response has no choices[0].message.content") from None
        if not isinstance(text, str):
            raise ProviderError("Message content is not a string")
        return Generation(text, data.get("usage"))


class GeminiProvider(_HTTPProvider):
    """POST .../models/{model}:generateContent with the key in a header."""
    kind = "gemini"

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", **kwargs):
        super().__init__(model, api_key_env, **kwargs)
        self.base_url = _check_base_url(base_url)

    @staticmethod
    def _auth(key: str) -> dict:
        return {"x-goog-api-key": key}

    def request_payload(self, prompt: str, image_paths: Sequence[str]) -> dict:
        parts = [{"text": prompt}]
        for path in image_paths:
            mime, data = encode_image(path)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        return {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_output_tokens,
                                     "responseMimeType": "application/json"}}

    def generate(self, prompt: str, image_paths: Sequence[str]) -> Generation:
        payload = self.request_payload(prompt, image_paths)
        url = f"{self.base_url}/models/{quote(self.model, safe='.-_')}:generateContent"
        data = self._post(url, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError):
            reason = (data.get("promptFeedback") or {}).get("blockReason") if isinstance(data, dict) else None
            raise ProviderError(f"Response has no candidate text (blockReason={reason})") from None
        return Generation(text, data.get("usageMetadata"))


class MockProvider:
    """Deterministic plumbing fixture with no semantic competence and no network."""
    kind = "mock"
    live = False

    def __init__(self, model: str = "mock"):
        self.model = model
        self.name = f"mock:{model}"

    def generate(self, prompt: str, image_paths: Sequence[str]) -> Generation:
        digest = hashlib.sha256((prompt + "|".join(image_paths)).encode()).digest()
        if '"selected_option"' in prompt:
            answer = {"selected_option": "ABCD"[digest[0] % 4]}
        else:
            answer = {"predicted_meaning": "Synthetic mock output; no inference was performed."}
        answer.update(confidence=0.0, evidence_used={"text": False, "image": False})
        return Generation(json.dumps(answer), {"synthetic": True})


def make_provider(kind: str, model: str, *, base_url: str | None = None,
                  api_key_env: str | None = None, timeout: float = 120.0,
                  transport: Transport | None = None, max_output_tokens: int = 512):
    if kind == "mock":
        return MockProvider(model)
    options = {"timeout": timeout, "transport": transport, "max_output_tokens": max_output_tokens}
    if api_key_env:
        options["api_key_env"] = api_key_env
    if base_url:
        options["base_url"] = base_url
    if kind == "openai":
        return OpenAICompatibleProvider(model, **options)
    if kind == "gemini":
        return GeminiProvider(model, **options)
    raise ValueError(f"Unknown provider kind: {kind!r}")
