"""Upstream model providers. Gemini and Ollama both expose an OpenAI-compatible chat endpoint."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from gateway.config import Settings
from gateway.errors import GatewayError

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Route:
    provider: str  # "gemini" | "ollama"
    upstream_model: str


class Provider(Protocol):
    name: str

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]: ...


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float,
        retries: int,
        client: httpx.AsyncClient | None = None,
    ):
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._retries = retries
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.TransportError as exc:  # timeouts, connection resets
                last = exc
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRY_STATUS:
                    raise GatewayError(_client_status(resp.status_code), "upstream_error", _upstream_message(resp))
                last = GatewayError(502, "upstream_unavailable", _upstream_message(resp))
            if attempt < self._retries:
                await asyncio.sleep(0.5 * 2**attempt)
        if isinstance(last, GatewayError):
            raise last
        raise GatewayError(504, "upstream_timeout", f"{self.name} did not answer: {last!r}")

    async def aclose(self) -> None:
        await self._client.aclose()


def _upstream_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return f"upstream returned {resp.status_code}: {resp.text[:200]}"
    if isinstance(data, list) and data:  # Gemini sometimes wraps the error in a list
        data = data[0]
    err = data.get("error", data) if isinstance(data, dict) else data
    msg = err.get("message") if isinstance(err, dict) else str(err)
    return f"upstream returned {resp.status_code}: {str(msg)[:300]}"


def _client_status(upstream: int) -> int:
    """A bad request is the caller's fault; a rejected key or anything else is the gateway's (502)."""
    return 400 if upstream in (400, 404, 413, 422) else 502


def resolve(model: str, settings: Settings) -> Route:
    """Map the model a client asked for to a provider. Unknown models are refused, not guessed."""
    if model.startswith("ollama/"):
        if not settings.ollama_enabled:
            raise GatewayError(400, "model_not_allowed", "Ollama models are turned off on this gateway")
        return Route("ollama", model.removeprefix("ollama/"))
    name = model.removeprefix("gemini/").removeprefix("models/")
    if name in settings.gemini_models:
        return Route("gemini", name)
    raise GatewayError(400, "model_not_allowed", f"model {model!r} is not served by this gateway")


def build_providers(settings: Settings) -> dict[str, Provider]:
    common = {"timeout": settings.upstream_timeout_seconds, "retries": settings.upstream_retries}
    providers: dict[str, Provider] = {}
    key = settings.gemini_api_key.get_secret_value()
    if key:
        providers["gemini"] = OpenAICompatProvider("gemini", settings.gemini_base_url, key, **common)
    if settings.ollama_enabled:
        providers["ollama"] = OpenAICompatProvider("ollama", settings.ollama_base_url, "", **common)
    return providers
