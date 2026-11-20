"""Embeddings through Gemini's own API, so callers can set the task type (query or document).

Gemini's OpenAI-compatible endpoint has no task type, and gemini-embedding-001 gives different
vectors for RETRIEVAL_QUERY and RETRIEVAL_DOCUMENT. An index built with one and searched with
the other ranks worse, so the gateway calls batchEmbedContents directly.
"""

import asyncio
import math
from typing import Any, Protocol

import httpx

from gateway.errors import GatewayError
from gateway.providers import RETRY_STATUS, _client_status, _upstream_message

TASK_TYPES = {
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}
MAX_INPUTS = 100  # batchEmbedContents takes at most 100 requests
MAX_INPUT_CHARS = 8_000  # gemini-embedding-001 reads 2,048 tokens per input; longer text is cut upstream
MAX_DIMENSIONS = 3_072


class Embedder(Protocol):
    name: str

    async def embed(
        self, model: str, texts: list[str], *, task_type: str | None, dimensions: int | None
    ) -> tuple[list[list[float]], int | None]:
        """Vectors in input order, and the input token count when the provider reports one."""
        ...


class GeminiEmbedder:
    name = "gemini"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float,
        retries: int,
        client: httpx.AsyncClient | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._headers = {"x-goog-api-key": api_key}
        self._retries = retries
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def embed(self, model, texts, *, task_type, dimensions):
        requests: list[dict[str, Any]] = []
        for text in texts:
            item: dict[str, Any] = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
            if task_type:
                item["taskType"] = task_type
            if dimensions:
                item["outputDimensionality"] = dimensions
            requests.append(item)
        url = f"{self._base}/models/{model}:batchEmbedContents"
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await self._client.post(url, json={"requests": requests}, headers=self._headers)
            except httpx.TransportError as exc:
                last = exc
            else:
                if resp.status_code < 400:
                    data = resp.json()
                    vectors = [e.get("values") or [] for e in data.get("embeddings") or []]
                    if len(vectors) != len(texts):
                        raise GatewayError(502, "upstream_error", "Gemini returned the wrong number of embeddings")
                    usage = data.get("usageMetadata") or {}
                    tokens = usage.get("promptTokenCount")
                    return vectors, int(tokens) if isinstance(tokens, int | float) else None
                if resp.status_code not in RETRY_STATUS:
                    raise GatewayError(_client_status(resp.status_code), "upstream_error", _upstream_message(resp))
                last = GatewayError(502, "upstream_unavailable", _upstream_message(resp))
            if attempt < self._retries:
                await asyncio.sleep(0.5 * 2**attempt)
        if isinstance(last, GatewayError):
            raise last
        raise GatewayError(504, "upstream_timeout", f"gemini did not answer: {last!r}")

    async def aclose(self) -> None:
        await self._client.aclose()


def estimate_tokens(texts: list[str]) -> int:
    """A rough count for when Gemini does not report one: 2 characters per token.

    English runs at about 4 characters per token and Thai at fewer, so this errs high,
    which is the safe side for the budget cap.
    """
    return sum(math.ceil(len(t) / 2) for t in texts)


def parse_request(body: dict[str, Any]) -> tuple[list[str], str | None, int | None]:
    """Validate an OpenAI-style embeddings body plus the gateway's task_type extension."""
    raw = body.get("input")
    texts = [raw] if isinstance(raw, str) else raw
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise GatewayError(400, "invalid_input", "input must be a non-empty string or a list of non-empty strings")
    if len(texts) > MAX_INPUTS:
        raise GatewayError(400, "too_many_inputs", f"Send at most {MAX_INPUTS} inputs per request")
    if any(len(t) > MAX_INPUT_CHARS for t in texts):
        raise GatewayError(400, "input_too_long", f"Each input must be at most {MAX_INPUT_CHARS} characters")
    if body.get("encoding_format") not in (None, "float"):
        raise GatewayError(400, "unsupported_encoding", "Only encoding_format 'float' is supported")
    task_type = body.get("task_type")
    if task_type is not None:
        if not isinstance(task_type, str) or task_type.upper() not in TASK_TYPES:
            raise GatewayError(400, "invalid_task_type", f"task_type must be one of {', '.join(sorted(TASK_TYPES))}")
        task_type = task_type.upper()
    dimensions = body.get("dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int) or isinstance(dimensions, bool) or not 1 <= dimensions <= MAX_DIMENSIONS
    ):
        raise GatewayError(400, "invalid_dimensions", f"dimensions must be a whole number from 1 to {MAX_DIMENSIONS}")
    return texts, task_type, dimensions


def build_embedders(settings) -> dict[str, Embedder]:
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        return {}
    return {
        "gemini": GeminiEmbedder(
            settings.gemini_native_base_url,
            key,
            timeout=settings.upstream_timeout_seconds,
            retries=settings.upstream_retries,
        )
    }
