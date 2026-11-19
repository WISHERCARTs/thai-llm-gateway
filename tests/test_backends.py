"""Providers and stores against mocked HTTP."""

import httpx
import pytest

from gateway.errors import GatewayError
from gateway.providers import OpenAICompatProvider
from gateway.state import UpstashStore
from gateway.usage import PostgresSink, UsageRecord


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_provider_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={"choices": [], "usage": {}})

    p = OpenAICompatProvider("gemini", "https://x/v1beta/openai", "k", timeout=5, retries=2, client=client(handler))
    assert await p.chat({"model": "m"}) == {"choices": [], "usage": {}}
    assert str(calls[0].url) == "https://x/v1beta/openai/chat/completions"
    assert calls[0].headers["authorization"] == "Bearer k"


async def test_provider_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "bad"}})

    p = OpenAICompatProvider("gemini", "https://x", "k", timeout=5, retries=2, client=client(handler))
    with pytest.raises(GatewayError) as exc:
        await p.chat({})
    assert exc.value.status == 400 and len(calls) == 1


async def test_provider_gives_up(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    p = OpenAICompatProvider(
        "gemini",
        "https://x",
        "k",
        timeout=5,
        retries=1,
        client=client(lambda r: httpx.Response(429, json={"error": "slow down"})),
    )
    with pytest.raises(GatewayError) as exc:
        await p.chat({})
    assert exc.value.status == 502


async def test_upstash_pipeline():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/pipeline":
            return httpx.Response(200, json=[{"result": None}, {"result": 3}])
        return httpx.Response(200, json={"result": "7"})

    store = UpstashStore("https://up.example", "tok", client=client(handler))
    assert await store.incr("k", 1, 60) == 3
    assert await store.get("k") == 7
    assert seen[0].headers["authorization"] == "Bearer tok"
    assert b'"SET","k","0","EX","60","NX"' in seen[0].content


async def test_postgres_sink_swallows_errors():
    sink = PostgresSink("postgresql://nobody@127.0.0.1:1/none")
    await sink.write(UsageRecord("r", "c", None, "gemini", "m", 200))  # logged, not raised


async def _no_sleep(_):
    return None


async def test_gemini_list_error_and_rejected_key():
    body = [{"error": {"code": 401, "message": "API key not valid"}}]
    p = OpenAICompatProvider(
        "gemini", "https://x", "k", timeout=5, retries=0, client=client(lambda r: httpx.Response(401, json=body))
    )
    with pytest.raises(GatewayError) as exc:
        await p.chat({})
    assert exc.value.status == 502 and "API key not valid" in exc.value.message
