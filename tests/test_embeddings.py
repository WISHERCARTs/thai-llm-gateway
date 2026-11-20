import json

import httpx
import pytest

from gateway.embeddings import GeminiEmbedder, estimate_tokens


def test_embeddings_shape_task_type_and_usage(gw):
    r = gw.embed(["หนึ่ง", "สองสาม"], task_type="retrieval_document", dimensions=768)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list" and body["model"] == "gemini-embedding-001"
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][1]["embedding"] == [6.0, 1.0]
    call = gw.embedder.calls[0]
    assert call["task_type"] == "RETRIEVAL_DOCUMENT" and call["dimensions"] == 768
    assert body["usage"]["prompt_tokens"] == estimate_tokens(["หนึ่ง", "สองสาม"])
    record = gw.sink.records[0]
    assert record.model == "gemini-embedding-001" and record.output_tokens == 0 and record.cost_usd > 0
    assert float(r.headers["x-gateway-cost-usd"]) == pytest.approx(record.cost_usd)


def test_reported_tokens_win_over_the_estimate(gw):
    gw.embedder.tokens = 7
    assert gw.embed().json()["usage"]["prompt_tokens"] == 7
    assert gw.span("embeddings gemini-embedding-001").attributes["gateway.tokens_estimated"] is False


def test_pii_is_redacted_before_embedding(gw):
    r = gw.embed("โทร 081-234-5678 ถามเรื่องวัดอรุณ")
    assert r.status_code == 200
    assert "081-234-5678" not in gw.embedder.calls[0]["texts"][0]
    assert r.headers["x-gateway-pii-redacted"] == "1"


@pytest.mark.parametrize(
    "body, code",
    [
        ({"input": []}, "invalid_input"),
        ({"input": ["ok", " "]}, "invalid_input"),
        ({"input": ["x"] * 101}, "too_many_inputs"),
        ({"input": "x" * 8001}, "input_too_long"),
        ({"task_type": "SUMMARY"}, "invalid_task_type"),
        ({"dimensions": 0}, "invalid_dimensions"),
        ({"dimensions": True}, "invalid_dimensions"),
        ({"encoding_format": "base64"}, "unsupported_encoding"),
        ({"model": "text-embedding-3-small"}, "model_not_allowed"),
    ],
)
def test_bad_requests_are_refused_before_the_provider(gw, body, code):
    r = gw.embed(**body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == code
    assert gw.embedder.calls == []


def test_embeddings_need_a_key_and_count_toward_limits(gw, settings):
    assert gw.client.post("/v1/embeddings", json={"model": "gemini-embedding-001", "input": "x"}).status_code == 401
    for _ in range(settings.rate_limit_user_per_minute):
        assert gw.embed(headers={"x-end-user": "1.2.3.4"}).status_code == 200
    assert gw.embed(headers={"x-end-user": "1.2.3.4"}).status_code == 429


def test_session_id_groups_calls_in_langfuse(gw):
    gw.chat(headers={"x-session-id": "run-42"})
    gw.embed(headers={"x-session-id": "run-42"})
    assert gw.span("gateway.chat").attributes["langfuse.session.id"] == "run-42"
    assert gw.span("gateway.embeddings").attributes["langfuse.session.id"] == "run-42"


async def test_gemini_embedder_calls_batch_embed_contents():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = GeminiEmbedder("https://example.test/v1beta", "k", timeout=5, retries=0, client=client)
    vectors, tokens = await embedder.embed(
        "gemini-embedding-001", ["a", "b"], task_type="RETRIEVAL_QUERY", dimensions=768
    )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]] and tokens is None
    assert seen["url"] == "https://example.test/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert seen["key"] == "k"
    first = seen["body"]["requests"][0]
    assert first["taskType"] == "RETRIEVAL_QUERY" and first["outputDimensionality"] == 768
    assert first["model"] == "models/gemini-embedding-001" and first["content"]["parts"][0]["text"] == "a"


async def test_gemini_embedder_retries_then_maps_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = GeminiEmbedder("https://example.test/v1beta", "k", timeout=5, retries=1, client=client)
    from gateway.errors import GatewayError

    with pytest.raises(GatewayError) as err:
        await embedder.embed("gemini-embedding-001", ["a"], task_type=None, dimensions=None)
    assert err.value.status == 502 and len(calls) == 2
