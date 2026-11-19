import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.app import create_app
from gateway.config import Settings
from gateway.state import MemoryStore
from gateway.usage import MemorySink


class FakeProvider:
    """Answers with a scripted reply and records what it was sent."""

    def __init__(self, name: str = "gemini", reply: str = "สวัสดีครับ", tokens=(12, 8)):
        self.name = name
        self.reply = reply
        self.tokens = tokens
        self.calls: list[dict] = []

    async def chat(self, body):
        self.calls.append(body)
        reply = self.reply(body) if callable(self.reply) else self.reply
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": self.tokens[0], "completion_tokens": self.tokens[1]},
        }


KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def settings():
    return Settings(_env_file=None, gateway_client_keys=f"tests:{KEY},other:other-key")


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def gw(settings, fake):
    """Build an app with in-memory state, a fake provider and captured spans."""

    class Gateway:
        def __init__(self):
            from fastapi.testclient import TestClient

            self.store = MemoryStore()
            self.sink = MemorySink()
            self.spans = InMemorySpanExporter()
            self.provider = fake
            self.app = create_app(
                settings,
                providers={"gemini": fake, "ollama": FakeProvider("ollama")},
                store=self.store,
                sink=self.sink,
                span_exporter=self.spans,
            )
            self.client = TestClient(self.app)

        def chat(self, content="สวัสดี", *, headers=None, **body):
            payload = {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": content}], **body}
            return self.client.post("/v1/chat/completions", json=payload, headers={**AUTH, **(headers or {})})

        def span(self, name):
            return next(s for s in self.spans.get_finished_spans() if s.name == name)

    return Gateway()
