"""The request path: limits -> input guardrails -> model -> output guardrails -> usage log."""

import copy
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from gateway.config import Settings
from gateway.embeddings import Embedder, estimate_tokens, parse_request
from gateway.errors import GatewayError
from gateway.guardrails.injection import score_injection
from gateway.guardrails.output import leaks_system_prompt
from gateway.guardrails.pii import Vault, mask_pii
from gateway.limits import Limiter
from gateway.pricing import estimate_cost
from gateway.providers import Provider, resolve
from gateway.usage import UsageRecord

CHECKED_ROLES = {"user", "tool"}  # where injected text comes from: people and retrieved content
REDACTED_ROLES = {"user", "assistant", "tool"}
LEAK_REPLY = "ขออภัย ไม่สามารถตอบคำขอนี้ได้ (Sorry, I can't help with that request.)"
SPAN_TEXT_LIMIT = 20_000


@dataclass
class Outcome:
    body: dict[str, Any]
    headers: dict[str, str]
    record: UsageRecord
    cost_usd: float = 0.0


@dataclass
class Context:
    request_id: str
    client: str
    end_user_hash: str | None
    session_id: str | None = None  # groups the calls of one agent run into one Langfuse session
    started: float = field(default_factory=time.perf_counter)


def hash_user(raw: str | None, salt: str = "") -> str | None:
    """Keyed hash of the end user's id (often an IP), so the usage log cannot be reversed by brute force."""
    if not raw:
        return None
    return hmac.new(salt.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]


def _text_parts(message: dict[str, Any]):
    """Yield (container, key) pairs that hold text in a chat message, so callers can rewrite in place."""
    content = message.get("content")
    if isinstance(content, str):
        yield message, "content"
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                yield part, "text"
    for call in message.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            yield fn, "arguments"


def _message_text(message: dict[str, Any]) -> str:
    return "\n".join(c[k] for c, k in _text_parts(message))


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)[:SPAN_TEXT_LIMIT]


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, Provider],
        limiter: Limiter,
        tracer: trace.Tracer,
        embedders: dict[str, Embedder] | None = None,
    ):
        self.settings = settings
        self.providers = providers
        self.limiter = limiter
        self.tracer = tracer
        self.embedders = embedders or {}

    def _start(self, root, ctx: Context) -> None:
        root.set_attribute("gateway.request_id", ctx.request_id)
        root.set_attribute("gateway.client", ctx.client)
        if ctx.end_user_hash:
            root.set_attribute("langfuse.user.id", ctx.end_user_hash)
        if ctx.session_id:
            root.set_attribute("langfuse.session.id", ctx.session_id)

    def _fail(self, err: GatewayError, record: UsageRecord, ctx: Context, root) -> None:
        record.status = err.status
        record.blocked_reason = record.blocked_reason or err.code
        record.latency_ms = int((time.perf_counter() - ctx.started) * 1000)
        root.set_status(Status(StatusCode.ERROR, err.code))
        root.set_attribute("gateway.error_code", err.code)
        err.record = record  # type: ignore[attr-defined]

    async def chat(self, body: dict[str, Any], ctx: Context) -> Outcome:
        record = UsageRecord(ctx.request_id, ctx.client, ctx.end_user_hash, None, body.get("model"), 200)
        with self.tracer.start_as_current_span("gateway.chat") as root:
            self._start(root, ctx)
            try:
                outcome = await self._chat(body, ctx, record, root)
            except GatewayError as err:
                self._fail(err, record, ctx, root)
                raise
            root.set_attribute("gateway.cost_usd", outcome.cost_usd)
            return outcome

    async def embed(self, body: dict[str, Any], ctx: Context) -> Outcome:
        record = UsageRecord(ctx.request_id, ctx.client, ctx.end_user_hash, None, body.get("model"), 200)
        with self.tracer.start_as_current_span("gateway.embeddings") as root:
            self._start(root, ctx)
            try:
                outcome = await self._embed(body, ctx, record)
            except GatewayError as err:
                self._fail(err, record, ctx, root)
                raise
            root.set_attribute("gateway.cost_usd", outcome.cost_usd)
            return outcome

    async def _embed(self, body: dict[str, Any], ctx: Context, record: UsageRecord) -> Outcome:
        s = self.settings
        model = body.get("model")
        if not isinstance(model, str):
            raise GatewayError(400, "invalid_model", "model is required")
        name = model.removeprefix("gemini/").removeprefix("models/")
        if name not in s.embedding_models:
            raise GatewayError(400, "model_not_allowed", f"embedding model {model!r} is not served by this gateway")
        record.provider, record.model = "gemini", name
        texts, task_type, dimensions = parse_request(body)
        embedder = self.embedders.get("gemini")
        if embedder is None:
            raise GatewayError(503, "provider_not_configured", "gemini has no API key on this gateway")

        with self.tracer.start_as_current_span("gateway.limits"):
            await self.limiter.check(ctx.client, ctx.end_user_hash)

        vault = Vault()
        with self.tracer.start_as_current_span("guardrail.input") as span:
            # No injection check: an embedding model follows no instructions, and nothing it returns is text.
            if s.pii_redaction:
                texts = [vault.redact(t).text for t in texts]
            record.pii_redactions = vault.total
            span.set_attribute("guardrail.pii.redacted", vault.total)
            span.set_attribute("guardrail.pii.kinds", sorted(vault.counts))

        with self.tracer.start_as_current_span(f"embeddings {name}") as span:
            span.set_attribute("langfuse.observation.type", "embedding")
            span.set_attribute("gen_ai.system", "gemini")
            span.set_attribute("gen_ai.operation.name", "embeddings")
            span.set_attribute("gen_ai.request.model", name)
            span.set_attribute("gateway.embeddings.count", len(texts))
            if task_type:
                span.set_attribute("gateway.embeddings.task_type", task_type)
            if dimensions:
                span.set_attribute("gen_ai.request.embedding.dimensions", dimensions)
            span.set_attribute("langfuse.observation.input", _dump(texts))
            vectors, reported = await embedder.embed(name, texts, task_type=task_type, dimensions=dimensions)
            tokens = reported if reported is not None else estimate_tokens(texts)
            cost = estimate_cost("gemini", name, tokens, 0)
            span.set_attribute("gen_ai.usage.input_tokens", tokens)
            span.set_attribute("gateway.tokens_estimated", reported is None)
            span.set_attribute("gateway.cost_usd", cost)
        record.input_tokens, record.cost_usd = tokens, cost
        record.latency_ms = int((time.perf_counter() - ctx.started) * 1000)
        response = {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            "model": name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
        headers = {"x-gateway-cost-usd": f"{cost:.8f}", "x-gateway-pii-redacted": str(vault.total)}
        return Outcome(response, headers, record, cost)

    async def _chat(self, body: dict[str, Any], ctx: Context, record: UsageRecord, root) -> Outcome:
        s = self.settings
        if body.get("stream"):
            raise GatewayError(
                400,
                "stream_not_supported",
                "Streaming is not supported yet: the output guardrails need the full answer",
            )
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise GatewayError(400, "invalid_messages", "messages must be a non-empty list of objects")
        model = body.get("model")
        if not isinstance(model, str):
            raise GatewayError(400, "invalid_model", "model is required")
        route = resolve(model, s)
        record.provider, record.model = route.provider, route.upstream_model
        provider = self.providers.get(route.provider)
        if provider is None:
            raise GatewayError(503, "provider_not_configured", f"{route.provider} has no API key on this gateway")

        with self.tracer.start_as_current_span("gateway.limits"):
            limit = await self.limiter.check(ctx.client, ctx.end_user_hash)

        messages = copy.deepcopy(messages)
        vault = Vault()
        with self.tracer.start_as_current_span("guardrail.input") as span:
            score, rules = 0.0, []
            if s.injection_mode != "off":
                for m in messages:
                    if m.get("role") in CHECKED_ROLES:
                        result = score_injection(_message_text(m))
                        if result.score > score:
                            score, rules = result.score, result.rules
            record.injection_score = score
            span.set_attribute("guardrail.injection.score", score)
            span.set_attribute("guardrail.injection.rules", rules)
            if s.pii_redaction:
                for m in messages:
                    if m.get("role") in REDACTED_ROLES:
                        for container, key in _text_parts(m):
                            container[key] = vault.redact(container[key]).text
            record.pii_redactions = vault.total
            span.set_attribute("guardrail.pii.redacted", vault.total)
            span.set_attribute("guardrail.pii.kinds", sorted(vault.counts))
            if s.injection_mode == "block" and score >= s.injection_threshold:
                span.set_status(Status(StatusCode.ERROR, "prompt_injection"))
                raise GatewayError(
                    400, "prompt_injection", f"The request looks like a prompt-injection attempt ({', '.join(rules)})"
                )

        upstream = {**body, "model": route.upstream_model, "messages": messages}
        upstream.pop("user", None)  # the provider does not need the end user's id
        with self.tracer.start_as_current_span(f"llm {route.upstream_model}") as span:
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("gen_ai.system", route.provider)
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", route.upstream_model)
            span.set_attribute("langfuse.observation.input", _dump(messages))
            for key in ("temperature", "max_tokens", "top_p"):
                if isinstance(body.get(key), int | float):
                    span.set_attribute(f"gen_ai.request.{key}", body[key])
            response = await provider.chat(upstream)
            usage = response.get("usage") or {}
            tokens_in, tokens_out, reasoning = _billed_tokens(usage)
            cost = estimate_cost(route.provider, route.upstream_model, tokens_in, tokens_out)
            span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
            span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning)
            span.set_attribute("gen_ai.response.model", str(response.get("model") or route.upstream_model))
            span.set_attribute("gateway.cost_usd", cost)
            span.set_attribute("langfuse.observation.output", _dump(response.get("choices")))
        record.input_tokens, record.output_tokens, record.cost_usd = tokens_in, tokens_out, cost

        with self.tracer.start_as_current_span("guardrail.output") as span:
            system_text = "\n".join(_message_text(m) for m in messages if m.get("role") in ("system", "developer"))
            leaked, masked = False, 0
            for choice in response.get("choices") or []:
                msg = choice.get("message") if isinstance(choice, dict) else None
                if not isinstance(msg, dict):
                    continue
                if s.system_leak_check and system_text and leaks_system_prompt(system_text, _message_text(msg)):
                    leaked = True
                    msg["content"], msg["tool_calls"] = LEAK_REPLY, None
                    choice["finish_reason"] = "content_filter"
                    continue
                for container, key in _text_parts(msg):
                    text = container[key]
                    if s.pii_redaction:
                        # PII the model wrote itself (it never saw the caller's values) stays hidden.
                        text, n = mask_pii(text)
                        masked += n
                    if s.pii_restore:
                        text = vault.restore(text)
                    container[key] = text
            span.set_attribute("guardrail.output.system_leak", leaked)
            span.set_attribute("guardrail.output.pii_masked", masked)
            if leaked:
                record.blocked_reason = "system_prompt_leak"

        record.latency_ms = int((time.perf_counter() - ctx.started) * 1000)
        headers = {
            "x-gateway-cost-usd": f"{cost:.8f}",
            "x-gateway-pii-redacted": str(vault.total),
            "x-gateway-injection-score": f"{score:.3f}",
        }
        if limit.remaining_user_minute is not None:
            headers["x-ratelimit-remaining-requests"] = str(limit.remaining_user_minute)
        return Outcome(response, headers, record, cost)


def _billed_tokens(usage: dict) -> tuple[int, int, int]:
    """Input, billed output and reasoning tokens.

    Gemini 2.5 thinks before it answers and bills those tokens as output, but its OpenAI-compatible
    usage leaves them out of completion_tokens: they only show in total_tokens (or in
    completion_tokens_details.reasoning_tokens where a provider reports them).
    """
    tokens_in = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)
    hidden = max(0, total - tokens_in - completion)  # thinking tokens missing from completion_tokens
    details = usage.get("completion_tokens_details")
    reported = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
    return tokens_in, completion + hidden, hidden or reported
