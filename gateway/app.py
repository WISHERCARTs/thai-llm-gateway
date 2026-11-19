"""FastAPI app: an OpenAI-compatible /v1/chat/completions in front of Gemini and Ollama."""

import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry.sdk.trace.export import SpanExporter
from starlette.background import BackgroundTask, BackgroundTasks

from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.limits import Limiter
from gateway.pipeline import Context, Pipeline, hash_user
from gateway.providers import Provider, build_providers
from gateway.state import CounterStore, MemoryStore, UpstashStore
from gateway.telemetry import setup_tracing, tracer
from gateway.usage import MemorySink, PostgresSink, UsageRecord, UsageSink

log = logging.getLogger("gateway")


def authenticate(request: Request, settings: Settings) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise GatewayError(401, "missing_api_key", "Send the gateway key as 'Authorization: Bearer <key>'")
    client = None
    for key, name in settings.client_keys.items():  # compare every key so timing does not leak which one
        if hmac.compare_digest(key.encode(), token.encode()):
            client = name
    if client is None:
        raise GatewayError(401, "invalid_api_key", "Unknown gateway key")
    return client


def create_app(
    settings: Settings | None = None,
    *,
    providers: dict[str, Provider] | None = None,
    store: CounterStore | None = None,
    sink: UsageSink | None = None,
    span_exporter: SpanExporter | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if store is None:
        token = settings.upstash_redis_rest_token.get_secret_value()
        url = settings.upstash_redis_rest_url
        store = UpstashStore(url, token) if url else MemoryStore()
    if sink is None:
        dsn = settings.database_url.get_secret_value()
        sink = PostgresSink(dsn) if dsn else MemorySink()
    providers = providers if providers is not None else build_providers(settings)
    tracer_provider = setup_tracing(settings, span_exporter, batch=span_exporter is None)
    limiter = Limiter(
        store,
        client_per_minute=settings.rate_limit_client_per_minute,
        user_per_minute=settings.rate_limit_user_per_minute,
        user_per_day=settings.rate_limit_user_per_day,
        daily_budget_usd=settings.daily_budget_usd,
    )
    pipeline = Pipeline(settings, providers, limiter, tracer(tracer_provider))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for closable in [*providers.values(), store, sink]:
            if hasattr(closable, "aclose"):
                await closable.aclose()
        tracer_provider.shutdown()

    app = FastAPI(title="thai-llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.sink, app.state.limiter = settings, sink, limiter

    async def after(record: UsageRecord, cost: float) -> None:
        await sink.write(record)
        if cost:
            try:
                await limiter.add_spend(cost)
            except Exception as exc:  # never fail the request after the answer was sent
                log.warning("could not add spend for %s: %s", record.request_id, exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        request_id = uuid.uuid4().hex
        try:
            authenticate(request, settings)
        except GatewayError as err:
            return err.response(request_id)
        names = list(settings.gemini_models)
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "gateway"} for n in names]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            client = authenticate(request, settings)
        except GatewayError as err:
            return err.response(request_id)
        try:
            body = await request.json()
        except ValueError:
            return GatewayError(400, "invalid_json", "Body must be JSON").response(request_id)
        if not isinstance(body, dict):
            return GatewayError(400, "invalid_json", "Body must be a JSON object").response(request_id)
        end_user = request.headers.get("x-end-user") or body.get("user")
        raw_user = end_user if isinstance(end_user, str) else None
        ctx = Context(request_id, client, hash_user(raw_user, settings.user_hash_salt.get_secret_value()))
        try:
            outcome = await pipeline.chat(body, ctx)
        except GatewayError as err:
            response = err.response(request_id)
            record = getattr(err, "record", None)
            if record is not None:
                response.background = BackgroundTask(after, record, 0.0)
            return response
        tasks = BackgroundTasks()
        tasks.add_task(after, outcome.record, outcome.cost_usd)
        return JSONResponse(outcome.body, headers={**outcome.headers, "x-request-id": request_id}, background=tasks)

    return app
