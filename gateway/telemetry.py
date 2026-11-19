"""OpenTelemetry tracing. Spans go to Langfuse over OTLP/HTTP when keys are set.

Span attributes follow the OpenTelemetry GenAI conventions (gen_ai.*), plus the langfuse.*
attributes Langfuse reads for input/output. Only redacted text is ever put on a span.
"""

import base64

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from gateway.config import Settings

TRACER_NAME = "thai_llm_gateway"


def langfuse_exporter(settings: Settings) -> SpanExporter | None:
    secret = settings.langfuse_secret_key.get_secret_value()
    if not (settings.langfuse_public_key and secret):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    auth = base64.b64encode(f"{settings.langfuse_public_key}:{secret}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )


def setup_tracing(settings: Settings, exporter: SpanExporter | None = None, *, batch: bool = True) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.service_name,
                "deployment.environment": settings.environment,
            }
        )
    )
    exporter = exporter or langfuse_exporter(settings)
    if exporter is not None:
        if batch:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def tracer(provider: TracerProvider) -> trace.Tracer:
    return provider.get_tracer(TRACER_NAME)
