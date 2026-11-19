"""One row per request in Supabase (table usage_log), for cost reports in SQL.

Writes happen after the response is sent and never fail a request; a failed write is logged.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    request_id: str
    client: str
    end_user_hash: str | None
    provider: str | None
    model: str | None
    status: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    pii_redactions: int = 0
    injection_score: float = 0.0
    blocked_reason: str | None = None
    cache_hit: bool = False


class UsageSink(Protocol):
    async def write(self, record: UsageRecord) -> None: ...


class MemorySink:
    def __init__(self):
        self.records: list[UsageRecord] = []

    async def write(self, record: UsageRecord) -> None:
        self.records.append(record)


class SupabaseSink:
    def __init__(self, url: str, service_key: str, client: httpx.AsyncClient | None = None):
        self._url = url.rstrip("/") + "/rest/v1/usage_log"
        self._headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Prefer": "return=minimal"}
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def write(self, record: UsageRecord) -> None:
        try:
            resp = await self._client.post(self._url, json=asdict(record), headers=self._headers)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("usage_log write failed for %s: %s", record.request_id, exc)

    async def aclose(self) -> None:
        await self._client.aclose()
