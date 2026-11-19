"""One row per request in Postgres (table usage_log), for cost reports in SQL.

Any Postgres works (Neon in production, a local one for tests). Writes happen after the
response is sent and never fail a request; a failed write is logged.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass, fields
from typing import Protocol

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


COLUMNS = [f.name for f in fields(UsageRecord)]
INSERT = (
    f"insert into public.usage_log ({', '.join(COLUMNS)}) "
    f"values ({', '.join(f'${i}' for i in range(1, len(COLUMNS) + 1))})"
)


class UsageSink(Protocol):
    async def write(self, record: UsageRecord) -> None: ...


class MemorySink:
    def __init__(self):
        self.records: list[UsageRecord] = []

    async def write(self, record: UsageRecord) -> None:
        self.records.append(record)


class PostgresSink:
    """asyncpg pool, opened on first write so the app starts even if the database is asleep."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    import asyncpg

                    # statement_cache_size=0: Neon's pooled endpoint (PgBouncer) does not keep prepared statements.
                    self._pool = await asyncpg.create_pool(
                        self._dsn, min_size=0, max_size=3, statement_cache_size=0, command_timeout=10
                    )
        return self._pool

    async def write(self, record: UsageRecord) -> None:
        try:
            pool = await self._get_pool()
            await pool.execute(INSERT, *asdict(record).values())
        except Exception as exc:
            log.warning("usage_log write failed for %s: %s", record.request_id, exc)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
