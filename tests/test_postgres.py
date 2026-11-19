"""Migrations and the usage log against a real Postgres. Set TEST_DATABASE_URL to run (CI does)."""

import os

import pytest

from gateway.migrate import migrate
from gateway.usage import PostgresSink, UsageRecord

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL is not set")


async def test_migrate_is_idempotent_and_usage_rows_land():
    import asyncpg

    conn = await asyncpg.connect(DSN)
    await conn.execute(
        "drop view if exists public.usage_daily; drop table if exists public.usage_log, public.schema_migrations"
    )
    await conn.close()

    assert await migrate(DSN) == ["20261119000000_usage_log.sql"]
    assert await migrate(DSN) == []

    sink = PostgresSink(DSN)
    await sink.write(
        UsageRecord(
            "r1",
            "tests",
            "abc",
            "gemini",
            "gemini-2.5-flash",
            200,
            input_tokens=12,
            output_tokens=8,
            cost_usd=0.0000236,
            latency_ms=40,
        )
    )
    await sink.write(
        UsageRecord(
            "r2",
            "tests",
            None,
            "gemini",
            "gemini-2.5-flash",
            400,
            blocked_reason="prompt_injection",
            injection_score=0.9,
        )
    )
    await sink.aclose()

    conn = await asyncpg.connect(DSN)
    try:
        row = await conn.fetchrow("select requests, blocked, input_tokens, cost_usd from public.usage_daily")
    finally:
        await conn.close()
    assert (row["requests"], row["blocked"], row["input_tokens"]) == (2, 1, 12)
    assert float(row["cost_usd"]) == pytest.approx(0.0000236)
