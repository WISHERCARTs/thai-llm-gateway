"""Shared counters for rate limits and the daily budget.

Cloud Run starts and stops instances, so counters live in Upstash Redis (over its REST API).
The in-memory store is for tests and a single local process only.
"""

import time
from typing import Protocol

import httpx


class CounterStore(Protocol):
    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        """Add amount to key (creating it with the TTL) and return the new value."""

    async def get(self, key: str) -> int: ...


class MemoryStore:
    def __init__(self, clock=time.monotonic):
        self._data: dict[str, tuple[int, float]] = {}
        self._clock = clock

    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        now = self._clock()
        value, expires = self._data.get(key, (0, now + ttl_seconds))
        if expires <= now:
            value, expires = 0, now + ttl_seconds
        value += amount
        self._data[key] = (value, expires)
        return value

    async def get(self, key: str) -> int:
        value, expires = self._data.get(key, (0, 0.0))
        return value if expires > self._clock() else 0


class UpstashStore:
    def __init__(self, url: str, token: str, client: httpx.AsyncClient | None = None):
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        # SET NX gives a new key its TTL; INCRBY then adds. One round trip.
        cmds = [["SET", key, "0", "EX", str(ttl_seconds), "NX"], ["INCRBY", key, str(amount)]]
        resp = await self._client.post(f"{self._url}/pipeline", json=cmds, headers=self._headers)
        resp.raise_for_status()
        result = resp.json()[1]
        if "error" in result:
            raise RuntimeError(f"upstash: {result['error']}")
        return int(result["result"])

    async def get(self, key: str) -> int:
        resp = await self._client.post(self._url, json=["GET", key], headers=self._headers)
        resp.raise_for_status()
        value = resp.json().get("result")
        return int(value) if value is not None else 0

    async def aclose(self) -> None:
        await self._client.aclose()
