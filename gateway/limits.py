"""Rate limits (fixed windows) and a daily spend cap shared by all clients."""

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from gateway.errors import GatewayError
from gateway.state import CounterStore

MICRO = 1_000_000  # spend is stored in micro-USD so Redis can use integer INCRBY


@dataclass
class LimitResult:
    remaining_user_minute: int | None


class Limiter:
    def __init__(
        self,
        store: CounterStore,
        *,
        client_per_minute: int,
        user_per_minute: int,
        user_per_day: int,
        daily_budget_usd: float,
        clock=time.time,
    ):
        self.store = store
        self.client_per_minute = client_per_minute
        self.user_per_minute = user_per_minute
        self.user_per_day = user_per_day
        self.daily_budget_micro = int(daily_budget_usd * MICRO)
        self._clock = clock

    def _day(self) -> str:
        return datetime.fromtimestamp(self._clock(), UTC).strftime("%Y%m%d")

    async def check(self, client: str, end_user: str | None) -> LimitResult:
        now = self._clock()
        minute = int(now // 60)
        if await self.store.get(f"spend:{self._day()}") >= self.daily_budget_micro:
            raise GatewayError(
                429,
                "daily_budget_exceeded",
                "The gateway's daily budget is used up. It resets at 00:00 UTC.",
                headers={"retry-after": str(_seconds_to_midnight(now))},
            )

        n = await self.store.incr(f"rl:c:{client}:{minute}", 1, 120)
        if n > self.client_per_minute:
            raise GatewayError(
                429,
                "rate_limited",
                f"Client {client} is over {self.client_per_minute} requests per minute",
                headers={"retry-after": str(60 - int(now % 60))},
            )
        if not end_user:
            return LimitResult(None)

        per_min = await self.store.incr(f"rl:u:{client}:{end_user}:{minute}", 1, 120)
        if per_min > self.user_per_minute:
            raise GatewayError(
                429,
                "rate_limited",
                f"Over {self.user_per_minute} requests per minute for this user",
                headers={"retry-after": str(60 - int(now % 60))},
            )
        per_day = await self.store.incr(f"rl:u:{client}:{end_user}:{self._day()}", 1, 86_400 + 3_600)
        if per_day > self.user_per_day:
            raise GatewayError(
                429,
                "rate_limited",
                f"Over {self.user_per_day} requests today for this user",
                headers={"retry-after": str(_seconds_to_midnight(now))},
            )
        return LimitResult(max(self.user_per_minute - per_min, 0))

    async def add_spend(self, cost_usd: float) -> None:
        micro = round(cost_usd * MICRO)
        if micro > 0:
            await self.store.incr(f"spend:{self._day()}", micro, 2 * 86_400)


def _seconds_to_midnight(now: float) -> int:
    return 86_400 - int(now % 86_400)
