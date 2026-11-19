import pytest

from gateway.errors import GatewayError
from gateway.limits import Limiter
from gateway.state import MemoryStore


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def limiter(clock, **kw):
    args = {"client_per_minute": 100, "user_per_minute": 2, "user_per_day": 3, "daily_budget_usd": 0.01}
    return Limiter(MemoryStore(clock=clock), clock=clock, **{**args, **kw})


async def test_user_per_minute_then_next_minute():
    clock = Clock()
    lim = limiter(clock)
    assert (await lim.check("c", "u")).remaining_user_minute == 1
    await lim.check("c", "u")
    with pytest.raises(GatewayError) as exc:
        await lim.check("c", "u")
    assert exc.value.status == 429 and "retry-after" in exc.value.headers
    clock.t += 60
    await lim.check("c", "u")
    with pytest.raises(GatewayError):  # the fourth request today
        await lim.check("c", "u")


async def test_users_are_separate_and_no_user_skips_user_limits():
    lim = limiter(Clock())
    for _ in range(2):
        await lim.check("c", "a")
        await lim.check("c", "b")
    for _ in range(10):
        assert (await lim.check("c", None)).remaining_user_minute is None


async def test_client_limit():
    lim = limiter(Clock(), client_per_minute=3)
    for _ in range(3):
        await lim.check("c", None)
    with pytest.raises(GatewayError):
        await lim.check("c", None)


async def test_budget_cap():
    lim = limiter(Clock())
    await lim.add_spend(0.006)
    await lim.check("c", None)
    await lim.add_spend(0.006)
    with pytest.raises(GatewayError) as exc:
        await lim.check("c", None)
    assert exc.value.code == "daily_budget_exceeded"
