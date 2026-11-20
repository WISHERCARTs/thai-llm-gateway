"""List prices in USD per 1M tokens, used to estimate cost per request.

These are paid-tier list prices as known when this file was written (Nov 2026); check
https://ai.google.dev/gemini-api/docs/pricing before relying on them. The free tier costs nothing,
but the gateway still records the list-price estimate, so the budget cap and the cost report
show what the traffic would cost on a paid key.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


PRICES: dict[str, Price] = {
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-flash-lite": Price(0.10, 0.40),
    "gemini-embedding-001": Price(0.15, 0.0),
}
# Unknown hosted models are priced at the most expensive known rate so the budget errs on the safe side.
FALLBACK = Price(1.25, 10.0)


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    if provider == "ollama":
        return 0.0
    price = PRICES.get(model, FALLBACK)
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
