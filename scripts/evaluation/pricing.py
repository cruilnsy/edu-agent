# scripts/evaluation/pricing.py
# Token prices and the cost of one request, for eval_system_health.py.
#
# Defaults are DeepSeek's off-peak list prices for deepseek-flash (USD per 1M tokens), from
# https://api-docs.deepseek.com/quick_start/pricing/ as read on 2026-11-10. Peak hours cost
# twice as much (01:00-04:00 and 06:00-10:00 UTC on weekdays), so costs are shown for both.
# Prices change: override with --price-* on the command line or EVAL_PRICE_*_PER_1M.

import os
from dataclasses import dataclass

PRICES_SOURCE = "DeepSeek list price for deepseek-flash, 2026-11-10"


@dataclass(frozen=True)
class Prices:
    """Off-peak USD per 1M tokens."""
    cache_hit: float = 0.003      # Input tokens served from DeepSeek's context cache
    cache_miss: float = 0.15      # Other input tokens
    output: float = 0.6
    peak_multiplier: float = 2.0  # Peak price / off-peak price


_ENV = {"cache_hit": "EVAL_PRICE_CACHE_HIT_PER_1M", "cache_miss": "EVAL_PRICE_CACHE_MISS_PER_1M",
        "output": "EVAL_PRICE_OUTPUT_PER_1M"}


def load_prices(**cli: float | None) -> tuple[Prices, str]:
    """Prices from the command line, then the environment, then the defaults; plus where they came from."""
    values, overridden = {}, False
    for name, env_name in _ENV.items():
        value = cli.get(name)
        if value is None and os.getenv(env_name):
            value = float(os.environ[env_name])
        if value is not None:
            values[name], overridden = value, True
    return Prices(**values), ("custom prices" if overridden else PRICES_SOURCE)


def request_cost(usage: dict, prices: Prices, peak: bool = False) -> float:
    """USD for one request. Without cache data every input token counts as a miss (an upper bound)."""
    cached = usage.get("cached_input_tokens") or 0
    cost = (cached * prices.cache_hit + (usage["input_tokens"] - cached) * prices.cache_miss
            + usage["output_tokens"] * prices.output) / 1_000_000
    return cost * prices.peak_multiplier if peak else cost
