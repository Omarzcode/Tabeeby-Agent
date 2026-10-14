"""USD prices per 1M tokens, looked up by model_id to turn token counts into cost.

Prices are OpenAI's Standard tier, from https://developers.openai.com/api/docs/pricing
(checked 2026-10-14). gpt-5.5 and gpt-5.4 rates apply to prompts under 272K tokens.

Add or override models without code changes with the MODEL_PRICES_JSON setting:
    MODEL_PRICES_JSON={"my-model": [input, cached_input, output]}   # USD per 1M tokens
Models found nowhere fall back to INPUT_TOKEN_COST_USD / OUTPUT_TOKEN_COST_USD
(USD per single token) and are flagged as "unpriced" in the usage dashboard.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


MODEL_PRICES: Dict[str, ModelPrice] = {
    "gpt-5.5": ModelPrice(5.00, 0.50, 30.00),
    "gpt-5.4": ModelPrice(2.50, 0.25, 15.00),
    "gpt-5.4-mini": ModelPrice(0.75, 0.075, 4.50),
    "gpt-5.4-nano": ModelPrice(0.20, 0.02, 1.25),
    "gpt-5.2": ModelPrice(1.75, 0.175, 14.00),
    "gpt-5.1": ModelPrice(1.25, 0.125, 10.00),
    "gpt-5": ModelPrice(1.25, 0.125, 10.00),
    "gpt-5-mini": ModelPrice(0.25, 0.025, 2.00),
    "gpt-5-nano": ModelPrice(0.05, 0.005, 0.40),
    "gpt-4.1": ModelPrice(2.00, 0.50, 8.00),
    "gpt-4.1-mini": ModelPrice(0.40, 0.10, 1.60),
    "gpt-4.1-nano": ModelPrice(0.10, 0.025, 0.40),
    "gpt-4o": ModelPrice(2.50, 1.25, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.075, 0.60),
    "o4-mini": ModelPrice(1.10, 0.275, 4.40),
}

_overrides: Dict[str, ModelPrice] = {}


def load_price_overrides(raw: Optional[str]) -> None:
    """Read MODEL_PRICES_JSON: {"model": [input, cached_input, output]} in USD per 1M tokens."""
    _overrides.clear()
    if not raw or not raw.strip():
        return
    try:
        for name, values in json.loads(raw).items():
            inp, cached, out = (float(v) for v in values)
            _overrides[name] = ModelPrice(inp, cached, out)
    except Exception as e:
        logger.error("Invalid MODEL_PRICES_JSON, ignored: %s", e)


def price_for(model_id: Optional[str]) -> Optional[ModelPrice]:
    """Price for a model id. Dated snapshots ("gpt-5.4-mini-2026-03-17") use the base model's price."""
    if not model_id:
        return None
    table = {**MODEL_PRICES, **_overrides}
    if model_id in table:
        return table[model_id]
    for name in sorted(table, key=len, reverse=True):
        if model_id.startswith(name + "-20"):
            return table[name]
    return None


def token_cost(
    model_id: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    fallback_input_per_token: float = 0.0,
    fallback_output_per_token: float = 0.0,
) -> float:
    """USD cost of one call. Cached tokens are included in prompt_tokens and billed at the cached rate."""
    price = price_for(model_id)
    cached = max(0, min(cached_tokens, prompt_tokens))
    if price is None:
        return prompt_tokens * fallback_input_per_token + completion_tokens * fallback_output_per_token
    return (
        (prompt_tokens - cached) * price.input_per_m
        + cached * price.cached_input_per_m
        + completion_tokens * price.output_per_m
    ) / 1_000_000
