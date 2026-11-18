"""Thin LLM wrapper over LiteLLM: one place to add caching, cost tracking, and routing later."""

from __future__ import annotations

import json
import re

import litellm

# Week 11: plug Langfuse in here with litellm.callbacks = ["langfuse"]
USAGE_LOG: list[dict] = []


def complete(model: str, system: str, user: str, max_tokens: int = 4000) -> str:
    resp = litellm.completion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
    )
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:
        cost = None
    USAGE_LOG.append(
        {
            "model": model,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cost_usd": cost,
        }
    )
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    return json.loads(raw)
