"""Public first-party API price of a rollout, from its logged token usage."""

import json
from pathlib import Path

from . import catalog


def _rows(path: Path):
    try:
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _rate(price: dict, context: int) -> dict:
    """The price block a request falls into once its context passes a tier."""
    for tier in price.get("tiers") or []:
        size = (tier.get("tier") or {}).get("size")
        if size and context > size:
            return tier
    return price


def _usd(price: dict, context: int, input=0, output=0, cache_read=0, cache_write=0) -> float:
    rate = _rate(price, context)
    return (input * (rate.get("input") or 0) + output * (rate.get("output") or 0)
            + cache_read * (rate.get("cache_read") or 0)
            + cache_write * (rate.get("cache_write") or 0)) / 1_000_000


def _tau_usd(messages: Path, price: dict):
    """USD for a Tau rollout; its input count excludes the cached part."""
    total = None
    for row in _rows(messages):
        usage = row.get("usage") if row.get("role") == "assistant" else None
        if not isinstance(usage, dict):
            continue
        input = usage.get("input") or 0
        cache_read = usage.get("cacheRead") or 0
        cache_write = usage.get("cacheWrite") or 0
        total = (total or 0) + _usd(price, input + cache_read + cache_write, input,
                                    usage.get("output") or 0, cache_read, cache_write)
    return total


def _codex_usd(trace: Path, price: dict):
    """USD for a Codex rollout; its input count already includes the cached part."""
    total = None
    for row in _rows(trace):
        usage = row.get("usage") if row.get("type") == "turn.completed" else None
        if not isinstance(usage, dict):
            continue
        input = usage.get("input_tokens") or 0
        cached = usage.get("cached_input_tokens") or 0
        total = (total or 0) + _usd(price, input, input - cached, usage.get("output_tokens") or 0,
                                    cached, usage.get("cache_write_input_tokens") or 0)
    return total


def rollout_cost(path: Path, model: str, *, codex: bool = False) -> float | None:
    """Public API price for one rollout, or None when it cannot be priced.

    Pass the run's ``messages.jsonl`` for the Tau harness, ``codex.jsonl`` for
    the Codex one. The price is the model producer's own list price, so runs
    made through a discounted gateway still compare on the same axis.
    """
    if not path.is_file():
        return None
    producer = catalog.producer(model)
    price = catalog.price(producer, model) if producer else None
    if not price:
        return None
    return _codex_usd(path, price) if codex else _tau_usd(path, price)
