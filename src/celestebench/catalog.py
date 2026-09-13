"""models.dev metadata: pricing, model producers and producer logos.

Kept free of Tau so the viewer can price rollouts without the LLM extra.
"""

import json
import time
import urllib.request
from pathlib import Path

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE = Path.home() / ".cache" / "celestebench" / "models-dev.json"
_MODELS_DEV_TTL_SECONDS = 4 * 60 * 60
_loaded: dict = {}

# models.dev tags every model with the AI-SDK package serving it, while Tau's
# catalog only keeps a provider-level protocol. Multi-protocol gateways (the
# Zen endpoints serve qwen/minimax over Anthropic, gpt over Responses, the rest
# over Completions) therefore resolve their wire format per model from here.
_MODELS_DEV_PROVIDER_KEYS = {
    "kimi-code": "kimi-for-coding",
    "together": "togetherai",
}

# models.dev exposes no lab field and gateway model ids hide the producer, so
# we match the brands we run by name.
_LABS = (
    ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("claude", "anthropic"), ("gemini", "google"), ("gemma", "google"),
    ("grok", "xai"), ("glm", "zai"), ("kimi", "moonshotai"),
    ("qwen", "alibaba"), ("minimax", "minimax"), ("mimo", "xiaomi"),
    ("muse", "meta"), ("deepseek", "deepseek"), ("mistral", "mistral"),
    ("mixtral", "mistral"), ("nex", "nex-agi"),
)
_LAB_NAMES = {
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "xai": "xAI",
    "meta": "Meta", "zai": "Z.AI", "moonshotai": "Moonshot AI", "deepseek": "DeepSeek",
    "alibaba": "Alibaba", "minimax": "MiniMax", "xiaomi": "Xiaomi",
    "nex-agi": "Nex AGI", "mistral": "Mistral",
}


def _models_dev() -> dict:
    """Raw models.dev data, cached between runs; falls back to a stale cache."""
    if _loaded.get("data") and time.time() - _loaded.get("at", 0) < _MODELS_DEV_TTL_SECONDS:
        return _loaded["data"]
    try:
        cache = json.loads(_MODELS_DEV_CACHE.read_text())
        if time.time() - cache.get("fetched_at", 0) < _MODELS_DEV_TTL_SECONDS:
            _loaded.update(at=time.time(), data=cache.get("data") or {})
            return _loaded["data"]
    except (OSError, ValueError):
        cache = None
    try:
        request = urllib.request.Request(
            MODELS_DEV_URL, headers={"Accept": "application/json",
                                     "User-Agent": "celestebench/0.1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
    except (OSError, ValueError):
        return (cache or {}).get("data") or {}
    try:
        _MODELS_DEV_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _MODELS_DEV_CACHE.write_text(json.dumps({"fetched_at": time.time(), "data": data}))
    except OSError:
        pass
    _loaded.update(at=time.time(), data=data)
    return data


def _basename(model: str) -> str:
    # A free program is a discount; the contributor rate is the model's own price.
    return model.rsplit("/", 1)[-1].removesuffix(":free").removesuffix("-free")


def producer(model: str) -> str | None:
    """The first-party lab that made a model, or None when we cannot tell."""
    name = _basename(model).lower()
    return next((lab for prefix, lab in _LABS if name.startswith(prefix)), None)


def producer_name(producer_id: str) -> str:
    return _LAB_NAMES.get(producer_id, producer_id.replace("-", " ").title())


def price(producer_id: str, model: str) -> dict | None:
    """models.dev cost (USD per million tokens) under a model's first-party lab."""
    name = _basename(model).lower()
    data = _models_dev()
    for key, entry in (data.get(producer_id, {}).get("models") or {}).items():
        if _basename(key).lower() == name or (entry.get("id") or "").lower() == name:
            return entry.get("cost")
    wanted = f"{producer_id}/{name}"
    for provider in data.values():
        for key, entry in (provider.get("models") or {}).items():
            if key.lower() == wanted:
                return entry.get("cost")
    return None
