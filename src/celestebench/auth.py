"""Which external-harness credentials are OAuth subscriptions.

An OAuth login is one shared session with per-account limits, so runs using it
must serialize; an API key can drive many runs at once. Each CLI keeps its
credentials in a different place, so this module owns that mapping.
"""

import json
import os
from pathlib import Path


def agent_dir(harness: str) -> Path | None:
    """The CLI's host state directory, where its login and settings live."""
    if harness == "opencode":
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        return base / "opencode"
    if harness == "pi":
        return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")
    return None


def _read(harness: str, name: str):
    directory = agent_dir(harness)
    if directory is None:
        return {}
    try:
        return json.loads((directory / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def provider(harness: str, model: str) -> str:
    """The provider a model id names: its prefix, or the CLI's own default."""
    if "/" in model:
        return model.split("/", 1)[0]
    # A bare Pi id resolves through the defaultProvider in its settings.
    return _read(harness, "settings.json").get("defaultProvider") or ""


def oauth(harness: str, model: str) -> bool:
    """Whether the model signs in through an OAuth subscription for this CLI."""
    entry = _read(harness, "auth.json").get(provider(harness, model))
    return isinstance(entry, dict) and entry.get("type") == "oauth"
