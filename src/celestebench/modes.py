"""The two CelesteBench modes, loaded from ``modes.json`` at the project root."""

import json
from pathlib import Path

MODES = json.loads((Path(__file__).resolve().parents[2] / "modes.json").read_text(encoding="utf-8"))
# Only these keys reach the harnesses; label and description are viewer metadata.
BUDGETS = ("timeout", "fps", "max_frames", "max_images")


def mode_budgets(name: str) -> tuple[str, dict]:
    """The mode's name and its harness budgets, or a ValueError for anything else."""
    if name not in MODES:
        raise ValueError(f"Unknown mode '{name}'; use one of {', '.join(MODES)}.")
    return name, {key: MODES[name][key] for key in BUDGETS if MODES[name].get(key) is not None}


def mode_of(fps) -> str:
    """The mode a run belongs to, from its frame rate."""
    return "lite" if fps is None else "rtc"
