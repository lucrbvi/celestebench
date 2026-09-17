"""Build the public static bundle the Cloudflare Worker serves.

Reuses the internal viewer's own logic (leaderboard, decisions, scoring) so the
public site can never drift from what we see locally. Writes only files: JSON
data, the CSS and logos, and the open8 WASM player. No server, no runtime.

Usage: uv run python -m web.publish [-o site/public] [--limit N]
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

from celestebench import BENCHMARK_VERSION, catalog
from celestebench.harnesses import HARNESSES

from . import viewer

ROOT = viewer.ROOT
RUNS = viewer.RUNS
STATIC = viewer.STATIC
WASM = ROOT / "build" / "web"
DEFAULT_OUT = ROOT / "site" / "public"


def _run_facts(name: str) -> tuple[dict, str | None, int]:
    """Everything the player needs for one run: decisions plus its input trace."""
    folder = viewer.apply_nested(RUNS / name)
    data = viewer.load_decisions(name)
    outcomes = {row["decision"]: row for row in viewer._jsonl(folder / "decisions.jsonl")
                if type(row.get("decision")) is int}
    for decision in data["decisions"]:
        decision.pop("screenshot", None)
        outcome = outcomes.get(decision["decision"], {})
        decision["frame_start"] = outcome.get("frame_start")
        decision["frame_end"] = outcome.get("frame_end")
    state = folder / "checkpoint.state"
    if not state.is_file():
        return data, None, 0
    raw = state.read_bytes()
    return data, base64.b64encode(raw).decode(), len(raw)


def _run_row(run: dict) -> dict:
    name = run["name"]
    folder = viewer.apply_nested(RUNS / name)
    config = viewer._json(RUNS / name / "config.json")
    if folder != RUNS / name:
        config.update(viewer._json(folder / "config.json"))
    settings = viewer._run_settings(run, config, viewer._json(folder / "score.json"))
    model = run.get("model", "?")
    harness = run.get("harness", "tau")
    return {"name": name, "model": model,
            "label": catalog.display_model(model), "harness": harness,
            "harness_label": HARNESSES[harness].label if harness in HARNESSES else harness,
            "mode": settings.get("mode"), "version": settings.get("benchmark_version"),
            "frames": run.get("frames", 0), "duration": run.get("duration", 0),
            "progress": run.get("progress")}


def publish_runs(out: Path, limit: int | None = None) -> list[dict]:
    rows = []
    for run in viewer.scan_runs():
        folder = viewer.apply_nested(RUNS / run["name"])
        state = folder / "checkpoint.state"
        # Only the current methodology, and only rollouts that actually played:
        # a run that failed before producing an input trace has nothing to show.
        if run.get("score_status") != "completed" or not state.is_file() or not state.stat().st_size:
            continue
        row = _run_row(run)
        if row["version"] != BENCHMARK_VERSION:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (row["label"], row["name"]))
    if limit:
        rows = rows[:limit]
    for row in rows:
        data, trace, frames = _run_facts(row["name"])
        row["frames"] = frames or row["frames"]
        payload = {**row, "fps": 30, "timeline_exact": data["timeline_exact"],
                   "input": trace, "decisions": data["decisions"]}
        # The file mirrors the run name, so the client builds the URL itself.
        path = out / "data" / "runs" / f"{row['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return rows


def publish_leaderboards(out: Path) -> dict:
    """One file holding every mode and budget, so the client never refetches."""
    first = viewer.leaderboard()
    entries = {}
    for mode in first["modes"]:
        base = viewer.leaderboard(None, mode)
        entries[mode] = {"budgets": base["budgets"],
                         "byBudget": {str(b): viewer.leaderboard(b, mode) for b in base["budgets"]}}
    document = {"version": BENCHMARK_VERSION, "modes": first["modes"],
                "defaultBudget": first["budget"], "entries": entries}
    path = out / "data" / "leaderboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    return document


def copy_assets(out: Path) -> None:
    """The reused stylesheet, the lab logos, and the built player."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STATIC / "viewer.css", out / "site.css")
    logos = out / "logos"
    logos.mkdir(parents=True, exist_ok=True)
    for logo in (STATIC / "logos").glob("*.svg"):
        shutil.copy2(logo, logos / logo.name)
    player = out / "player"
    player.mkdir(parents=True, exist_ok=True)
    for name in ("open8.js", "open8.wasm", "open8.data"):
        shutil.copy2(WASM / name, player / name)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="publish only the first N runs")
    args = parser.parse_args(argv)
    out = args.out.resolve()

    shutil.rmtree(out / "data", ignore_errors=True)
    copy_assets(out)
    rows = publish_runs(out, args.limit)
    document = publish_leaderboards(out)

    print(f"{out}")
    print(f"  leaderboard: {len(document['modes'])} mode(s), "
          f"{sum(len(e['budgets']) for e in document['entries'].values())} budget file(s)")
    print(f"  runs: {len(rows)}")


if __name__ == "__main__":
    main()
