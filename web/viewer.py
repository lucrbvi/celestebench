"""A minimal web viewer for rollout runs. Usage: uv run python -m web.viewer"""

import json
import mimetypes
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import av

from celestebench import catalog, cost

from . import evals

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STATIC = Path(__file__).resolve().parent / "static"
FPS = 30.0


def _jsonl(path: Path) -> list[dict]:
    """Read complete JSON objects and ignore a truncated final write."""
    if not path.is_file():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


_video_cache: dict = {}


def _video_info(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    stat = path.stat()
    cached = _video_cache.get(path)
    if cached and cached[0] == (stat.st_mtime_ns, stat.st_size):
        return cached[1]
    try:
        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            fps = float(stream.average_rate) if stream and stream.average_rate else None
            duration = (float(stream.duration * stream.time_base) if stream and stream.duration
                        and stream.time_base else None)
            if duration is None and container.duration:
                duration = float(container.duration / av.time_base)
            info = duration, fps
    except Exception:
        info = None, None
    _video_cache[path] = (stat.st_mtime_ns, stat.st_size), info
    return info


def apply_nested(folder: Path) -> Path:
    """External harnesses (Codex) nest the rollout engine one level deeper."""
    return folder / "rollout" if (folder / "rollout").is_dir() else folder


_export_locks: dict[str, threading.Lock] = {}
_export_locks_guard = threading.Lock()


def annotated(path: Path) -> Path:
    """The annotated export for a recording, generated once and cached next to it.

    The recording and the export share fps and frame count, so the viewer
    timeline stays aligned. Falls back to the raw recording if generation
    fails."""
    from . import export

    run_name = path.parent.relative_to(RUNS.resolve()).as_posix()
    if run_name.endswith("/rollout"):
        run_name = run_name[: -len("/rollout")]
    output = path.parent / "export.mp4"
    stamp = max(path.stat().st_mtime_ns, Path(export.__file__).stat().st_mtime_ns)
    with _export_locks_guard:
        lock = _export_locks.setdefault(str(output), threading.Lock())
    with lock:
        if output.is_file() and output.stat().st_mtime_ns >= stamp:
            return output
        try:
            export.export(run_name, output)
        except (Exception, SystemExit):
            return path
    return output if output.is_file() else path


def scan_runs() -> list[dict]:
    if not RUNS.is_dir():
        return []
    jobs = {job["name"]: job for job in evals.list_evals(RUNS)}
    folders = {RUNS / name for name in jobs}
    for path in RUNS.rglob("*"):
        if path.is_file() and path.name in {"config.json", "messages.jsonl", "actions.jsonl", "decisions.jsonl", "rollout.mp4"}:
            folders.add(path.parent)
    for name, job in jobs.items():
        if job.get("harness") == "codex":
            folders.discard(RUNS / name / "rollout")  # merged into its wrapper row
    runs = []
    for folder in sorted(folders):
        try:
            rel = folder.relative_to(RUNS).as_posix()
            job = jobs.get(rel, {})
            # Harness wrappers hold the metadata; game files sit one level down,
            # so the wrapper row already covers the nested rollout directory.
            data_folder = apply_nested(folder)
            config = json.loads((data_folder / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                config = {}
        except (OSError, ValueError, UnicodeError):
            continue
        outcomes = _jsonl(data_folder / "decisions.jsonl")
        score = _json(data_folder / "score.json")
        video_path = data_folder / "rollout.mp4"
        duration, fps = _video_info(video_path)
        video = None
        if video_path.is_file() and (duration is not None or fps is not None):
            video = "/video/" + quote(data_folder.relative_to(RUNS).as_posix() + "/rollout.mp4")
        runs.append({"name": rel, "model": job.get("model", config.get("model", "?")),
                     "harness": job.get("harness", "tau"),
                     "status": job.get("status", "archived"),
                     "timeout": job.get("timeout", config.get("timeout")),
                     "frames": job.get("frames", outcomes[-1].get("frame_end", 0) if outcomes else 0),
                     "elapsed": job.get("elapsed"), "tokens": job.get("tokens"),
                     "video": video,
                     "duration": round(duration, 2) if duration is not None else 0,
                     "fps": fps,
                     "progress": score.get("progress"),
                     "score_timing": score.get("timing"),
                     "score_status": score.get("status")})
    return runs


# A leaderboard row is one model on one harness with one set of episode
# settings. The level is folded (Tau used to call no reasoning "none"), and
# prompt provenance is ignored so a verified and an unverified run of the same
# setup share a row.
_LEADERBOARD_SETTINGS = ("harness", "provider", "api", "base_url", "thinking_level",
                         "thinking_budget", "fps", "max_frames", "max_actions",
                         "max_images", "frames", "strict_timeout", "lite")

_THINKING_ALIASES = {"none": "off"}


def _thinking_level(config: dict) -> str | None:
    """The canonical level, whether the run stored thinking_level or the older
    reasoning_effort. Tau renamed "none" to "off" without a data migration."""
    level = config.get("thinking_level") or config.get("reasoning_effort")
    return _THINKING_ALIASES.get(level.lower(), level) if isinstance(level, str) else None


def _run_settings(run: dict, config: dict, score: dict) -> dict:
    """The settings that can change a policy's episode, from the config merged
    with the options the scorer recorded."""
    options = score.get("options")
    if isinstance(options, dict):
        config = config | options
    settings = {key: config.get(key) for key in _LEADERBOARD_SETTINGS if key in config}
    settings["harness"] = run.get("harness") or settings.get("harness") or "tau"
    settings["thinking_level"] = _thinking_level(config)
    settings["benchmark_version"] = config.get("benchmark_version")
    return {key: settings.get(key) for key in (*_LEADERBOARD_SETTINGS, "benchmark_version")}


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and value not in {float("inf"), float("-inf")} else None


def _progress_at(folder: Path, budget: float) -> float | None:
    rows = _jsonl(folder / "progress.jsonl")
    best = None
    for row in rows:
        elapsed = _number(row.get("elapsed"))
        progress = _number(row.get("progress"))
        if elapsed is None or progress is None or not 0 <= elapsed <= budget or not 0 <= progress <= 100:
            continue
        if best is None or elapsed >= best[0]:
            best = elapsed, progress
    return best[1] if best is not None else None


def leaderboard(budget: float | None = None) -> dict:
    """Aggregate grounded progress at one shared wall-clock budget."""
    candidates = []
    budgets = set()
    excluded = {"replay": 0, "unknown_timing": 0}
    for run in scan_runs():
        folder = apply_nested(RUNS / run["name"])
        score = _json(folder / "score.json")
        # A launcher failure (e.g. a provider usage limit) must not hide a
        # rollout that still finished its full budget and scored.
        if (run.get("status") not in {"completed", "archived"}
                and score.get("status") != "completed"):
            continue
        if score.get("invalid_reason"):
            continue
        timing = score.get("timing")
        if timing == "replay":
            excluded["replay"] += 1
        elif score and timing != "wall_clock":
            excluded["unknown_timing"] += 1
        elapsed = _number(score.get("elapsed"))
        timeout = _number(run.get("timeout"))
        if timeout is not None and timeout > 0:
            budgets.add(timeout)
        config = _json(RUNS / run["name"] / "config.json")
        if folder != RUNS / run["name"]:
            config.update(_json(folder / "config.json"))
        codex = run.get("harness") == "codex"
        candidates.append({"run": run, "folder": folder, "score": score,
                          "timing": timing, "elapsed": elapsed, "codex": codex,
                          "usage": (RUNS / run["name"] / "codex.jsonl" if codex
                                    else folder / "messages.jsonl"),
                          "settings": _run_settings(run, config, score)})
    if not budgets:
        budgets = {item["elapsed"] for item in candidates if item["elapsed"] is not None}
    choices = sorted(budgets)
    if budget is None or _number(budget) is None or budget <= 0:
        budget = choices[0] if choices else None
    groups = {}
    for item in candidates:
        run, score = item["run"], item["score"]
        key = (run.get("model", "?"), json.dumps(item["settings"], sort_keys=True,
                                                   separators=(",", ":")))
        group = groups.setdefault(key, {"model": run.get("model", "?"),
                                        "settings": item["settings"], "scores": [],
                                        "scored_runs": [], "costs": [], "runs": 0, "unscored": 0})
        group["runs"] += 1
        value = None
        if (budget is not None and score.get("metric") == "grounded_height_v1"
                and score.get("status") == "completed"
                and item["timing"] == "wall_clock"
                and item["elapsed"] is not None and item["elapsed"] >= budget
                and score.get("progress") is not None):
            value = _progress_at(item["folder"], budget)
        if value is None:
            group["unscored"] += 1
        else:
            group["scores"].append(value)
            group["scored_runs"].append({"name": run["name"], "progress": value})
            priced = cost.rollout_cost(item["usage"], run.get("model", "?"), codex=item["codex"])
            if priced is not None:
                group["costs"].append(priced)
    rows = []
    for group in groups.values():
        scores = group.pop("scores")
        costs = group.pop("costs")
        group["scored_runs"].sort(key=lambda link: (-link["progress"], link["name"]))
        group["scored"] = len(scores)
        group["progress"] = round(sum(scores) / len(scores), 3) if scores else None
        group["cost"] = round(sum(costs) / len(costs), 6) if costs else None
        producer = catalog.producer(group["model"])
        group["producer"] = producer
        group["producer_name"] = catalog.producer_name(producer) if producer else None
        group["status"] = "scored" if scores else "unscored"
        rows.append(group)
    rows.sort(key=lambda row: (row["progress"] is None, -(row["progress"] or 0), row["model"]))
    return {"budget": budget, "budgets": choices, "groups": rows, "excluded": excluded}


def _codex_frame(item: dict) -> int | None:
    """The last frame a completed Codex `play` observed, kept in its MCP reply."""
    result = item.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            payload = json.loads(block.get("text") or "")
        except (TypeError, ValueError):
            continue
        frames = payload.get("frame_ids") if isinstance(payload, dict) else None
        if isinstance(frames, list) and frames and type(frames[-1]) is int:
            return frames[-1]
    return None


def _codex_trace(run_folder: Path, outcomes: list[dict]) -> dict[int, dict]:
    """Codex keeps its LLM trace beside the nested rollout, not in messages.jsonl.
    Map each successful `play` to the decision whose episode ends on the frame it
    observed, so the viewer shows the same CoT, reply, and tool call as Tau."""
    trace = run_folder / "codex.jsonl"
    if not trace.is_file():
        return {}
    by_frame = {row["frame_end"]: row["decision"] for row in outcomes
                if type(row.get("decision")) is int and type(row.get("frame_end")) is int}
    merged, thinking, text = {}, [], []
    for row in _jsonl(trace):
        item = row.get("item") if row.get("type") == "item.completed" else None
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            thinking.append(item.get("text") or "")
        elif kind == "agent_message":
            text.append(item.get("text") or "")
        elif kind == "mcp_tool_call" and item.get("tool") == "play":
            decision = by_frame.get(_codex_frame(item))
            if decision is None:
                continue  # a play against an ended episode has no decision
            merged[decision] = {"thinking": "\n".join(x for x in thinking if x) or None,
                                "text": "\n".join(x for x in text if x) or None,
                                "tool": item.get("arguments")}
            thinking, text = [], []
    return merged


def load_decisions(name: str) -> dict:
    folder = apply_nested(RUNS / name)
    rows = _jsonl(folder / "actions.jsonl")
    outcomes = _jsonl(folder / "decisions.jsonl")
    messages = _jsonl(folder / "messages.jsonl")
    _, native_fps = _video_info(folder / "rollout.mp4")
    fps = native_fps or FPS
    decisions, current = [], None
    for msg in messages:
        if msg.get("role") == "user":
            if current is not None and not current.get("_assistant"):
                current["status"], current["partial"] = "interrupted", True
            current = {"decision": len(decisions), "screenshot": None, "thinking": None,
                       "text": None, "tool": None, "actions": [], "latency": 0.0,
                       "t": 0.0, "from": 0.0, "status": "unexecuted", "partial": False}
            content = msg.get("content", [])
            if isinstance(content, dict):
                content = [content]
            for c in content if isinstance(content, list) else []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "image":
                    if c.get("data"):
                        current["screenshot"] = f"data:{c.get('mimeType', c.get('mime_type', 'image/png'))};base64,{c['data']}"
            decisions.append(current)
        elif msg.get("role") == "assistant":
            if current is None or current.get("_assistant"):
                current = {"decision": len(decisions), "screenshot": None, "thinking": None,
                           "text": None, "tool": None, "actions": [], "latency": 0.0,
                           "t": 0.0, "from": 0.0, "status": "unexecuted", "partial": True}
                decisions.append(current)
            current["_assistant"] = True
            stop = msg.get("status") or msg.get("stop_reason") or msg.get("stopReason")
            if stop in {"timeout", "cancelled", "canceled", "error", "aborted"}:
                current["status"] = "cancelled" if stop in {"cancelled", "canceled"} else stop
                current["error"] = (msg.get("error") or msg.get("error_message")
                                     or msg.get("errorMessage"))
                current["partial"] = True
            content = msg.get("content", [])
            if isinstance(content, dict):
                content = [content]
            for c in content if isinstance(content, list) else []:
                if not isinstance(c, dict):
                    continue
                kind = c.get("type")
                if kind == "thinking":
                    current["thinking"] = "\n".join(x for x in [current["thinking"], c.get("thinking") or c.get("text")] if x)
                elif kind == "text":
                    current["text"] = "\n".join(x for x in [current["text"], c.get("text")] if x)
                elif kind in {"toolCall", "tool_call", "toolUse", "tool_use"}:
                    arguments = c.get("arguments", c.get("input"))
                    if current["tool"] is None:
                        current["tool"] = arguments
                    elif isinstance(current["tool"], list):
                        current["tool"].append(arguments)
                    else:
                        current["tool"] = [current["tool"], arguments]
            for call in msg.get("tool_calls", []) if isinstance(msg.get("tool_calls", []), list) else []:
                arguments = call.get("arguments", call.get("input")) if isinstance(call, dict) else call
                if current["tool"] is None:
                    current["tool"] = arguments
                elif isinstance(current["tool"], list):
                    current["tool"].append(arguments)
                else:
                    current["tool"] = [current["tool"], arguments]
            if msg.get("latency") is not None:
                current["latency"] = float(msg["latency"])
    if current is not None and not current.get("_assistant"):
        current["status"], current["partial"] = "interrupted", True
    # Outcomes keep failed attempts visible even for policies without LLM traces.
    for row in outcomes + rows:
        index = row.get("decision")
        if type(index) is not int or index < 0:
            continue
        while len(decisions) <= index:
            decisions.append({"decision": len(decisions), "screenshot": None, "thinking": None,
                              "text": None, "tool": None, "actions": [], "latency": 0.0,
                              "t": 0.0, "from": 0.0, "status": "unknown", "partial": False})
    # Rollouts persist observations separately, so they remain available when
    # an LLM trace is missing or contains no image content.
    for row in outcomes:
        index = row.get("decision")
        screenshot = row.get("screenshot")
        if (type(index) is int and 0 <= index < len(decisions)
                and decisions[index]["screenshot"] is None and isinstance(screenshot, str)):
            path = (folder / screenshot).resolve()
            if path.is_relative_to(folder.resolve()) and path.is_file():
                rel = path.relative_to(RUNS.resolve()).as_posix()
                decisions[index]["screenshot"] = "/screenshot/" + quote(rel)
    outcome_by_id = {r["decision"]: r for r in outcomes
                     if type(r.get("decision")) is int and r["decision"] >= 0}
    for decision, trace in _codex_trace(RUNS / name, outcomes).items():
        if 0 <= decision < len(decisions):
            decisions[decision].update(trace)
    for row in rows:
        index = row.get("decision")
        if type(index) is not int or index < 0:
            continue
        try:
            action = {"buttons": int(row["buttons"]), "frames": int(row["frames"])}
            if row.get("action") == "wait":
                action["wait"] = True
            if action["frames"] < 1:
                continue
            for key in ("frame_start", "frame_end"):
                if row.get(key) is not None:
                    action[key] = int(row[key])
            decisions[index]["actions"].append(action)
            decisions[index]["latency"] = max(decisions[index]["latency"], float(row.get("latency") or 0))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        pace = float(config.get("fps") or 0) if isinstance(config, dict) else 0
    except (OSError, ValueError, UnicodeError):
        pace = 0
    cursor = 1
    timeline_exact = bool(decisions)
    for d in decisions:
        outcome = outcome_by_id.get(d["decision"], {})
        start = outcome.get("frame_start", cursor)
        d["from"] = round(start / fps, 6)
        d["latency"] = outcome.get("latency", d["latency"])
        action_cursor = start + round(d["latency"] * pace)
        timeline_exact &= "frame_start" in outcome and "frame_end" in outcome
        for action in d["actions"]:
            exact = "frame_start" in action and "frame_end" in action
            timeline_exact &= exact
            action["frame_start"] = action.get("frame_start", action_cursor)
            action["frame_end"] = action.get("frame_end", action["frame_start"] + action["frames"])
            action["t0"] = round(action["frame_start"] / fps, 6)
            action["t1"] = round(action["frame_end"] / fps, 6)
            action_cursor = action["frame_end"]
        d["t"] = d["actions"][0]["t0"] if d["actions"] else d["from"]
        cursor = outcome.get("frame_end", action_cursor if d["actions"] else start)
        if outcome.get("status"):
            d["status"] = outcome["status"]
            d["error"] = outcome.get("error")
        elif d["actions"]:
            d["status"] = "played"
        d.pop("_assistant", None)
    return {"decisions": decisions, "timeline_exact": bool(timeline_exact)}


class Handler(BaseHTTPRequestHandler):
    timeout = 60

    def do_GET(self):
        request = urlsplit(self.path)
        request_path = request.path
        if request_path == "/":
            return self.asset("leaderboard.html", "text/html; charset=utf-8")
        if request_path == "/runs":
            return self.asset("viewer.html", "text/html; charset=utf-8")
        if request_path == "/evals":
            return self.asset("evals.html", "text/html; charset=utf-8")
        if request_path == "/leaderboard":
            return self.asset("leaderboard.html", "text/html; charset=utf-8")
        if request_path == "/viewer.css":
            return self.asset("viewer.css", "text/css; charset=utf-8")
        if re.fullmatch(r"/logos/[a-z0-9-]+\.svg", request_path):
            return self.asset(request_path[1:], "image/svg+xml")
        if request_path == "/api/runs":
            return self.json(scan_runs())
        if request_path == "/api/evals":
            return self.json(evals.list_evals(RUNS))
        if request_path == "/api/leaderboard":
            raw_budget = parse_qs(request.query).get("budget", [None])[0]
            try:
                selected = float(raw_budget) if raw_budget is not None else None
            except ValueError:
                selected = None
            return self.json(leaderboard(selected))
        if request_path == "/api/harnesses":
            return self.json(evals.harness_catalog())
        if request_path.startswith("/api/run/"):
            name = unquote(request_path[len("/api/run/"):])
            folder = (RUNS / name).resolve()
            root = RUNS.resolve()
            if not folder.is_relative_to(root):
                return self.send_error(404)
            name = folder.relative_to(root).as_posix()
            job = next((job for job in evals.list_evals(RUNS) if job["name"] == name), None)
            if not folder.is_dir() and job is None:
                return self.send_error(404)
            data = load_decisions(name)
            if job and job["status"] == "running" and data["decisions"]:
                last = data["decisions"][-1]
                if last["status"] == "interrupted":
                    last["status"], last["partial"] = "thinking", False
            return self.json(data)
        if request_path.startswith("/live/"):
            name = unquote(request_path[len("/live/"):])
            folder = (RUNS / name).resolve()
            root = RUNS.resolve()
            if not folder.is_relative_to(root):
                return self.send_error(404)
            name = folder.relative_to(root).as_posix()
            job = next((job for job in evals.list_evals(RUNS) if job["name"] == name), None)
            if job is None or job["status"] != "running":
                return self.send_error(404)
            return self.live(folder, lambda: evals.job_running(job.get("id")))
        if request_path.startswith("/video/"):
            path = (RUNS / unquote(request_path[len("/video/"):])).resolve()
            if not path.is_relative_to(RUNS.resolve()):
                return self.send_error(404)
            return self.video(path)
        if request_path.startswith("/export/"):
            folder = (RUNS / unquote(request_path[len("/export/"):])).resolve()
            root = RUNS.resolve()
            if not folder.is_relative_to(root) or not folder.is_dir():
                return self.send_error(404)
            name = folder.relative_to(root).as_posix()
            recording = apply_nested(folder) / "rollout.mp4"
            if not recording.is_file():
                return self.send_error(404)
            return self.video(annotated(recording), name.replace("/", "_") + ".mp4")
        if request_path.startswith("/screenshot/"):
            path = (RUNS / unquote(request_path[len("/screenshot/"):])).resolve()
            if not path.is_relative_to(RUNS.resolve()):
                return self.send_error(404)
            return self.image(path)
        self.send_error(404)

    def do_POST(self):
        # A page on another origin must not be able to spend local API credentials.
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if (urlsplit("http://" + host).hostname not in {"localhost", "127.0.0.1", "::1"}
                or (origin is not None and origin != "http://" + host)):
            return self.json({"error": "Requests must come from this local viewer."}, 403)
        if self.headers.get_content_type() != "application/json":
            return self.json({"error": "Expected application/json."}, 415)
        path = urlsplit(self.path).path
        stop = re.fullmatch(r"/api/evals/([a-zA-Z0-9_-]+)/stop", path)
        if path != "/api/evals" and not stop:
            return self.json({"error": "Unknown endpoint."}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Expected a JSON body up to 64 KiB.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            job = evals.stop_eval(stop[1]) if stop else evals.start_eval(payload, RUNS)
        except (ValueError, UnicodeError) as error:
            return self.json({"error": str(error)}, 400)
        except KeyError:
            return self.json({"error": "Unknown evaluation."}, 404)
        except RuntimeError as error:
            return self.json({"error": str(error)}, 409)
        return self.json(job, 200 if stop else 201)

    def do_DELETE(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if (urlsplit("http://" + host).hostname not in {"localhost", "127.0.0.1", "::1"}
                or (origin is not None and origin != "http://" + host)):
            return self.json({"error": "Requests must come from this local viewer."}, 403)
        path = urlsplit(self.path).path
        if not path.startswith("/api/run/"):
            return self.json({"error": "Unknown endpoint."}, 404)
        raw = RUNS / unquote(path[len("/api/run/"):])
        root = RUNS.resolve()
        folder = raw.resolve()
        if (folder == root or not folder.is_relative_to(root) or not folder.is_dir()
                or folder.relative_to(root).parts[0] == ".evals"
                or any(part.is_symlink() for part in [raw, *raw.parents]
                       if part != RUNS and part.is_relative_to(RUNS))):
            return self.json({"error": "Unknown run."}, 404)
        name = folder.relative_to(root).as_posix()
        try:
            evals.forget_run(name, RUNS)
            shutil.rmtree(folder)
            if folder.parent != root:
                try:
                    folder.parent.rmdir()
                except OSError:
                    pass
        except RuntimeError as error:
            return self.json({"error": str(error)}, 409)
        except OSError:
            return self.json({"error": "Could not delete the run."}, 409)
        return self.json({"deleted": name})

    def asset(self, name, content_type):
        try:
            body = (STATIC / name).read_bytes()
        except OSError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def live(self, folder, running=None):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        modified = None
        last_push = 0.0
        # A disconnected client must end this thread: write errors propagate to
        # the outer handler instead of being swallowed as missing-file noise,
        # and the send timeout unblocks a client that stopped reading without
        # closing (its slot in the browser's per-host connection pool frees up).
        self.close_connection = True
        self.connection.settimeout(10)
        try:
            while True:
                # External harnesses create the nested rollout directory only
                # once the game starts, well after this request: re-resolve it
                # every iteration or the stream would watch an empty path.
                directory = apply_nested(folder)
                path, done = directory / "live.png", directory / "live.done"
                try:
                    current = path.stat().st_mtime_ns
                    frame = path.read_bytes() if current != modified else None
                except OSError:
                    frame = None
                    current = modified
                if frame is not None and time.monotonic() - last_push >= 0.08:
                    # Cap pushed frames well below the game rate: several live
                    # streams decoding full-rate PNGs can starve a browser tab.
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/png\r\nContent-Length: "
                        + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                    )
                    self.wfile.flush()
                    modified = current
                    last_push = time.monotonic()
                elif frame is not None:
                    modified = current
                if done.is_file():
                    break
                # Boot can take a while (starting Codex); keep the stream open
                # while the job runs instead of forcing a reload for a late frame.
                if running is not None and not running():
                    break
                time.sleep(1 / 30)
            self.wfile.write(b"--frame--\r\n")
        except OSError:
            pass

    def video(self, path: Path, download: str | None = None):
        if not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        if size == 0:
            return self.send_error(416)
        start, end, status = 0, size - 1, 200
        value = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
        if value and not match:
            return self.send_error(416)
        if match:
            left, right = match.groups()
            if not left and not right:
                return self.send_error(416)
            if left:
                start = int(left)
                if start >= size:
                    return self.send_error(416)
                end = int(right) if right else size - 1
            else:
                length = int(right)
                if length < 1:
                    return self.send_error(416)
                start = max(size - length, 0)
                end = size - 1
            if end < start:
                return self.send_error(416)
            end = min(end, size - 1)
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    data = f.read(min(remaining, 64 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def image(self, path: Path):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return self.send_error(404)
        try:
            body = path.read_bytes()
        except OSError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("http://localhost:8123")
    with ThreadingHTTPServer(("127.0.0.1", 8123), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            evals.shutdown()
