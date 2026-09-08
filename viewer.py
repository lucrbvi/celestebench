"""A minimal web viewer for rollout runs. Usage: uv run python viewer.py"""

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import av

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
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


def _video_info(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    try:
        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            fps = float(stream.average_rate) if stream and stream.average_rate else None
            duration = (float(stream.duration * stream.time_base) if stream and stream.duration
                        and stream.time_base else None)
            if duration is None and container.duration:
                duration = float(container.duration / av.time_base)
            return duration, fps
    except Exception:
        return None, None


def scan_runs() -> list[dict]:
    if not RUNS.is_dir():
        return []
    folders = set()
    for path in RUNS.rglob("*"):
        if path.is_file() and path.name in {"config.json", "messages.jsonl", "actions.jsonl", "decisions.jsonl", "rollout.mp4"}:
            folders.add(path.parent)
    runs = []
    for folder in sorted(folders):
        try:
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                config = {}
        except (OSError, ValueError, UnicodeError):
            config = {}
        rel = folder.relative_to(RUNS).as_posix()
        video_path = folder / "rollout.mp4"
        duration, fps = _video_info(video_path)
        runs.append({"name": rel, "model": config.get("model", "?"),
                     "video": "/video/" + quote(rel + "/rollout.mp4") if video_path.is_file()
                     and (duration is not None or fps is not None) else None,
                     "duration": round(duration, 2) if duration is not None else 0,
                     "fps": fps})
    return runs


def load_decisions(name: str) -> dict:
    folder = RUNS / name
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
    for row in rows:
        index = row.get("decision")
        if type(index) is not int or index < 0:
            continue
        try:
            action = {"buttons": int(row["buttons"]), "frames": int(row["frames"])}
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
    def do_GET(self):
        request_path = urlsplit(self.path).path
        if request_path == "/":
            return self.html()
        if request_path == "/api/runs":
            return self.json(scan_runs())
        if request_path.startswith("/api/run/"):
            name = unquote(request_path[len("/api/run/"):])
            folder = (RUNS / name).resolve()
            root = RUNS.resolve()
            if not folder.is_relative_to(root) or not folder.is_dir():
                return self.send_error(404)
            return self.json(load_decisions(folder.relative_to(RUNS).as_posix()))
        if request_path.startswith("/video/"):
            path = (RUNS / unquote(request_path[len("/video/"):])).resolve()
            if not path.is_relative_to(RUNS.resolve()):
                return self.send_error(404)
            return self.video(path)
        if request_path.startswith("/screenshot/"):
            path = (RUNS / unquote(request_path[len("/screenshot/"):])).resolve()
            if not path.is_relative_to(RUNS.resolve()):
                return self.send_error(404)
            return self.image(path)
        self.send_error(404)

    def html(self):
        try:
            body = (ROOT / "viewer.html").read_bytes()
        except OSError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def video(self, path: Path):
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
    ThreadingHTTPServer(("127.0.0.1", 8123), Handler).serve_forever()
