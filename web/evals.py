"""Launch the existing CLI in isolated processes and keep a small job registry."""

import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "examples" / "llm.py"
APIS = ("openai-responses", "openai-completions", "anthropic",
        "google-generative-ai", "mistral-conversations")
_MAX_PROCESSES = 4
_lock = threading.RLock()
_jobs = {}
_processes = {}


def _rows(path):
    try:
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
                except ValueError:
                    pass  # A running process may still be writing the last line.
    except OSError:
        pass


def _public(job):
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _write(job):
    path = Path(job["_meta"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_public(job), indent=2) + "\n")
    temporary.replace(path)


def _progress(job):
    if job["status"] == "queued":
        job.update(decisions=0, frames=0, elapsed=0, tokens=None)
        return
    folder = Path(job["_folder"])
    outcomes = list(_rows(folder / "decisions.jsonl"))
    job["decisions"] = len(outcomes)
    job["frames"] = max((row.get("frame_end", 0) for row in outcomes), default=0)
    job["elapsed"] = max(0, job.get("finished_at", time.time()) - job["started_at"])
    usages = [row["usage"] for row in _rows(folder / "messages.jsonl")
              if row.get("role") == "assistant" and isinstance(row.get("usage"), dict)]
    job["tokens"] = sum(u.get("totalTokens", u.get("total_tokens", 0)) or 0 for u in usages) if usages else None


def _watch(job, proc, secret):
    _, stderr = proc.communicate()
    with _lock:
        job["finished_at"] = time.time()
        job["status"] = "cancelled" if job.get("_cancelled") else "completed" if proc.returncode == 0 else "failed"
        if job["status"] == "failed":
            # Never persist raw provider stderr: it can include credentials.
            lines = (stderr or "").replace(secret, "[redacted]").strip().splitlines()
            last = lines[-1][-1000:] if lines else f"Evaluation exited with code {proc.returncode}."
            job["error"] = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception): ", "", last)
        _progress(job)
        _write(job)
        _processes.pop(job["id"], None)
        _pump()


def _launch(job, options, secret):
    """Spawn the CLI subprocess. Caller holds the lock; job becomes running or failed."""
    args = [sys.executable, str(CLI), f"--output={job['_folder']}", f"--model={job['model']}"]
    args += [f"--{key.replace('_', '-')}={value}" for key, value in options.items()]
    env = os.environ.copy()
    env["CELESTEBENCH_API_KEY"] = secret
    job["status"], job["started_at"], job["error"] = "running", time.time(), None
    _write(job)
    try:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError:
        job.update(status="failed", error="Could not start the evaluation process.",
                   finished_at=time.time())
        _progress(job)
        _write(job)
        return
    _processes[job["id"]] = proc
    threading.Thread(target=_watch, args=(job, proc, secret), daemon=True).start()


def _pump():
    """Start queued evaluations in submission order while a process slot is free."""
    with _lock:
        for job in list(_jobs.values()):
            if len(_processes) >= _MAX_PROCESSES:
                break
            if job["status"] != "queued":
                continue
            options, secret = job.pop("_options", None), job.pop("_secret", None)
            if options is None:
                job.update(status="failed", error="Queued evaluation lost its settings.",
                           finished_at=time.time())
                continue
            _launch(job, options, secret)


def start_eval(payload, runs):
    defaults = {"timeout": 120, "max_frames": 30, "max_actions": 4, "max_images": 3}
    allowed = set(defaults) | {"model", "models", "api", "base_url", "key_env", "api_key",
                               "fps", "reasoning_effort", "thinking_budget"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown evaluation settings.")
    models = payload.get("models", [payload.get("model")])
    if not isinstance(models, list) or not models or len(models) > 24:
        raise ValueError("Provide one to 24 models.")
    entries = []
    for raw in models:
        raw = raw.strip() if isinstance(raw, str) else None
        if not raw:
            raise ValueError("Enter a model identifier on every line.")
        model, sep, override = raw.partition("@")
        if sep and override not in APIS:
            raise ValueError(f"Unknown API '{override}' for model '{model}'; use one of {', '.join(APIS)}.")
        if not model or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError("Enter a model identifier on every line.")
        entries.append((model, override or None))
    if not entries:
        raise ValueError("Enter a model identifier on every line.")
    api = payload.get("api", "openai-responses")
    if api not in APIS:
        raise ValueError("Unknown API protocol.")
    base_url = payload.get("base_url", {
        "anthropic": "https://api.anthropic.com/v1",
        "google-generative-ai": "https://generativelanguage.googleapis.com/v1beta",
        "mistral-conversations": "https://api.mistral.ai/v1",
    }.get(api, "https://api.openai.com/v1"))
    if not isinstance(base_url, str):
        raise ValueError("Enter an HTTP(S) base URL.")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or any(c.isspace() for c in base_url)):
        raise ValueError("Base URL must be HTTP(S), without credentials, query or fragment.")
    key_env = payload.get("key_env") or {
        "anthropic": "ANTHROPIC_API_KEY", "google-generative-ai": "GEMINI_API_KEY",
        "mistral-conversations": "MISTRAL_API_KEY",
    }.get(api, "OPENAI_API_KEY")
    if not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key_env):
        raise ValueError("Invalid key environment variable name.")
    secret = payload.get("api_key") or os.environ.get(key_env)
    if not isinstance(secret, str) or not secret or "\0" in secret:
        raise ValueError(f"Enter an API key or set {key_env} before starting the viewer.")
    options = {"api": api, "base_url": base_url, "key_env": "CELESTEBENCH_API_KEY"}
    for key in ("max_frames", "max_actions", "max_images"):
        value = payload.get(key, defaults[key])
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
        options[key] = value
    value = payload.get("timeout", defaults["timeout"])
    if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a positive number of seconds.")
    options["timeout"] = value
    for key in ("thinking_budget", "fps"):
        value = payload.get(key)
        if value is None:
            continue
        if (type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                or (key == "thinking_budget" and type(value) is not int)):
            raise ValueError(f"Invalid {key} budget.")
        options[key] = value
    # Tau resolves the output budget itself (and raises it for thinking), so
    # only the provider choice for thinking needs validation here. The flag is
    # only forwarded to anthropic-routed jobs; other providers have no such knob.
    effort = payload.get("reasoning_effort", "low")
    if not isinstance(effort, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", effort):
        raise ValueError("Invalid reasoning effort.")
    options["reasoning_effort"] = effort
    if importlib.util.find_spec("tau_ai") is None:
        raise RuntimeError("Install LLM support: uv sync --extra llm; then restart the viewer.")

    runs = Path(runs).resolve()
    with _lock:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        jobs = []
        for model, override in entries:
            job_id = uuid.uuid4().hex
            safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model).strip("._")[:100] or "model"
            name = f"{safe_model}/{stamp}-{job_id[:8]}"
            folder = runs / name
            meta = runs / ".evals" / f"{job_id}.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            folder.parent.mkdir(parents=True, exist_ok=True)
            entry_options = dict(options)
            if override:
                entry_options["api"] = override
            if entry_options["api"] != "anthropic":
                entry_options.pop("thinking_budget", None)
            job = {"id": job_id, "name": name, "model": model, "api": entry_options["api"],
                   "status": "queued", "error": None, "decisions": 0, "timeout": options["timeout"],
                   "frames": 0, "elapsed": 0, "tokens": None, "started_at": time.time(),
                   "_folder": str(folder), "_meta": str(meta)}
            if len(_processes) >= _MAX_PROCESSES:
                # Options and the secret stay in memory only; never persisted to disk.
                job["_options"], job["_secret"] = entry_options, secret
                _write(job)
            else:
                _launch(job, entry_options, secret)
            _jobs[job_id] = job
            jobs.append(_public(job))
        return jobs


def list_evals(runs):
    runs = Path(runs).resolve()
    with _lock:
        result = []
        for path in (runs / ".evals").glob("*.json"):
            try:
                data = json.loads(path.read_text())
                job = _jobs.get(data["id"])
                if job is None:
                    folder = (runs / data["name"]).resolve()
                    if not folder.is_relative_to(runs):
                        continue
                    job = data | {"_folder": str(folder), "_meta": str(path)}
                    if job["status"] in {"running", "queued"}:
                        job.update(status="interrupted", error="Viewer stopped before this evaluation finished.",
                                   finished_at=path.stat().st_mtime)
                        _write(job)
                _progress(job)
                result.append(_public(job))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda job: job["started_at"], reverse=True)


def stop_eval(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        proc = _processes.get(job_id)
        if proc is None:
            if job["status"] == "queued":
                job.pop("_options", None)
                job.pop("_secret", None)
                job.update(status="cancelled", finished_at=time.time())
                _write(job)
            return _public(job)
        job["_cancelled"] = True
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    except ProcessLookupError:
        pass
    with _lock:
        job.update(status="cancelled", finished_at=time.time())
        _progress(job)
        _write(job)
        _pump()
        return _public(job)


def forget_run(name, runs):
    """Forget terminal evaluation metadata associated with a deleted run."""
    runs = Path(runs).resolve()
    with _lock:
        matches = []
        for path in (runs / ".evals").glob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError, TypeError):
                continue
            if data.get("name") != name:
                continue
            job = _jobs.get(data.get("id"))
            if job and job.get("status") in {"running", "queued"}:
                raise RuntimeError("Stop the evaluation before deleting its run.")
            matches.append((path, data.get("id")))
        for path, job_id in matches:
            path.unlink(missing_ok=True)
            _jobs.pop(job_id, None)


def shutdown():
    with _lock:
        # Cancel queued jobs first so stopping runners cannot pump them back in.
        queued = [job["id"] for job in _jobs.values() if job["status"] == "queued"]
    for job_id in queued + list(_processes):
        stop_eval(job_id)
