"""Launch the existing CLI in isolated processes and keep a small job registry."""

import importlib.util
import json
import math
import os
import re
import shutil
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
CODEX_CLI = ROOT / "examples" / "codex_vm.py"


def _find_limactl() -> Path:
    found = shutil.which("limactl")
    if found:
        return Path(found)
    for probe in ("/opt/homebrew/bin/limactl", "/usr/local/bin/limactl"):
        if Path(probe).is_file():  # brew installs may be missing from PATH
            return Path(probe)
    return Path("limactl")  # keeps PATH failures visible in the error


LIMACTL = _find_limactl()
APIS = ("openai-responses", "openai-completions", "anthropic",
        "google-generative-ai", "mistral-conversations")
# A provider tag reroutes a model line to the official endpoint and key env.
PROVIDERS = {
    "openai": {"api": "openai-responses", "base_url": "https://api.openai.com/v1",
               "key_env": "OPENAI_API_KEY"},
    "anthropic": {"api": "anthropic", "base_url": "https://api.anthropic.com/v1",
                  "key_env": "ANTHROPIC_API_KEY"},
    "google": {"api": "google-generative-ai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta",
               "key_env": "GEMINI_API_KEY"},
    "mistral": {"api": "mistral-conversations", "base_url": "https://api.mistral.ai/v1",
                "key_env": "MISTRAL_API_KEY"},
}
FAMILIES = {"claude": "anthropic", "gemini": "google", "gemma": "google",
            "gpt": "openai", "codex": "openai", "o1": "openai", "o3": "openai",
            "o4": "openai", "mistral": "mistral", "magistral": "mistral",
            "codestral": "mistral", "ministral": "mistral", "pixtral": "mistral"}
_OFFICIAL = {(p["api"], p["base_url"]): tag for tag, p in PROVIDERS.items()}
_MAX_PROCESSES = 4
_lock = threading.RLock()
_jobs = {}
_processes = {}


def _routing(model, override, api, base_url):
    """Per-model provider settings, or None to keep the dialog endpoint."""
    if override:
        return (PROVIDERS.get(override) or {"api": override}).copy()
    family = PROVIDERS.get(FAMILIES.get(re.split(r"[-_.:/@]", model, 1)[0].lower(), ""))
    if (api, base_url) in _OFFICIAL and family and (api, base_url) != (
            family["api"], family["base_url"]):
        return family.copy()
    return None


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
    folder = Path(job["_folder"]) / "rollout" if job.get("harness") == "codex" else Path(job["_folder"])
    engine = folder / "decisions.jsonl"
    if job["status"] == "queued":
        job.update(decisions=0, frames=0, elapsed=0, tokens=None, engine=True)
        return
    if job["status"] == "running" and not engine.is_file():
        # Countdown starts when the game actually runs, not while any harness
        # boots its emulator, VM or provider before the first decision.
        job.update(decisions=0, frames=0, elapsed=0, tokens=None, engine=False)
        return
    job["engine"] = True
    outcomes = list(_rows(engine))
    job["decisions"] = len(outcomes)
    job["frames"] = max((row.get("frame_end", 0) for row in outcomes), default=0)
    try:  # The rollout engine writes config.json at the first decision.
        engine_start = (folder / "config.json").stat().st_mtime
    except OSError:
        engine_start = job["started_at"]
    job["elapsed"] = max(0, job.get("finished_at", time.time()) - engine_start)
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
            stderr = stderr.replace(secret, "[redacted]") if secret else stderr
            lines = (stderr or "").strip().splitlines()
            last = lines[-1][-1000:] if lines else f"Evaluation exited with code {proc.returncode}."
            job["error"] = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception): ", "", last)
        if job["status"] == "failed" and job.get("harness") == "codex":
            # The guest Codex CLI reports its own whole failures in the JSON trace
            # (e.g. an authentication problem), which reads much better than the
            # process traceback.
            messages = [row.get("message") for row in _rows(Path(job["_folder"]) / "codex.jsonl")
                        if row.get("type") == "error" and row.get("message")]
            if messages:
                job["error"] = messages[-1]
                if ("authorizat" in job["error"].lower()
                        or "authentication" in job["error"].lower()):
                    job["error"] += (" — the viewer VM has no Codex access yet;"
                                     " run examples/codex_vm.py login once")
        _progress(job)
        _write(job)
        _processes.pop(job["id"], None)
        _pump()


def _launch(job, options, secret):
    """Spawn the harness subprocess. Caller holds the lock; job becomes running."""
    if job["harness"] == "codex":
        args = [sys.executable, str(CODEX_CLI), "run", f"--output={job['_folder']}",
                f"--model={job['model']}", f"--timeout={options['timeout']}",
                f"--max-frames={options['max_frames']}",
                f"--prompt={options['prompt']}"]
        if options.get("frames"):
            args.append(f"--frames={options['frames']}")
        if options.get("lite"):
            args.append("--lite")
        env = os.environ.copy()
    else:
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
        codex_running = any(_jobs[_id].get("harness") == "codex" for _id in _processes)
        for job in list(_jobs.values()):
            if len(_processes) >= _MAX_PROCESSES:
                break
            if job["status"] != "queued":
                continue
            # One Codex CLI evaluation at a time: they share the single VM and port.
            if job.get("harness") == "codex":
                if codex_running:
                    continue
                codex_running = True
            options, secret = job.pop("_options", None), job.pop("_secret", None)
            if options is None:
                job.update(status="failed", error="Queued evaluation lost its settings.",
                           finished_at=time.time())
                continue
            _launch(job, options, secret)


def _run_rows(payload, codex=False):
    """Per-run dicts; legacy `models` strings become one row each."""
    rows = payload.get("evals")
    if rows is None:
        return [{"model": model} if override is None else {"model": model, "tag": override}
                for model, override in _model_entries(payload, codex=codex)]
    if not isinstance(rows, list) or not rows or len(rows) > 24:
        raise ValueError("Provide one to 24 runs.")
    allowed = {"model", "timeout"} if codex else {"model", "tag", "reasoning_effort",
                                                  "thinking_budget", "timeout", "max_frames"}
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every run must be an object.")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"Unknown run setting: {sorted(unknown)[0]}.")
        model = row.get("model")
        if (not isinstance(model, str) or not model.strip() or not model or len(model) > 200
                or any(ord(c) < 32 for c in model)):
            raise ValueError("Enter a model identifier in every run.")
        if codex and row.get("tag"):
            raise ValueError("Provider overrides do not apply to the Codex harness.")
        parsed.append(dict(row))
    return parsed


def _model_entries(payload, codex=False):
    models = payload.get("models", [payload.get("model")])
    if not isinstance(models, list) or not models or len(models) > 24:
        raise ValueError("Provide one to 24 models.")
    entries = []
    for raw in models:
        raw = raw.strip() if isinstance(raw, str) else None
        if not raw:
            raise ValueError("Enter a model identifier on every line.")
        model, sep, override = raw.partition("@")
        if not model or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError("Enter a model identifier on every line.")
        if override and (codex or override not in APIS and override not in PROVIDERS):
            raise ValueError(f"Unknown API '{override}' for model '{model}';"
                             f" use one of {', '.join((*APIS, *PROVIDERS))}.")
        entries.append((model, override or None))
    if not entries:
        raise ValueError("Enter a model identifier on every line.")
    return entries


def _positive(payload, key, default, integer=False):
    value = payload.get(key, default)
    if (type(value) not in {int, float} or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0
            or key == "thinking_budget" and type(value) is not int
            or integer and type(value) is not int):
        raise ValueError("timeout must be a positive number of seconds." if key == "timeout"
                         else f"Invalid {key} budget.")
    return value


def _enqueue(runs, harness, plans):
    runs = Path(runs).resolve()
    with _lock:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        jobs = []
        for model, entry_options, secret in plans:
            job_id = uuid.uuid4().hex
            safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model).strip("._")[:100] or "model"
            name = f"{safe_model}/{stamp}-{job_id[:8]}"
            folder = runs / name
            meta = runs / ".evals" / f"{job_id}.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            folder.parent.mkdir(parents=True, exist_ok=True)
            job = {"id": job_id, "name": name, "model": model, "harness": harness,
                   "api": entry_options.get("api"), "status": "queued", "error": None,
                   "decisions": 0, "timeout": entry_options["timeout"], "frames": 0,
                   "elapsed": 0, "tokens": None, "started_at": time.time(),
                   "_folder": str(folder), "_meta": str(meta)}
            # Options and the secret stay in memory only; never persisted to disk.
            # Codex evaluations queue even with free slots: one VM, one port.
            if (harness == "codex"
                    and any(_jobs[_id].get("harness") == "codex" for _id in _processes)
                    or len(_processes) >= _MAX_PROCESSES):
                job["_options"], job["_secret"] = entry_options, secret
                _write(job)
            else:
                _launch(job, entry_options, secret)
            _jobs[job_id] = job
            jobs.append(_public(job))
        return jobs


def _support_tau_ai():
    return importlib.util.find_spec("tau_ai")


def start_eval(payload, runs):
    if not isinstance(payload, dict):
        raise ValueError("Unknown evaluation settings.")
    harness = payload.get("harness", "tau")
    if harness == "codex":
        return _codex_jobs(payload, runs)
    if harness != "tau":
        raise ValueError("Unknown harness.")
    defaults = {"timeout": 120, "max_frames": 30, "max_actions": 4, "max_images": 3}
    allowed = set(defaults) | {"harness", "evals", "model", "models", "api", "base_url",
                               "key_env", "api_key", "fps", "reasoning_effort",
                               "thinking_budget"}
    if set(payload) - allowed:
        raise ValueError("Unknown evaluation settings.")
    rows = _run_rows(payload)
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
    for key, integer in (("max_frames", True), ("max_actions", True), ("max_images", True),
                         ("timeout", False)):
        options[key] = _positive(payload, key, defaults[key], integer=integer)
    for key in ("thinking_budget", "fps"):
        if payload.get(key) is not None:
            options[key] = _positive(payload, key, 1)
    # Tau resolves the output budget itself (and raises it for thinking), so
    # only the provider choice for thinking needs validation here. The flag is
    # only forwarded to anthropic-routed jobs; other providers have no such knob.
    effort = payload.get("reasoning_effort", "low")
    if not isinstance(effort, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", effort):
        raise ValueError("Invalid reasoning effort.")
    options["reasoning_effort"] = effort
    if _support_tau_ai() is None:
        raise RuntimeError("Install LLM support: uv sync --extra llm; then restart the viewer.")

    # Resolve every run's endpoint and key before touching the filesystem.
    plans = []
    for row in rows:
        model, override = row["model"], row.get("tag")
        routing = _routing(model, override, api, base_url)
        entry_options, entry_secret = dict(options), secret
        for key in ("timeout", "max_frames"):
            if row.get(key) is not None:
                entry_options[key] = _positive(row, key, entry_options[key],
                                               integer=key != "timeout")
        for key in ("thinking_budget", "reasoning_effort"):
            if row.get(key) is None:
                continue
            value = row[key]
            if key == "thinking_budget":
                entry_options[key] = _positive(row, key, 1)
            elif not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", value):
                raise ValueError("Invalid reasoning effort.")
            else:
                entry_options[key] = value
        if routing:
            entry_options["api"] = routing["api"]
            if "base_url" in routing:  # provider tag or inferred family: official route
                entry_options["base_url"] = routing["base_url"]
                entry_secret = os.environ.get(routing["key_env"])
                if not entry_secret:
                    raise ValueError(f"Set {routing['key_env']} to launch {model} on the "
                                     f"{routing['api']} API.")
        if entry_options["api"] != "anthropic":
            entry_options.pop("thinking_budget", None)
        plans.append((model, entry_options, entry_secret))
    return _enqueue(runs, "tau", plans)


def _codex_jobs(payload, runs):
    allowed = {"harness", "evals", "model", "models", "prompt", "timeout", "max_frames",
               "frames", "lite"}
    if set(payload) - allowed:
        raise ValueError("Unknown Codex evaluation settings.")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 60000 or "\0" in prompt:
        raise ValueError("Write a task prompt for the Codex evaluation.")
    options = {"timeout": _positive(payload, "timeout", 120),
               "max_frames": _positive(payload, "max_frames", 30, integer=True),
               "prompt": prompt}
    if payload.get("frames") is not None:
        options["frames"] = _positive(payload, "frames", 1, integer=True)
    if payload.get("lite"):
        options["lite"] = True
    if not LIMACTL.is_file():
        raise RuntimeError("Install Lima before launching a Codex evaluation: brew install lima.")
    rows = _run_rows(payload, codex=True)
    return _enqueue(runs, "codex", [(row["model"], {**options, "timeout": _positive(
        row, "timeout", options["timeout"])}, None) for row in rows])


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
