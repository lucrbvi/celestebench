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
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from celestebench.harnesses import HARNESSES

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "examples" / "llm.py"


_lock = threading.RLock()
_jobs = {}
_processes = {}


def _provider_env(provider):
    """Resolve a provider name to its key environment variable, or fail loudly."""
    from celestebench import providers
    try:
        return providers.key_env(provider)
    except ValueError:
        raise ValueError(f"Unknown provider '{provider}'; use one of "
                         f"{', '.join((*providers.provider_names(), providers.CUSTOM))}.") from None


def _thinking_level(value):
    from tau_coding.thinking import normalize_thinking_level
    return normalize_thinking_level(value)


def _harness(name):
    return HARNESSES.get(name) or HARNESSES["tau"]


_REQUIREMENTS = {"codex": lambda: shutil.which("codex") is not None}


def _require(harness):
    """Refuse an external harness whose host tool is missing."""
    check = _REQUIREMENTS.get(harness.requires)
    if harness.requires and (check is None or not check()):
        raise RuntimeError(f"Install {harness.requires} before launching {harness.label}.")


def _field_value(field, value):
    """Validate one registry field value, or return None when it is left out."""
    if value is None:
        if field.required:
            raise ValueError(f"Enter {field.label}.")
        return None
    if field.kind == "bool":
        if type(value) is not bool:
            raise ValueError(f"{field.label} must be true or false.")
        return value
    if field.kind == "choice":
        if value not in field.choices:
            raise ValueError(f"Invalid {field.label}.")
        return value
    if field.kind in {"int", "number"}:
        if (type(value) not in {int, float} or isinstance(value, bool)
                or not math.isfinite(value) or value < field.minimum
                or field.kind == "int" and type(value) is not int):
            raise ValueError(f"Invalid {field.label}.")
        return value
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"Invalid {field.label}.")
    if field.required and not value.strip():
        raise ValueError(f"Enter {field.label}.")
    return value


def _harness_flags(harness, options):
    """Command-line flags for an external harness; booleans are bare switches."""
    flags = []
    for field in (*harness.run, *harness.options):
        if field.key == "model" or field.key not in options:
            continue
        name = "--" + field.key.replace("_", "-")
        if field.kind == "bool":
            if options[field.key]:
                flags.append(name)
        else:
            flags.append(f"{name}={options[field.key]}")
    return flags


def harness_catalog():
    """Harness descriptors for the new-eval form, with dynamic choice lists."""
    choices = []
    if _support_tau_ai() is not None:
        from celestebench import providers
        choices = [{"value": name, "label": name} for name in providers.provider_names()]
    choices.append({"value": "custom", "label": "custom endpoint"})
    catalog = []
    for harness in HARNESSES.values():
        entry = {"key": harness.key, "label": harness.label, "builtin": harness.builtin,
                 "note": harness.note, "run": [field.spec() for field in harness.run],
                 "options": [field.spec() for field in harness.options]}
        for field in entry["run"] + entry["options"]:
            field["choices"] = choices if field["source"] == "providers" else [
                {"value": value, "label": value} for value in field["choices"]]
        catalog.append(entry)
    return catalog


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
    harness = _harness(job.get("harness"))
    folder = Path(job["_folder"]) / "rollout" if harness.nested else Path(job["_folder"])
    engine = folder / "decisions.jsonl"
    # The viewer only lists a run once its engine wrote config.json, so the
    # evaluation tab may link to it from that moment on.
    job["run_ready"] = (folder / "config.json").is_file()
    if job["status"] == "queued":
        job.update(decisions=0, frames=0, elapsed=0, tokens=None, engine=True)
        return
    if job["status"] == "running" and not engine.is_file():
        # Countdown starts when the game actually runs, not while any harness
        # boots its emulator or provider before the first decision.
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
        harness = _harness(job.get("harness"))
        if job["status"] == "failed" and harness.trace:
            # The external CLI reports its own whole failures in the JSON trace
            # (e.g. an authentication problem), which reads much better than the
            # process traceback.
            messages = [row.get("message") for row in _rows(Path(job["_folder"]) / harness.trace)
                        if row.get("type") == "error" and row.get("message")]
            if messages:
                job["error"] = messages[-1]
                if harness.login_hint and ("authorizat" in job["error"].lower()
                                           or "authentication" in job["error"].lower()):
                    job["error"] += (f" — {harness.label} has no access in the viewer yet;"
                                     f" {harness.login_hint}")
        _progress(job)
        _write(job)
        _processes.pop(job["id"], None)
        _pump()


def _launch(job, options, secret):
    """Spawn the harness subprocess. Caller holds the lock; job becomes running."""
    harness = _harness(job["harness"])
    if harness.builtin:
        args = [sys.executable, str(CLI), f"--output={job['_folder']}", f"--model={job['model']}"]
        args += [f"--{key.replace('_', '-')}={value}" for key, value in options.items()]
        env = os.environ.copy()
        env["CELESTEBENCH_API_KEY"] = secret
    else:
        args = [sys.executable, str(ROOT / harness.script), *harness.command,
                f"--output={job['_folder']}", f"--model={job['model']}"]
        args += _harness_flags(harness, options)
        env = os.environ.copy()
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


def _concurrency(harness):
    """Codex shares one ChatGPT login without an API key, so it runs one at a time
    there and in parallel once CODEX_API_KEY lets every run authenticate on its own."""
    if harness.key == "codex" and not os.environ.get("CODEX_API_KEY"):
        return 1
    return harness.concurrency


def _pump():
    """Start queued evaluations in submission order, up to each harness's cap."""
    with _lock:
        busy = {}
        for _id in _processes:
            key = _harness(_jobs[_id].get("harness")).key
            busy[key] = busy.get(key, 0) + 1
        for job in list(_jobs.values()):
            if job["status"] != "queued":
                continue
            harness = _harness(job.get("harness"))
            # Capped harnesses share a resource; wait for a free slot.
            limit = _concurrency(harness)
            if limit and busy.get(harness.key, 0) >= limit:
                continue
            options, secret = job.pop("_options", None), job.pop("_secret", None)
            if options is None:
                job.update(status="failed", error="Queued evaluation lost its settings.",
                           finished_at=time.time())
                continue
            busy[harness.key] = busy.get(harness.key, 0) + 1
            _launch(job, options, secret)


def _run_rows(payload, harness):
    """Per-run dicts; legacy `models` strings become one row each."""
    rows = payload.get("evals")
    if rows is None:
        return [{"model": model} if override is None else {"model": model, "tag": override}
                for model, override in _model_entries(payload, harness)]
    if not isinstance(rows, list) or not rows or len(rows) > 24:
        raise ValueError("Provide one to 24 runs.")
    allowed = {field.key for field in harness.run} | ({"tag"} if harness.builtin else set())
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
        if not harness.builtin and row.get("tag"):
            raise ValueError(f"Provider overrides do not apply to the {harness.label} harness.")
        parsed.append(dict(row))
    return parsed


def _model_entries(payload, harness):
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
        if override and not harness.builtin:
            raise ValueError(f"Provider overrides do not apply to the {harness.label} harness.")
        if override and not _known_provider(override):
            raise ValueError(f"Unknown provider '{override}' for model '{model}';"
                             f" use one of {_provider_choices()}.")
        entries.append((model, override or None))
    if not entries:
        raise ValueError("Enter a model identifier on every line.")
    return entries


def _known_provider(name):
    from celestebench import providers
    return name != providers.CUSTOM and name in providers.provider_names()


def _provider_choices():
    from celestebench import providers
    return ", ".join((*providers.provider_names(), providers.CUSTOM))


def _positive(payload, key, default, integer=False):
    value = payload.get(key, default)
    if (type(value) not in {int, float} or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0
            or integer and type(value) is not int):
        raise ValueError("timeout must be a positive number of seconds." if key == "timeout"
                         else f"Invalid {key} budget.")
    return value


def _enqueue(runs, harness, plans):
    runs = Path(runs).resolve()
    with _lock:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
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
                   "provider": entry_options.get("provider"), "status": "queued", "error": None,
                   "decisions": 0, "timeout": entry_options["timeout"], "frames": 0,
                   "elapsed": 0, "tokens": None, "started_at": time.time(),
                   "_folder": str(folder), "_meta": str(meta)}
            # Options and the secret stay in memory only; never persisted to disk.
            # Capped harnesses queue once their share of the resource is running.
            limit = _concurrency(HARNESSES[harness])
            running = sum(1 for _id in _processes
                          if _harness(_jobs[_id].get("harness")).key == harness)
            if limit and running >= limit:
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
    name = payload.get("harness", "tau")
    if name not in HARNESSES:
        raise ValueError("Unknown harness.")
    harness = HARNESSES[name]
    return _tau_jobs(harness, payload, runs) if harness.builtin else _external_jobs(
        harness, payload, runs)


def _tau_jobs(harness, payload, runs):
    if _support_tau_ai() is None:
        raise RuntimeError("Install LLM support: uv sync --extra llm; then restart the viewer.")
    defaults = {"timeout": 120, "max_frames": 30, "max_actions": 4, "max_images": 3}
    allowed = set(defaults) | {"harness", "evals", "model", "models", "provider",
                               "base_url", "api_key", "fps", "thinking_level"}
    if set(payload) - allowed:
        raise ValueError("Unknown evaluation settings.")
    rows = _run_rows(payload, harness)
    provider = payload.get("provider", "opencode-go")
    base_url = payload.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("Enter an HTTP(S) base URL.")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or any(c.isspace() for c in base_url)):
            raise ValueError("Base URL must be HTTP(S), without credentials, query or fragment.")
    elif provider == "custom":
        raise ValueError("Custom providers need a base URL.")
    key_env_name = _provider_env(provider)
    secret = payload.get("api_key")
    if not isinstance(secret, str) or not secret or "\0" in secret:
        secret = os.environ.get(key_env_name)
    if not isinstance(secret, str) or not secret or "\0" in secret:
        raise ValueError(f"Enter an API key or set {key_env_name} before starting the viewer.")
    options = {"provider": provider, "thinking_level": _thinking_level(
        payload.get("thinking_level", "low"))}
    if base_url is not None:
        options["base_url"] = base_url
    for key, integer in (("max_frames", True), ("max_actions", True), ("max_images", True),
                         ("timeout", False)):
        options[key] = _positive(payload, key, defaults[key], integer=integer)
    if payload.get("fps") is not None:
        options["fps"] = _positive(payload, "fps", 1)

    # Resolve every run's endpoint and key before touching the filesystem.
    plans = []
    for row in rows:
        model, override = row["model"], row.get("tag")
        entry_options, entry_secret = dict(options), secret
        if row.get("timeout") is not None:
            entry_options["timeout"] = _positive(row, "timeout", entry_options["timeout"])
        if row.get("thinking_level") is not None:
            entry_options["thinking_level"] = _thinking_level(row["thinking_level"])
        if override is not None and override != provider:
            # A provider tag reroutes the run to that provider's own key env.
            entry_options["provider"] = override
            entry_secret = os.environ.get(_provider_env(override))
            if not entry_secret:
                raise ValueError(f"Set {_provider_env(override)} to launch {model} on {override}.")
        plans.append((model, entry_options, entry_secret))
    return _enqueue(runs, harness.key, plans)


def _external_jobs(harness, payload, runs):
    allowed = {"harness", "evals", "model", "models"} | {
        field.key for field in (*harness.run, *harness.options)}
    if set(payload) - allowed:
        raise ValueError(f"Unknown {harness.label} evaluation settings.")
    _require(harness)
    options = {field.key: _field_value(field, payload.get(field.key, field.default))
               for field in harness.options}
    # Top-level run fields are shared defaults; each row may override them.
    defaults = {field.key: _field_value(field, payload.get(field.key, field.default))
                for field in harness.run if field.key != "model"}
    options = {key: value for key, value in options.items() if value is not None}
    rows = _run_rows(payload, harness)
    plans = []
    for row in rows:
        entry_options = options | {key: value for key, value in defaults.items() if value is not None}
        for field in harness.run:
            if field.key != "model" and row.get(field.key) is not None:
                entry_options[field.key] = _field_value(field, row[field.key])
        plans.append((row["model"], entry_options, None))
    return _enqueue(runs, harness.key, plans)


def job_running(job_id):
    """Whether a live job can still produce frames; lets a stream end early."""
    with _lock:
        job = _jobs.get(job_id)
        return job is not None and job["status"] == "running"


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
