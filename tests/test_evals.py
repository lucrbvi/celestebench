import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import wait_until
from web import evals

class FakeProcess:
    def __init__(self, args, **kwargs):
        self.args, self.kwargs, self.returncode = args, kwargs, 0

    def wait(self, timeout=None):
        return self.returncode

    def communicate(self):
        return None, ""

    def send_signal(self, _signal):
        self.returncode = 0

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class Blocking(FakeProcess):
    def __init__(self, args, gate, **kwargs):
        super().__init__(args, **kwargs)
        self.gate = gate

    def communicate(self):
        self.gate.wait(2)
        return None, ""


@pytest.fixture(autouse=True)
def clean_registry():
    evals._jobs.clear()
    evals._processes.clear()


@pytest.fixture
def processes(monkeypatch):
    recorded = []

    def launch(args, **kwargs):
        process = FakeProcess(args, **kwargs)
        recorded.append(process)
        return process

    monkeypatch.setattr(evals.subprocess, "Popen", launch)
    return recorded


def test_validation_happens_before_filesystem_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with pytest.raises(ValueError):
            evals.start_eval({"model": "demo", "base_url": "https://user:pass@example.com/v1"}, root)
        assert list(root.iterdir()) == []


def test_invalid_model_in_batch_starts_nothing():
    with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
        with pytest.raises(ValueError):
            evals.start_eval({"models": ["good", "  ", "bad"], "api_key": "s"}, Path(directory))
        assert list(Path(directory).iterdir()) == []
        launch.assert_not_called()


def test_models_share_settings_and_each_gets_its_own_job(processes):
    with tempfile.TemporaryDirectory() as directory:
        jobs = evals.start_eval({"models": ["m/a", "m/b", "m/a"], "api_key": "s"},
                                Path(directory))
        assert [job["model"] for job in jobs] == ["m/a", "m/b", "m/a"]
        assert len({job["name"] for job in jobs}) == 3
        models = [next(a for a in p.args if a.startswith("--model=")) for p in processes]
        assert models == ["--model=m/a", "--model=m/b", "--model=m/a"]
        # Let the watcher threads finish writing metadata before cleanup.
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        time.sleep(0.1)


def test_provider_tags_reroute_single_jobs_to_their_own_key(processes):
    with tempfile.TemporaryDirectory() as directory, \
            patch.dict(os.environ, {"OPENCODE_API_KEY": "go-secret",
                                    "MINIMAX_API_KEY": "mm-secret"}):
        jobs = evals.start_eval({"models": ["glm-5.3@opencode-go", "minimax-m3@minimax"],
                                 "provider": "openai", "api_key": "s"},
                                Path(directory))
        assert [job["provider"] for job in jobs] == ["opencode-go", "minimax"]
        providers = [next(a for a in p.args if a.startswith("--provider=")) for p in processes]
        assert providers == ["--provider=opencode-go", "--provider=minimax"]
        # The dialog key serves the preset provider; tagged runs use their own env.
        assert processes[0].kwargs["env"]["CELESTEBENCH_API_KEY"] == "go-secret"
        assert processes[1].kwargs["env"]["CELESTEBENCH_API_KEY"] == "mm-secret"
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        time.sleep(0.1)


def test_runs_default_to_the_dialog_provider(processes):
    with tempfile.TemporaryDirectory() as directory:
        jobs = evals.start_eval({"models": ["qwen3.8-flash", "glm-5.3"],
                                 "provider": "opencode-go", "api_key": "s"},
                                Path(directory))
        assert [job["provider"] for job in jobs] == ["opencode-go", "opencode-go"]
        assert all("--provider=opencode-go" in process.args for process in processes)
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        time.sleep(0.1)


def test_provider_tag_without_its_key_refuses_the_batch():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen") as launch, \
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            evals.start_eval({"models": ["claude-sonnet-4-6@anthropic"],
                              "provider": "opencode-go", "api_key": "s"},
                             Path(directory))
        launch.assert_not_called()
        assert list(Path(directory).iterdir()) == []


def test_unknown_provider_override_starts_nothing():
    with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
        with pytest.raises(ValueError):
            evals.start_eval({"models": ["m@grpc"], "api_key": "s"}, Path(directory))
        assert list(Path(directory).iterdir()) == []
        launch.assert_not_called()


def test_thinking_level_reaches_every_job_without_special_casing(processes):
    with tempfile.TemporaryDirectory() as directory, \
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-secret"}):
        evals.start_eval({"models": ["claude-sonnet-4-6@anthropic", "glm-5.3"],
                          "api_key": "s", "thinking_level": "high"},
                         Path(directory))
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        assert all("--thinking-level=high" in process.args for process in processes)
        time.sleep(0.1)


def test_runs_apply_per_run_thinking_and_the_mode_budget(processes):
    with tempfile.TemporaryDirectory() as directory:
        jobs = evals.start_eval({"harness": "tau", "api_key": "s", "mode": "lite",
                                 "evals": [
                                     {"model": "glm-5.3", "thinking_level": "low"},
                                     {"model": "glm-5.3", "thinking_level": "high"},
                                 ]}, Path(directory))
        assert [job["model"] for job in jobs] == ["glm-5.3", "glm-5.3"]
        assert all(job["mode"] == "lite" for job in jobs)
        # Lite pauses the game, so no --fps reaches the CLI.
        assert not any(a.startswith("--fps=") for p in processes for a in p.args)
        assert all("--timeout=300" in p.args for p in processes)
        level_arg = [next(a for a in p.args if a.startswith("--thinking-level=")) for p in processes]
        assert level_arg == ["--thinking-level=low", "--thinking-level=high"]
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        time.sleep(0.1)


def test_unknown_run_setting_starts_nothing():
    with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
        with pytest.raises(ValueError, match="sandbox"):
            evals.start_eval({"evals": [{"model": "m", "sandbox": "off"}], "api_key": "s"},
                             Path(directory))
        launch.assert_not_called()


def test_stop_cancels_a_queued_evaluation():
    gate = threading.Event()

    with tempfile.TemporaryDirectory() as directory, patch.object(
            evals.subprocess, "Popen", lambda a, **k: Blocking(a, gate, **k)), \
            patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
        jobs = evals.start_eval({"harness": "codex", "models": ["a", "b", "c", "d", "e"]},
                                Path(directory))
        stopped = evals.stop_eval(jobs[4]["id"])
        assert stopped["status"] == "cancelled"
        gate.set()
        wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] == "completed")
        assert evals._jobs[jobs[4]["id"]]["status"] == "cancelled"


def test_start_uses_fresh_rollout_and_keeps_secret_out_of_argv_and_metadata(processes):
    with tempfile.TemporaryDirectory() as directory:
        jobs = evals.start_eval({"model": "demo/model", "provider": "mistral",
                                 "api_key": "secret-value"}, Path(directory))
        job = jobs[0]
        time.sleep(0.02)
        process = processes[0]
        assert process is not None
        assert "--provider=mistral" in process.args
        output_arg = next(value for value in process.args if value.startswith("--output="))
        output = Path(output_arg.split("=", 1)[1])
        assert not output.exists()
        assert "secret-value" not in " ".join(process.args)
        assert process.kwargs["env"]["CELESTEBENCH_API_KEY"] == "secret-value"
        metadata = Path(directory) / ".evals" / f"{job['id']}.json"
        assert "secret-value" not in metadata.read_text()
        assert evals.list_evals(Path(directory))[0]["status"] == "completed"


def test_restart_marks_unowned_running_job_interrupted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = root / ".evals"
        metadata.mkdir()
        (metadata / "job.json").write_text(json.dumps({
            "id": "old-job", "name": "demo/run", "model": "demo",
            "provider": "opencode-go", "status": "running",
            "timeout": 60, "decisions": 1, "started_at": time.time(),
        }))
        jobs = evals.list_evals(root)
        assert jobs[0]["status"] == "interrupted"
        assert "stopped" in jobs[0]["error"]


def test_run_readiness_follows_the_first_decision():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".evals").mkdir()
        (root / ".evals" / "job.json").write_text(json.dumps({
            "id": "job", "name": "demo/run", "model": "demo",
            "provider": "opencode-go", "status": "completed", "error": None,
            "timeout": 60, "decisions": 1, "started_at": time.time(),
        }))
        jobs = evals.list_evals(root)
        assert not jobs[0]["run_ready"]
        folder = root / "demo" / "run"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
        assert evals.list_evals(root)[0]["run_ready"]


@pytest.mark.parametrize("key,value", [("mode", "turbo"), ("timeout", 120),
                                       ("max_images", 3), ("fps", 30)])
def test_rejects_unknown_modes_and_budgets_before_writing(key, value):
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError):
            evals.start_eval({"model": "demo", "api_key": "secret", key: value}, Path(directory))
        assert list(Path(directory).iterdir()) == []


def test_codex_job_failure_reports_the_last_cli_error():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".evals").mkdir()
        (root / "codex.jsonl").write_text(
            '{"type":"error","message":"Reconnecting... 2/5"}\n'
            '{"type":"error","message":"401 Unauthorized: Missing bearer in header"}\n')
        job = {"id": "job", "name": "m/run", "model": "m", "harness": "codex",
               "status": "failed", "error": None, "decisions": 0, "timeout": 5,
               "frames": 0, "elapsed": 0, "tokens": None, "started_at": time.time(),
               "_folder": str(root), "_meta": str(root / ".evals" / "job.json")}
        process = FakeProcess([], cwd=str(root))
        process.returncode = 1
        process.communicate = lambda: (None, "")
        evals._watch(job, process, None)
        assert job["error"] == "401 Unauthorized: Missing bearer in header"


def test_codex_countdown_starts_when_the_engine_runs_not_at_launch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        job = {"id": "j", "name": "m/run", "model": "m", "harness": "codex",
               "status": "running", "decisions": 0, "frames": 0, "elapsed": 0,
               "tokens": None, "started_at": time.time() - 100,
               "_folder": str(root), "_meta": str(root / ".evals" / "j.json")}
        evals._progress(job)
        assert job["elapsed"] == 0
        assert not job["engine"], "booting state must be visible to the UI"
        rollout = root / "rollout"
        (rollout / "screenshots").mkdir(parents=True)
        (rollout / "config.json").write_text("{}")
        old = time.time() - 4
        os.utime(rollout / "config.json", (old, old))
        (rollout / "decisions.jsonl").write_text('{"frame_end": 3}\n')
        evals._progress(job)
        assert job["engine"]
        assert job["elapsed"] >= 4
        assert job["elapsed"] < 30, "boot time must stay out of the game clock"
        assert job["frames"] == 3


@pytest.mark.parametrize("codex_key", ["", "k"])
def test_codex_jobs_queue_on_a_shared_login(codex_key):
    gate = threading.Event()

    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, gate, **k)), \
            patch.object(evals.shutil, "which", return_value="/usr/bin/codex"), \
            patch.dict(evals.os.environ, {"CODEX_API_KEY": codex_key}):
        jobs = evals.start_eval({"harness": "codex", "models": ["m1", "m2", "m3", "m4", "m5"]},
                                Path(directory))
        assert [job["status"] for job in jobs] == ["running"] + ["queued"] * 4
        gate.set()
        wait_until(lambda: all(evals._jobs[job["id"]]["status"] == "completed"
                               for job in jobs))


def test_opencode_oauth_runs_serialize_while_api_runs_parallelize():
    gate = threading.Event()

    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, gate, **k)), \
            patch.object(evals.shutil, "which", return_value="/usr/bin/opencode"), \
            patch.object(evals.auth, "oauth",
                         side_effect=lambda _h, model: model.startswith("openai/")), \
            patch.object(evals.auth, "provider",
                         side_effect=lambda _h, model: model.split("/", 1)[0]):
        jobs = evals.start_eval({"harness": "opencode", "models": [
            "openai/gpt-5", "opencode-go/deepseek", "openai/gpt-5.1"]}, Path(directory))
        # The two OAuth runs share one login and queue; the API run is free.
        assert [job["status"] for job in jobs] == ["running", "running", "queued"]
        gate.set()
        wait_until(lambda: all(evals._jobs[job["id"]]["status"] == "completed"
                               for job in jobs))


def test_pi_oauth_runs_serialize_while_api_runs_parallelize():
    gate = threading.Event()

    def oauth(_harness, model):
        return model.startswith("openai-codex/")

    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, gate, **k)), \
            patch.object(evals.shutil, "which", return_value="/usr/bin/pi"), \
            patch.object(evals.auth, "oauth", side_effect=oauth), \
            patch.object(evals.auth, "provider",
                         side_effect=lambda _h, model: model.split("/", 1)[0]):
        jobs = evals.start_eval({"harness": "pi", "models": [
            "openai-codex/gpt-5.6-sol", "openrouter/llama",
            "openai-codex/gpt-5.6-terra"]}, Path(directory))
        assert [job["status"] for job in jobs] == ["running", "running", "queued"]
        gate.set()
        wait_until(lambda: all(evals._jobs[job["id"]]["status"] == "completed"
                               for job in jobs))


def test_codex_harness_launches_the_local_cli_with_mode_budgets(processes):
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
        jobs = evals.start_eval({"harness": "codex", "model": "gpt-5.2", "mode": "rtc"},
                                Path(directory))
        job = jobs[0]
        assert job["harness"] == "codex"
        assert job["mode"] == "rtc"
        args = processes[0].args
        assert str(evals.ROOT / evals.HARNESSES["codex"].script) in args
        assert "--model=gpt-5.2" in args
        assert "--prompt=Play Celeste Classic." in args
        assert "--timeout=300" in args
        assert "--max-frames=30" in args
        assert "--fps=30" in args
        wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] != "running")
        final = next(j for j in evals.list_evals(Path(directory)) if j["id"] == jobs[0]["id"])
        assert final["status"] == "completed"
        time.sleep(0.1)


def test_codex_runs_carry_reasoning_and_a_paused_mode(processes):
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
        jobs = evals.start_eval({"harness": "codex", "mode": "lite", "evals": [
            {"model": "gpt-5.2", "thinking_level": "high"},
        ]}, Path(directory))
        args = processes[0].args
        assert "--thinking-level=high" in args
        assert "--timeout=300" in args
        assert "--fps=30" not in args
        assert "--prompt=Play Celeste Classic." in args
        wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] == "completed")
        time.sleep(0.1)


def test_harness_catalog_describes_the_two_modes_and_every_field():
    catalog = evals.harness_catalog()
    by_key = {entry["key"]: entry for entry in catalog["harnesses"]}
    assert [mode["name"] for mode in catalog["modes"]] == ["rtc", "lite"]
    assert by_key["tau"]["builtin"]
    codex = by_key["codex"]
    assert [field["key"] for field in codex["run"]] == ["model", "thinking_level"]
    assert [field["key"] for field in codex["options"]] == ["prompt"]
    thinking = next(field for field in codex["run"] if field["key"] == "thinking_level")
    assert "high" in [choice["value"] for choice in thinking["choices"]]


def test_codex_harness_validation_starts_nothing():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen") as launch, \
            patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
        for bad in ({"prompt": "x"},
                    {"harness": "codex"},
                    {"harness": "codex", "model": "m", "api": "anthropic"},
                    {"harness": "codex", "models": ["m"], "frames": 0},
                    {"harness": "codex", "model": "m@openai"}):
            with pytest.raises(ValueError):
                evals.start_eval(bad, Path(directory))
        launch.assert_not_called()
        assert list(Path(directory).iterdir()) == []


def test_external_harnesses_catalog_and_launch(processes):
    catalog = {entry["key"]: entry for entry in evals.harness_catalog()["harnesses"]}
    for key in ("opencode", "pi"):
        assert not catalog[key]["builtin"]
        assert [field["key"] for field in catalog[key]["run"]] == ["model", "thinking_level"]
        assert [field["key"] for field in catalog[key]["options"]] == ["prompt"]

    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.shutil, "which", return_value="/usr/bin/opencode"):
        evals.start_eval({"harness": "opencode", "model": "opencode-go/deepseek",
                          "mode": "rtc"}, Path(directory))
        args = processes[0].args
        assert str(evals.ROOT / evals.HARNESSES["opencode"].script) in args
        assert "--model=opencode-go/deepseek" in args
        assert "--prompt=Play Celeste Classic." in args
        assert "--max-frames=30" in args
        assert "--max-images=3" in args
        assert "--thinking-level=low" in args
        assert "--timeout=300" in args
        assert "--fps=30" in args
        wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
        time.sleep(0.1)


def test_pi_harness_requires_pi_before_touching_the_filesystem():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen") as launch, \
            patch.object(evals.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match="pi"):
            evals.start_eval({"harness": "pi", "model": "openai-codex/gpt-5.6-sol"},
                             Path(directory))
        launch.assert_not_called()
        assert list(Path(directory).iterdir()) == []


def test_codex_harness_requires_codex_before_touching_the_filesystem():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(evals.subprocess, "Popen") as launch, \
            patch.object(evals.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match="codex"):
            evals.start_eval({"harness": "codex", "model": "m"}, Path(directory))
        launch.assert_not_called()
        assert list(Path(directory).iterdir()) == []
