import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


class EvalTests(unittest.TestCase):
    def setUp(self):
        evals._jobs.clear()
        evals._processes.clear()

    def wait_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Condition not reached before timeout.")

    def test_validation_happens_before_filesystem_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                evals.start_eval({"model": "demo", "base_url": "https://user:pass@example.com/v1"}, root)
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_model_in_batch_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
            with self.assertRaises(ValueError):
                evals.start_eval({"models": ["good", "  ", "bad"], "api_key": "s"}, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
            launch.assert_not_called()

    def test_models_share_settings_and_each_gets_its_own_job(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"models": ["m/a", "m/b", "m/a"], "api_key": "s", "timeout": 2},
                                    Path(directory))
            self.assertEqual([job["model"] for job in jobs], ["m/a", "m/b", "m/a"])
            self.assertEqual(len({job["name"] for job in jobs}), 3)
            models = [next(a for a in p.args if a.startswith("--model=")) for p in processes]
            self.assertEqual(models, ["--model=m/a", "--model=m/b", "--model=m/a"])
            self.assertTrue(all("--max-actions=4" in process.args for process in processes))
            # Let the watcher threads finish writing metadata before cleanup.
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_provider_tags_reroute_single_jobs_to_their_own_key(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.dict("os.environ", {"OPENCODE_API_KEY": "go-secret",
                                          "MINIMAX_API_KEY": "mm-secret"}):
            jobs = evals.start_eval({"models": ["glm-5.3@opencode-go", "minimax-m3@minimax"],
                                     "provider": "openai", "api_key": "s", "timeout": 2},
                                    Path(directory))
            self.assertEqual([job["provider"] for job in jobs], ["opencode-go", "minimax"])
            providers = [next(a for a in p.args if a.startswith("--provider=")) for p in processes]
            self.assertEqual(providers, ["--provider=opencode-go", "--provider=minimax"])
            # The dialog key serves the preset provider; tagged runs use their own env.
            self.assertEqual(processes[0].kwargs["env"]["CELESTEBENCH_API_KEY"], "go-secret")
            self.assertEqual(processes[1].kwargs["env"]["CELESTEBENCH_API_KEY"], "mm-secret")
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_runs_default_to_the_dialog_provider(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"models": ["qwen3.8-flash", "glm-5.3"],
                                     "provider": "opencode-go", "api_key": "s", "timeout": 2},
                                    Path(directory))
            self.assertEqual([job["provider"] for job in jobs], ["opencode-go", "opencode-go"])
            self.assertTrue(all("--provider=opencode-go" in process.args for process in processes))
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_provider_tag_without_its_key_refuses_the_batch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                evals.start_eval({"models": ["claude-sonnet-4-6@anthropic"],
                                  "provider": "opencode-go", "api_key": "s",
                                  "timeout": 2}, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_provider_override_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
            with self.assertRaises(ValueError):
                evals.start_eval({"models": ["m@grpc"], "api_key": "s"}, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
            launch.assert_not_called()

    def test_thinking_level_reaches_every_job_without_special_casing(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-secret"}):
            jobs = evals.start_eval({"models": ["claude-sonnet-4-6@anthropic", "glm-5.3"],
                                     "api_key": "s", "thinking_level": "high", "timeout": 2},
                                    Path(directory))
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            self.assertTrue(all("--thinking-level=high" in process.args for process in processes))
            time.sleep(0.1)

    def test_runs_apply_per_run_settings_like_thinking_and_timeout(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"harness": "tau", "api_key": "s", "timeout": 10,
                                     "evals": [
                                         {"model": "glm-5.3", "thinking_level": "low", "timeout": 10},
                                         {"model": "glm-5.3", "thinking_level": "high",
                                          "timeout": 300},
                                     ]}, Path(directory))
            self.assertEqual([job["model"] for job in jobs], ["glm-5.3", "glm-5.3"])
            timeout_arg = [next(a for a in p.args if a.startswith("--timeout=")) for p in processes]
            self.assertEqual(timeout_arg, ["--timeout=10", "--timeout=300"])
            level_arg = [next(a for a in p.args if a.startswith("--thinking-level=")) for p in processes]
            self.assertEqual(level_arg, ["--thinking-level=low", "--thinking-level=high"])
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_unknown_run_setting_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "sandbox"):
                evals.start_eval({"evals": [{"model": "m", "sandbox": "off"}], "api_key": "s"},
                                 Path(directory))
            launch.assert_not_called()

    def test_stop_cancels_a_queued_evaluation(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        with tempfile.TemporaryDirectory() as directory, patch.object(
                evals.subprocess, "Popen", lambda a, **k: Blocking(a, **k)), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
            jobs = evals.start_eval({"harness": "codex", "models": ["a", "b", "c", "d", "e"],
                                     "timeout": 5}, Path(directory))
            stopped = evals.stop_eval(jobs[4]["id"])
            self.assertEqual(stopped["status"], "cancelled")
            gate.set()
            self.wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] == "completed")
            self.assertEqual(evals._jobs[jobs[4]["id"]]["status"], "cancelled")

    def test_start_uses_fresh_rollout_and_keeps_secret_out_of_argv_and_metadata(self):
        process = None

        def launch(args, **kwargs):
            nonlocal process
            process = FakeProcess(args, **kwargs)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"model": "demo/model", "provider": "mistral",
                                     "api_key": "secret-value", "timeout": 2}, Path(directory))
            job = jobs[0]
            time.sleep(0.02)
            self.assertIsNotNone(process)
            self.assertIn("--provider=mistral", process.args)
            output_arg = next(value for value in process.args if value.startswith("--output="))
            output = Path(output_arg.split("=", 1)[1])
            self.assertFalse(output.exists())
            self.assertNotIn("secret-value", " ".join(process.args))
            self.assertEqual(process.kwargs["env"]["CELESTEBENCH_API_KEY"], "secret-value")
            metadata = Path(directory) / ".evals" / f"{job['id']}.json"
            self.assertNotIn("secret-value", metadata.read_text())
            self.assertEqual(evals.list_evals(Path(directory))[0]["status"], "completed")

    def test_restart_marks_unowned_running_job_interrupted(self):
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
            self.assertEqual(jobs[0]["status"], "interrupted")
            self.assertIn("stopped", jobs[0]["error"])

    def test_run_readiness_follows_the_first_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".evals").mkdir()
            (root / ".evals" / "job.json").write_text(json.dumps({
                "id": "job", "name": "demo/run", "model": "demo",
                "provider": "opencode-go", "status": "completed", "error": None,
                "timeout": 60, "decisions": 1, "started_at": time.time(),
            }))
            jobs = evals.list_evals(root)
            self.assertFalse(jobs[0]["run_ready"])
            folder = root / "demo" / "run"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}")
            self.assertTrue(evals.list_evals(root)[0]["run_ready"])

    def test_rejects_bool_fraction_and_nan_budgets_before_writing(self):
        for key, value in (("timeout", True), ("timeout", 0), ("max_images", 2.5), ("fps", float("nan"))):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    evals.start_eval({"model": "demo", "api_key": "secret", key: value}, Path(directory))
                self.assertEqual(list(Path(directory).iterdir()), [])


    def test_codex_job_failure_reports_the_last_cli_error(self):
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
            self.assertEqual(job["error"], "401 Unauthorized: Missing bearer in header")

    def test_codex_countdown_starts_when_the_engine_runs_not_at_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {"id": "j", "name": "m/run", "model": "m", "harness": "codex",
                   "status": "running", "decisions": 0, "frames": 0, "elapsed": 0,
                   "tokens": None, "started_at": time.time() - 100,
                   "_folder": str(root), "_meta": str(root / ".evals" / "j.json")}
            evals._progress(job)
            self.assertEqual(job["elapsed"], 0)
            self.assertFalse(job["engine"], "booting state must be visible to the UI")
            rollout = root / "rollout"
            (rollout / "screenshots").mkdir(parents=True)
            (rollout / "config.json").write_text("{}")
            old = time.time() - 4
            os.utime(rollout / "config.json", (old, old))
            (rollout / "decisions.jsonl").write_text('{"frame_end": 3}\n')
            evals._progress(job)
            self.assertTrue(job["engine"])
            self.assertGreaterEqual(job["elapsed"], 4)
            self.assertLess(job["elapsed"], 30, "boot time must stay out of the game clock")
            self.assertEqual(job["frames"], 3)

    def test_codex_jobs_run_in_parallel_on_a_shared_login(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, **k)), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"), \
                patch.dict(evals.os.environ, {"CODEX_API_KEY": ""}):
            jobs = evals.start_eval({"harness": "codex", "models": ["m1", "m2", "m3", "m4", "m5"],
                                     "timeout": 5}, Path(directory))
            self.assertEqual([job["status"] for job in jobs], ["running"] * 5)
            gate.set()
            self.wait_until(lambda: all(evals._jobs[job["id"]]["status"] != "running"
                                        for job in jobs))

    def test_codex_jobs_run_in_parallel_with_an_api_key(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, **k)), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"), \
                patch.dict(evals.os.environ, {"CODEX_API_KEY": "k"}):
            jobs = evals.start_eval({"harness": "codex", "models": ["m1", "m2", "m3", "m4", "m5"],
                                     "timeout": 5}, Path(directory))
            self.assertEqual([job["status"] for job in jobs], ["running"] * 5)
            gate.set()
            self.wait_until(lambda: all(evals._jobs[job["id"]]["status"] != "running"
                                        for job in jobs))

    def test_codex_harness_launches_the_local_cli_with_budgets(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
            jobs = evals.start_eval({"harness": "codex", "model": "gpt-5.2", "frames": 9,
                                     "timeout": 30}, Path(directory))
            job = jobs[0]
            self.assertEqual(job["harness"], "codex")
            args = processes[0].args
            self.assertIn(str(evals.ROOT / evals.HARNESSES["codex"].script), args)
            self.assertIn("--model=gpt-5.2", args)
            self.assertIn("--prompt=Play Celeste Classic.", args)
            self.assertIn("--timeout=30", args)
            self.assertIn("--frames=9", args)
            self.wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] != "running")
            final = next(j for j in evals.list_evals(Path(directory)) if j["id"] == jobs[0]["id"])
            self.assertEqual(final["status"], "completed")
            time.sleep(0.1)

    def test_codex_runs_carry_reasoning_and_budgets(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
            jobs = evals.start_eval({"harness": "codex", "max_frames": 5, "evals": [
                {"model": "gpt-5.2", "thinking_level": "high", "timeout": 45},
            ]}, Path(directory))
            args = processes[0].args
            self.assertIn("--thinking-level=high", args)
            self.assertIn("--timeout=45", args)
            self.assertIn("--max-frames=5", args)
            self.assertIn("--prompt=Play Celeste Classic.", args)
            self.wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] == "completed")
            time.sleep(0.1)

    def test_harness_catalog_describes_every_form_field(self):
        catalog = {entry["key"]: entry for entry in evals.harness_catalog()}
        self.assertTrue(catalog["tau"]["builtin"])
        codex = catalog["codex"]
        self.assertEqual([field["key"] for field in codex["run"]],
                         ["model", "thinking_level", "timeout"])
        self.assertEqual([field["key"] for field in codex["options"]],
                         ["prompt", "max_frames", "frames", "fps"])
        thinking = next(field for field in codex["run"] if field["key"] == "thinking_level")
        self.assertIn("high", [choice["value"] for choice in thinking["choices"]])


    def test_codex_harness_validation_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.object(evals.shutil, "which", return_value="/usr/bin/codex"):
            for bad in ({"prompt": "x"},
                        {"harness": "codex"},
                        {"harness": "codex", "model": "m", "api": "anthropic"},
                        {"harness": "codex", "models": ["m"], "frames": 0},
                        {"harness": "codex", "model": "m@openai"}):
                with self.assertRaises(ValueError):
                    evals.start_eval(bad, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_external_harnesses_catalog_and_launch(self):
        catalog = {entry["key"]: entry for entry in evals.harness_catalog()}
        for key in ("opencode", "pi"):
            self.assertFalse(catalog[key]["builtin"])
            self.assertEqual([field["key"] for field in catalog[key]["run"]],
                             ["model", "thinking_level", "timeout"])
            self.assertEqual([field["key"] for field in catalog[key]["options"]],
                             ["prompt", "max_frames", "max_images", "frames", "fps"])
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen", launch), \
                patch.object(evals.shutil, "which", return_value="/usr/bin/opencode"):
            evals.start_eval({"harness": "opencode", "model": "opencode-go/deepseek",
                              "max_frames": 5, "timeout": 10}, Path(directory))
            args = processes[0].args
            self.assertIn(str(evals.ROOT / evals.HARNESSES["opencode"].script), args)
            self.assertIn("--model=opencode-go/deepseek", args)
            self.assertIn("--prompt=Play Celeste Classic.", args)
            self.assertIn("--max-frames=5", args)
            self.assertIn("--max-images=3", args)
            self.assertIn("--thinking-level=low", args)
            self.assertIn("--timeout=10", args)
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_pi_harness_requires_pi_before_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.object(evals.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "pi"):
                evals.start_eval({"harness": "pi", "model": "openai-codex/gpt-5.6-sol"},
                                 Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_codex_harness_requires_codex_before_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.object(evals.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "codex"):
                evals.start_eval({"harness": "codex", "model": "m"}, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
