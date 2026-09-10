import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from web import evals

LIMACTL_OK = Path(__file__)  # an existing file passes the Lima check
MISSING_LIMACTL = Path(__file__).parent / "missing" / "limactl"


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

    def test_model_api_override_routes_a_single_job_to_its_own_api(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-secret"}):
            jobs = evals.start_eval({"models": ["deepseek@openai-completions", "minimax-m3@anthropic"],
                                     "api": "openai-responses", "api_key": "s", "timeout": 2},
                                    Path(directory))
            self.assertEqual([job["api"] for job in jobs], ["openai-completions", "anthropic"])
            apis = [next(a for a in p.args if a.startswith("--api=")) for p in processes]
            self.assertEqual(apis, ["--api=openai-completions", "--api=anthropic"])
            # A protocol tag keeps the dialog endpoint; a provider tag reroutes.
            self.assertIn("--base-url=https://api.openai.com/v1", processes[0].args)
            self.assertIn("--base-url=https://api.anthropic.com/v1", processes[1].args)
            self.assertEqual(processes[1].kwargs["env"]["CELESTEBENCH_API_KEY"], "ant-secret")
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_model_families_auto_route_without_a_tag_on_official_endpoints(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-secret"}):
            jobs = evals.start_eval(
                {"models": ["gpt-5.2", "claude-sonnet-4-5", "muse-spark-1.3-contributor"],
                 "api": "openai-responses", "base_url": "https://api.openai.com/v1",
                 "key_env": "OPENAI_API_KEY", "api_key": "s", "timeout": 2},
                Path(directory))
            self.assertEqual([job["api"] for job in jobs],
                             ["openai-responses", "anthropic", "openai-responses"])
            bases = [next(a for a in p.args if a.startswith("--base-url=")) for p in processes]
            self.assertEqual(bases, ["--base-url=https://api.openai.com/v1",
                                     "--base-url=https://api.anthropic.com/v1",
                                     "--base-url=https://api.openai.com/v1"])
            self.assertEqual(processes[1].kwargs["env"]["CELESTEBENCH_API_KEY"], "ant-secret")
            # An aggregator endpoint serves every family: no rerouting there.
            aggregated = evals.start_eval(
                {"models": ["claude-sonnet-4-5"], "base_url": "https://opencode.ai/zen/go/v1",
                 "api_key": "s", "timeout": 2}, Path(directory))
            self.assertEqual(aggregated[0]["api"], "openai-responses")
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_provider_tag_without_its_key_refuses_the_batch(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                evals.start_eval({"models": ["claude-sonnet-4-5"], "api": "openai-responses",
                                  "base_url": "https://api.openai.com/v1", "api_key": "s",
                                  "timeout": 2}, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_api_override_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
            with self.assertRaises(ValueError):
                evals.start_eval({"models": ["m@grpc"], "api_key": "s"}, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
            launch.assert_not_called()

    def test_thinking_budget_only_reaches_anthropic_jobs(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-secret"}):
            jobs = evals.start_eval({"models": ["claude@anthropic", "minimax-m3"],
                                     "api_key": "s", "thinking_budget": 1024}, Path(directory))
            self.assertEqual([job["api"] for job in jobs], ["anthropic", "openai-responses"])
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            anthropic_args = processes[0].args
            openai_args = processes[1].args
            self.assertIn("--thinking-budget=1024", anthropic_args)
            self.assertNotIn("--thinking-budget=1024", openai_args)
            time.sleep(0.1)

    def test_runs_apply_per_run_settings_like_effort_and_timeout(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"harness": "tau", "api_key": "s", "timeout": 10,
                                     "evals": [
                                         {"model": "gpt-5.2", "reasoning_effort": "low", "timeout": 10},
                                         {"model": "gpt-5.2", "reasoning_effort": "high",
                                          "timeout": 300},
                                     ]}, Path(directory))
            self.assertEqual([job["model"] for job in jobs], ["gpt-5.2", "gpt-5.2"])
            timeout_arg = [next(a for a in p.args if a.startswith("--timeout=")) for p in processes]
            self.assertEqual(timeout_arg, ["--timeout=10", "--timeout=300"])
            effort_arg = [next(a for a in p.args if a.startswith("--reasoning-effort=")) for p in processes]
            self.assertEqual(effort_arg, ["--reasoning-effort=low", "--reasoning-effort=high"])
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

    def test_unknown_run_setting_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "sandbox"):
                evals.start_eval({"evals": [{"model": "m", "sandbox": "off"}], "api_key": "s"},
                                 Path(directory))
            launch.assert_not_called()

    def test_extra_models_queue_until_a_slot_frees(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        def launch(args, **kwargs):
            return Blocking(args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.object(evals, "_MAX_PROCESSES", 1):
            jobs = evals.start_eval({"models": ["a", "b"], "api_key": "s", "timeout": 5}, Path(directory))
            self.assertEqual([job["status"] for job in jobs], ["running", "queued"])
            queued = next(j for j in evals.list_evals(Path(directory)) if j["model"] == "b")
            self.assertEqual(queued["elapsed"], 0)
            evals.stop_eval(jobs[0]["id"])
            self.wait_until(lambda: jobs[0]["id"] not in evals._processes)
            self.wait_until(lambda: evals._jobs[jobs[1]["id"]]["status"] == "running")
            gate.set()
            self.wait_until(lambda: evals._jobs[jobs[1]["id"]]["status"] == "completed")

    def test_stop_cancels_a_queued_evaluation(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        with tempfile.TemporaryDirectory() as directory, patch.object(
                evals.subprocess, "Popen", lambda a, **k: Blocking(a, **k)), \
                patch.object(evals, "_MAX_PROCESSES", 1):
            jobs = evals.start_eval({"models": ["a", "b"], "api_key": "s", "timeout": 5}, Path(directory))
            stopped = evals.stop_eval(jobs[1]["id"])
            self.assertEqual(stopped["status"], "cancelled")
            gate.set()
            self.wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] == "completed")
            self.assertEqual(evals._jobs[jobs[1]["id"]]["status"], "cancelled")

    def test_start_uses_fresh_rollout_and_keeps_secret_out_of_argv_and_metadata(self):
        process = None

        def launch(args, **kwargs):
            nonlocal process
            process = FakeProcess(args, **kwargs)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"model": "demo/model", "api": "mistral-conversations",
                                     "api_key": "secret-value", "timeout": 2}, Path(directory))
            job = jobs[0]
            time.sleep(0.02)
            self.assertIsNotNone(process)
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
                "api": "openai-responses", "status": "running",
                "timeout": 60, "decisions": 1, "started_at": time.time(),
            }))
            jobs = evals.list_evals(root)
            self.assertEqual(jobs[0]["status"], "interrupted")
            self.assertIn("stopped", jobs[0]["error"])

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

    def test_codex_jobs_run_one_at_a_time_on_the_shared_vm(self):
        gate = threading.Event()

        class Blocking(FakeProcess):
            def communicate(self):
                gate.wait(2)
                return None, ""

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen", lambda a, **k: Blocking(a, **k)), \
                patch.object(evals, "LIMACTL", LIMACTL_OK):
            jobs = evals.start_eval({"harness": "codex", "models": ["m1", "m2"],
                                     "prompt": "x", "timeout": 5}, Path(directory))
            self.assertEqual([job["status"] for job in jobs], ["running", "queued"])
            evals.stop_eval(jobs[0]["id"])
            self.wait_until(lambda: evals._jobs[jobs[1]["id"]]["status"] == "running")
            gate.set()
            self.wait_until(lambda: evals._jobs[jobs[1]["id"]]["status"] == "completed")

    def test_codex_harness_launches_the_vm_cli_with_prompt_and_budgets(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch), \
                patch.object(evals, "LIMACTL", LIMACTL_OK):
            jobs = evals.start_eval({"harness": "codex", "model": "gpt-5.2", "frames": 9,
                                     "prompt": "Play the game.", "timeout": 30}, Path(directory))
            job = jobs[0]
            self.assertEqual(job["harness"], "codex")
            args = processes[0].args
            self.assertIn(str(evals.CODEX_CLI), args)
            self.assertIn("run", args)
            self.assertIn("--model=gpt-5.2", args)
            self.assertIn("--prompt=Play the game.", args)
            self.assertIn("--timeout=30", args)
            self.assertIn("--frames=9", args)
            self.assertNotIn("api", args)
            self.wait_until(lambda: evals._jobs[jobs[0]["id"]]["status"] != "running")
            final = next(j for j in evals.list_evals(Path(directory)) if j["id"] == jobs[0]["id"])
            self.assertEqual(final["status"], "completed")
            time.sleep(0.1)

    def test_codex_harness_validation_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.object(evals, "LIMACTL", LIMACTL_OK):
            for bad in ({"harness": "codex", "model": "m"},
                        {"prompt": "x"},
                        {"harness": "codex", "model": "m", "prompt": "x", "api": "anthropic"},
                        {"harness": "codex", "models": ["m"], "prompt": "x", "frames": 0},
                        {"harness": "codex", "model": "m@openai", "prompt": "x"}):
                with self.assertRaises(ValueError):
                    evals.start_eval(bad, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_codex_harness_requires_lima_before_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(evals.subprocess, "Popen") as launch, \
                patch.object(evals, "LIMACTL", MISSING_LIMACTL):
            with self.assertRaisesRegex(RuntimeError, "Lima"):
                evals.start_eval({"harness": "codex", "model": "m", "prompt": "x"}, Path(directory))
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
