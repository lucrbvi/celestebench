import json
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

    def test_model_api_override_routes_a_single_job_to_its_own_api(self):
        processes = []

        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"models": ["deepseek@openai-responses", "minimax-m3@anthropic"],
                                     "api": "openai-completions", "api_key": "s", "timeout": 2},
                                    Path(directory))
            self.assertEqual([job["api"] for job in jobs], ["openai-responses", "anthropic"])
            apis = [next(a for a in p.args if a.startswith("--api=")) for p in processes]
            self.assertEqual(apis, ["--api=openai-responses", "--api=anthropic"])
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            time.sleep(0.1)

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

        with tempfile.TemporaryDirectory() as directory, patch.object(evals.subprocess, "Popen", launch):
            jobs = evals.start_eval({"models": ["claude@anthropic", "minimax-m3"],
                                     "api_key": "s", "thinking_budget": 1024}, Path(directory))
            self.assertEqual([job["api"] for job in jobs], ["anthropic", "openai-responses"])
            self.wait_until(lambda: all(j["status"] == "completed" for j in evals._jobs.values()))
            anthropic_args = processes[0].args
            openai_args = processes[1].args
            self.assertIn("--thinking-budget=1024", anthropic_args)
            self.assertNotIn("--thinking-budget=1024", openai_args)
            time.sleep(0.1)

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


if __name__ == "__main__":
    unittest.main()
