"""Exercise the real CLI/emulator against a local streaming API, without paid calls."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from web import evals


class LocalAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        if request["model"] == "local-failure":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid model for local-test-key"}}')
            return
        time.sleep(0.2 if request["model"] != "local-slow" else 1)
        call_id = f"call-{len(self.server.requests)}"
        call = {"index": 0, "id": call_id, "type": "function",
                "function": {"name": "play", "arguments": '{"actions":[{"buttons":16,"frames":2}]}'}}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.path.endswith("/responses"):
            item = {"type": "function_call", "id": call_id, "call_id": call_id,
                    "name": "play", "arguments": call["function"]["arguments"]}
            chunks = [
                {"type": "response.output_item.added", "output_index": 0, "item": item},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"status": "completed",
                    "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}}},
            ]
        else:
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Local test action.",
                              "reasoning_content": "Local test reasoning.", "tool_calls": [call]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
            ]
        try:
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@unittest.skipUnless(importlib.util.find_spec("tau_ai"), "LLM extra is not installed")
class EvalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runs = Path(self.directory.name)
        self.api = ThreadingHTTPServer(("127.0.0.1", 0), LocalAPI)
        self.api.requests = []
        self.thread = threading.Thread(target=self.api.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        for job in evals.list_evals(self.runs):
            if job["status"] in {"running", "queued"}:
                evals.stop_eval(job["id"])
        self.api.shutdown()
        self.api.server_close()
        self.thread.join()
        self.directory.cleanup()

    def launch(self, model, timeout=60, **settings):
        payload = {"model": model, "provider": settings.pop("provider", "custom"),
                   "timeout": timeout,
                   "base_url": f"http://127.0.0.1:{self.api.server_port}/v1",
                   "api_key": "local-test-key", **settings}
        return evals.start_eval(payload, self.runs)[0]

    def wait_for(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("Timed out waiting for the local evaluation")

    def test_real_cli_writes_progress_history_and_video(self):
        job = self.launch("local-test", timeout=4)
        self.wait_for(lambda: (self.runs / job["name"] / "live.png").is_file())
        self.wait_for(lambda: evals.list_evals(self.runs)[0]["status"] != "running")
        final = evals.list_evals(self.runs)[0]
        self.assertEqual(final["status"], "completed", final["error"])
        self.assertGreaterEqual(final["decisions"], 1)
        self.assertGreater(final["frames"], 0)
        self.assertGreaterEqual(final["tokens"], 20 * (final["decisions"] - 1))
        folder = self.runs / job["name"]
        self.assertEqual(len(list((folder / "screenshots").glob("*.png"))), final["decisions"])
        self.assertTrue((folder / "rollout.mp4").stat().st_size)
        self.assertEqual(len(self.api.requests), final["decisions"])
        self.assertGreater(len(self.api.requests[-1]["messages"]), len(self.api.requests[0]["messages"]))
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(self.api.requests[0]["messages"][0]["role"], "system")
        self.assertEqual(self.api.requests[0]["messages"][0]["content"], config["system"])
        self.assertNotIn("local-test-key", json.dumps(config))
        elapsed = final["elapsed"]
        self.assertEqual(evals.list_evals(self.runs)[0]["elapsed"], elapsed)

    def test_responses_request_contains_persisted_system_prompt(self):
        job = self.launch("local-responses", timeout=4, provider="openai", fps=12,
                          max_frames=7, max_actions=2, max_images=2)
        self.wait_for(lambda: job["id"] not in evals._processes)
        final = next(item for item in evals.list_evals(self.runs) if item["id"] == job["id"])
        self.assertEqual(final["status"], "completed", final["error"])
        self.assertTrue(self.api.requests)
        request = self.api.requests[0]
        self.assertEqual(request["model"], "local-responses")
        instructions = request["instructions"]
        self.assertIsInstance(instructions, str)
        self.assertTrue(instructions)
        self.assertIn("This episode is RTC at 12 fps", instructions)
        self.assertIn("Frames must be integers from 1 to 7", instructions)
        self.assertIn("up to 2 sampled frames", instructions)
        self.assertEqual(request["tools"][0]["parameters"]["properties"]["actions"]["maxItems"], 2)

        config = json.loads((self.runs / final["name"] / "config.json").read_text())
        self.assertEqual(config["system"], instructions)
        self.assertTrue(config["system_prompt_sent"])
        self.assertEqual(config["fps"], 12)
        self.assertEqual(config["max_frames"], 7)
        self.assertEqual(config["max_actions"], 2)
        self.assertEqual(config["max_images"], 2)

    def test_cancel_keeps_partial_run_and_failure_is_visible(self):
        job = self.launch("local-slow")
        self.wait_for(lambda: bool(self.api.requests))
        stopped = evals.stop_eval(job["id"])
        self.assertEqual(stopped["status"], "cancelled")
        self.wait_for(lambda: job["id"] not in evals._processes)
        self.assertTrue((self.runs / job["name"] / "checkpoint.state").is_file())
        failed = self.launch("local-failure")
        self.wait_for(lambda: failed["id"] not in evals._processes)
        final = next(job for job in evals.list_evals(self.runs) if job["id"] == failed["id"])
        self.assertEqual(final["status"], "failed")
        self.assertIn("Invalid model", final["error"])
        self.assertNotIn("local-test-key", final["error"])
        self.assertNotIn("local-test-key", (self.runs / failed["name"] / "messages.jsonl").read_text())
