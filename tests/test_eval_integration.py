"""Exercise the real CLI/emulator against a local streaming API, without paid calls."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
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
        call = {"index": 0, "id": f"call-{len(self.server.requests)}", "type": "function",
                "function": {"name": "play", "arguments": '{"actions":[{"buttons":16,"frames":2}]}'}}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
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

    def launch(self, model, timeout=60):
        return evals.start_eval({"model": model, "api": "openai-completions", "timeout": timeout,
                                 "base_url": f"http://127.0.0.1:{self.api.server_port}/v1",
                                 "api_key": "local-test-key"}, self.runs)[0]

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
        self.assertTrue((folder / "live.pcm").stat().st_size)
        self.assertTrue((folder / "rollout.mp4").stat().st_size)
        with av.open(str(folder / "rollout.mp4")) as recording:
            self.assertEqual(recording.streams.audio[0].rate, 22050)
        self.assertEqual(len(self.api.requests), final["decisions"])
        self.assertGreater(len(self.api.requests[-1]["messages"]), len(self.api.requests[0]["messages"]))
        self.assertNotIn("local-test-key", (folder / "config.json").read_text())
        elapsed = final["elapsed"]
        self.assertEqual(evals.list_evals(self.runs)[0]["elapsed"], elapsed)

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
