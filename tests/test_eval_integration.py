"""Exercise the real CLI/emulator against a local streaming API, without paid calls."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from conftest import wait_until
from web import evals

pytestmark = pytest.mark.skipif(not importlib.util.find_spec("tau_ai"),
                                reason="LLM extra is not installed")


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


@pytest.fixture
def local_api(tmp_path):
    api = ThreadingHTTPServer(("127.0.0.1", 0), LocalAPI)
    api.requests = []
    thread = threading.Thread(target=api.serve_forever, daemon=True)
    thread.start()
    try:
        yield tmp_path, api
    finally:
        for job in evals.list_evals(tmp_path):
            if job["status"] in {"running", "queued"}:
                evals.stop_eval(job["id"])
        api.shutdown()
        api.server_close()
        thread.join()


def launch(api, runs, model, timeout=60, **settings):
    payload = {"model": model, "provider": settings.pop("provider", "custom"),
               "base_url": f"http://127.0.0.1:{api.server_port}/v1",
               "api_key": "local-test-key", **settings}
    payload.setdefault("mode", "rtc")
    # The modes pin the wall-clock budget; keep the test's short deadlines.
    mode = {**evals.MODES[payload["mode"]], "timeout": timeout}
    with patch.dict(evals.MODES, {payload["mode"]: mode}):
        return evals.start_eval(payload, runs)[0]


def test_real_cli_writes_progress_history_and_video(local_api):
    runs, api = local_api
    job = launch(api, runs, "local-test", timeout=4)
    wait_until(lambda: (runs / job["name"] / "live.png").is_file(), timeout=15, interval=0.05)
    wait_until(lambda: evals.list_evals(runs)[0]["status"] != "running", timeout=15, interval=0.05)
    final = evals.list_evals(runs)[0]
    assert final["status"] == "completed", final["error"]
    assert final["decisions"] >= 1
    assert final["frames"] > 0
    assert final["tokens"] >= 20 * (final["decisions"] - 1)
    folder = runs / job["name"]
    assert len(list((folder / "screenshots").glob("*.png"))) == final["decisions"]
    assert (folder / "rollout.mp4").stat().st_size
    assert len(api.requests) == final["decisions"]
    assert len(api.requests[-1]["messages"]) > len(api.requests[0]["messages"])
    config = json.loads((folder / "config.json").read_text())
    assert api.requests[0]["messages"][0]["role"] == "system"
    assert api.requests[0]["messages"][0]["content"] == config["system"]
    assert "local-test-key" not in json.dumps(config)
    elapsed = final["elapsed"]
    assert evals.list_evals(runs)[0]["elapsed"] == elapsed


def test_responses_request_contains_persisted_system_prompt(local_api):
    runs, api = local_api
    custom = {**evals.MODES["rtc"], "fps": 12, "max_frames": 7, "max_images": 2}
    with patch.dict(evals.MODES, {"test": custom}):
        job = launch(api, runs, "local-responses", timeout=4, provider="openai", mode="test")
    wait_until(lambda: job["id"] not in evals._processes, timeout=15, interval=0.05)
    final = next(item for item in evals.list_evals(runs) if item["id"] == job["id"])
    assert final["status"] == "completed", final["error"]
    assert api.requests
    request = api.requests[0]
    assert request["model"] == "local-responses"
    instructions = request["instructions"]
    assert isinstance(instructions, str)
    assert instructions
    assert "This episode is RTC at 12 fps" in instructions
    assert "Frames must be integers from 1 to 7" in instructions
    assert "up to 2 sampled frames" in instructions

    config = json.loads((runs / final["name"] / "config.json").read_text())
    assert config["system"] == instructions
    assert config["system_prompt_sent"]
    assert config["fps"] == 12
    assert config["max_frames"] == 7
    assert config["max_images"] == 2


def test_cancel_keeps_partial_run_and_failure_is_visible(local_api):
    runs, api = local_api
    job = launch(api, runs, "local-slow")
    wait_until(lambda: bool(api.requests), timeout=15, interval=0.05)
    stopped = evals.stop_eval(job["id"])
    assert stopped["status"] == "cancelled"
    wait_until(lambda: job["id"] not in evals._processes, timeout=15, interval=0.05)
    assert (runs / job["name"] / "checkpoint.state").is_file()
    failed = launch(api, runs, "local-failure")
    wait_until(lambda: failed["id"] not in evals._processes, timeout=15, interval=0.05)
    final = next(job for job in evals.list_evals(runs) if job["id"] == failed["id"])
    assert final["status"] == "failed"
    assert "Invalid model" in final["error"]
    assert "local-test-key" not in final["error"]
    assert "local-test-key" not in (runs / failed["name"] / "messages.jsonl").read_text()
