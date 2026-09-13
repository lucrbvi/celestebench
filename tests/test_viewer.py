import http.client
import io
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import av
import numpy as np

from web import export, viewer


class ViewerTest(unittest.TestCase):
    def run_dir(self, root, name, *, video=None):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"model": "same-model"}))
        if video is not None:
            (folder / "rollout.mp4").write_bytes(video)
        return folder

    def test_scan_and_decisions_keep_partial_runs_and_exact_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.run_dir(root, "a", video=b"corrupt")
            self.run_dir(root, "nested/b")
            (first / "messages.jsonl").write_text(
                json.dumps({"role": "user", "content": [{"type": "image", "data": "AA=="}]}) + "\n"
                + json.dumps({"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "one"},
                    {"type": "thinking", "thinking": "two"},
                    {"type": "text", "text": "reply"},
                    {"type": "toolCall", "arguments": {"actions": [{"buttons": 0, "frames": 2}]}}]}) + "\n"
                + json.dumps({"role": "user", "content": []}) + "\n"
                + "{incomplete\n"
            )
            (first / "actions.jsonl").write_text(json.dumps({
                "decision": 0, "buttons": 0, "frames": 2, "frame_start": 4, "frame_end": 6,
                "latency": 0.25}) + "\n")
            (first / "decisions.jsonl").write_text(json.dumps({
                "decision": 0, "status": "played", "frame_start": 1, "frame_end": 6}) + "\n" + json.dumps({
                "decision": 1, "status": "timeout", "latency": 2, "error": "deadline",
                "frame_start": 6, "frame_end": 60}) + "\n")
            with patch.object(viewer, "RUNS", root):
                runs = viewer.scan_runs()
                data = viewer.load_decisions("a")
            self.assertEqual({r["name"] for r in runs}, {"a", "nested/b"})
            self.assertIsNone(next(r for r in runs if r["name"] == "a")["video"])
            self.assertEqual(data["timeline_exact"], True)
            self.assertEqual(len(data["decisions"]), 2)
            self.assertAlmostEqual(data["decisions"][0]["actions"][0]["t0"], 4 / 30, places=5)
            self.assertEqual(data["decisions"][0]["thinking"], "one\ntwo")
            self.assertEqual(data["decisions"][1]["status"], "timeout")
            self.assertAlmostEqual(data["decisions"][1]["from"], 6 / 30)
            self.assertEqual(data["decisions"][1]["actions"], [])

    def test_codex_trace_is_merged_into_the_nested_rollout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapper = root / "gpt" / "run"
            rollout = wrapper / "rollout"
            rollout.mkdir(parents=True)
            (wrapper / "config.json").write_text(json.dumps({"model": "gpt"}))
            (rollout / "config.json").write_text(json.dumps({"model": "gpt"}))
            (rollout / "decisions.jsonl").write_text('\n'.join(json.dumps({
                "decision": index, "status": "played", "frame_start": start, "frame_end": end,
            }) for index, (start, end) in enumerate([(1, 60), (60, 200)])) + "\n")
            (rollout / "actions.jsonl").write_text('\n'.join(json.dumps({
                "decision": index, "buttons": 1, "frames": 2, "frame_start": start,
                "frame_end": start + 2}) for index, start in enumerate([1, 60])) + "\n")
            (wrapper / "codex.jsonl").write_text('\n'.join(json.dumps(row) for row in [
                {"type": "item.completed", "item": {"type": "reasoning", "text": "plan one"}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "go"}},
                {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
                 "arguments": {"actions": [{"buttons": 1, "frames": 2}]},
                 "result": {"content": [{"type": "text", "text": json.dumps({"frame_ids": [60]})}]}}},
                {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "observe",
                 "arguments": {}}},
                {"type": "item.completed", "item": {"type": "reasoning", "text": "plan two"}},
                {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
                 "arguments": {"actions": [{"buttons": 2, "frames": 1}]},
                 "result": {"content": [{"type": "text", "text": json.dumps({"frame_ids": [200]})}]}}},
                {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
                 "arguments": {"actions": [{"buttons": 0, "frames": 1}]},
                 "result": {"content": [{"type": "text", "text": "Error executing tool play"}]}}},
            ]) + "\n")
            with patch.object(viewer, "RUNS", root):
                decisions = viewer.load_decisions("gpt/run")["decisions"]
            self.assertEqual(len(decisions), 2)
            self.assertEqual(decisions[0]["thinking"], "plan one")
            self.assertEqual(decisions[0]["text"], "go")
            self.assertEqual(decisions[0]["tool"], {"actions": [{"buttons": 1, "frames": 2}]})
            self.assertEqual(decisions[1]["thinking"], "plan two")
            self.assertEqual(decisions[1]["tool"], {"actions": [{"buttons": 2, "frames": 1}]})

    def test_legacy_actions_are_approximate_and_start_after_observation_frame(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.run_dir(root, "legacy")
            (folder / "config.json").write_text(json.dumps({"model": "m", "fps": 20}))
            (folder / "messages.jsonl").write_text(json.dumps({"role": "user", "content": []}) + "\n"
                                                     + json.dumps({"role": "assistant", "content": []}) + "\n")
            (folder / "actions.jsonl").write_text(json.dumps({"decision": 0, "buttons": 2,
                                                               "frames": 3, "latency": 0.5}) + "\n")
            with patch.object(viewer, "RUNS", root):
                data = viewer.load_decisions("legacy")
            action = data["decisions"][0]["actions"][0]
            self.assertFalse(data["timeline_exact"])
            self.assertEqual(action["frame_start"], 11)  # one initial frame + ten idle frames
            # Pacing is 20 Hz, but video is native 30 fps, not wall-clock time.
            self.assertAlmostEqual(action["t0"], 11 / 30, places=5)

    def test_http_ranges_and_path_containment(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_dir(root, "run", video=b"0123456789")
            class Fake:
                path = "/video/run/rollout.mp4"
                headers = {"Range": "bytes=2-5"}
                wfile = io.BytesIO()
                status = None
                response_headers = {}
                def send_response(self, status): self.status = status
                def send_header(self, name, value): self.response_headers[name] = value
                def end_headers(self): pass
                def send_error(self, status): self.status = status
            fake = Fake()
            viewer.Handler.video(fake, root / "run" / "rollout.mp4")
            self.assertEqual(fake.status, 206)
            self.assertEqual(fake.wfile.getvalue(), b"2345")
            fake.headers = {"Range": "bytes=-3"}
            fake.wfile = io.BytesIO()
            viewer.Handler.video(fake, root / "run" / "rollout.mp4")
            self.assertEqual(fake.wfile.getvalue(), b"789")
            fake.headers = {"Range": "bytes=20-30"}
            viewer.Handler.video(fake, root / "run" / "rollout.mp4")
            self.assertEqual(fake.status, 416)
            with patch.object(viewer, "RUNS", root):
                fake.path = "/video/../viewer.py"
                fake.headers = {}
                viewer.Handler.do_GET(fake)
                self.assertEqual(fake.status, 404)

    def test_live_stream_waits_for_a_late_frame_and_ends_with_the_job(self):
        class Fake:
            close_connection = False
            connection = type("Connection", (), {"settimeout": lambda self, value: None})()
            wfile = io.BytesIO()
            status = None
            def send_response(self, status): self.status = status
            def send_header(self, name, value): pass
            def end_headers(self): pass
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            folder.mkdir()
            fake = Fake()
            calls = {"count": 0}
            def running():
                calls["count"] += 1
                return calls["count"] < 3
            # No frame on disk yet: the stream must stay open while the job runs,
            # then stop on its own when the harness reports completion.
            viewer.Handler.live(fake, folder, running)
            self.assertEqual(fake.status, 200)
            self.assertTrue(fake.wfile.getvalue().endswith(b"--frame--\r\n"))

    def test_live_stream_finds_a_nested_rollout_created_after_it_opens(self):
        class Fake:
            close_connection = False
            connection = type("Connection", (), {"settimeout": lambda self, value: None})()
            wfile = io.BytesIO()
            status = None
            def send_response(self, status): self.status = status
            def send_header(self, name, value): pass
            def end_headers(self): pass
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            folder.mkdir()
            frame = b"\x89PNG\r\nnested"
            calls = {"count": 0}
            def running():
                calls["count"] += 1
                if calls["count"] == 2:  # the harness starts the game mid-stream
                    rollout = folder / "rollout"
                    rollout.mkdir()
                    (rollout / "live.png").write_bytes(frame)
                    (rollout / "live.done").touch()
                return calls["count"] < 4
            fake = Fake()
            viewer.Handler.live(fake, folder, running)
            self.assertIn(frame, fake.wfile.getvalue())

    def test_job_running_only_reports_live_jobs(self):
        with patch.dict(viewer.evals._jobs, {
                "alive": {"status": "running"}, "done": {"status": "completed"}}):
            self.assertTrue(viewer.evals.job_running("alive"))
            self.assertFalse(viewer.evals.job_running("done"))
            self.assertFalse(viewer.evals.job_running("missing"))

    def test_timeout_reasoning_without_video_and_legacy_batch_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.run_dir(root, "partial")
            (folder / "messages.jsonl").write_text('\n'.join(json.dumps(m) for m in [
                {"role": "user", "content": []},
                {"role": "assistant", "status": "timeout", "stopReason": "aborted",
                 "errorMessage": "deadline", "content": [{"type": "thinking", "thinking": "partial text"}]},
            ]))
            with patch.object(viewer, "RUNS", root):
                d = viewer.load_decisions("partial")["decisions"][0]
            self.assertEqual((d["status"], d["thinking"], d["actions"]), ("timeout", "partial text", []))
            self.assertTrue(d["partial"])
            (folder / "actions.jsonl").write_text('\n'.join(json.dumps({
                "decision": 0, "buttons": b, "frames": n, "latency": 0}) for b, n in [(0, 2), (2, 3)]))
            with patch.object(viewer, "RUNS", root):
                actions = viewer.load_decisions("partial")["decisions"][0]["actions"]
            self.assertEqual([(a["frame_start"], a["frame_end"]) for a in actions], [(1, 3), (3, 6)])

    def test_persisted_screenshot_is_used_and_served_safely(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.run_dir(root, "run")
            screenshot = folder / "screenshots" / "000000.png"
            screenshot.parent.mkdir()
            screenshot.write_bytes(b"\x89PNG\r\n")
            (folder / "decisions.jsonl").write_text(json.dumps({
                "decision": 0, "status": "timeout", "screenshot": "screenshots/000000.png"}) + "\n")
            with patch.object(viewer, "RUNS", root):
                decision = viewer.load_decisions("run")["decisions"][0]
                self.assertEqual(decision["screenshot"], "/screenshot/run/screenshots/000000.png")
                fake = type("Fake", (), {
                    "path": "/screenshot/run/screenshots/000000.png", "headers": {},
                    "wfile": io.BytesIO(), "send_response": lambda s, n: setattr(s, "status", n),
                    "send_header": lambda s, k, v: None, "end_headers": lambda s: None,
                    "send_error": lambda s, n: setattr(s, "status", n), "status": None,
                })()
                fake.image = viewer.Handler.image.__get__(fake)
                viewer.Handler.do_GET(fake)
                self.assertEqual(fake.status, 200)
                self.assertEqual(fake.wfile.getvalue(), b"\x89PNG\r\n")
                fake.path = "/screenshot/../viewer.py"
                viewer.Handler.do_GET(fake)
                self.assertEqual(fake.status, 404)

    def test_wait_marker_is_preserved_for_the_viewer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.run_dir(root, "run")
            (folder / "actions.jsonl").write_text(json.dumps({
                "decision": 0, "buttons": 0, "frames": 3, "action": "wait",
            }) + "\n")
            with patch.object(viewer, "RUNS", root):
                action = viewer.load_decisions("run")["decisions"][0]["actions"][0]
            self.assertTrue(action["wait"])

    def test_leaderboard_uses_one_wall_clock_event_and_keeps_unscored_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, progress, actions in (("a", 40, 4), ("b", 60, 4), ("different", 90, 2)):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({
                    "fps": 30, "max_actions": actions, "max_frames": 30,
                }))
                (folder / "score.json").write_text(json.dumps({
                    "version": 1, "metric": "grounded_height_v1", "progress": progress,
                    "status": "completed",
                    "elapsed": 10, "timing": "wall_clock",
                }))
                (folder / "progress.jsonl").write_text("\n".join(json.dumps(row) for row in [
                    {"elapsed": 5, "progress": progress - 10},
                    {"elapsed": 10, "progress": progress},
                ]))
                runs.append({"name": name, "model": "model", "harness": "tau",
                             "status": "completed", "timeout": 10, "elapsed": 10})
            short = root / "short"
            short.mkdir()
            (short / "config.json").write_text(json.dumps({"fps": 30, "max_actions": 4,
                                                            "max_frames": 30}))
            (short / "score.json").write_text(json.dumps({
                "version": 1, "metric": "grounded_height_v1", "progress": 80,
                "elapsed": 4, "timing": "wall_clock", "status": "completed",
            }))
            (short / "progress.jsonl").write_text(json.dumps({"elapsed": 4, "progress": 80}))
            runs.append({"name": "short", "model": "model", "harness": "tau",
                         "status": "completed", "timeout": 10, "elapsed": 4})
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            self.assertEqual(result["excluded"], {"replay": 0, "unknown_timing": 0})
            grouped = {(row["settings"]["max_actions"]): row for row in result["groups"]}
            self.assertEqual(grouped[4]["progress"], 50)
            self.assertEqual(grouped[4]["scored"], 2)
            self.assertEqual(grouped[4]["unscored"], 1)
            self.assertEqual(grouped[4]["scored_runs"], [{"name": "b", "progress": 60},
                                                         {"name": "a", "progress": 40}])
            self.assertEqual(grouped[2]["progress"], 90)
            self.assertEqual(grouped[2]["scored_runs"], [{"name": "different", "progress": 90}])

    def test_leaderboard_does_not_score_replay_or_legacy_scoreless_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            replay.mkdir()
            (replay / "config.json").write_text(json.dumps({"max_actions": 4}))
            (replay / "score.json").write_text(json.dumps({
                "metric": "grounded_height_v1", "progress": 70, "elapsed": 10,
                "timing": "replay", "status": "completed",
            }))
            (replay / "progress.jsonl").write_text(json.dumps({"elapsed": 10, "progress": 70}))
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "config.json").write_text(json.dumps({"max_actions": 4}))
            runs = [{"name": "replay", "model": "m", "harness": "tau", "status": "completed", "timeout": 10},
                    {"name": "legacy", "model": "m", "harness": "tau", "status": "completed", "timeout": 10}]
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            self.assertEqual(result["excluded"]["replay"], 1)
            self.assertTrue(all(row["status"] == "unscored" for row in result["groups"]))

    def test_leaderboard_reads_score_options_and_excludes_failed_archives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, actions, status, system in [
                    ("a", 4, "completed", "first"), ("b", 2, "completed", "first"),
                    ("c", 4, "error", "first"), ("d", 4, "completed", "other")]:
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"system": system, "max_actions": 1}))
                (folder / "score.json").write_text(json.dumps({
                    "metric": "grounded_height_v1", "progress": 2, "elapsed": 20,
                    "timing": "wall_clock", "status": status,
                    "options": {"max_actions": actions, "fps": 30},
                }))
                (folder / "progress.jsonl").write_text(
                    '\n'.join(json.dumps(row) for row in [
                        {"elapsed": 0, "progress": 0}, {"elapsed": 9, "progress": 1},
                        {"elapsed": 11, "progress": 2}]))
                runs.append({"name": name, "model": "m", "status": "archived", "timeout": 20})
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual(sum(row["scored"] for row in result["groups"]), 3)
            self.assertTrue(all(row["progress"] == 1 for row in result["groups"]))
            self.assertEqual({row["settings"]["max_actions"] for row in result["groups"]}, {4, 2})

    def test_leaderboard_excludes_invalid_scores_and_merges_prompt_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, config, progress, invalid in [
                    ("verified", {"system_prompt_sent": True}, 10, None),
                    ("unknown", {}, 20, None),
                    ("invalid", {"system_prompt_sent": True}, 999, "missing_system_prompt")]:
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps(config))
                score = {"metric": "grounded_height_v1", "progress": progress,
                         "elapsed": 10, "timing": "wall_clock", "status": "completed"}
                if invalid:
                    score["invalid_reason"] = invalid
                (folder / "score.json").write_text(json.dumps(score))
                (folder / "progress.jsonl").write_text(json.dumps({"elapsed": 10, "progress": progress}))
                runs.append({"name": name, "model": "m", "harness": "tau",
                             "status": "completed", "timeout": 10})
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            self.assertEqual(len(result["groups"]), 1)
            self.assertEqual(result["groups"][0]["progress"], 15)
            self.assertEqual(result["groups"][0]["scored"], 2)

    def test_leaderboard_prices_rollouts_and_separates_versions(self):
        from celestebench import catalog
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, version in (("a", "0.1"), ("b", "0.2")):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({
                    "model": "gpt-5.6-sol", "fps": 30, "max_frames": 30,
                    "max_actions": 4, "benchmark_version": version}))
                (folder / "score.json").write_text(json.dumps({
                    "metric": "grounded_height_v1", "progress": 50, "elapsed": 10,
                    "timing": "wall_clock", "status": "completed"}))
                (folder / "progress.jsonl").write_text(json.dumps({"elapsed": 10, "progress": 50}))
                (folder / "messages.jsonl").write_text(json.dumps({
                    "role": "assistant",
                    "usage": {"input": 1000, "output": 2000, "cacheRead": 0,
                              "cacheWrite": 0}}) + "\n")
                runs.append({"name": name, "model": "gpt-5.6-sol", "harness": "tau",
                             "status": "completed", "timeout": 10})
            prices = {"openai": {"models": {"gpt-5.6-sol": {
                "id": "gpt-5.6-sol", "cost": {"input": 4, "output": 20}}}}}
            with patch.object(viewer, "RUNS", root), \
                    patch.object(viewer, "scan_runs", return_value=runs), \
                    patch.object(catalog, "_models_dev", return_value=prices):
                result = viewer.leaderboard(10)
            self.assertEqual(len(result["groups"]), 2)
            row = next(r for r in result["groups"]
                       if r["settings"]["benchmark_version"] == "0.1")
            self.assertEqual(row["producer"], "openai")
            self.assertEqual(row["producer_name"], "OpenAI")
            self.assertAlmostEqual(row["cost"], (1000 * 4 + 2000 * 20) / 1_000_000, places=6)

    def test_leaderboard_folds_old_none_into_thinking_off(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, level in (("old", {"reasoning_effort": "none"}),
                                ("new", {"thinking_level": "off"}),
                                ("hot", {"reasoning_effort": "high"})):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({
                    "fps": 30, "max_actions": 4, "max_frames": 30, **level}))
                (folder / "score.json").write_text(json.dumps({
                    "metric": "grounded_height_v1", "progress": 50, "elapsed": 10,
                    "timing": "wall_clock", "status": "completed"}))
                (folder / "progress.jsonl").write_text(json.dumps({"elapsed": 10, "progress": 50}))
                runs.append({"name": name, "model": "model", "harness": "tau",
                             "status": "completed", "timeout": 10})
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            merged = next(row for row in result["groups"]
                          if row["settings"]["thinking_level"] == "off")
            self.assertEqual(merged["scored"], 2)
            self.assertNotIn("reasoning_effort", merged["settings"])
            hot = next(row for row in result["groups"]
                       if row["settings"]["thinking_level"] == "high")
            self.assertEqual(hot["scored"], 1)

    def test_leaderboard_keeps_harnesses_and_setting_generations_apart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, harness, config, progress in (
                    ("tau-old", "tau", {"api": "openai-responses", "max_actions": 4}, 40),
                    ("tau-new", "tau", {"provider": "opencode-go", "max_actions": 4}, 60),
                    ("codex", "codex", {"max_actions": 4}, 80)):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({
                    "thinking_level": "low", "fps": 30, "max_frames": 30, **config}))
                (folder / "score.json").write_text(json.dumps({
                    "metric": "grounded_height_v1", "progress": progress, "elapsed": 10,
                    "timing": "wall_clock", "status": "completed"}))
                (folder / "progress.jsonl").write_text(
                    json.dumps({"elapsed": 10, "progress": progress}))
                runs.append({"name": name, "model": "model", "harness": harness,
                             "status": "completed", "timeout": 10})
            with patch.object(viewer, "RUNS", root), patch.object(viewer, "scan_runs", return_value=runs):
                result = viewer.leaderboard(10)
            self.assertEqual(len(result["groups"]), 3)
            self.assertEqual({row["settings"]["harness"] for row in result["groups"]},
                             {"tau", "codex"})


class ViewerHTTPTest(unittest.TestCase):
    def setUp(self):
        self.server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, payload, **headers):
        self.connection.request("POST", path, json.dumps(payload),
                                {"Content-Type": "application/json", **headers})
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    def get(self, path):
        self.connection.request("GET", path)
        response = self.connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read().decode()

    def get_bytes(self, path):
        self.connection.request("GET", path)
        response = self.connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()

    def test_json_ignores_a_client_that_disconnects_before_the_body(self):
        class ClosedClient:
            def write(self, body):
                raise BrokenPipeError

        handler = type("Handler", (), {
            "wfile": ClosedClient(),
            "send_response": lambda self, status: None,
            "send_header": lambda self, name, value: None,
            "end_headers": lambda self: None,
        })()
        viewer.Handler.json(handler, {"ok": True})

    def delete(self, path, **headers):
        self.connection.request("DELETE", path, headers=headers)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    def test_harnesses_endpoint_feeds_the_new_eval_form(self):
        status, content_type, body = self.get("/api/harnesses")
        self.assertEqual((status, content_type), (200, "application/json"))
        catalog = {entry["key"]: entry for entry in json.loads(body)}
        self.assertTrue(catalog["tau"]["builtin"])
        self.assertFalse(catalog["codex"]["builtin"])
        self.assertEqual([field["key"] for field in catalog["codex"]["run"]][:2],
                         ["model", "thinking_level"])
        self.assertEqual([field["key"] for field in catalog["codex"]["options"]],
                         ["prompt", "max_frames", "frames", "fps"])

    def test_runs_and_evaluations_have_separate_linkable_pages(self):
        status, content_type, root = self.get("/")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('id="leaderboard"', root)
        self.assertIn('id="budget"', root)
        self.assertIn('href="/runs"', root)

        status, content_type, runs = self.get("/runs")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('href="/evals"', runs)
        self.assertIn('href="/leaderboard"', runs)
        self.assertIn('new URLSearchParams(location.search).get("run")', runs)
        self.assertNotIn('id="evalJobs"', runs)
        self.assertNotIn('"/api/evals"', runs)

        status, content_type, evaluations = self.get("/evals")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('id="evalJobs"', evaluations)
        self.assertIn('jsonRequest("/api/evals", {timeout: 10000})', evaluations)
        self.assertIn('href="/runs?run=${encodeURIComponent(job.name)}"', evaluations)
        self.assertIn('.filter(job => ["queued", "running"].includes(job.status))', evaluations)
        self.assertIn('job.run_ready !== false', evaluations)
        self.assertIn('class="eval-live" src="/live/${encodeURIComponent(job.name)}"', evaluations)
        self.assertNotIn("live-audio", evaluations)

        status, content_type, css = self.get("/viewer.css")
        self.assertEqual((status, content_type), (200, "text/css; charset=utf-8"))
        self.assertIn(".card", css)

        status, content_type, leaderboard = self.get("/leaderboard")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('id="budget"', leaderboard)
        self.assertIn('id="leaderboard"', leaderboard)
        self.assertIn('href="/runs?run=${encodeURIComponent(', leaderboard)

    def test_delete_run_removes_files_and_terminal_evaluation_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            folder = root / "bad-model" / "run"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}")
            metadata = root / ".evals" / "job.json"
            metadata.parent.mkdir()
            metadata.write_text(json.dumps({
                "id": "job", "name": "bad-model/run", "status": "failed",
            }))
            viewer.evals._jobs["job"] = {"id": "job", "name": "bad-model/run", "status": "failed"}
            try:
                with patch.object(viewer, "RUNS", root):
                    self.assertEqual(self.delete("/api/run/bad-model%2Frun"),
                                     (200, {"deleted": "bad-model/run"}))
                self.assertFalse(folder.exists())
                self.assertFalse(metadata.exists())
                self.assertNotIn("job", viewer.evals._jobs)
            finally:
                viewer.evals._jobs.pop("job", None)

    def test_delete_run_rejects_active_jobs_and_unsafe_paths(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "runs"
            folder = root / "model" / "run"
            folder.mkdir(parents=True)
            outside = base / "outside"
            outside.mkdir()
            metadata = root / ".evals" / "job.json"
            metadata.parent.mkdir()
            metadata.write_text(json.dumps({
                "id": "job", "name": "model/run", "status": "running",
            }))
            viewer.evals._jobs["job"] = {"id": "job", "name": "model/run", "status": "running"}
            try:
                with patch.object(viewer, "RUNS", root):
                    self.assertEqual(self.delete("/api/run/model%2Frun")[0], 409)
                    self.assertEqual(self.delete("/api/run/..%2Foutside")[0], 404)
                    self.assertEqual(self.delete("/api/run/.evals")[0], 404)
                self.assertTrue(folder.exists())
                self.assertTrue(outside.exists())
                self.assertTrue(metadata.exists())
            finally:
                viewer.evals._jobs.pop("job", None)

    def test_runs_page_confirms_before_deleting_the_selected_run(self):
        runs = self.get("/runs")[2]
        self.assertIn('id="deleteRun"', runs)
        self.assertIn("confirm(`Delete run", runs)
        self.assertIn('{method:"DELETE"}', runs)
        self.assertIn('"/live/" + encodeURIComponent(run.name)', runs)
        self.assertNotIn('id="liveSound"', runs)

    def test_live_endpoint_streams_the_atomic_frame(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "model" / "run"
            folder.mkdir(parents=True)
            frame = b"\x89PNG\r\nlatest"
            (folder / "live.png").write_bytes(frame)
            (folder / "live.done").touch()
            job = {"name": "model/run", "status": "running"}
            with patch.object(viewer, "RUNS", root), patch.object(
                    viewer.evals, "list_evals", return_value=[job]):
                status, content_type, body = self.get_bytes("/live/model%2Frun")
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "multipart/x-mixed-replace; boundary=frame")
            self.assertIn(b"Content-Type: image/png", body)
            self.assertIn(frame, body)

    def test_launch_stop_and_validation_errors_are_json(self):
        job = {"id": "abc", "name": "model/run", "status": "running"}
        payload = {"model": "custom-vlm", "provider": "custom",
                   "base_url": "http://localhost:9000/v1", "timeout": 120}
        with patch.object(viewer.evals, "start_eval", return_value=[job]) as launch:
            self.assertEqual(self.post("/api/evals", payload), (201, [job]))
            launch.assert_called_once_with(payload, viewer.RUNS)
        with patch.object(viewer.evals, "stop_eval", return_value=job) as stop:
            self.assertEqual(self.post("/api/evals/abc/stop", {}), (200, job))
            stop.assert_called_once_with("abc")
        with patch.object(viewer.evals, "start_eval", side_effect=ValueError("Invalid model")):
            self.assertEqual(self.post("/api/evals", {}), (400, {"error": "Invalid model"}))

    def test_video_stays_raw_and_export_endpoint_downloads_the_annotation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"model": "m"}))
            (folder / "actions.jsonl").write_text(json.dumps({
                "decision": 0, "buttons": 2, "frames": 2,
                "frame_start": 1, "frame_end": 3}) + "\n")
            with av.open(str(folder / "rollout.mp4"), "w", format="mp4") as out:
                stream = out.add_stream("libx264", rate=30)
                stream.width = stream.height = 512
                stream.pix_fmt = "yuv420p"
                for _ in range(3):
                    out.mux(stream.encode(av.VideoFrame.from_ndarray(
                        np.zeros((512, 512, 3), np.uint8), format="rgb24")))
                out.mux(stream.encode())
            with patch.object(viewer, "RUNS", root), patch.object(export, "RUNS", root):
                status, content_type, body = self.get_bytes("/video/run/rollout.mp4")
                self.assertEqual((status, content_type), (200, "video/mp4"))
                self.assertEqual(body, (folder / "rollout.mp4").read_bytes())
                self.assertFalse((folder / "export.mp4").is_file())
                self.connection.request("GET", "/export/run")
                response = self.connection.getresponse()
                body = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Content-Type"), "video/mp4")
                self.assertEqual(response.getheader("Content-Disposition"),
                                 'attachment; filename="run.mp4"')
            self.assertTrue((folder / "export.mp4").is_file())
            self.assertEqual(body, (folder / "export.mp4").read_bytes())

    def test_cross_origin_and_non_json_requests_cannot_launch(self):
        with patch.object(viewer.evals, "start_eval") as launch:
            self.assertEqual(self.post("/api/evals", {}, Origin="https://example.com")[0], 403)
            self.assertEqual(self.post("/api/evals", {}, Host="example.com")[0], 403)
            self.assertEqual(self.post("/api/evals", {}, **{"Content-Type": "text/plain"})[0], 415)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
