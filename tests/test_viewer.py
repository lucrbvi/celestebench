import json
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import viewer


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


if __name__ == "__main__":
    unittest.main()
