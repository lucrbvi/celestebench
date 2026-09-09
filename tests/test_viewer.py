import json
import io
import http.client
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from web import viewer


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

    def delete(self, path, **headers):
        self.connection.request("DELETE", path, headers=headers)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    def test_runs_and_evaluations_have_separate_linkable_pages(self):
        status, content_type, runs = self.get("/")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('href="/evals"', runs)
        self.assertIn('new URLSearchParams(location.search).get("run")', runs)
        self.assertNotIn('id="evalJobs"', runs)
        self.assertNotIn('"/api/evals"', runs)

        status, content_type, evaluations = self.get("/evals")
        self.assertEqual((status, content_type), (200, "text/html; charset=utf-8"))
        self.assertIn('id="evalJobs"', evaluations)
        self.assertIn('jsonRequest("/api/evals")', evaluations)
        self.assertIn('href="/?run=${encodeURIComponent(job.name)}"', evaluations)
        self.assertIn('.filter(job => ["queued", "running"].includes(job.status))', evaluations)
        self.assertIn('class="eval-live" src="/live/${encodeURIComponent(job.name)}"', evaluations)
        self.assertIn('toggleLiveSound(sound, job.name)', evaluations)

        status, content_type, audio = self.get("/live-audio.js")
        self.assertEqual((status, content_type), (200, "text/javascript; charset=utf-8"))
        self.assertIn("new AudioContext()", audio)

        status, content_type, css = self.get("/viewer.css")
        self.assertEqual((status, content_type), (200, "text/css; charset=utf-8"))
        self.assertIn(".card", css)

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
        runs = self.get("/")[2]
        self.assertIn('id="deleteRun"', runs)
        self.assertIn("confirm(`Delete run", runs)
        self.assertIn('{method:"DELETE"}', runs)
        self.assertIn('"/live/" + encodeURIComponent(run.name)', runs)
        self.assertIn('id="liveSound"', runs)
        self.assertIn('src="/live-audio.js"', runs)

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

    def test_live_audio_reads_pcm_incrementally_and_can_start_at_tail(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.pcm"
            path.write_bytes(b"01234567")

            class Fake:
                wfile = io.BytesIO()
                status = None
                response_headers = {}
                def send_response(self, status): self.status = status
                def send_header(self, name, value): self.response_headers[name] = value
                def end_headers(self): pass
                def send_error(self, status): self.status = status

            fake = Fake()
            viewer.Handler.audio(fake, path, "4")
            self.assertEqual((fake.status, fake.wfile.getvalue()), (200, b"4567"))
            self.assertEqual(fake.response_headers["X-Audio-Offset"], "8")
            self.assertEqual(fake.response_headers["X-Audio-Rate"], "22050")

            fake.wfile = io.BytesIO()
            viewer.Handler.audio(fake, path, "tail")
            self.assertEqual(fake.wfile.getvalue(), b"")
            self.assertEqual(fake.response_headers["X-Audio-Offset"], "8")

    def test_launch_stop_and_validation_errors_are_json(self):
        job = {"id": "abc", "name": "model/run", "status": "running"}
        payload = {"model": "custom-vlm", "api": "openai-completions",
                   "base_url": "http://localhost:9000/v1", "timeout": 120}
        with patch.object(viewer.evals, "start_eval", return_value=[job]) as launch:
            self.assertEqual(self.post("/api/evals", payload), (201, [job]))
            launch.assert_called_once_with(payload, viewer.RUNS)
        with patch.object(viewer.evals, "stop_eval", return_value=job) as stop:
            self.assertEqual(self.post("/api/evals/abc/stop", {}), (200, job))
            stop.assert_called_once_with("abc")
        with patch.object(viewer.evals, "start_eval", side_effect=ValueError("Invalid model")):
            self.assertEqual(self.post("/api/evals", {}), (400, {"error": "Invalid model"}))

    def test_cross_origin_and_non_json_requests_cannot_launch(self):
        with patch.object(viewer.evals, "start_eval") as launch:
            self.assertEqual(self.post("/api/evals", {}, Origin="https://example.com")[0], 403)
            self.assertEqual(self.post("/api/evals", {}, Host="example.com")[0], 403)
            self.assertEqual(self.post("/api/evals", {}, **{"Content-Type": "text/plain"})[0], 415)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
