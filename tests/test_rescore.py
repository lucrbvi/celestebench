import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from celestebench.rescore import rescore_root, rescore_run
from celestebench.rollout import _png


class FakeProgress:
    def __init__(self):
        self.value = 0

    def update(self, state):
        old = self.value
        self.value = state["value"]
        return self.value > old

    def snapshot(self):
        return {"version": 1, "metric": "fake", "progress": self.value,
                "rooms_completed": 0, "room_progress": self.value / 100}


class FakeEnv:
    def __init__(self):
        self.buttons = []
        self.game_state = {"value": 0}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def step(self, buttons, frames):
        self.buttons.extend([buttons] * frames)
        self.game_state = {"value": len(self.buttons)}


class RescoreTest(unittest.TestCase):
    def make_run(self, root, name="model/run"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "live.done").touch()
        (directory / "checkpoint.state").write_bytes(bytes([0, 2, 0, 16]))
        return directory

    def test_replays_every_checkpoint_byte_and_writes_replay_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.make_run(Path(tmp))
            env = FakeEnv()
            with patch("celestebench.rescore.Open8", return_value=env), \
                    patch("celestebench.rescore.Progress", return_value=FakeProgress()):
                result = rescore_run(directory)

            self.assertEqual(result["status"], "rescored")
            self.assertEqual(env.buttons, [0, 2, 0, 16])
            progress = [json.loads(line) for line in
                        (directory / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([event["frame"] for event in progress], [0, 1, 2, 3, 4])
            self.assertEqual([event["elapsed"] for event in progress],
                             [0.0, 1 / 30, 2 / 30, 3 / 30, 4 / 30])
            score = json.loads((directory / "score.json").read_text())
            self.assertEqual(score["timing"], "replay")
            self.assertEqual(score["frame"], 4)
            self.assertEqual(score["elapsed"], 4 / 30)
            self.assertEqual(score["status"], "completed")
            self.assertIn("checkpoint_sha256", score["provenance"])
            self.assertEqual(score["provenance"]["verification"], "unverified")

    def test_skips_active_and_scored_runs_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.make_run(root, "candidate")
            active = self.make_run(root, "active")
            (active / "live.done").unlink()
            scored = self.make_run(root, "scored")
            (scored / "score.json").write_text("{}")

            results = rescore_root(root, dry_run=True)
            by_run = {Path(row["run"]).name: row["status"] for row in results}
            self.assertEqual(by_run, {"candidate": "candidate", "active": "skipped-missing-completion",
                                      "scored": "skipped-scored"})
            self.assertFalse((candidate / "score.json").exists())

    def test_missing_game_state_is_reported_without_writing_scores(self):
        class NoState:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def step(self, buttons, frames):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            directory = self.make_run(Path(tmp))
            with patch("celestebench.rescore.Open8", return_value=NoState()), \
                    patch("celestebench.rescore.Progress", return_value=FakeProgress()):
                result = rescore_root(root=Path(tmp))
            self.assertEqual(result[0]["status"], "error")
            self.assertIn("game_state", result[0]["error"])
            self.assertFalse((directory / "score.json").exists())

    def test_reference_frame_mismatch_does_not_write_score(self):
        class FrameEnv(FakeEnv):
            def step(self, buttons, frames):
                super().step(buttons, frames)
                self.framebuffer = np.zeros((2, 2, 4), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            directory = self.make_run(Path(tmp))
            env = FrameEnv()
            (directory / "live.png").write_bytes(
                _png(np.ones((2, 2, 4), dtype=np.uint8)))
            with patch("celestebench.rescore.Open8", return_value=env), \
                    patch("celestebench.rescore.Progress", return_value=FakeProgress()):
                result = rescore_root(Path(tmp))
            self.assertEqual(result[0]["status"], "skipped-mismatch")
            self.assertFalse((directory / "score.json").exists())
            self.assertFalse((directory / "progress.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
