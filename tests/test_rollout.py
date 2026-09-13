import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import av
import numpy as np

from celestebench import Button
from celestebench.rollout import rollout


class RolloutTest(unittest.IsolatedAsyncioTestCase):
    async def test_progress_records_idle_frames_and_uses_wall_clock(self):
        async def policy(frames):
            await asyncio.Event().wait()

        state = dict(room=0, alive=True, grounded=True, feet_y=58,
                     spawn_feet_y=112, exit_feet_y=4, deaths=0)
        with tempfile.TemporaryDirectory() as root, patch(
                "celestebench.open8.Open8.game_state", new_callable=PropertyMock,
                return_value=state):
            output = Path(root) / "run"
            await rollout(policy, output, frames=5, fps=100)
            events = [json.loads(line) for line in
                      (output / "progress.jsonl").read_text().splitlines()]
            score = json.loads((output / "score.json").read_text())
            self.assertEqual([event["frame"] for event in events], [0, 3])
            self.assertGreater(events[1]["elapsed"], 0)
            self.assertLessEqual(events[1]["elapsed"], score["elapsed"])
            self.assertEqual(score["timing"], "wall_clock")
            self.assertEqual(score["status"], "completed")
            self.assertEqual(score["room_progress"], .5)
            self.assertEqual(score["frame"], 5)

    async def test_wait_is_distinct_from_releasing_buttons(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(lambda frame: [(2, 1), ("wait", 2), (0, 1)], output,
                                   frames=5, max_actions=3)
            rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]

            self.assertEqual((result["frames"], result["actions"]), (5, 3))
            self.assertNotIn("action", rows[0])
            self.assertEqual(rows[1]["action"], "wait")
            self.assertNotIn("action", rows[2])
            self.assertEqual([row["buttons"] for row in rows], [2, 0, 0])
            self.assertEqual((output / "checkpoint.state").read_bytes(), b"\0\2\0\0\0")
            self.assertTrue((output / "live.png").is_file())
            self.assertTrue((output / "live.done").is_file())

    async def test_multi_action_policy_writes_video_and_history(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            observations = []
            def policy(frames):
                observations.append([frame.copy() for frame in frames])
                return [(2, 2), (0, 1)]
            result = await rollout(policy, output, frames=10, max_actions=2)
            self.assertEqual(result["frames"], 10)  # one initial screenshot
            self.assertEqual(result["decisions"], 3)
            self.assertEqual(result["actions"], 6)
            # Decision 0 sees the initial frame; each later decision sees every
            # frame played since the previous one, oldest to newest.
            self.assertEqual([len(frames) for frames in observations], [1, 3, 3])
            rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
            self.assertEqual([row["buttons"] for row in rows], [2, 0] * 3)
            self.assertEqual([row["decision"] for row in rows], [0, 0, 1, 1, 2, 2])
            self.assertTrue(all(row["latency"] == 0 for row in rows[1::2]))
            self.assertTrue(all(row["latency"] > 0 for row in rows[::2]))
            self.assertEqual([(r["frame_start"], r["frame_end"]) for r in rows],
                             [(1, 3), (3, 4), (4, 6), (6, 7), (7, 9), (9, 10)])
            outcomes = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in outcomes], ["played"] * 3)
            self.assertEqual([r["frame_start"] for r in outcomes], [1, 4, 7])
            self.assertEqual([r["screenshot"] for r in outcomes],
                             [f"screenshots/{n:06d}.png" for n in range(3)])
            for row, frames in zip(outcomes, observations, strict=True):
                with av.open(str(output / row["screenshot"])) as png:
                    decoded = next(png.decode(video=0)).to_ndarray(format="rgba")
                np.testing.assert_array_equal(decoded, frames[-1])
            with av.open(str(output / "rollout.mp4")) as video:
                self.assertEqual(len(list(video.decode(video=0))), 10)
            self.assertEqual((output / "checkpoint.state").read_bytes(), b"\x00" + b"\x02\x02\x00" * 3)

    async def test_single_action_policy_accepts_bare_pair(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(lambda frame: (Button.RIGHT, np.int64(2)), output, frames=5)
            self.assertEqual(result["frames"], 5)
            self.assertEqual(result["decisions"], 2)
            self.assertEqual(result["actions"], 2)
            self.assertEqual((output / "checkpoint.state").read_bytes(), b"\x00\x02\x02\x02\x02")

    async def test_async_policy_and_invalid_action_keep_partial_checkpoint(self):
        async def policy(frame):
            await asyncio.sleep(0)
            return (1, 1)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            with self.assertRaises(ValueError):
                await rollout(lambda frame: (1, 0), output, frames=3)
            self.assertTrue((output / "rollout.mp4").stat().st_size)
            self.assertEqual(len((output / "checkpoint.state").read_bytes()), 1)
            self.assertEqual(json.loads((output / "score.json").read_text())["status"], "error")

            result = await rollout(policy, Path(root) / "async", frames=2)
            self.assertEqual(result["frames"], 2)

    async def test_output_must_be_fresh(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                await rollout(lambda frame: (0, 1), output, frames=1)

    async def test_invalid_batch_never_partially_executes(self):
        for batch, limit in [([(2, 1), (0, 0)], 2), ([(2, 1), (0, 1)], 1),
                             ([(2, 1), (True, 1)], 2), ([], 2)]:
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "run"
                with self.assertRaises((ValueError, TypeError)):
                    await rollout(lambda frame: batch, output, frames=2, max_actions=limit)
                self.assertEqual((output / "checkpoint.state").read_bytes(), b"\x00")
                self.assertEqual((output / "actions.jsonl").read_text(), "")

    async def test_realtime_runs_environment_while_policy_thinks(self):
        observations = []
        calls = 0

        async def policy(frames):
            nonlocal calls
            observations.append(len(frames))
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.05)  # ~10 idle frames tick meanwhile
            return (2, 10)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output, frames=60, fps=200)
            self.assertEqual(result["frames"], 60)
            self.assertEqual(result["actions"], 5)
            # Decision 0 sees the initial frame; decision 1 sees the idle frames
            # ticked while thinking plus the frames of the played action.
            self.assertEqual(observations[0], 1)
            self.assertGreaterEqual(observations[1], 11)
            self.assertLess(result["elapsed"], 5)
            with av.open(str(output / "rollout.mp4")) as video:
                self.assertEqual(len(list(video.decode(video=0))), 60)

    async def test_wall_clock_timeout_cancels_pending_policy_and_ends_gracefully(self):
        async def policy(frame):
            await asyncio.sleep(5)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output, timeout=0.1)
            self.assertLess(result["elapsed"], 5)
            self.assertEqual((result["frames"], result["decisions"], result["actions"]), (1, 0, 0))
            outcome = json.loads((output / "decisions.jsonl").read_text())
            self.assertEqual(outcome["status"], "timeout")
            self.assertEqual((outcome["frame_start"], outcome["frame_end"]), (1, 1))
            self.assertEqual(outcome["screenshot"], "screenshots/000000.png")
            self.assertTrue((output / outcome["screenshot"]).is_file())
            self.assertEqual((output / "actions.jsonl").read_text(), "")
            with av.open(str(output / "rollout.mp4")) as video:
                self.assertEqual(len(list(video.decode(video=0))), 1)

    async def test_strict_timeout_truncates_synchronous_action_batch(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(lambda frame: (2, 100000), output,
                                   timeout=0.1, strict_timeout=True,
                                   max_frames=100000)
            outcome = json.loads((output / "decisions.jsonl").read_text())
            action = json.loads((output / "actions.jsonl").read_text())
            self.assertEqual(outcome["status"], "timeout")
            self.assertEqual(result["frames"], outcome["frame_end"])
            self.assertEqual(action["frames"], result["frames"] - 1)
            self.assertLess(action["frames"], 100000)

    async def test_realtime_strict_deadline_cuts_batch_but_default_finishes_it(self):
        for strict in (True, False):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "run"
                await rollout(lambda frames: [(2, 10), (1, 2)], output,
                              fps=20, timeout=0.25, max_actions=2,
                              strict_timeout=strict)
                actions = [json.loads(line) for line in
                           (output / "actions.jsonl").read_text().splitlines()]
                outcome = json.loads((output / "decisions.jsonl").read_text())
                if strict:
                    self.assertEqual(outcome["status"], "timeout")
                    self.assertEqual(len(actions), 1)
                    self.assertLess(actions[0]["frames"], 10)
                else:
                    self.assertEqual(outcome["status"], "played")
                    self.assertEqual([a["frames"] for a in actions], [10, 2])

    async def test_policy_timeout_is_logged_and_ends_gracefully(self):
        async def policy(frame):
            raise TimeoutError("deadline")

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output)
            self.assertEqual(result["decisions"], 0)
            outcome = json.loads((output / "decisions.jsonl").read_text())
            self.assertEqual((outcome["status"], outcome["error"]), ("timeout", "deadline"))
            self.assertEqual((output / "actions.jsonl").read_text(), "")

    async def test_expired_budget_makes_no_decision(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(lambda frame: (0, 1), output, timeout=1e-6)
            self.assertEqual((result["frames"], result["decisions"]), (1, 0))
            self.assertEqual((output / "decisions.jsonl").read_text(), "")

    async def test_frame_budget_cancels_and_logs_pending_policy(self):
        cancelled = asyncio.Event()

        async def policy(frame):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output, frames=3, fps=1000)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(result["frames"], 3)
            outcome = json.loads((output / "decisions.jsonl").read_text())
            self.assertEqual(outcome["status"], "frame_limit")
            self.assertEqual((outcome["frame_start"], outcome["frame_end"]), (1, 3))
            self.assertEqual((output / "actions.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
