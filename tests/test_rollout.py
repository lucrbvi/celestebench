import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from celestebench import Button
from celestebench.rollout import rollout


class RolloutTest(unittest.IsolatedAsyncioTestCase):
    async def test_multi_action_policy_writes_video_and_history(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            observations = []
            def policy(frame):
                observations.append(frame.copy())
                return [(2, 2), (0, 1)]
            result = await rollout(policy, output, decisions=3, max_actions=2)
            self.assertEqual(result["frames"], 10)  # one initial screenshot
            self.assertEqual(result["decisions"], 3)
            self.assertEqual(result["actions"], 6)
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
            for row, observation in zip(outcomes, observations, strict=True):
                with av.open(str(output / row["screenshot"])) as png:
                    decoded = next(png.decode(video=0)).to_ndarray(format="rgba")
                np.testing.assert_array_equal(decoded, observation)
            with av.open(str(output / "rollout.mp4")) as video:
                self.assertEqual(len(list(video.decode(video=0))), 10)
            self.assertEqual((output / "checkpoint.state").read_bytes(), b"\x00" + b"\x02\x02\x00" * 3)

    async def test_single_action_policy_accepts_bare_pair(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(lambda frame: (Button.RIGHT, np.int64(2)), output, decisions=2)
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
                await rollout(lambda frame: (1, 0), output, decisions=2)
            self.assertTrue((output / "rollout.mp4").stat().st_size)
            self.assertEqual(len((output / "checkpoint.state").read_bytes()), 1)

            result = await rollout(policy, Path(root) / "async", decisions=1)
            self.assertEqual(result["frames"], 2)

    async def test_output_must_be_fresh(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                await rollout(lambda frame: (0, 1), output, decisions=0)

    async def test_invalid_batch_never_partially_executes(self):
        for batch, limit in [([(2, 1), (0, 0)], 2), ([(2, 1), (0, 1)], 1),
                             ([(2, 1), (True, 1)], 2), ([], 2)]:
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "run"
                with self.assertRaises((ValueError, TypeError)):
                    await rollout(lambda frame: batch, output, decisions=1, max_actions=limit)
                self.assertEqual((output / "checkpoint.state").read_bytes(), b"\x00")
                self.assertEqual((output / "actions.jsonl").read_text(), "")

    async def test_realtime_runs_environment_while_policy_thinks(self):
        async def policy(frame):
            await asyncio.sleep(0.05)
            return (2, 10)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output, decisions=1, frames=20, fps=200)
            self.assertEqual(result["frames"], 20)  # ~10 idle frames during thinking
            self.assertEqual(result["actions"], 1)
            self.assertLess(result["elapsed"], 5)
            with av.open(str(output / "rollout.mp4")) as video:
                self.assertEqual(len(list(video.decode(video=0))), 20)

    async def test_timeout_is_logged_without_executed_actions(self):
        async def policy(frame):
            raise TimeoutError("deadline")

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            with self.assertRaises(TimeoutError):
                await rollout(policy, output, decisions=1)
            result = json.loads((output / "decisions.jsonl").read_text())
            self.assertEqual(result["status"], "timeout")
            self.assertEqual((result["frame_start"], result["frame_end"]), (1, 1))
            self.assertEqual(result["screenshot"], "screenshots/000000.png")
            self.assertTrue((output / result["screenshot"]).is_file())
            self.assertEqual((output / "actions.jsonl").read_text(), "")

    async def test_frame_budget_cancels_and_logs_pending_policy(self):
        cancelled = asyncio.Event()

        async def policy(frame):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            result = await rollout(policy, output, decisions=1, frames=3, fps=1000)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(result["frames"], 3)
            outcome = json.loads((output / "decisions.jsonl").read_text())
            self.assertEqual(outcome["status"], "frame_limit")
            self.assertEqual((outcome["frame_start"], outcome["frame_end"]), (1, 3))
            self.assertEqual((output / "actions.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
