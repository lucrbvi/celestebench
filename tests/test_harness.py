import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from celestebench import harness


class HarnessTest(unittest.TestCase):
    def test_free_port_is_a_loopback_port(self):
        self.assertIn(harness.free_port(), range(1024, 65536))

    def test_the_shared_task_prompt_is_minimal(self):
        self.assertEqual(harness.PROMPT, "Play Celeste Classic.")

    def test_stop_mcp_lets_a_running_episode_finalize(self):
        process = MagicMock()
        process.poll.return_value = None
        stop = MagicMock()
        with tempfile.TemporaryDirectory() as root:
            rollout = Path(root) / "rollout"
            rollout.mkdir(parents=True)
            (rollout / "config.json").write_text("{}")  # the engine started
            # Missing live.done: the episode still runs, so it gets its grace
            # window instead of dying before the mp4 index is written.
            with patch.object(harness.time, "monotonic",
                              side_effect=[0] + [i * 0.25 for i in range(1, 10000)]), \
                    patch.object(harness.time, "sleep") as sleep:
                harness.stop_mcp(process, rollout, timeout=1, grace=0, stop=stop)
            sleep.assert_any_call(0.25)
            stop.assert_called_once_with(process)
            # live.done present: short wait, then stop.
            (rollout / "live.done").write_text("")
            with patch.object(harness.time, "monotonic", return_value=0), \
                    patch.object(harness.time, "sleep") as sleep:
                harness.stop_mcp(process, rollout, stop=stop)
            sleep.assert_any_call(1)
        # No engine ever started: skip the whole wait.
        with tempfile.TemporaryDirectory() as root:
            with patch.object(harness.time, "sleep") as sleep:
                harness.stop_mcp(process, Path(root) / "rollout", timeout=300, stop=stop)
            sleep.assert_not_called()

    def test_mcp_command_matches_the_server_flags(self):
        live = harness.mcp_command("out", port=8124, timeout=60, frames=90,
                                   max_frames=10, max_images=2, fps=30)
        self.assertIn("--frames", live)
        self.assertIn("--fps", live)
        self.assertIn("--max-images", live)
        paused = harness.mcp_command("out", port=8124, timeout=60)
        self.assertIn("--lite", paused)
        self.assertNotIn("--frames", paused)

    def test_prepare_writes_the_public_config_and_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run"
            rollout = harness.prepare(output, {"model": "m", "timeout": 5}, "play")
            self.assertEqual(rollout, output / "rollout")
            self.assertFalse(rollout.exists())
            self.assertIn('"model": "m"', (output / "config.json").read_text())
            self.assertEqual((output / "prompt.txt").read_text(), "play")
            with self.assertRaises(FileExistsError):
                harness.prepare(output, {}, "again")

    def test_usage_and_assistant_helpers_shape_rows_for_the_viewer(self):
        self.assertEqual(harness.usage(input=1, output=2, cache_read=3, cache_write=4),
                         {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4,
                          "totalTokens": 10})
        row = harness.assistant("think", "say", {"actions": []},
                                harness.usage(input=1, output=1))
        self.assertEqual(row["role"], "assistant")
        self.assertEqual([block["type"] for block in row["content"]],
                         ["thinking", "text", "toolCall"])
        self.assertEqual(row["content"][2]["arguments"], {"actions": []})


if __name__ == "__main__":
    unittest.main()
