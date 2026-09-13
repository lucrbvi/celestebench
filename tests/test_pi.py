import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PATH = Path(__file__).parents[1] / "examples" / "pi.py"
spec = importlib.util.spec_from_file_location("pi", PATH)
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)


class PiTest(unittest.TestCase):
    def test_normalize_emits_one_assistant_row_per_completed_play(self):
        play = {"actions": [{"buttons": 2, "frames": 4}]}
        events = [
            {"type": "message_end", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "hello"},
                {"type": "toolCall", "name": "play", "arguments": play},
            ], "usage": {"input": 8, "output": 2, "cacheRead": 3, "cacheWrite": 0,
                         "totalTokens": 13, "reasoning": 1}}},
            {"type": "tool_execution_end", "toolName": "play", "isError": False, "result": {}},
            # An observe-only turn must not add a decision.
            {"type": "message_end", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "observe", "arguments": {}}],
                "usage": {"input": 5, "output": 1, "totalTokens": 6}}},
            {"type": "tool_execution_end", "toolName": "observe", "isError": False, "result": {}},
        ]
        with tempfile.TemporaryDirectory() as root:
            trace, messages = Path(root) / "pi.jsonl", Path(root) / "messages.jsonl"
            trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            pi.normalize(trace, messages)
            rows = [json.loads(line) for line in messages.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual([block["type"] for block in rows[0]["content"]],
                         ["thinking", "text", "toolCall"])
        self.assertEqual(rows[0]["content"][2]["arguments"], play)
        self.assertEqual(rows[0]["usage"]["cacheRead"], 3)
        self.assertEqual(rows[0]["usage"]["totalTokens"], 13)

    def test_normalize_skips_failed_play_calls(self):
        events = [
            {"type": "message_end", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": "play", "arguments": {"actions": []}}]}},
            {"type": "tool_execution_end", "toolName": "play", "isError": True, "result": {}},
        ]
        with tempfile.TemporaryDirectory() as root:
            trace, messages = Path(root) / "pi.jsonl", Path(root) / "messages.jsonl"
            trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            pi.normalize(trace, messages)
            self.assertEqual(messages.read_text(), "")

    def test_cli_error_reads_the_provider_error(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "pi.jsonl"
            trace.write_text(
                '{"type":"message_end","message":{"role":"assistant","errorMessage":"first"}}\n'
                '{"type":"message_end","message":{"role":"assistant","errorMessage":"Codex error"}}\n')
            self.assertEqual(pi.cli_error(trace), "Codex error")

    @patch.object(pi, "_stop_mcp")
    @patch.object(pi, "wait_for_mcp")
    @patch.object(pi.secrets, "token_urlsafe", return_value="generated-token")
    @patch.object(pi, "free_port", return_value=9124)
    @patch.object(pi.subprocess, "Popen")
    @patch.object(pi.subprocess, "run")
    def test_run_loads_the_bundled_extension_without_builtin_tools(
        self, run, popen, _port, _token, ready, stop
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result"
            rollout = output / "rollout"

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write((json.dumps({
                    "type": "message_end",
                    "message": {"role": "assistant", "content": [
                        {"type": "toolCall", "name": "play",
                         "arguments": {"actions": [{"buttons": 2, "frames": 3}]}}],
                        "usage": {"input": 4, "output": 1, "totalTokens": 5}},
                }) + "\n").encode())
                kwargs["stdout"].write((json.dumps({
                    "type": "tool_execution_end", "toolName": "play", "isError": False,
                    "result": {}}) + "\n").encode())
                rollout.mkdir(parents=True, exist_ok=True)
                (rollout / "config.json").write_text("{}")
                return MagicMock(returncode=0)

            run.side_effect = fake_run
            pi.run("Begin.", "openai-codex/gpt-5.6-sol", output, 45, 30, thinking_level="low")
            argv = run.call_args.args[0]
            kwargs = run.call_args.kwargs
            self.assertEqual(argv[0], "pi")
            self.assertIn("--no-builtin-tools", argv)
            self.assertIn("--system-prompt", argv)
            self.assertIn("--extension", argv)
            self.assertIn(str(pi.EXTENSION), argv)
            self.assertEqual(argv[-1], "Begin.")
            self.assertIn("--thinking", argv)
            self.assertIn("low", argv)
            self.assertNotIn("generated-token", " ".join(argv))
            self.assertEqual(kwargs["env"]["CELESTEBENCH_MCP_TOKEN"], "generated-token")
            self.assertEqual(kwargs["env"]["CELESTEBENCH_MCP_URL"], "http://127.0.0.1:9124/mcp")
            self.assertEqual(kwargs["env"]["CELESTEBENCH_MAX_FRAMES"], "30")
            self.assertEqual(kwargs["cwd"], output / "workspace")
            self.assertIn(str(rollout), " ".join(popen.call_args_list[0].args[0]))
            self.assertEqual((rollout / "messages.jsonl").read_text().count("\n"), 1)
        ready.assert_called_once_with(process, "generated-token", 9124)
        stop.assert_called_once_with(process, rollout, 45)


if __name__ == "__main__":
    unittest.main()
