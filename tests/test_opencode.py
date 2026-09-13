import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PATH = Path(__file__).parents[1] / "examples" / "opencode.py"
spec = importlib.util.spec_from_file_location("opencode", PATH)
opencode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opencode)


class OpenCodeTest(unittest.TestCase):
    def test_config_points_at_mcp_and_denies_every_builtin_tool(self):
        config = opencode.config(9123, "secret-token")
        server = config["mcp"]["celeste"]
        self.assertEqual(server["type"], "remote")
        self.assertEqual(server["url"], "http://127.0.0.1:9123/mcp")
        self.assertEqual(server["headers"]["Authorization"], "Bearer secret-token")
        agent = config["agent"][opencode.AGENT]
        # A file reference: OpenCode expands braces in inline strings.
        self.assertEqual(agent["prompt"], f"{{file:./{opencode.INSTRUCTIONS}}}")
        self.assertEqual(agent["tools"], {"*": False, "celeste*": True})

    def test_variant_maps_tau_thinking_levels(self):
        self.assertIsNone(opencode.variant(None))
        self.assertIsNone(opencode.variant("off"))
        self.assertEqual(opencode.variant("high"), "high")
        self.assertEqual(opencode.variant("minimal"), "minimal")

    def test_normalize_emits_one_assistant_row_per_play(self):
        events = [
            {"type": "step_start"},
            {"type": "reasoning", "part": {"type": "reasoning", "text": "think"}},
            {"type": "text", "part": {"type": "text", "text": "hello"}},
            {"type": "tool_use", "part": {"type": "tool", "tool": "celeste_play",
                                          "state": {"input": {"actions": [{"buttons": 2, "frames": 4}]}}}},
            {"type": "step_finish", "part": {"type": "step-finish",
                                             "tokens": {"total": 10, "input": 8, "output": 2,
                                                        "reasoning": 1, "cache": {"write": 0, "read": 3}}}},
            # An observe call must not add a decision.
            {"type": "tool_use", "part": {"type": "tool", "tool": "celeste_observe",
                                          "state": {"input": {}}}},
            {"type": "step_finish", "part": {"type": "step-finish",
                                             "tokens": {"total": 5, "input": 5, "output": 0}}},
        ]
        with tempfile.TemporaryDirectory() as root:
            trace, messages = Path(root) / "opencode.jsonl", Path(root) / "messages.jsonl"
            trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            opencode.normalize(trace, messages)
            rows = [json.loads(line) for line in messages.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "assistant")
        self.assertEqual([block["type"] for block in rows[0]["content"]],
                         ["thinking", "text", "toolCall"])
        self.assertEqual(rows[0]["content"][0]["thinking"], "think")
        self.assertEqual(rows[0]["content"][2]["arguments"],
                         {"actions": [{"buttons": 2, "frames": 4}]})
        self.assertEqual(rows[0]["usage"]["cacheRead"], 3)
        self.assertEqual(rows[0]["usage"]["totalTokens"], 10)

    @patch.object(opencode, "models", return_value=["opencode-go/deepseek-v4.1-flash"])
    @patch.object(opencode, "_stop_mcp")
    @patch.object(opencode, "wait_for_mcp")
    @patch.object(opencode.secrets, "token_urlsafe", return_value="generated-token")
    @patch.object(opencode, "free_port", return_value=9123)
    @patch.object(opencode.subprocess, "Popen")
    @patch.object(opencode.subprocess, "run")
    def test_run_jails_opencode_and_keeps_the_token_out_of_argv(
        self, run, popen, _port, _token, ready, stop, _models
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result"
            rollout = output / "rollout"

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write((json.dumps({
                    "type": "tool_use",
                    "part": {"type": "tool", "tool": "celeste_play",
                             "state": {"input": {"actions": [{"buttons": 2, "frames": 3}]}}}}) + "\n").encode())
                kwargs["stdout"].write((json.dumps({
                    "type": "step_finish", "part": {"tokens": {"total": 1, "input": 1}}}) + "\n").encode())
                rollout.mkdir(parents=True, exist_ok=True)
                (rollout / "config.json").write_text("{}")
                return MagicMock(returncode=0)

            run.side_effect = fake_run
            opencode.run("Begin.", "opencode-go/deepseek-v4.1-flash", output, 30, 30,
                         thinking_level="high")
            argv = run.call_args.args[0]
            kwargs = run.call_args.kwargs
            self.assertEqual(argv[:5], ["opencode", "run", "--pure", "--format", "json"])
            self.assertIn("--agent", argv)
            self.assertIn("--model", argv)
            self.assertIn("opencode-go/deepseek-v4.1-flash", argv)
            self.assertIn("--variant", argv)
            self.assertIn("high", argv)
            self.assertNotIn("generated-token", " ".join(argv))
            self.assertEqual(kwargs["env"]["CELESTEBENCH_MCP_TOKEN"], "generated-token")
            self.assertEqual(kwargs["cwd"], output / "workspace")
            self.assertIn(str(rollout), " ".join(popen.call_args_list[0].args[0]))
            config = json.loads((output / "workspace" / "opencode.json").read_text())
            self.assertIn("generated-token", config["mcp"]["celeste"]["headers"]["Authorization"])
            self.assertEqual((rollout / "messages.jsonl").read_text().count("\n"), 1)
        ready.assert_called_once_with(process, "generated-token", 9123)
        stop.assert_called_once_with(process, rollout, 30)

    def test_resolve_model_blocks_bare_and_near_miss_ids(self):
        listing = ["opencode-go/deepseek-v4-flash", "opencode-go/deepseek-v4-pro"]
        with patch.object(opencode, "models", return_value=listing):
            with self.assertRaisesRegex(SystemExit, "provider/model"):
                opencode.resolve_model("deepseek-flash")
            with self.assertRaisesRegex(SystemExit, "deepseek-v4-flash"):
                opencode.resolve_model("opencode-go/deepseek-flash")
            self.assertEqual(opencode.resolve_model("opencode-go/deepseek-v4-flash"),
                             "opencode-go/deepseek-v4-flash")
            # A namespaced id with no near match may be a dynamic provider.
            self.assertEqual(opencode.resolve_model("custom/whatever"), "custom/whatever")

    @patch.object(opencode, "models", return_value=None)
    def test_resolve_model_skips_validation_without_a_catalog(self, _models):
        self.assertEqual(opencode.resolve_model("deepseek-flash"), "deepseek-flash")

    def test_cli_error_reads_the_last_error_event(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "opencode.jsonl"
            trace.write_text(
                '{"type":"error","message":"boom"}\n'
                '{"type":"error","message":"final failure"}\n')
            self.assertEqual(opencode.cli_error(trace), "final failure")


if __name__ == "__main__":
    unittest.main()
