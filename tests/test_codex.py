import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PATH = Path(__file__).parents[1] / "examples" / "codex.py"
spec = importlib.util.spec_from_file_location("codex", PATH)
codex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codex)


class CodexTest(unittest.TestCase):
    def test_free_port_is_a_loopback_port(self):
        self.assertIn(codex.free_port(), range(1024, 65536))

    def test_profile_locks_the_shell_to_the_workspace(self):
        self.assertIn('default_permissions = "celestebench"', codex.PERMISSIONS)
        self.assertIn('":root" = "deny"', codex.PERMISSIONS)
        self.assertIn('":workspace_roots" = "read"', codex.PERMISSIONS)
        self.assertIn("enabled = false", codex.PERMISSIONS)

    def test_reasoning_effort_maps_tau_thinking_levels(self):
        self.assertIsNone(codex.reasoning_effort(None))
        self.assertIsNone(codex.reasoning_effort(""))
        self.assertEqual(codex.reasoning_effort("off"), "none")
        self.assertEqual(codex.reasoning_effort("high"), "high")
        # Clamp to what the model accepts: gpt-6-astra has no none/minimal.
        self.assertEqual(codex.reasoning_effort("off", ("low", "medium")), "low")
        self.assertEqual(codex.reasoning_effort("minimal", ("low", "max")), "low")
        self.assertEqual(codex.reasoning_effort("max", ("low", "high")), "high")
        self.assertEqual(codex.reasoning_effort("medium", ("low", "high")), "high")

    @patch.object(codex, "_stop_mcp")
    @patch.object(codex, "wait_for_mcp")
    @patch.object(codex.secrets, "token_urlsafe", return_value="generated-token")
    @patch.object(codex, "free_port", return_value=9123)
    @patch.object(codex.subprocess, "Popen")
    @patch.object(codex.subprocess, "run")
    def test_run_jails_codex_and_keeps_secrets_out_of_argv(
        self, run, popen, _port, _token, ready, stop
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        seen = []

        def fake_run(argv, **kwargs):
            home = Path(kwargs["env"]["CODEX_HOME"])
            seen.append((home, (home / "config.toml").is_file(),
                         (home / "auth.json").is_symlink()))
            return MagicMock(stdout="")

        run.side_effect = fake_run
        hostile = "hi ' $(touch /tmp/nope)\nsecond line"
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(codex.os.environ, {"CODEX_API_KEY": "api-secret"}),
        ):
            output = Path(root) / "result"
            codex.run(hostile, None, output, 12, 30, 90, 7, "high")
            argv = run.call_args.args[0]
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["input"], hostile)
            self.assertNotIn("generated-token", " ".join(argv))
            self.assertNotIn("api-secret", " ".join(argv))
            self.assertNotIn(hostile, " ".join(argv))
            self.assertNotIn("--sandbox", argv)
            self.assertIn("model_reasoning_effort=high", argv)
            self.assertIn("--skip-git-repo-check", argv)
            self.assertIn("--ephemeral", argv)
            self.assertIn("shell_environment_policy.inherit='core'", argv)
            self.assertEqual(kwargs["env"]["CODEX_API_KEY"], "api-secret")
            self.assertEqual(kwargs["env"]["CELESTEBENCH_MCP_TOKEN"], "generated-token")
            self.assertEqual(kwargs["cwd"], output / "workspace")
            self.assertIn(str(output / "rollout"), " ".join(popen.call_args_list[0].args[0]))
            self.assertEqual((output / "prompt.txt").read_text(), hostile)
            self.assertNotIn("secret", (output / "config.json").read_text())
        self.assertEqual((seen[0][1], seen[0][2]), (True, False),
                         "key mode carries the profile but no auth.json")
        ready.assert_called_once_with(process, "generated-token", 9123)
        stop.assert_called_once_with(process, output / "rollout", 12)

    @patch.object(codex, "_stop_mcp")
    @patch.object(codex, "wait_for_mcp")
    @patch.object(codex, "oauth_login")
    @patch.object(codex.secrets, "token_urlsafe", return_value="token")
    @patch.object(codex, "free_port", return_value=9125)
    @patch.object(codex.subprocess, "Popen")
    @patch.object(codex.subprocess, "run")
    def test_run_without_a_key_uses_the_chatgpt_login(
        self, run, popen, _port, _token, login, ready, stop
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        seen = []

        def fake_run(argv, **kwargs):
            home = Path(kwargs["env"]["CODEX_HOME"])
            seen.append((kwargs["env"].get("CODEX_API_KEY"),
                         (home / "auth.json").is_symlink()))
            return MagicMock(stdout="")

        run.side_effect = fake_run
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(codex.os.environ, {"CODEX_API_KEY": ""}):
            login.return_value = Path(root) / "auth.json"
            login.return_value.write_text('{"auth_mode": "chatgpt"}')
            codex.run("prompt", None, Path(root) / "out", 2, None)
        self.assertEqual(seen, [(None, True)])

    @patch.object(codex, "oauth_login")
    def test_run_without_a_key_or_login_refuses(self, login):
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(codex.os.environ, {"CODEX_API_KEY": ""}):
            login.return_value = Path(root) / "missing" / "auth.json"
            with self.assertRaisesRegex(SystemExit, "codex login"):
                codex.run("prompt", None, Path(root) / "out", 2, None)
            self.assertFalse((Path(root) / "out").exists())

    @patch.object(codex, "_stop_mcp")
    @patch.object(codex, "wait_for_mcp", side_effect=RuntimeError("collision"))
    @patch.object(codex.secrets, "token_urlsafe", return_value="token")
    @patch.object(codex, "free_port", return_value=9124)
    @patch.object(codex.subprocess, "Popen")
    def test_mcp_start_failure_still_cleans_up(
        self, popen, _port, _token, _ready, stop
    ):
        process = MagicMock()
        popen.return_value = process
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(codex.os.environ, {"CODEX_API_KEY": "api-secret"}),
            self.assertRaisesRegex(RuntimeError, "collision"),
        ):
            codex.run("prompt", None, Path(root) / "out", 2, None)
        stop.assert_called_once_with(process, Path(root) / "out" / "rollout", 2)

    @patch.object(codex, "stop_process")
    def test_stop_mcp_lets_a_running_episode_finalize(self, stop):
        process = MagicMock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as root:
            rollout = Path(root) / "rollout"
            rollout.mkdir(parents=True)
            (rollout / "config.json").write_text("{}")  # the engine started
            # Missing live.done: the episode still runs, so it gets its grace
            # window instead of dying before the mp4 index is written.
            with patch.object(codex.time, "monotonic",
                              side_effect=[0] + [i * 0.25 for i in range(1, 10000)]), \
                    patch.object(codex.time, "sleep") as sleep:
                codex._stop_mcp(process, rollout, timeout=1, grace=0)
            sleep.assert_any_call(0.25)
            stop.assert_called_once_with(process)
            # live.done present: short wait, then stop.
            (rollout / "live.done").write_text("")
            with patch.object(codex.time, "monotonic", return_value=0), \
                    patch.object(codex.time, "sleep") as sleep:
                codex._stop_mcp(process, rollout)
            sleep.assert_any_call(1)
            stop.assert_called_with(process)
        # No engine ever started: skip the whole wait.
        with tempfile.TemporaryDirectory() as root:
            with patch.object(codex.time, "sleep") as sleep:
                codex._stop_mcp(process, Path(root) / "rollout", timeout=300)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
