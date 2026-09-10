import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PATH = Path(__file__).parents[1] / "examples" / "codex_vm.py"
spec = importlib.util.spec_from_file_location("codex_vm", PATH)
codex_vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codex_vm)


class CodexVMTest(unittest.TestCase):
    def test_vm_config_matches_the_host(self):
        if codex_vm.sys.platform == "darwin":
            self.assertEqual(codex_vm.host_config(Path("/tmp")), codex_vm.CONFIG)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                body = codex_vm.host_config(Path(tmp)).read_text()
                self.assertIn("vmType: qemu", body)
                self.assertNotIn("vmType: vz", body)
                self.assertIn(f"arch: {codex_vm.platform.machine()}", body)

    def test_limactl_resolution(self):
        self.assertIn("limactl", str(codex_vm.LIMACTL))

    @patch.object(codex_vm, "command")
    def test_setup_refuses_an_instance_with_host_mounts(self, command):
        command.return_value.stdout = '[{"location":"/Users/luc"}]\n'
        with self.assertRaisesRegex(RuntimeError, "host mounts"):
            codex_vm.setup("vm")

    @patch.object(codex_vm, "ssh_config", return_value="/tmp/ssh config")
    def test_ssh_quotes_remote_arguments_and_tunnel_options_precede_destination(
        self, _
    ):
        command = codex_vm.ssh("vm", "printf", "%s", "$(touch /tmp/nope); ' quoted")
        self.assertEqual(command[-2], "lima-vm")
        self.assertEqual(command[-1], "printf %s '$(touch /tmp/nope); '\"'\"' quoted'")
        tunnel = codex_vm.tunnel_command("vm")
        self.assertLess(tunnel.index("-N"), tunnel.index("lima-vm"))
        self.assertLess(tunnel.index("-R"), tunnel.index("lima-vm"))
        # Lima's shared ControlMaster strands the forward and exits the command.
        self.assertIn("ControlMaster=no", tunnel)
        self.assertIn("ControlPath=none", tunnel)

    @patch.object(codex_vm, "stop_process")
    @patch.object(codex_vm, "wait_for_mcp")
    @patch.object(codex_vm, "setup")
    @patch.object(codex_vm, "start")
    @patch.object(codex_vm, "ssh_config", return_value="config")
    @patch.object(codex_vm.secrets, "token_urlsafe", return_value="generated-token")
    @patch.object(codex_vm.socket, "socket")
    @patch.object(codex_vm.time, "sleep")
    @patch.object(codex_vm.subprocess, "Popen")
    @patch.object(codex_vm.subprocess, "run")
    def test_run_sends_prompt_and_secrets_only_in_stdin_and_cleans_up(
        self, run, popen, _sleep, _socket, _token, _config, start, setup, ready, stop
    ):
        processes = [MagicMock(), MagicMock()]
        processes[1].poll.return_value = None
        popen.side_effect = processes
        hostile = "hi ' $(touch /tmp/nope)\nsecond line"
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(codex_vm.os.environ, {"CODEX_API_KEY": "api-secret"}),
        ):
            output = Path(root) / "result"
            codex_vm.run("vm", hostile, None, output, 12, True, 90, 7)
            payload = json.loads(run.call_args.kwargs["input"])
            argv = run.call_args.args[0]
            self.assertEqual(payload["prompt"], hostile)
            self.assertEqual(payload["token"], "generated-token")
            self.assertEqual(payload["api_key"], "api-secret")
            self.assertNotIn("generated-token", " ".join(argv))
            self.assertNotIn("api-secret", " ".join(argv))
            self.assertNotIn(hostile, " ".join(argv))
            self.assertNotIn("--model", payload["args"])
            self.assertIn("--frames", popen.call_args_list[0].args[0])
            self.assertIn("--max-frames", popen.call_args_list[0].args[0])
            self.assertEqual((output / "prompt.txt").read_text(), hostile)
            self.assertNotIn("secret", (output / "config.json").read_text())
        start.assert_called_once_with("vm")
        setup.assert_called_once_with("vm")
        ready.assert_called_once_with(processes[0], "generated-token")
        self.assertEqual(
            [call.args[0] for call in stop.call_args_list], processes[::-1]
        )

    @patch.object(codex_vm, "stop_process")
    @patch.object(codex_vm, "wait_for_mcp", side_effect=RuntimeError("collision"))
    @patch.object(codex_vm, "setup")
    @patch.object(codex_vm, "start")
    @patch.object(codex_vm.secrets, "token_urlsafe", return_value="token")
    @patch.object(codex_vm.socket, "socket")
    @patch.object(codex_vm.subprocess, "Popen")
    def test_mcp_start_failure_still_cleans_up(
        self, popen, _socket, _token, _start, _setup, _ready, stop
    ):
        process = MagicMock()
        popen.return_value = process
        with (
            tempfile.TemporaryDirectory() as root,
            self.assertRaisesRegex(RuntimeError, "collision"),
        ):
            codex_vm.run("vm", "prompt", None, Path(root) / "out", 2, False)
        stop.assert_any_call(None)
        stop.assert_any_call(process)

    @patch.object(codex_vm, "stop_process")
    def test_stop_mcp_lets_a_running_episode_finalize(self, stop):
        process = MagicMock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as root:
            rollout = Path(root) / "rollout"
            # Missing live.done: the episode still runs, so it gets its grace
            # window instead of dying before the mp4 index is written.
            with patch.object(codex_vm.time, "monotonic",
                              side_effect=[0] + list(i * 0.25 for i in range(1, 10000))), \
                    patch.object(codex_vm.time, "sleep"):
                codex_vm._stop_mcp(process, rollout, grace=1)
        stop.assert_called_once_with(process)
        # live.done present: short wait, then stop.
        rollout.mkdir(parents=True)
        (rollout / "live.done").write_text("")
        with patch.object(codex_vm.time, "monotonic", return_value=0), \
                patch.object(codex_vm.time, "sleep") as sleep:
            codex_vm._stop_mcp(process, rollout)
        sleep.assert_any_call(1)
        stop.assert_called_with(process)

    def test_guest_runner_drops_privileges_times_out_and_limits_output(self):
        self.assertIn("os.setuid", codex_vm.GUEST_RUNNER)
        self.assertIn("RLIMIT_FSIZE", codex_vm.GUEST_RUNNER)
        self.assertIn("killpg", codex_vm.GUEST_RUNNER)
        self.assertNotIn("CODEX_HOME", codex_vm.GUEST_RUNNER)


if __name__ == "__main__":
    unittest.main()
