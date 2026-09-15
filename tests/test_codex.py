import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PATH = Path(__file__).parents[1] / "examples" / "codex.py"
spec = importlib.util.spec_from_file_location("codex", PATH)
codex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codex)


def test_profile_locks_the_shell_to_the_workspace():
    assert 'default_permissions = "celestebench"' in codex.PERMISSIONS
    assert '":root" = "deny"' in codex.PERMISSIONS
    assert '":workspace_roots" = "read"' in codex.PERMISSIONS
    assert "enabled = false" in codex.PERMISSIONS


def test_profile_shares_one_file_auth_across_parallel_runs():
    assert 'cli_auth_credentials_store = "file"' in codex.PERMISSIONS


def test_reasoning_effort_maps_tau_thinking_levels():
    assert codex.reasoning_effort(None) is None
    assert codex.reasoning_effort("") is None
    assert codex.reasoning_effort("off") == "none"
    assert codex.reasoning_effort("high") == "high"
    # Clamp to what the model accepts: gpt-6-astra has no none/minimal.
    assert codex.reasoning_effort("off", ("low", "medium")) == "low"
    assert codex.reasoning_effort("minimal", ("low", "max")) == "low"
    assert codex.reasoning_effort("max", ("low", "high")) == "high"
    assert codex.reasoning_effort("medium", ("low", "high")) == "high"


def test_run_jails_codex_and_keeps_secrets_out_of_argv():
    with (
        patch.object(codex, "_stop_mcp") as stop,
        patch.object(codex, "wait_for_mcp") as ready,
        patch.object(codex.secrets, "token_urlsafe", return_value="generated-token"),
        patch.object(codex.subprocess, "Popen") as popen,
        patch.object(codex.subprocess, "run") as run,
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        ready.return_value = 9123
        seen = []

        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(codex.os.environ, {"CODEX_API_KEY": "api-secret"}),
        ):
            output = Path(root) / "result"

            def fake_run(argv, **kwargs):
                home = Path(kwargs["env"]["CODEX_HOME"])
                seen.append((home, (home / "config.toml").is_file(),
                             (home / "auth.json").is_symlink()))
                (output / "rollout").mkdir(exist_ok=True)
                (output / "rollout" / "config.json").write_text("{}")
                return MagicMock(stdout="", stderr="", returncode=0)

            run.side_effect = fake_run
            hostile = "hi ' $(touch /tmp/nope)\nsecond line"
            codex.run(hostile, None, output, 12, 30, 90, 7, "high")
            argv = run.call_args.args[0]
            kwargs = run.call_args.kwargs
            assert kwargs["input"] == hostile
            assert "generated-token" not in " ".join(argv)
            assert "api-secret" not in " ".join(argv)
            assert hostile not in " ".join(argv)
            assert "--sandbox" not in argv
            assert "model_reasoning_effort=high" in argv
            assert "--skip-git-repo-check" in argv
            assert "--ephemeral" in argv
            assert "shell_environment_policy.inherit='core'" in argv
            # A workspace inside the repo must not make Codex read the repo AGENTS.md.
            assert "project_doc_max_bytes=0" in argv
            assert kwargs["env"]["CODEX_API_KEY"] == "api-secret"
            assert kwargs["env"]["CELESTEBENCH_MCP_TOKEN"] == "generated-token"
            assert kwargs["cwd"] == output / "workspace"
            assert str(output / "rollout") in " ".join(popen.call_args_list[0].args[0])
            assert (output / "prompt.txt").read_text() == hostile
            assert "secret" not in (output / "config.json").read_text()
        assert (seen[0][1], seen[0][2]) == (True, False), \
            "key mode carries the profile but no auth.json"
        ready.assert_called_once_with(process, "generated-token", output)
        stop.assert_called_once_with(process, output / "rollout", 12)


def test_run_without_a_key_uses_the_chatgpt_login():
    with (
        patch.object(codex, "_stop_mcp"),
        patch.object(codex, "wait_for_mcp") as ready,
        patch.object(codex, "oauth_login") as login,
        patch.object(codex.secrets, "token_urlsafe", return_value="token"),
        patch.object(codex.subprocess, "Popen") as popen,
        patch.object(codex.subprocess, "run") as run,
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        ready.return_value = 9125
        seen = []

        with tempfile.TemporaryDirectory() as root, \
                patch.dict(codex.os.environ, {"CODEX_API_KEY": ""}):
            out = Path(root) / "out"

            def fake_run(argv, **kwargs):
                home = Path(kwargs["env"]["CODEX_HOME"])
                seen.append((kwargs["env"].get("CODEX_API_KEY"),
                             (home / "auth.json").is_symlink()))
                (out / "rollout").mkdir(exist_ok=True)
                (out / "rollout" / "config.json").write_text("{}")
                return MagicMock(stdout="", stderr="", returncode=0)

            run.side_effect = fake_run
            login.return_value = Path(root) / "auth.json"
            login.return_value.write_text('{"auth_mode": "chatgpt"}')
            codex.run("prompt", None, out, 2, None)
        assert seen == [(None, True)]


def test_codex_failure_reports_its_own_stderr():
    with (
        patch.object(codex, "_stop_mcp"),
        patch.object(codex, "wait_for_mcp") as ready,
        patch.object(codex, "oauth_login") as login,
        patch.object(codex.secrets, "token_urlsafe", return_value="token"),
        patch.object(codex.subprocess, "Popen") as popen,
        patch.object(codex.subprocess, "run") as run,
    ):
        popen.return_value = MagicMock()
        ready.return_value = 9126
        run.return_value = MagicMock(
            returncode=1, stdout="", stderr="boot\nstream error: connection reset\n")
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(codex.os.environ, {"CODEX_API_KEY": ""}):
            login.return_value = Path(root) / "auth.json"
            login.return_value.write_text("{}")
            with pytest.raises(SystemExit, match="connection reset"):
                codex.run("prompt", "gpt-6-astra", Path(root) / "out", 5, 30)
            trace = (Path(root) / "out" / "codex.jsonl").read_text()
        assert "connection reset" in trace
        assert "developer_instructions" not in trace


def test_run_without_a_key_or_login_refuses():
    with (
        patch.object(codex, "oauth_login") as login,
        tempfile.TemporaryDirectory() as root,
        patch.dict(codex.os.environ, {"CODEX_API_KEY": ""}),
    ):
        login.return_value = Path(root) / "missing" / "auth.json"
        with pytest.raises(SystemExit, match="codex login"):
            codex.run("prompt", None, Path(root) / "out", 2, None)
        assert not (Path(root) / "out").exists()


def test_mcp_start_failure_still_cleans_up():
    with (
        patch.object(codex, "_stop_mcp") as stop,
        patch.object(codex, "wait_for_mcp", side_effect=RuntimeError("collision")),
        patch.object(codex.secrets, "token_urlsafe", return_value="token"),
        patch.object(codex.subprocess, "Popen") as popen,
    ):
        process = MagicMock()
        popen.return_value = process
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(codex.os.environ, {"CODEX_API_KEY": "api-secret"}),
            pytest.raises(RuntimeError, match="collision"),
        ):
            codex.run("prompt", None, Path(root) / "out", 2, None)
        stop.assert_called_once_with(process, Path(root) / "out" / "rollout", 2)


def test_stop_mcp_delegates_and_lets_the_episode_finalize():
    with patch.object(codex, "stop_process") as stop:
        process = MagicMock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as root:
            rollout = Path(root) / "rollout"  # no config.json: skip the wait
            codex._stop_mcp(process, rollout, timeout=1)
        stop.assert_called_once_with(process)
