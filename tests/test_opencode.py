import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import load_example

opencode = load_example("opencode")


def test_config_points_at_mcp_and_denies_every_builtin_tool():
    config = opencode.config(9123, "secret-token")
    server = config["mcp"]["celeste"]
    assert server["type"] == "remote"
    assert server["url"] == "http://127.0.0.1:9123/mcp"
    assert server["headers"]["Authorization"] == "Bearer secret-token"
    agent = config["agent"][opencode.AGENT]
    # A file reference: OpenCode expands braces in inline strings.
    assert agent["prompt"] == f"{{file:./{opencode.INSTRUCTIONS}}}"
    assert agent["tools"] == {"*": False, "celeste*": True}
    # No sharing or self-updating while a run owns the host.
    assert config["share"] == "disabled"
    assert config["snapshot"] is False
    assert config["autoupdate"] is False


def test_variant_maps_tau_thinking_levels():
    assert opencode.variant(None) is None
    assert opencode.variant("off") is None
    assert opencode.variant("high") == "high"
    assert opencode.variant("minimal") == "minimal"


def test_normalize_emits_one_assistant_row_per_play():
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
    assert len(rows) == 1
    assert rows[0]["role"] == "assistant"
    assert [block["type"] for block in rows[0]["content"]] == ["thinking", "text", "toolCall"]
    assert rows[0]["content"][0]["thinking"] == "think"
    assert rows[0]["content"][2]["arguments"] == {"actions": [{"buttons": 2, "frames": 4}]}
    assert rows[0]["usage"]["cacheRead"] == 3
    assert rows[0]["usage"]["totalTokens"] == 10


@patch.object(opencode, "models", return_value=["opencode-go/deepseek-v4.1-flash"])
@patch.object(opencode.harness, "stop_mcp")
@patch.object(opencode.harness, "wait_for_mcp", return_value=9123)
@patch.object(opencode.secrets, "token_urlsafe", return_value="generated-token")
@patch.object(opencode.subprocess, "Popen")
@patch.object(opencode.subprocess, "run")
def test_run_jails_opencode_and_keeps_the_token_out_of_argv(
    run, popen, _token, ready, stop, _models
):
    process = MagicMock()
    process.poll.return_value = None
    popen.return_value = process

    with tempfile.TemporaryDirectory() as root:
        output = Path(root) / "result"
        rollout = output / "rollout"
        seen = {}

        def fake_run(argv, **kwargs):
            kwargs["stdout"].write((json.dumps({
                "type": "tool_use",
                "part": {"type": "tool", "tool": "celeste_play",
                         "state": {"input": {"actions": [{"buttons": 2, "frames": 3}]}}}}) + "\n").encode())
            kwargs["stdout"].write((json.dumps({
                "type": "step_finish", "part": {"tokens": {"total": 1, "input": 1}}}) + "\n").encode())
            rollout.mkdir(parents=True, exist_ok=True)
            (rollout / "config.json").write_text("{}")
            # The token-bearing config is deleted after run returns, so read it now.
            seen["config"] = json.loads((output / "workspace" / "opencode.json").read_text())
            return MagicMock(returncode=0)

        run.side_effect = fake_run
        opencode.run("Begin.", "opencode-go/deepseek-v4.1-flash", output, 30, 30,
                     thinking_level="high")
        argv = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        assert argv[:5] == ["opencode", "run", "--pure", "--format", "json"]
        assert "--agent" in argv
        assert "--model" in argv
        assert "opencode-go/deepseek-v4.1-flash" in argv
        assert "--variant" in argv
        assert "high" in argv
        assert "generated-token" not in " ".join(argv)
        assert kwargs["env"]["CELESTEBENCH_MCP_TOKEN"] == "generated-token"
        assert kwargs["cwd"] == output / "workspace"
        # The run points at our config and keeps every host config out.
        assert kwargs["env"]["OPENCODE_CONFIG"] == str(output / "workspace" / "opencode.json")
        assert kwargs["env"]["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
        assert kwargs["env"]["XDG_CONFIG_HOME"] == str(output / "opencode" / "config")
        assert str(rollout) in " ".join(popen.call_args_list[0].args[0])
        assert "generated-token" in seen["config"]["mcp"]["celeste"]["headers"]["Authorization"]
        assert (rollout / "messages.jsonl").read_text().count("\n") == 1
    ready.assert_called_once_with(process, "generated-token", output)
    stop.assert_called_once_with(process, rollout, 30)


def test_resolve_model_blocks_bare_and_near_miss_ids():
    listing = ["opencode-go/deepseek-v4-flash", "opencode-go/deepseek-v4-pro"]
    with patch.object(opencode, "models", return_value=listing):
        with pytest.raises(SystemExit, match="provider/model"):
            opencode.resolve_model("deepseek-flash")
        with pytest.raises(SystemExit, match="deepseek-v4-flash"):
            opencode.resolve_model("opencode-go/deepseek-flash")
        assert opencode.resolve_model("opencode-go/deepseek-v4-flash") == \
            "opencode-go/deepseek-v4-flash"
        # A namespaced id with no near match may be a dynamic provider.
        assert opencode.resolve_model("custom/whatever") == "custom/whatever"


def test_resolve_model_skips_validation_without_a_catalog():
    with patch.object(opencode, "models", return_value=None):
        assert opencode.resolve_model("deepseek-flash") == "deepseek-flash"


def test_cli_error_reads_the_last_error_event():
    with tempfile.TemporaryDirectory() as root:
        trace = Path(root) / "opencode.jsonl"
        trace.write_text(
            '{"type":"error","message":"boom"}\n'
            '{"type":"error","message":"final failure"}\n')
        assert opencode.cli_error(trace) == "final failure"
