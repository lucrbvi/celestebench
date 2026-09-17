import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import load_example

claude = load_example("claude")


def test_config_points_at_mcp_with_the_bearer_token():
    config = claude.config(9123, "secret-token")
    server = config["mcpServers"][claude.SERVER]
    assert server["type"] == "http"
    assert server["url"] == "http://127.0.0.1:9123/mcp"
    assert server["headers"]["Authorization"] == "Bearer secret-token"


def test_effort_maps_tau_thinking_levels():
    assert claude.effort(None) is None
    assert claude.effort("off") is None
    assert claude.effort("minimal") is None
    assert claude.effort("high") == "high"
    assert claude.effort("max") == "max"


def test_normalize_emits_one_assistant_row_per_completed_play():
    play = {"actions": [{"buttons": 2, "frames": 4}]}
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "think"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "hello"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": claude.PLAY, "input": play}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "claude.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        claude.normalize(trace, messages)
        rows = [json.loads(line) for line in messages.read_text().splitlines()]
    assert len(rows) == 1
    assert [block["type"] for block in rows[0]["content"]] == ["thinking", "text", "toolCall"]
    assert rows[0]["content"][0]["thinking"] == "think"
    assert rows[0]["content"][2]["arguments"] == play


def test_normalize_drops_observe_turns():
    events = [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "watching"},
            {"type": "tool_use", "id": "o1", "name": "mcp__celeste__observe", "input": {}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "o1", "content": "frame"}]}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "claude.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        claude.normalize(trace, messages)
        assert messages.read_text() == ""


def test_normalize_skips_failed_play_calls():
    events = [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": claude.PLAY, "input": {"actions": []}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ended", "is_error": True}]}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "claude.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        claude.normalize(trace, messages)
        assert messages.read_text() == ""


def test_normalize_attaches_the_result_usage_to_the_last_play():
    events = [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": claude.PLAY, "input": {"actions": []}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "usage": {
            "input_tokens": 150, "output_tokens": 70,
            "cache_read_input_tokens": 800, "cache_creation_input_tokens": 5}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "claude.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        claude.normalize(trace, messages)
        rows = [json.loads(line) for line in messages.read_text().splitlines()]
    usage = rows[0]["usage"]
    assert (usage["input"], usage["output"]) == (150, 70)
    assert (usage["cacheRead"], usage["cacheWrite"]) == (800, 5)
    assert usage["totalTokens"] == 1025


def test_cli_error_reads_the_last_error():
    with tempfile.TemporaryDirectory() as root:
        trace = Path(root) / "claude.jsonl"
        trace.write_text(
            '{"type":"assistant","message":{"role":"assistant","error":"first"}}\n'
            '{"type":"result","subtype":"error_during_execution","is_error":true,'
            '"result":"final failure"}\n')
        assert claude.cli_error(trace) == "final failure"


def test_run_jails_claude_and_keeps_the_token_out_of_argv():
    with (
        patch.object(claude.harness, "stop_mcp") as stop,
        patch.object(claude.harness, "wait_for_mcp") as ready,
        patch.object(claude.secrets, "token_urlsafe", return_value="generated-token"),
        patch.object(claude.subprocess, "Popen") as popen,
        patch.object(claude.subprocess, "run") as run,
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        ready.return_value = 9123
        seen = {}

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result"
            rollout = output / "rollout"

            def fake_run(argv, **kwargs):
                kwargs["stdout"].write((json.dumps({
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "t1", "name": claude.PLAY,
                         "input": {"actions": [{"buttons": 2, "frames": 3}]}}]},
                }) + "\n").encode())
                kwargs["stdout"].write((json.dumps({
                    "type": "user",
                    "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
                }) + "\n").encode())
                # The token-bearing config is deleted after run returns, so read it now.
                seen["config"] = json.loads((output / "workspace" / "mcp.json").read_text())
                rollout.mkdir(parents=True, exist_ok=True)
                (rollout / "config.json").write_text("{}")
                return MagicMock(returncode=0)

            run.side_effect = fake_run
            claude.run("Begin.", "sonnet", output, 45, 30, thinking_level="high")
            argv = run.call_args.args[0]
            kwargs = run.call_args.kwargs
            assert argv[0] == "claude"
            assert argv[1:4] == ["--print", "--output-format", "stream-json"]
            assert argv[argv.index("--tools") + 1] == ""
            assert argv[argv.index("--setting-sources") + 1] == ""
            assert argv[argv.index("--allowedTools") + 1] == "mcp__celeste"
            assert argv[argv.index("--permission-prompts") + 1] == "none"
            assert argv[argv.index("--mcp-config") + 1] == str(output / "workspace" / "mcp.json")
            assert "--strict-mcp-config" in argv
            assert "--system-prompt" in argv
            assert "--disable-slash-commands" in argv
            assert argv[argv.index("--effort") + 1] == "high"
            assert "generated-token" not in " ".join(argv)
            assert kwargs["env"]["CELESTEBENCH_MCP_TOKEN"] == "generated-token"
            assert kwargs["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
            assert kwargs["cwd"] == output / "workspace"
            assert str(rollout) in " ".join(popen.call_args_list[0].args[0])
            assert "generated-token" in seen["config"]["mcpServers"]["celeste"]["headers"]["Authorization"]
            assert (rollout / "messages.jsonl").read_text().count("\n") == 1
            assert not (output / "workspace" / "mcp.json").exists()
    ready.assert_called_once_with(process, "generated-token", output)
    stop.assert_called_once_with(process, rollout, 45)
