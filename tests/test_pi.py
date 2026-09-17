import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import load_example

pi = load_example("pi")


def test_normalize_emits_one_assistant_row_per_completed_play():
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
    assert len(rows) == 1
    assert [block["type"] for block in rows[0]["content"]] == ["thinking", "text", "toolCall"]
    assert rows[0]["content"][2]["arguments"] == play
    assert rows[0]["usage"]["cacheRead"] == 3
    assert rows[0]["usage"]["totalTokens"] == 13


def test_normalize_observe_turn_usage_does_not_leak_into_play_row():
    play = {"actions": [{"buttons": 1, "frames": 2}]}
    events = [
        # An observe-only turn must be replaced by the next play turn.
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "watching"},
            {"type": "toolCall", "name": "observe", "arguments": {}}],
            "usage": {"input": 40, "output": 7, "cacheRead": 99, "totalTokens": 47,
                      "reasoning": 5}}},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "play", "arguments": play}],
            "usage": {"input": 8, "output": 2, "cacheRead": 3, "totalTokens": 13,
                      "reasoning": 1}}},
        {"type": "tool_execution_end", "toolName": "play", "isError": False, "result": {}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "pi.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        pi.normalize(trace, messages)
        rows = [json.loads(line) for line in messages.read_text().splitlines()]
    assert len(rows) == 1
    assert [block["type"] for block in rows[0]["content"]] == ["toolCall"]
    assert rows[0]["content"][0]["arguments"] == play
    assert rows[0]["usage"]["input"] == 8
    assert rows[0]["usage"]["cacheRead"] == 3
    assert rows[0]["usage"]["totalTokens"] == 13
    assert rows[0]["usage"]["reasoning"] == 1


def test_normalize_skips_failed_play_calls():
    events = [
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "play", "arguments": {"actions": []}}]}},
        {"type": "tool_execution_end", "toolName": "play", "isError": True, "result": {}},
    ]
    with tempfile.TemporaryDirectory() as root:
        trace, messages = Path(root) / "pi.jsonl", Path(root) / "messages.jsonl"
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        pi.normalize(trace, messages)
        assert messages.read_text() == ""


def test_cli_error_reads_the_provider_error():
    with tempfile.TemporaryDirectory() as root:
        trace = Path(root) / "pi.jsonl"
        trace.write_text(
            '{"type":"message_end","message":{"role":"assistant","errorMessage":"first"}}\n'
            '{"type":"message_end","message":{"role":"assistant","errorMessage":"Codex error"}}\n')
        assert pi.cli_error(trace) == "Codex error"


def test_run_loads_the_bundled_extension_without_builtin_tools():
    with (
        patch.object(pi.subprocess, "run") as run,
        patch.object(pi.subprocess, "Popen") as popen,
        patch.object(pi.secrets, "token_urlsafe", return_value="generated-token"),
        patch.object(pi.harness, "wait_for_mcp") as ready,
        patch.object(pi.harness, "stop_mcp") as stop,
        patch.object(pi, "host_agent_dir") as agent_dir,
    ):
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        ready.return_value = 9124

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "result"
            rollout = output / "rollout"
            host = Path(root) / "host-pi"
            host.mkdir()
            (host / "auth.json").write_text("{}")
            agent_dir.return_value = host

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
            assert argv[0] == "pi"
            assert "--no-builtin-tools" in argv
            assert "--no-approve" in argv
            assert "--offline" in argv
            assert "--system-prompt" in argv
            assert "--extension" in argv
            assert str(pi.EXTENSION) in argv
            assert argv[-1] == "Begin."
            assert "--thinking" in argv
            assert "low" in argv
            assert "generated-token" not in " ".join(argv)
            assert kwargs["env"]["CELESTEBENCH_MCP_TOKEN"] == "generated-token"
            assert kwargs["env"]["CELESTEBENCH_MCP_URL"] == "http://127.0.0.1:9124/mcp"
            assert kwargs["env"]["CELESTEBENCH_MAX_FRAMES"] == "30"
            assert kwargs["cwd"] == output / "workspace"
            # The run gets its own agent directory and borrows just the login.
            agent = Path(kwargs["env"]["PI_CODING_AGENT_DIR"])
            assert agent == output / "pi" / "agent"
            assert (agent / "auth.json").is_symlink()
            assert str(rollout) in " ".join(popen.call_args_list[0].args[0])
            assert (rollout / "messages.jsonl").read_text().count("\n") == 1
            ready.assert_called_once_with(process, "generated-token", output)
        stop.assert_called_once_with(process, rollout, 45)
