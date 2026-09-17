"""Run the Claude Code CLI against our MCP game server, jailed to an empty workspace.

Claude Code connects to MCP through the ``--mcp-config`` we write; every
built-in tool is turned off, so the model can only play through the
``celeste`` server. The run loads no settings source, no memory files and
strictly only our MCP server, so the host's config cannot reach the model.
The host config directory stays the default on purpose: Claude Code keeps the
subscription login in the OS Keychain (or ``~/.claude/.credentials.json``) and
a custom ``CLAUDE_CONFIG_DIR`` makes it stop reading that login. Its
``stream-json`` events are normalized into the same messages.jsonl the viewer
reads for Tau runs.
"""

import json
import os
import secrets
import subprocess
from pathlib import Path

from celestebench import harness
from celestebench.prompt import system_prompt

SERVER = "celeste"
PLAY = f"mcp__{SERVER}__play"

# WebFetch, WebSearch and Bash are the only tools that reach the internet.
# Disabling every built-in tool already removes them; the deny rules keep them
# blocked if a future CLI ever ignores `--tools`.
SETTINGS = {"permissions": {"deny": ["Bash", "WebFetch", "WebSearch"]}}

# Claude Code's own effort ladder; off and minimal send no flag.
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def config(port, token):
    """The one MCP server Claude Code loads, pointing at this run's episode."""
    return {"mcpServers": {SERVER: {
        "type": "http",
        "url": f"http://127.0.0.1:{port}/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
    }}}


def effort(level):
    """Tau's thinking level as the closest effort Claude Code accepts."""
    return level if level in EFFORTS else None


def cli_error(path):
    """The last error Claude Code reported in its stream, if any."""
    message = None
    for row in harness.rows(path):
        kind = row.get("type")
        if kind == "assistant":
            inner = row.get("message") if isinstance(row.get("message"), dict) else {}
            message = inner.get("error") or message
        elif kind == "result" and row.get("is_error"):
            errors = row.get("errors")
            last = errors[-1] if isinstance(errors, list) and errors else None
            message = last or row.get("result") or row.get("subtype") or str(row)
    return message


def normalize(trace, messages):
    """Fold Claude Code's stream into one assistant row per completed play call.

    A pending turn is replaced by the next assistant message, so observe-only
    or failed turns never leak their reasoning into the next play's row. Usage
    is only reported once, on the final result event, and is attached to the
    last play row so the viewer can still price the run."""
    rows, thinking, text, tool, tool_id, usage = [], [], [], None, None, None
    for event in harness.rows(trace):
        kind = event.get("type")
        if kind in {"assistant", "user"}:
            for block in _content(event):
                if kind == "assistant":
                    _absorb(block, thinking, text)
                    if block.get("type") == "tool_use":
                        if block.get("name") == PLAY:
                            tool, tool_id = block.get("input"), block.get("id")
                        elif tool is None:
                            thinking, text = [], []
                elif block.get("type") == "tool_result" and block.get("tool_use_id") == tool_id:
                    if tool is not None and not block.get("is_error"):
                        rows.append(harness.assistant(
                            "\n".join(x for x in thinking if x) or None,
                            "\n".join(x for x in text if x) or None, tool))
                    thinking, text, tool, tool_id = [], [], None, None
        elif kind == "result" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    if rows and usage:
        rows[-1]["usage"] = _usage(usage)
    with Path(messages).open("x", encoding="utf-8") as out:
        out.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def _content(event):
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = message.get("content")
    return content if isinstance(content, list) else []


def _absorb(block, thinking, text):
    if block.get("type") == "thinking":
        thinking.append(block.get("thinking") or "")
    elif block.get("type") == "text":
        text.append(block.get("text") or "")


def _usage(usage):
    return harness.usage(
        input=usage.get("input_tokens") or 0,
        output=usage.get("output_tokens") or 0,
        cache_read=usage.get("cache_read_input_tokens") or 0,
        cache_write=usage.get("cache_creation_input_tokens") or 0,
    )


def run(prompt, model, output, timeout, fps, frames=None, max_frames=30,
        thinking_level=None, max_images=3):
    token = secrets.token_urlsafe(32)
    rollout = harness.prepare(
        output,
        harness.run_config(model, timeout, frames, max_frames, fps, thinking_level),
        prompt,
    )
    workspace = output / "workspace"
    workspace.mkdir()
    instructions = system_prompt(fps=fps, max_frames=max_frames,
                                 max_images=max_images, mcp=True, oneshot=True)
    env = os.environ.copy()
    # The host's own CLAUDE.md files must not reach the model.
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CELESTEBENCH_MCP_TOKEN"] = token
    trace_path = output / "claude.jsonl"
    mcp = subprocess.Popen(
        harness.mcp_command(rollout, timeout=timeout, frames=frames,
                            max_frames=max_frames, max_images=max_images, fps=fps),
        env=env, start_new_session=True)
    try:
        port = harness.wait_for_mcp(mcp, token, output)
        # The config carries the bearer token, so write it only once the port is
        # known and delete it before the run is archived (see finally).
        (workspace / "mcp.json").write_text(
            json.dumps(config(port, token), indent=2) + "\n", encoding="utf-8")
        command = [
            "claude", "--print", "--output-format", "stream-json", "--verbose",
            "--mcp-config", str(workspace / "mcp.json"),
            "--strict-mcp-config",
            # No settings source and no memory files: the host's config, agents,
            # plugins, hooks and CLAUDE.md stay out of the model's context.
            "--setting-sources", "",
            "--system-prompt", instructions,
            "--disable-slash-commands",
            "--no-session-persistence",
            # No built-in tool at all; only the celeste server's tools can run,
            # and anything else that would prompt is denied automatically.
            "--tools", "",
            "--allowedTools", f"mcp__{SERVER}",
            "--permission-prompts", "none",
            "--settings", json.dumps(SETTINGS),
        ]
        if model:
            command += ["--model", model]
        chosen = effort(thinking_level)
        if chosen:
            command += ["--effort", chosen]
        command.append(prompt)
        timed_out = False
        with trace_path.open("xb") as trace:
            try:
                completed = subprocess.run(
                    command, cwd=workspace, env=env, text=True, stdout=trace,
                    timeout=timeout + 30, preexec_fn=harness.limit_output, check=False)
            except subprocess.TimeoutExpired:
                timed_out = True
        # Fold whatever the CLI wrote before it stopped, so a run that ends on a
        # provider error still keeps its decisions and the usage its result event
        # reported. Failing first would drop messages.jsonl and leave the scored
        # rollout unpriced.
        normalize(trace_path, rollout / "messages.jsonl")
        if timed_out:
            harness.fail(trace_path, f"Claude Code timed out after {timeout + 30}s")
        error = cli_error(trace_path)
        if completed.returncode != 0 or error:
            harness.fail(trace_path, error or
                         f"Claude Code exited with code {completed.returncode}")
        if not (rollout / "config.json").is_file():
            harness.fail(trace_path, "Claude Code finished without using the game tools")
    finally:
        (workspace / "mcp.json").unlink(missing_ok=True)
        harness.stop_mcp(mcp, rollout, timeout)


def main():
    args = harness.cli_args(__doc__)
    run(args.prompt, args.model, args.output, args.timeout, args.fps, args.frames,
        args.max_frames, args.thinking_level, args.max_images)


if __name__ == "__main__":
    main()
