"""Run the Pi CLI against our MCP game server, jailed to an empty workspace.

Pi has no MCP client built in, so we load examples/pi_celeste.ts with -e: it
registers play/observe tools that forward to the episode server this script
owns. Built-in tools stay off. Pi's --mode json events are normalized into the
same messages.jsonl the viewer reads for Tau runs.
"""

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path

from celestebench import harness
from celestebench.prompt import system_prompt

EXTENSION = Path(__file__).with_name("pi_celeste.ts")

stop_process = harness.stop_process
free_port = harness.free_port
wait_for_mcp = harness.wait_for_mcp
limit_output = harness.limit_output


def _stop_mcp(process, rollout, timeout=0):
    harness.stop_mcp(process, rollout, timeout, stop=stop_process)


def cli_error(path):
    """The last provider error Pi reported in an assistant message, if any."""
    message = None
    for row in harness.rows(path):
        if row.get("type") == "message_end":
            inner = row.get("message") if isinstance(row.get("message"), dict) else {}
            if inner.get("errorMessage"):
                message = inner["errorMessage"]
    return message


def _sum(total, usage):
    if not isinstance(usage, dict):
        return total
    total = total or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                      "total": 0, "reasoning": 0}
    total["input"] += usage.get("input") or 0
    total["output"] += usage.get("output") or 0
    total["cacheRead"] += usage.get("cacheRead") or 0
    total["cacheWrite"] += usage.get("cacheWrite") or 0
    total["total"] += usage.get("totalTokens") or usage.get("total") or 0
    total["reasoning"] += usage.get("reasoning") or 0
    return total


def normalize(trace, messages):
    """Fold Pi's event stream into one assistant row per completed play call."""
    thinking, text, tool, tokens = [], [], None, None
    with Path(messages).open("x", encoding="utf-8") as out, Path(trace).open(
            encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "message_end" and isinstance(event.get("message"), dict):
                message = event["message"]
                if message.get("role") != "assistant":
                    continue
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking":
                        thinking.append(block.get("thinking") or "")
                    elif block.get("type") == "text":
                        text.append(block.get("text") or "")
                    elif block.get("type") == "toolCall" and block.get("name") == "play":
                        tool = block.get("arguments")
                tokens = _sum(tokens, message.get("usage"))
            elif kind == "tool_execution_end" and event.get("toolName") == "play":
                if event.get("isError") or tool is None:
                    continue
                row = harness.assistant(
                    "\n".join(x for x in thinking if x) or None,
                    "\n".join(x for x in text if x) or None,
                    tool,
                    harness.usage(input=tokens["input"], output=tokens["output"],
                                  cache_read=tokens["cacheRead"], cache_write=tokens["cacheWrite"],
                                  total=tokens["total"] or None,
                                  reasoning=tokens["reasoning"] or None) if tokens else None,
                )
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                thinking, text, tool, tokens = [], [], None, None


def run(prompt, model, output, timeout, fps, frames=None, max_frames=30,
        thinking_level=None, max_images=3):
    token, port = secrets.token_urlsafe(32), free_port()
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
    env |= {
        "CELESTEBENCH_MCP_URL": f"http://127.0.0.1:{port}/mcp",
        "CELESTEBENCH_MCP_TOKEN": token,
        "CELESTEBENCH_MAX_FRAMES": str(max_frames),
        # cwd= does not update $PWD; keep the child's idea of its directory sane.
        "PWD": str(workspace),
    }
    trace_path = output / "pi.jsonl"
    mcp = subprocess.Popen(
        harness.mcp_command(rollout, port=port, timeout=timeout, frames=frames,
                            max_frames=max_frames, max_images=max_images, fps=fps),
        env=env, start_new_session=True)
    try:
        wait_for_mcp(mcp, token, port)
        command = [
            "pi", "--print", "--mode", "json", "--no-session", "--no-extensions",
            "--no-builtin-tools", "--no-context-files", "--no-skills",
            "--no-prompt-templates", "--model", model,
            "--system-prompt", instructions, "--extension", str(EXTENSION),
        ]
        if thinking_level:
            command += ["--thinking", thinking_level]
        command.append(prompt)
        with trace_path.open("xb") as trace:
            completed = subprocess.run(
                command, cwd=workspace, env=env, text=True, stdout=trace,
                timeout=timeout + 30, preexec_fn=limit_output, check=False)
        error = cli_error(trace_path)
        if completed.returncode != 0 or error:
            harness.fail(trace_path, error or f"Pi exited with code {completed.returncode}")
        normalize(trace_path, rollout / "messages.jsonl")
        if not (rollout / "config.json").is_file():
            harness.fail(trace_path, "Pi finished without using the game tools")
    finally:
        _stop_mcp(mcp, rollout, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=harness.PROMPT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--thinking-level")
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or args.frames is not None
        and args.frames <= 0
        or args.max_frames <= 0
        or args.max_images <= 0
        or args.fps is not None
        and args.fps <= 0
    ):
        parser.error("timeout, frames, max-frames, max-images, and fps must be positive")
    run(args.prompt, args.model, args.output, args.timeout, args.fps, args.frames,
        args.max_frames, args.thinking_level, args.max_images)


if __name__ == "__main__":
    main()
