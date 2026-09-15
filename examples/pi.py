"""Run the Pi CLI against our MCP game server, jailed to an empty workspace.

Pi has no MCP client built in, so we load examples/pi_celeste.ts with -e: it
registers play/observe tools that forward to the episode server this script
owns. Built-in tools stay off, and the run gets a throwaway agent directory so
the host's settings, trust list, extensions and skills never reach it. Pi's
--mode json events are normalized into the same messages.jsonl the viewer reads
for Tau runs.
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
wait_for_mcp = harness.wait_for_mcp
limit_output = harness.limit_output


def _stop_mcp(process, rollout, timeout=0):
    harness.stop_mcp(process, rollout, timeout, stop=stop_process)


def host_agent_dir():
    """The host Pi agent directory whose login and model catalog we borrow."""
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")


def isolated_home(output):
    """Give Pi a throwaway agent directory for this run, so the host's settings,
    trust list, extensions and skills cannot leak in. Only the login and the
    model catalog are symlinked back, and --offline keeps Pi from refreshing them."""
    source = host_agent_dir()
    agent = output / "pi" / "agent"
    agent.mkdir(parents=True)
    for name in ("auth.json", "models.json", "models-store.json"):
        if (source / name).is_file():
            (agent / name).symlink_to(source / name)
    return {"PI_CODING_AGENT_DIR": str(agent)}


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
    total = total or {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                      "total": 0, "reasoning": 0}
    total["input"] += usage.get("input") or 0
    total["output"] += usage.get("output") or 0
    total["cache_read"] += usage.get("cacheRead") or 0
    total["cache_write"] += usage.get("cacheWrite") or 0
    total["total"] += usage.get("totalTokens") or usage.get("total") or 0
    total["reasoning"] += usage.get("reasoning") or 0
    return total


def normalize(trace, messages):
    """Fold Pi's event stream into one assistant row per completed play call.

    Each assistant turn replaces the pending one, so an observe-only or failed
    turn never leaks its reasoning or tokens into the next play's row."""
    turn = None
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
                thinking, text, tool = [], [], None
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking":
                        thinking.append(block.get("thinking") or "")
                    elif block.get("type") == "text":
                        text.append(block.get("text") or "")
                    elif block.get("type") == "toolCall" and block.get("name") == "play":
                        tool = block.get("arguments")
                turn = thinking, text, tool, _sum(None, message.get("usage"))
            elif kind == "tool_execution_end" and event.get("toolName") == "play":
                if event.get("isError") or turn is None or turn[2] is None:
                    continue
                thinking, text, tool, tokens = turn
                row = harness.assistant(
                    "\n".join(x for x in thinking if x) or None,
                    "\n".join(x for x in text if x) or None,
                    tool,
                    harness.usage(**tokens) if tokens else None,
                )
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                turn = None


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
    env |= isolated_home(output)
    env |= {
        "CELESTEBENCH_MCP_TOKEN": token,
        "CELESTEBENCH_MAX_FRAMES": str(max_frames),
        # cwd= does not update $PWD; keep the child's idea of its directory sane.
        "PWD": str(workspace),
    }
    trace_path = output / "pi.jsonl"
    mcp = subprocess.Popen(
        harness.mcp_command(rollout, timeout=timeout, frames=frames,
                            max_frames=max_frames, max_images=max_images, fps=fps),
        env=env, start_new_session=True)
    try:
        port = wait_for_mcp(mcp, token, output)
        env["CELESTEBENCH_MCP_URL"] = f"http://127.0.0.1:{port}/mcp"
        command = [
            "pi", "--print", "--mode", "json", "--no-session", "--no-extensions",
            "--no-builtin-tools", "--no-context-files", "--no-skills",
            "--no-prompt-templates", "--no-approve", "--offline",
            "--model", model,
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
