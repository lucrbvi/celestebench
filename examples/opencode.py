"""Run the OpenCode CLI against our MCP game server, jailed to an empty workspace.

OpenCode connects to MCP through the ``opencode.json`` we generate; the agent
denies every built-in tool except the ``celeste_*`` ones, so the model can only
play the game. The run gets a throwaway config/data/state/cache home and project
discovery is off, so the host's config, agents, plugins and MCP servers never
reach it. Its ``--format json`` events are normalized into the same
messages.jsonl the viewer reads for Tau runs.
"""

import json
import os
import re
import secrets
import subprocess
from pathlib import Path

from celestebench import auth, harness
from celestebench.prompt import system_prompt

AGENT = "celestebench"
INSTRUCTIONS = "instructions.txt"
VARIANTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


def config(port, token):
    """OpenCode expands ``{...}`` in config strings, so the system prompt must
    come from a file instead of being inlined (the game rules contain braces).
    Sharing, snapshots and self-updates stay off so a run owns nothing on the host."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
        "mcp": {
            "celeste": {
                "type": "remote",
                "url": f"http://127.0.0.1:{port}/mcp",
                "enabled": True,
                "headers": {"Authorization": f"Bearer {token}"},
            },
        },
        "agent": {
            AGENT: {
                "description": "Plays Celeste Classic through the CelesteBench MCP server.",
                "mode": "primary",
                "prompt": f"{{file:./{INSTRUCTIONS}}}",
                "tools": {"*": False, "celeste*": True},
            },
        },
    }


def variant(level):
    """OpenCode's model variants match Tau's levels, but there is no "off"."""
    return level if level in VARIANTS else None


def models():
    """The host's provider/model ids, or None when the catalog is unavailable."""
    try:
        result = subprocess.run(["opencode", "models"], capture_output=True, text=True,
                                timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if "/" in line.strip()]


def resolve_model(model):
    """An unknown model reaches OpenCode as a generic server error, so check its
    catalog first and hand back the closest ids instead of a mystery crash."""
    listing = models()
    if listing is None or model in listing:
        return model
    parts = [part for part in re.split(r"[-/.]", model.lower()) if part]
    close = [item for item in listing if all(part in item.lower() for part in parts)][:5]
    # A bare id is never valid (OpenCode wants provider/model); a namespaced id
    # we cannot find may still be a dynamic provider, so only block near misses.
    if "/" in model and not close:
        return model
    hint = f" Try {', '.join(close)}." if close else ""
    raise SystemExit(f"Unknown OpenCode model '{model}'; OpenCode wants provider/model "
                     f"(for example opencode-go/deepseek-v4-flash).{hint}")


def isolated_env(output):
    """Give OpenCode a throwaway home for this run, so the host's config, agents,
    plugins, MCP servers and database cannot leak in. Only the login is symlinked
    back; the project config we wrote in the workspace loads through OPENCODE_CONFIG
    while project discovery stays off, so no ancestor config can add tools."""
    home = output / "opencode"
    credentials = home / "data" / "opencode"
    credentials.mkdir(parents=True)
    source = auth.agent_dir("opencode")
    for name in ("auth.json", "account.json"):
        if (source / name).is_file():
            (credentials / name).symlink_to(source / name)
    return {
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    }


def cli_error(path):
    """The last error OpenCode reported in its JSON stream, if any."""
    message = None
    for row in harness.rows(path):
        if row.get("type") != "error":
            continue
        part = row.get("part") if isinstance(row.get("part"), dict) else {}
        error = row.get("error") if isinstance(row.get("error"), dict) else {}
        data = error.get("data") if isinstance(error.get("data"), dict) else {}
        message = (row.get("message") or part.get("message") or data.get("message")
                   or error.get("name") or str(row))
    return message


def _usage(total):
    if not total:
        return None
    return harness.usage(input=total["input"], output=total["output"],
                         cache_read=total.get("read", 0), cache_write=total.get("write", 0),
                         total=total.get("total") or None, reasoning=total.get("reasoning"))


def normalize(trace, messages):
    """Fold OpenCode's event stream into one assistant row per play call.

    Every step boundary clears the pending turn, so observe-only or errored
    steps never leak their reasoning or tokens into the next play's row."""
    thinking, text, tool, tool_error, tokens = [], [], None, False, None
    with Path(messages).open("x", encoding="utf-8") as out, Path(trace).open(
            encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            if kind in {"text", "reasoning"} and part.get("text"):
                (thinking if kind == "reasoning" else text).append(part["text"])
            elif kind == "tool_use" and part.get("tool") == "celeste_play":
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool = state.get("input")
                tool_error = state.get("status") == "error" or bool(state.get("error"))
            elif kind == "step_finish":
                tokens = _sum(tokens, part.get("tokens"))
                if tool is not None and not tool_error:
                    row = harness.assistant("\n".join(thinking) or None, "\n".join(text) or None,
                                            tool, _usage(tokens))
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                thinking, text, tool, tool_error, tokens = [], [], None, False, None


def _sum(total, tokens):
    if not isinstance(tokens, dict):
        return total
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    total = total or {"input": 0, "output": 0, "total": 0, "reasoning": 0, "read": 0, "write": 0}
    total["input"] += tokens.get("input") or 0
    total["output"] += tokens.get("output") or 0
    total["total"] += tokens.get("total") or 0
    total["reasoning"] += tokens.get("reasoning") or 0
    total["read"] += cache.get("read") or 0
    total["write"] += cache.get("write") or 0
    return total


def run(prompt, model, output, timeout, fps, frames=None, max_frames=30,
        thinking_level=None, max_images=3):
    model = resolve_model(model)
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
    (workspace / INSTRUCTIONS).write_text(instructions, encoding="utf-8")
    env = os.environ.copy()
    # The host OpenCode (or a parent agent session) leaks inline config and
    # identity vars into children; drop them so our own config is the only one.
    for name in ("OPENCODE", "OPENCODE_PID", "OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT",
                 "OPENCODE_CONFIG_DIR", "OPENCODE_PERMISSION", "OPENCODE_DB",
                 "OPENCODE_SERVER_PASSWORD", "OPENCODE_SERVER_USERNAME", "OPENCODE_CLIENT"):
        env.pop(name, None)
    env["CELESTEBENCH_MCP_TOKEN"] = token
    # cwd= does not update $PWD, and OpenCode locates the project from $PWD, so
    # without this it loads our config from the wrong directory.
    env["PWD"] = str(workspace)
    # Project discovery is off, so point OpenCode straight at our config file.
    env["OPENCODE_CONFIG"] = str(workspace / "opencode.json")
    env |= isolated_env(output)
    trace_path = output / "opencode.jsonl"
    mcp = subprocess.Popen(
        harness.mcp_command(rollout, timeout=timeout, frames=frames,
                            max_frames=max_frames, max_images=max_images, fps=fps),
        env=env, start_new_session=True)
    try:
        port = harness.wait_for_mcp(mcp, token, output)
        # The config carries the bearer token, so write it only once the port is
        # known and delete it before the run is archived (see finally).
        (workspace / "opencode.json").write_text(
            json.dumps(config(port, token), indent=2) + "\n", encoding="utf-8")
        command = ["opencode", "run", "--pure", "--format", "json", "--agent", AGENT,
                   "--model", model, "--title", AGENT]
        chosen = variant(thinking_level)
        if chosen:
            command += ["--variant", chosen]
        command.append(prompt)
        with trace_path.open("xb") as trace:
            completed = subprocess.run(
                command, cwd=workspace, env=env, text=True, stdout=trace,
                timeout=timeout + 30, preexec_fn=harness.limit_output, check=False)
        error = cli_error(trace_path)
        if completed.returncode != 0 or error:
            harness.fail(trace_path, error or f"OpenCode exited with code {completed.returncode}")
        normalize(trace_path, rollout / "messages.jsonl")
        if not (rollout / "config.json").is_file():
            harness.fail(trace_path, "OpenCode finished without using the game tools")
    finally:
        (workspace / "opencode.json").unlink(missing_ok=True)
        harness.stop_mcp(mcp, rollout, timeout)


def main():
    args = harness.cli_args(__doc__)
    run(args.prompt, args.model, args.output, args.timeout, args.fps, args.frames,
        args.max_frames, args.thinking_level, args.max_images)


if __name__ == "__main__":
    main()
