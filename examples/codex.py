"""Run Codex against our MCP game server, jailed to an empty read-only workspace."""

import argparse
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

from celestebench import harness
from celestebench.prompt import system_prompt

# Codex keeps a read-only view of its empty workspace root and nothing else:
# the repo, the home directory and the network stay unreadable to the shell
# commands it runs. A plain `--sandbox read-only` is not enough here, it still
# lets commands read the whole disk; the profile below is what locks that down.
PERMISSIONS = """\
approval_policy = "never"
default_permissions = "celestebench"
# Force file auth so parallel runs share `auth.json` instead of each talking to
# the OS keyring, whose per-CODEX_HOME key would strand every refreshed token.
cli_auth_credentials_store = "file"

[permissions.celestebench]
extends = ":read-only"

[permissions.celestebench.filesystem]
":root" = "deny"
":minimal" = "read"
":workspace_roots" = "read"

[permissions.celestebench.network]
enabled = false
"""


free_port = harness.free_port
wait_for_mcp = harness.wait_for_mcp
stop_process = harness.stop_process
limit_output = harness.limit_output


def _stop_mcp(process, rollout, timeout=0, grace=60):
    # `stop` stays a codex module attribute so tests can patch it.
    harness.stop_mcp(process, rollout, timeout, grace, stop=stop_process)


# Codex reasoning efforts, weakest to strongest. Tau's "off" and "minimal" have
# no direct equivalent: models like gpt-6-astra reject them, so clamp per model.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def codex_efforts(env, model):
    """The reasoning efforts Codex accepts for one model, from its own catalog."""
    if not model:
        return ()
    result = subprocess.run(
        ["codex", "debug", "models"], env=env, check=False,
        capture_output=True, text=True,
    )
    try:
        models = json.loads(result.stdout).get("models", [])
    except ValueError:
        return ()
    for entry in models:
        if entry.get("slug") == model:
            return tuple(level["effort"]
                         for level in entry.get("supported_reasoning_levels", ()))
    return ()


def reasoning_effort(level, supported=()):
    """Tau's thinking level as the closest effort the model actually accepts."""
    if not level:
        return None
    wanted = "none" if level == "off" else level
    if wanted not in EFFORTS or not supported or wanted in supported:
        return wanted
    at = EFFORTS.index(wanted)
    stronger = [effort for effort in EFFORTS[at + 1:] if effort in supported]
    weaker = [effort for effort in reversed(EFFORTS[:at]) if effort in supported]
    return (stronger or weaker)[0]


def oauth_login():
    """The host ChatGPT session Codex falls back to when no API key is set."""
    return Path.home() / ".codex" / "auth.json"


def run(prompt, model, output, timeout, fps, frames=None, max_frames=30,
        thinking_level=None):
    api_key = os.environ.get("CODEX_API_KEY") or None
    if api_key and "\n" in api_key:
        raise SystemExit("CODEX_API_KEY must not contain a newline")
    login = oauth_login()
    if api_key is None and not login.is_file():
        raise SystemExit("Set CODEX_API_KEY or run `codex login` before launching Codex")
    token, port = secrets.token_urlsafe(32), free_port()
    rollout = harness.prepare(
        output,
        harness.run_config(model, timeout, frames, max_frames, fps, thinking_level),
        prompt,
    )
    workspace = output / "workspace"
    workspace.mkdir()
    mcp_args = harness.mcp_command(
        rollout, port=port, timeout=timeout, frames=frames, max_frames=max_frames, fps=fps)
    env = os.environ.copy()
    env.pop("CODEX_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env["CELESTEBENCH_MCP_TOKEN"] = token
    # A throwaway CODEX_HOME carries our sandbox profile; auth is either the
    # key we inject or a symlink to the host's `codex login` session. Parallel
    # runs all write through that one file, which is how Codex settles a shared
    # token refresh: the first refresher persists the new bundle and the others
    # reload it instead of asking the token authority again.
    with tempfile.TemporaryDirectory(prefix="celestebench-codex-") as tmp:
        home = Path(tmp)
        (home / "config.toml").write_text(PERMISSIONS, encoding="utf-8")
        if api_key is None:
            (home / "auth.json").symlink_to(login)
        mcp = subprocess.Popen(mcp_args, env=env, start_new_session=True)
        try:
            wait_for_mcp(mcp, token, port)
            # Codex never forwards the MCP server's `instructions` to the model,
            # so the game rules must be injected as developer instructions.
            instructions = system_prompt(fps=fps, max_frames=max_frames, mcp=True, oneshot=True)
            codex = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--json",
                "-c",
                f"mcp_servers.celeste.url='http://127.0.0.1:{port}/mcp'",
                "-c",
                "mcp_servers.celeste.bearer_token_env_var='CELESTEBENCH_MCP_TOKEN'",
                "-c",
                f"mcp_servers.celeste.tool_timeout_sec={timeout + 30}",
                "-c",
                "mcp_servers.celeste.required=true",
                "-c",
                "mcp_servers.celeste.default_tools_approval_mode='approve'",
                "-c",
                f"developer_instructions='''{instructions}'''",
                "-c",
                "features.apps=false",
                "-c",
                "features.plugins=false",
                # Codex spawns shell commands with the whole environment by
                # default, which would hand the model our API key and MCP token.
                "-c",
                "shell_environment_policy.inherit='core'",
            ]
            if model:
                codex += ["--model", model]
            codex_env = os.environ.copy()
            codex_env.pop("CODEX_API_KEY", None)
            codex_env.pop("OPENAI_API_KEY", None)
            if api_key is not None:
                codex_env["CODEX_API_KEY"] = api_key
            codex_env["CODEX_HOME"] = str(home)
            codex_env["CELESTEBENCH_MCP_TOKEN"] = token
            supported = codex_efforts(codex_env, model) if thinking_level else ()
            effort = reasoning_effort(thinking_level, supported)
            if effort:
                codex += ["-c", f"model_reasoning_effort={effort}"]
            codex.append("-")
            with (output / "codex.jsonl").open("xb") as trace:
                subprocess.run(
                    codex,
                    cwd=workspace,
                    env=codex_env,
                    input=prompt,
                    text=True,
                    stdout=trace,
                    check=True,
                    timeout=timeout + 30,
                    preexec_fn=limit_output,
                )
        finally:
            _stop_mcp(mcp, rollout, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=harness.PROMPT)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--thinking-level")
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    if (
        args.timeout <= 0
        or args.frames is not None
        and args.frames <= 0
        or args.max_frames <= 0
        or args.fps is not None
        and args.fps <= 0
    ):
        parser.error("timeout, frames, max-frames, and fps must be positive")
    run(
        args.prompt,
        args.model,
        args.output,
        args.timeout,
        args.fps,
        args.frames,
        args.max_frames,
        args.thinking_level,
    )


if __name__ == "__main__":
    main()
