"""Shared plumbing for the external agent harnesses.

An external harness runs the host CLI in a throwaway workspace against a
per-run CelesteBench MCP server. The CLI plays through MCP and writes its own
JSON trace; the harness script then normalizes that trace into the
messages.jsonl the viewer already reads. Only the CLI-specific parts (config,
command line, trace schema) live in each script under examples/.
"""

import json
import resource
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import BENCHMARK_VERSION

# The one task prompt the external harnesses send; the game rules ride in the
# system prompt, so the user message adds nothing else.
PROMPT = "Play Celeste Classic."


def free_port():
    """Bind and release a loopback port so parallel runs never share one."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_mcp(process, token, port, timeout=20):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            urllib.request.urlopen(request, timeout=0.2).close()
            return
        except urllib.error.HTTPError as error:
            if error.code in (400, 406):
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError("this run's host MCP server did not become ready")


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def stop_mcp(process, rollout, timeout=0, grace=60, stop=stop_process):
    """Let a still-running episode reach its own deadline and finish recording: a
    termination mid-finalization writes an mp4 without its moov index. The MCP
    server owns the wall-clock budget, so the game keeps running until it times
    out even after the model ends its turn."""
    rollout = Path(rollout)
    if process is not None and process.poll() is None:
        done = rollout / "live.done"
        # A run that never called a tool has no episode to let finish.
        wait = timeout + grace if (rollout / "config.json").is_file() else 0
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and process.poll() is None and not done.is_file():
            time.sleep(0.25)
        if done.is_file():
            time.sleep(1)  # the recorder closes its container shortly after
    stop(process)


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (67108864, 67108864))


def mcp_command(output, *, port, timeout, frames=None, max_frames=30, max_images=3, fps=None):
    """Serve one bounded episode over HTTP for the CLI's MCP client."""
    command = [
        sys.executable, "-m", "celestebench.mcp",
        "--transport", "http",
        "--output", str(output),
        "--port", str(port),
        "--timeout", str(timeout),
        "--max-frames", str(max_frames),
        "--max-images", str(max_images),
    ]
    if frames is not None:
        command += ["--frames", str(frames)]
    if fps is None:
        command.append("--lite")  # empty frame rate pauses the game
    else:
        command += ["--fps", str(fps)]
    return command


def prepare(output, config, prompt):
    """Create the wrapper directory and persist the run's public settings."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    return output / "rollout"


def rows(path):
    """Yield each parseable JSON object from a JSONL file, ignoring partial writes."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def usage(input=0, output=0, cache_read=0, cache_write=0, total=None, reasoning=None):
    """One assistant turn's token usage in the viewer's own shape."""
    row = {"input": input, "output": output, "cacheRead": cache_read, "cacheWrite": cache_write,
           "totalTokens": total if total is not None else input + output + cache_read + cache_write}
    if reasoning is not None:
        row["reasoning"] = reasoning
    return row


def assistant(thinking=None, text=None, tool=None, usage_row=None):
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    if text:
        content.append({"type": "text", "text": text})
    if tool is not None:
        content.append({"type": "toolCall", "arguments": tool})
    row = {"role": "assistant", "content": content}
    if usage_row:
        row["usage"] = usage_row
    return row


def fail(trace, message):
    """Record a CLI failure as a trace error row and exit non-zero, so the viewer
    reports the CLI's own wording instead of a Python traceback."""
    with Path(trace).open("a", encoding="utf-8") as file:
        file.write(json.dumps({"type": "error", "message": message}) + "\n")
    raise SystemExit(message)


def run_config(model, timeout, frames, max_frames, fps, thinking_level, **extra):
    return {
        "model": model,
        "benchmark_version": BENCHMARK_VERSION,
        "timeout": timeout,
        "frames": frames,
        "max_frames": max_frames,
        "fps": fps,
        "thinking_level": thinking_level,
        **extra,
    }
