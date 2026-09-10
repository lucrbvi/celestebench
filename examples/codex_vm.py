"""Run Codex in a small Lima VM while Celeste stays on the host."""

import argparse
import json
import os
import platform
import resource
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).with_name("codex-vm.yaml")
NAME, PORT = "celestebench-codex", 8124


def _find_limactl() -> Path:
    found = shutil.which("limactl")
    if found:
        return Path(found)
    for probe in ("/opt/homebrew/bin/limactl", "/usr/local/bin/limactl"):
        if Path(probe).is_file():  # brew installs may be missing from PATH
            return Path(probe)
    return Path("limactl")  # keeps PATH failures visible in the error


LIMACTL = _find_limactl()


def command(args, **kwargs):
    return subprocess.run([str(LIMACTL), *args], check=True, **kwargs)


def ssh_config(name):
    return command(
        ["ls", "--format={{.SSHConfigFile}}", name], capture_output=True, text=True
    ).stdout.strip()


def ssh(name, *remote):
    args = [
        "ssh",
        "-F",
        ssh_config(name),
        "-o",
        "ForwardAgent=no",
        "-o",
        "ExitOnForwardFailure=yes",
        f"lima-{name}",
    ]
    return args + ([shlex.join(remote)] if remote else [])


def tunnel_command(name):
    # Lima's ssh config shares one ControlMaster; multiplexing on it exits this
    # command immediately and strands the forward. Own the connection instead.
    return [
        "ssh",
        "-F",
        ssh_config(name),
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-R",
        f"127.0.0.1:{PORT}:127.0.0.1:{PORT}",
        f"lima-{name}",
    ]


def host_config(tmp: Path) -> Path:
    """The best VM backend for the host: VZ (Apple Virtualization) on macOS,
    QEMU elsewhere, with the guest arch matching the host outside Mac ARM."""
    if sys.platform == "darwin":
        return CONFIG
    body = CONFIG.read_text(encoding="utf-8").replace("vmType: vz", "vmType: qemu")
    body = body.replace("arch: aarch64", f"arch: {platform.machine()}")
    config = tmp / "codex-vm.yaml"
    config.write_text(body, encoding="utf-8")
    return config


def start(name):
    with tempfile.TemporaryDirectory(prefix="celestebench-codex-") as tmp:
        config = host_config(Path(tmp))
        command(["validate", str(config)])
        names = command(
            ["list", "--format={{.Name}}"], capture_output=True, text=True
        ).stdout.splitlines()
        command(
            ["start", name]
            if name in names
            else ["start", "--tty=false", f"--name={name}", str(config)]
        )


def setup(name):
    mounts = command(
        ["list", "--format={{json .Config.Mounts}}", name],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if mounts not in ("null", "[]"):
        raise RuntimeError("refusing a VM with host mounts")
    subprocess.run(
        ssh(name, "sudo", "nft", "list", "table", "inet", "celestebench"),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ssh(name, "sudo", "runuser", "-u", "bench", "--", "codex", "--version"),
        check=True,
    )


def login(name):
    start(name)
    setup(name)
    subprocess.run(
        ssh(
            name,
            "sudo",
            "runuser",
            "-u",
            "bench",
            "--",
            "bash",
            "-lc",
            "cd /home/bench && exec codex login --device-auth",
        ),
        check=True,
    )


def wait_for_mcp(process, token, timeout=20):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/mcp",
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
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _stop_mcp(process, rollout, grace=75):
    """Give a still-running episode its deadline and let the recorder finish:
    a SIGTERM mid-finalization writes an mp4 without its moov index."""
    if process is not None and process.poll() is None:
        done = rollout / "live.done"
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and process.poll() is None and not done.is_file():
            time.sleep(0.25)
        if done.is_file():
            time.sleep(1)  # the recorder closes its container shortly after
    stop_process(process)


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (67108864, 67108864))


GUEST_RUNNER = r"""import json, os, pwd, resource, signal, subprocess, sys
d=json.load(sys.stdin); u=pwd.getpwnam("bench")
env={"HOME":u.pw_dir,"PATH":"/usr/local/bin:/usr/bin:/bin","CELESTEBENCH_MCP_TOKEN":d["token"]}
if d.get("api_key"): env["CODEX_API_KEY"]=d["api_key"]
def demote():
 os.initgroups(u.pw_name,u.pw_gid); os.setgid(u.pw_gid); os.setuid(u.pw_uid)
 resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864))
p=subprocess.Popen(d["args"],stdin=subprocess.PIPE,text=True,cwd=u.pw_dir,env=env,
 start_new_session=True,preexec_fn=demote)
try: p.communicate(d["prompt"],timeout=d["timeout"])
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGTERM)
 try: p.wait(5)
 except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
 raise SystemExit(124)
raise SystemExit(p.returncode)"""


def run(name, prompt, model, output, timeout, lite, frames=None, max_frames=30):
    api_key, token = os.environ.get("CODEX_API_KEY") or None, secrets.token_urlsafe(32)
    if api_key and "\n" in api_key:
        raise SystemExit("CODEX_API_KEY must not contain a newline")
    start(name)
    setup(name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", PORT))
    output.mkdir(parents=True, exist_ok=False)
    rollout = output / "rollout"
    config = {
        "vm": name,
        "model": model,
        "timeout": timeout,
        "frames": frames,
        "max_frames": max_frames,
        "lite": lite,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    mcp_args = [
        sys.executable,
        "-m",
        "celestebench.mcp",
        "--transport",
        "http",
        "--output",
        str(rollout),
        "--port",
        str(PORT),
        "--timeout",
        str(timeout),
        "--max-frames",
        str(max_frames),
    ]
    if frames is not None:
        mcp_args += ["--frames", str(frames)]
    if lite:
        mcp_args.append("--lite")
    env = os.environ.copy()
    env.pop("CODEX_API_KEY", None)
    env["CELESTEBENCH_MCP_TOKEN"] = token
    mcp = subprocess.Popen(mcp_args, env=env, start_new_session=True)
    tunnel = None
    try:
        wait_for_mcp(mcp, token)
        tunnel = subprocess.Popen(tunnel_command(name), start_new_session=True)
        time.sleep(0.3)
        if tunnel.poll() is not None:
            raise RuntimeError("SSH reverse tunnel exited early")
        codex = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--sandbox",
            "read-only",
            "-c",
            "mcp_servers.celeste.url='http://127.0.0.1:8124/mcp'",
            "-c",
            "mcp_servers.celeste.bearer_token_env_var='CELESTEBENCH_MCP_TOKEN'",
            "-c",
            f"mcp_servers.celeste.tool_timeout_sec={timeout + 30}",
            "-c",
            "mcp_servers.celeste.required=true",
            "-c",
            "mcp_servers.celeste.default_tools_approval_mode='approve'",
            "-",
        ]
        if model:
            codex[2:2] = ["--model", model]
        payload = json.dumps(
            {
                "api_key": api_key,
                "token": token,
                "prompt": prompt,
                "args": codex,
                "timeout": timeout + 30,
            }
        )
        with (output / "codex.jsonl").open("xb") as trace:
            subprocess.run(
                ssh(name, "sudo", "python3", "-c", GUEST_RUNNER),
                input=payload,
                text=True,
                stdout=trace,
                check=True,
                timeout=timeout + 45,
                preexec_fn=limit_output,
            )
    finally:
        stop_process(tunnel)
        _stop_mcp(mcp, output / "rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "setup", "login", "stop"):
        sub.add_parser(action)
    run_parser = sub.add_parser("run")
    prompts = run_parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    run_parser.add_argument("--model")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--timeout", type=int, default=300)
    run_parser.add_argument("--frames", type=int)
    run_parser.add_argument("--max-frames", type=int, default=30)
    run_parser.add_argument("--lite", action="store_true")
    args = parser.parse_args()
    if not all(c.isalnum() or c in "_-" for c in args.name):
        parser.error("invalid VM name")
    if args.action == "start":
        start(args.name)
    elif args.action == "setup":
        setup(args.name)
    elif args.action == "login":
        login(args.name)
    elif args.action == "stop":
        command(["stop", args.name])
    else:
        if (
            args.timeout <= 0
            or args.frames is not None
            and args.frames <= 0
            or args.max_frames <= 0
        ):
            parser.error("timeout, frames, and max-frames must be positive")
        prompt = (
            args.prompt if args.prompt is not None else args.prompt_file.read_text()
        )
        run(
            args.name,
            prompt,
            args.model,
            args.output,
            args.timeout,
            args.lite,
            args.frames,
            args.max_frames,
        )


if __name__ == "__main__":
    main()
