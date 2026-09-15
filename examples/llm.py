"""uv run --extra llm python examples/llm.py --timeout 120"""

import argparse
import asyncio
import io
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tau_coding.models_dev_store import (
    ModelsDevRefreshError,
    refresh_models_dev_catalog,
)

from celestebench import BENCHMARK_VERSION, providers
from celestebench.llm import TauPolicy
from celestebench.modes import mode_of
from celestebench.rollout import rollout


async def main():
    parser = argparse.ArgumentParser(description="Run a bounded Celeste vision policy using Tau.")
    parser.add_argument("--model", default="muse-spark-1.3-contributor")
    parser.add_argument("--provider", default="opencode-go",
                        help="Tau catalog provider serving the model (opencode-go, anthropic, google, …)")
    parser.add_argument("--base-url", help="override the provider endpoint (custom gateways)")
    parser.add_argument("--api-key",
                        help="API key; default: CELESTEBENCH_API_KEY, then the provider's own env var")
    parser.add_argument("--thinking-level", default="low",
                        help="off, minimal, low, medium, high, xhigh or max; Tau maps it to each API")
    parser.add_argument("--timeout", type=float, default=120,
                        help="wall-clock budget for the whole rollout, in seconds")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--frames", type=int, help="cap on total environment frames played")
    parser.add_argument("--fps", type=float, help="run the environment in real time at this rate")
    parser.add_argument("--max-images", type=int, default=3,
                        help="cap on images kept in the model context (providers reject more)")
    parser.add_argument("--output", type=Path, help="fresh directory (default: runs/MODEL/TIMESTAMP)")
    args = parser.parse_args()
    if min(args.max_frames, args.timeout) <= 0:
        parser.error("budgets must be positive")
    try:
        await refresh_models_dev_catalog()  # fresh model data; the bundled snapshot is the fallback
    except ModelsDevRefreshError:
        pass
    key = args.api_key or os.environ.get("CELESTEBENCH_API_KEY")
    secret = key or os.environ.get(providers.key_env(args.provider), "")

    class Trace(io.TextIOWrapper):
        def write(self, text):
            # Provider errors can echo credentials; keep them out of saved traces.
            for needle in (secret, json.dumps(secret)[1:-1]) if secret else ():
                text = text.replace(needle, "[redacted]")
            return super().write(text)

    # Go requires a per-conversation session header and a real client identity.
    session = uuid.uuid4().hex
    provider = providers.create(
        args.provider, args.model, api_key=key or None, base_url=args.base_url,
        timeout_seconds=args.timeout, max_retries=0,
        headers={"x-opencode-session": session, "User-Agent": "celestebench/0.1.0"},
        thinking_level=args.thinking_level)
    if args.output is None:
        model_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.model)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
        args.output = Path("runs") / model_name / stamp
        suffix = 2
        while args.output.exists():
            args.output = Path("runs") / model_name / f"{stamp}-{suffix}"
            suffix += 1
    # The runner owns directory creation, so refuse overwrites before opening logs.
    policy = TauPolicy(provider, args.model, max_frames=args.max_frames,
                       max_images=args.max_images, fps=args.fps)
    try:
        # Open the trace only after rollout has created its output directory.
        async def decide(frames):
            if policy.trace is None:
                policy.trace = Trace((args.output / "messages.jsonl").open("xb"), encoding="utf-8")
                config = {name: value for name, value in vars(args).items() if name != "api_key"} | {
                    "output": str(args.output), "system": policy.system,
                    "mode": mode_of(args.fps), "benchmark_version": BENCHMARK_VERSION,
                    "system_prompt_sent": True, "tau_version": "0.4.1", "max_retries": 0,
                    "session": session}
                (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            return await policy(frames)

        result = await rollout(decide, args.output, timeout=args.timeout,
                               max_frames=args.max_frames,
                               frames=args.frames, fps=args.fps)
        print(json.dumps(result))
    finally:
        if policy.trace is not None:
            policy.trace.close()
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
