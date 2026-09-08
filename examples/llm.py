"""uv run --extra llm python examples/llm.py --decisions 5"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
import uuid
from pathlib import Path

from tau_ai import (
    AnthropicConfig, AnthropicProvider, OpenAICompatibleConfig, OpenAICompatibleProvider,
)

from celestebench.llm import SYSTEM, TauPolicy
from celestebench.rollout import rollout


async def main():
    parser = argparse.ArgumentParser(description="Run a bounded Celeste vision policy using Tau.")
    parser.add_argument("--model", default="muse-spark-1.3-contributor")
    parser.add_argument("--base-url", default="https://opencode.ai/zen/go/v1")
    parser.add_argument("--api", choices=["openai-responses", "openai-completions", "anthropic"],
                        default="openai-responses")
    parser.add_argument("--key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument("--decisions", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-actions", type=int, default=1)
    parser.add_argument("--frames", type=int, help="cap on total environment frames played")
    parser.add_argument("--fps", type=float, help="run the environment in real time at this rate")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", default="low",
                        help="Responses API reasoning effort (model dependent; none disables it)")
    parser.add_argument("--thinking-budget", type=int,
                        help="enable Anthropic thinking with this token budget")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--action-history", type=int, default=16,
                        help="previous conversation turns to send (0 disables history)")
    parser.add_argument("--image-history", type=int, default=3,
                        help="recent screenshots to send, including the current image")
    parser.add_argument("--reasoning-history", action=argparse.BooleanOptionalAction, default=True,
                        help="replay assistant text and available reasoning; always saved in traces")
    parser.add_argument("--output", type=Path, help="fresh directory (default: runs/MODEL/TIMESTAMP)")
    args = parser.parse_args()
    if min(args.decisions, args.max_frames, args.max_actions, args.max_tokens, args.timeout) <= 0:
        parser.error("budgets must be positive")
    if args.action_history < 0 or args.image_history < 1:
        parser.error("action-history must be non-negative and image-history positive")
    if args.thinking_budget is not None and (
            args.api != "anthropic" or not 1024 <= args.thinking_budget <= args.max_tokens - 1024):
        parser.error("thinking-budget requires anthropic and 1024..max-tokens-1024 tokens")
    key = os.environ.get(args.key_env)
    if not key:
        parser.error(f"set {args.key_env} before running")
    # Go requires a per-conversation session header and a real client identity.
    session = uuid.uuid4().hex
    if args.output is None:
        model_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.model)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        args.output = Path("runs") / model_name / stamp
        suffix = 2
        while args.output.exists():
            args.output = Path("runs") / model_name / f"{stamp}-{suffix}"
            suffix += 1
    headers = {"x-opencode-session": session, "User-Agent": "celestebench/0.1.0"}
    if args.api == "anthropic":
        provider = AnthropicProvider(AnthropicConfig(
            api_key=key, base_url=args.base_url, supports_images=True,
            max_tokens=args.max_tokens, timeout_seconds=args.timeout,
            headers=headers, thinking_mode="budget" if args.thinking_budget else "disabled",
            thinking_budget_tokens=args.thinking_budget, max_retries=0,
        ))
    else:
        provider = OpenAICompatibleProvider(OpenAICompatibleConfig(
            api_key=key, base_url=args.base_url, api=args.api, supports_images=True,
            max_tokens=args.max_tokens, timeout_seconds=args.timeout, max_retries=0,
            reasoning_effort=args.reasoning_effort if args.api == "openai-responses" else None,
            headers=headers,
        ))
    # The runner owns directory creation, so refuse overwrites before opening logs.
    policy = TauPolicy(provider, args.model, max_frames=args.max_frames,
                       max_actions=args.max_actions, timeout=args.timeout,
                       action_history=args.action_history, image_history=args.image_history,
                       reasoning_history=args.reasoning_history)
    try:
        # Open the trace only after rollout has created its output directory.
        async def decide(frame):
            if policy.trace is None:
                policy.trace = (args.output / "messages.jsonl").open("x")
                config = vars(args) | {"output": str(args.output), "system": SYSTEM,
                                       "tau_version": "0.4.1", "max_retries": 0,
                                       "session": session}
                (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            return await policy(frame)

        result = await rollout(decide, args.output, decisions=args.decisions,
                               max_frames=args.max_frames, max_actions=args.max_actions,
                               frames=args.frames, fps=args.fps)
        print(json.dumps(result))
    finally:
        if policy.trace is not None:
            policy.trace.close()
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
