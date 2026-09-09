"""uv run --extra llm python examples/llm.py --timeout 120"""

import argparse
import asyncio
import io
import json
import os
import re
from datetime import datetime, timezone
import uuid
from pathlib import Path

from tau_ai import (
    AnthropicConfig, AnthropicProvider, GoogleGenerativeAIProvider,
    MistralConversationsProvider, OpenAICompatibleConfig, OpenAICompatibleProvider,
)

from celestebench.llm import SYSTEM, TauPolicy
from celestebench.rollout import rollout


async def main():
    parser = argparse.ArgumentParser(description="Run a bounded Celeste vision policy using Tau.")
    parser.add_argument("--model", default="muse-spark-1.3-contributor")
    parser.add_argument("--base-url", default="https://opencode.ai/zen/go/v1")
    parser.add_argument("--api", choices=["openai-responses", "openai-completions", "anthropic",
                                           "google-generative-ai", "mistral-conversations"],
                        default="openai-responses")
    parser.add_argument("--key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument("--timeout", type=float, default=120,
                        help="wall-clock budget for the whole rollout, in seconds")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--frames", type=int, help="cap on total environment frames played")
    parser.add_argument("--fps", type=float, help="run the environment in real time at this rate")
    parser.add_argument("--max-images", type=int, default=3,
                        help="cap on images kept in the model context (providers reject more)")
    parser.add_argument("--reasoning-effort", default="low",
                        help="Responses API reasoning effort (model dependent; none disables it)")
    parser.add_argument("--thinking-budget", type=int,
                        help="enable Anthropic thinking with this token budget")
    parser.add_argument("--output", type=Path, help="fresh directory (default: runs/MODEL/TIMESTAMP)")
    args = parser.parse_args()
    if min(args.max_frames, args.max_actions, args.timeout) <= 0:
        parser.error("budgets must be positive")
    if args.thinking_budget is not None and args.api != "anthropic":
        parser.error("thinking-budget requires anthropic")
    # Anthropic only exposes reasoning when thinking is enabled, so default it
    # on (0 explicitly disables) to preserve the CoT in the trace and context.
    budget = args.thinking_budget
    if args.api == "anthropic" and budget is None:
        budget = 2048
    key = os.environ.get(args.key_env)
    if not key:
        parser.error(f"set {args.key_env} before running")

    class Trace(io.TextIOWrapper):
        def write(self, text):
            # Provider errors can echo credentials; keep them out of saved traces.
            return super().write(text.replace(key, "[redacted]").replace(
                json.dumps(key)[1:-1], "[redacted]"))
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
            timeout_seconds=args.timeout,
            headers=headers, thinking_mode="budget" if budget else "disabled",
            thinking_budget_tokens=budget or None, max_retries=0,
        ))
    else:
        config = OpenAICompatibleConfig(
            api_key=key, base_url=args.base_url, api=args.api, supports_images=True,
            timeout_seconds=args.timeout, max_retries=0,
            reasoning_effort=args.reasoning_effort if args.api == "openai-responses" else None,
            headers=headers,
        )
        provider_type = {"google-generative-ai": GoogleGenerativeAIProvider,
                         "mistral-conversations": MistralConversationsProvider}.get(args.api,
                                                                                       OpenAICompatibleProvider)
        provider = provider_type(config)
    # The runner owns directory creation, so refuse overwrites before opening logs.
    policy = TauPolicy(provider, args.model, max_frames=args.max_frames, max_actions=args.max_actions,
                       max_images=args.max_images)
    try:
        # Open the trace only after rollout has created its output directory.
        async def decide(frames):
            if policy.trace is None:
                policy.trace = Trace((args.output / "messages.jsonl").open("xb"), encoding="utf-8")
                config = vars(args) | {"output": str(args.output), "system": SYSTEM,
                                       "tau_version": "0.4.1", "max_retries": 0,
                                       "session": session}
                (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            return await policy(frames)

        result = await rollout(decide, args.output, timeout=args.timeout,
                               max_frames=args.max_frames, max_actions=args.max_actions,
                               frames=args.frames, fps=args.fps)
        print(json.dumps(result))
    finally:
        if policy.trace is not None:
            policy.trace.close()
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
