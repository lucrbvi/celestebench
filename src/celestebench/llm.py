"""A vision baseline using Tau's agent core; install celestebench[llm]."""

import asyncio
import base64
import json
from contextlib import aclosing
from io import BytesIO

from PIL import Image
from tau_agent import (
    AgentHarness, AgentHarnessConfig, AgentTool, AgentToolResult,
    AssistantMessage, ImageContent, MessageEndEvent, MessageUpdateEvent,
    TextContent, TurnEndEvent, UserMessage,
)

from .prompt import system_prompt

# Kept as the default export: the RTC flavor with the standard budgets.
SYSTEM = system_prompt(fps=30)


async def _select_action(call_id, arguments, signal=None, on_update=None):
    # Only select here. The shared rollout validates and applies the actions.
    return AgentToolResult(content="Actions selected.")


def _parse_play(calls, max_frames: int) -> tuple[tuple[int | str, int], ...]:
    if len(calls) != 1 or calls[0].name != "play":
        raise ValueError("model must call play exactly once")
    arguments = calls[0].arguments
    if set(arguments) != {"actions"}:
        raise ValueError("play requires exactly actions")
    actions = arguments["actions"]
    if type(actions) is not list or not actions:
        raise ValueError("play requires at least one action")
    for action in actions:
        if type(action) is not dict or set(action) not in ({"buttons", "frames"}, {"action", "frames"}):
            raise ValueError("each action must press buttons or wait")
        waiting = action.get("action") == "wait"
        buttons, frames = ("wait", action["frames"]) if waiting else (action.get("buttons"), action["frames"])
        if (type(frames) is not int or not 1 <= frames <= max_frames
                or "action" in action and not waiting
                or not waiting and (type(buttons) is not int or not 0 <= buttons <= 63)):
            raise ValueError("model action is outside the allowed button/frame bounds")
    return tuple(("wait", a["frames"]) if a.get("action") == "wait" else (a["buttons"], a["frames"])
                 for a in actions)


class TauPolicy:
    """One persistent Tau harness per rollout, retaining its full conversation."""

    def __init__(self, provider, model: str, *, max_frames=30,
                 max_images=3, trace=None, system=None, fps=None):
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if type(max_images) is not int or max_images < 1:
            raise ValueError("max_images must be a positive integer")
        self.provider, self.model = provider, model
        self.max_frames = max_frames
        self.max_images = max_images
        self.trace = trace
        self.system = system if system is not None else system_prompt(
            fps=fps, max_frames=max_frames, max_images=max_images)
        self._step = 0
        self._feedback: str | None = None
        self.tool = AgentTool(
            name="play", label="Play actions", description="Play the next game actions.",
            parameters={
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array", "minItems": 1,
                        "items": {
                            "oneOf": [{
                                "type": "object",
                                "properties": {
                                    "buttons": {"type": "integer", "minimum": 0, "maximum": 63},
                                    "frames": {"type": "integer", "minimum": 1, "maximum": max_frames},
                                },
                                "required": ["buttons", "frames"], "additionalProperties": False,
                            }, {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["wait"]},
                                    "frames": {"type": "integer", "minimum": 1, "maximum": max_frames},
                                },
                                "required": ["action", "frames"], "additionalProperties": False,
                            }],
                        },
                    }
                },
                "required": ["actions"], "additionalProperties": False,
            },
            execute_fn=_select_action,
        )
        self.harness = AgentHarness(AgentHarnessConfig(
            provider=provider, model=model, system=self.system,
            tools=[self.tool], max_turns=1,
        ))

    def _evict_old_images(self, *, incoming: int) -> None:
        overflow = sum(isinstance(block, ImageContent) for message in self.harness._messages
                       if isinstance(message, UserMessage) and not isinstance(message.content, str)
                       for block in message.content) + incoming - self.max_images
        messages = self.harness._messages
        for index, message in enumerate(messages):
            if overflow <= 0:
                break
            if not isinstance(message, UserMessage) or isinstance(message.content, str):
                continue
            images = [block for block in message.content if isinstance(block, ImageContent)]
            drop = min(overflow, len(images))
            if drop:
                overflow -= drop
                messages[index] = message.model_copy(update={"content": [
                    block for block in message.content if not isinstance(block, ImageContent)]
                    + images[drop:]})

    async def __call__(self, frames) -> tuple[tuple[int, int], ...]:
        frames = list(frames)
        if not frames:
            raise ValueError("policy needs at least one frame")
        pngs = []
        for frame in frames:
            png = BytesIO()
            Image.fromarray(frame).save(png, format="PNG")
            pngs.append(base64.b64encode(png.getvalue()).decode())
        # Providers cap images per request (e.g. 600 on OpenAI-compatible Go).
        # Keep the newest frames, uniformly subsampling an oversized batch, and
        # evict the oldest images from earlier turns to stay within budget.
        if len(pngs) > self.max_images:
            stride = len(pngs) / self.max_images
            picked = [min(int(i * stride), len(pngs) - 1) for i in range(self.max_images)]
            picked[-1] = len(pngs) - 1
            pngs = [pngs[i] for i in picked]
        self._evict_old_images(incoming=len(pngs))
        prompt = f"Decision {self._step}: choose the next actions using play."
        if self._step == 0:
            prompt += "\nYou have not acted yet."
        elif self._feedback:
            prompt += "\nYour previous call was rejected: " + self._feedback + "; buttons were released for one frame."
        self._feedback = None
        observation = UserMessage(content=[TextContent(text=prompt)] + [
            ImageContent(data=data, mime_type="image/png") for data in pngs
        ])
        latest = None
        completed = None
        assistant_traced = False

        def write_message(message):
            nonlocal assistant_traced
            if self.trace is None or (message.role == "assistant" and assistant_traced):
                return
            self.trace.write(message.model_dump_json() + "\n")
            self.trace.flush()
            if message.role == "assistant":
                assistant_traced = True

        def write_interrupted(status, error):
            if self.trace is None or assistant_traced:
                return
            message = (latest if latest is not None else completed)
            if message is None:
                message = AssistantMessage(model=self.model)
            message = message.model_copy(update={"stop_reason": "aborted", "error_message": error})
            record = message.model_dump(by_alias=True)
            record["status"] = status
            self.trace.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.trace.flush()

        try:
            # The rollout's wall-clock budget cancels this call; there is no
            # per-decision timeout. Cancellation still traces partial reasoning.
            async with aclosing(self.harness.prompt_message(observation)) as events:
                async for event in events:
                    if isinstance(event, MessageUpdateEvent):
                        latest = event.message
                    if isinstance(event, MessageEndEvent):
                        if event.message.role == "assistant":
                            latest = completed = event.message
                        write_message(event.message)
                    if isinstance(event, TurnEndEvent):
                        message = event.message
                        write_message(completed if completed is not None else message)
                        if message.stop_reason == "length":
                            # Truncated before calling play (the token budget
                            # includes reasoning). BALROG-style fallback: log
                            # it, feed back, release the buttons and play on.
                            self._step += 1
                            self._feedback = ("your previous response hit the token limit "
                                              "before calling play; answer again and keep it short")
                            if self.trace is not None:
                                self.trace.write(json.dumps(
                                    {"status": "length", "step": self._step}, separators=(",", ":")) + "\n")
                                self.trace.flush()
                            return ((0, 1),)
                        if message.stop_reason in {"error", "aborted"}:
                            raise RuntimeError(message.error_message or f"Model stopped: {message.stop_reason}")
                        self._step += 1
                        try:
                            actions = _parse_play(message.tool_calls, self.max_frames)
                        except ValueError as error:
                            # BALROG-style fallback: feed back the invalidity,
                            # log it, release the buttons for one frame and play on.
                            self._feedback = error.args[0]
                            if self.trace is not None:
                                self.trace.write(json.dumps(
                                    {"status": "invalid_play", "step": self._step,
                                     "error": error.args[0]}, separators=(",", ":")) + "\n")
                                self.trace.flush()
                            return ((0, 1),)
                        return actions
        except asyncio.TimeoutError:
            write_interrupted("timeout", "Model response timed out; no complete decision was received.")
            raise
        except asyncio.CancelledError:
            write_interrupted("cancelled", "Model response was cancelled before a complete decision was received.")
            raise
        except Exception as error:
            write_interrupted("error", str(error) or "Model response failed before a complete decision was received.")
            raise
        raise RuntimeError("model returned no decision")
