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
    TextContent, ToolCall, TurnEndEvent, UserMessage,
)

SYSTEM = """Play Celeste Classic. Climb upward and don't die. Call play exactly once with your next actions, in order. Each action holds a button bitmask for a number of frames: LEFT=1, RIGHT=2, UP=4, DOWN=8, O=16 (jump), X=32 (dash). Combine buttons by adding their values; 0 releases all buttons. The game runs at 30 fps. Observations are ordered oldest to newest; the last image is current. Use past images and actions to infer movement and learn from mistakes. In real-time mode the game keeps moving with buttons released while you think.
"""


async def _select_action(call_id, arguments, signal=None, on_update=None):
    # Only select here. The shared rollout validates and applies the actions.
    return AgentToolResult(content="Actions selected.")


def _parse_play(calls, max_actions: int, max_frames: int) -> tuple[tuple[int, int], ...]:
    if len(calls) != 1 or calls[0].name != "play":
        raise ValueError("model must call play exactly once")
    arguments = calls[0].arguments
    if set(arguments) != {"actions"}:
        raise ValueError("play requires exactly actions")
    actions = arguments["actions"]
    if type(actions) is not list or not 1 <= len(actions) <= max_actions:
        raise ValueError(f"play requires 1..{max_actions} actions")
    for action in actions:
        if type(action) is not dict or set(action) != {"buttons", "frames"}:
            raise ValueError("each action requires exactly buttons and frames")
        buttons, frames = action["buttons"], action["frames"]
        if (type(buttons) is not int or type(frames) is not int
                or not 0 <= buttons <= 63 or not 1 <= frames <= max_frames):
            raise ValueError("model action is outside the allowed button/frame bounds")
    return tuple((a["buttons"], a["frames"]) for a in actions)


class TauPolicy:
    """One API turn per observation, with bounded native conversation history.

    Keep complete turns and provider reasoning blocks, including signatures.
    Image history includes the current image; traces always retain everything.
    """

    def __init__(self, provider, model: str, *, max_frames=30, max_actions=1,
                 timeout=60, trace=None, system=SYSTEM, action_history=16,
                 image_history=3, reasoning_history=True):
        if (type(max_frames) is not int or max_frames < 1 or type(max_actions) is not int
                or max_actions < 1 or timeout <= 0):
            raise ValueError("max_frames, max_actions and timeout must be positive")
        if type(action_history) is not int or action_history < 0:
            raise ValueError("action_history must be a non-negative integer")
        if type(image_history) is not int or image_history < 1:
            raise ValueError("image_history must be a positive integer")
        self.provider, self.model = provider, model
        self.max_frames, self.max_actions, self.timeout = max_frames, max_actions, timeout
        self.trace, self.system = trace, system
        self.window, self._step = action_history, 0
        self.image_history, self.reasoning_history = image_history, reasoning_history
        self._history: list[tuple] = []
        self._feedback: str | None = None
        self.tool = AgentTool(
            name="play", label="Play actions", description="Play the next game actions.",
            parameters={
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array", "minItems": 1, "maxItems": max_actions,
                        "items": {
                            "type": "object",
                            "properties": {
                                "buttons": {"type": "integer", "minimum": 0, "maximum": 63},
                                "frames": {"type": "integer", "minimum": 1, "maximum": max_frames},
                            },
                            "required": ["buttons", "frames"], "additionalProperties": False,
                        },
                    }
                },
                "required": ["actions"], "additionalProperties": False,
            },
            execute_fn=_select_action,
        )

    async def __call__(self, frame) -> tuple[tuple[int, int], ...]:
        png = BytesIO()
        Image.fromarray(frame).save(png, format="PNG")
        prompt = f"Decision {self._step}: choose the next actions using play."
        if self._step == 0:
            prompt += "\nYou have not acted yet."
        elif self._feedback:
            prompt += "\nYour previous call was rejected: " + self._feedback + "; buttons were released for one frame."
        self._feedback = None
        observation = UserMessage(content=[
            TextContent(text=prompt),
            ImageContent(data=base64.b64encode(png.getvalue()).decode(), mime_type="image/png"),
        ])
        context = []
        for i, turn in enumerate(self._history):
            for message in turn:
                if isinstance(message, UserMessage) and i < len(self._history) - self.image_history + 1:
                    message = message.model_copy(update={"content": [
                        block for block in message.content if not isinstance(block, ImageContent)]})
                if isinstance(message, AssistantMessage) and not self.reasoning_history:
                    message = message.model_copy(update={"content": [
                        block for block in message.content if isinstance(block, ToolCall)]})
                context.append(message)
        harness = AgentHarness(AgentHarnessConfig(
            provider=self.provider, model=self.model, system=self.system,
            tools=[self.tool], max_turns=1,
        ), messages=context)
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
            async with asyncio.timeout(self.timeout), aclosing(harness.prompt_message(observation)) as events:
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
                        if message.stop_reason in {"error", "aborted", "length"}:
                            raise RuntimeError(message.error_message or f"Model stopped: {message.stop_reason}")
                        self._step += 1
                        if self.window:
                            self._history.append((observation, message, *event.tool_results))
                            self._history = self._history[-self.window:]
                        try:
                            actions = _parse_play(message.tool_calls, self.max_actions, self.max_frames)
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
