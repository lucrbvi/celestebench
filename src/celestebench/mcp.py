"""Bounded MCP access to one CelesteBench rollout."""

import argparse
import asyncio
import base64
import hmac
import json
import math
import os
import sys
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from .prompt import system_prompt
from .rollout import _png, rollout


def _actions(value, max_frames):
    if type(value) is not list or not value:
        raise ValueError("actions must be a non-empty list")
    parsed = []
    for action in value:
        if type(action) is not dict or set(action) not in ({"buttons", "frames"}, {"action", "frames"}):
            raise ValueError("each action must contain exactly buttons and frames, or action and frames")
        frames = action["frames"]
        if type(frames) is not int or not 1 <= frames <= max_frames:
            raise ValueError(f"frames must be an integer from 1 to {max_frames}")
        if "action" in action:
            if action["action"] != "wait":
                raise ValueError("action must be wait")
            parsed.append(("wait", frames))
        else:
            buttons = action["buttons"]
            if type(buttons) is not int or not 0 <= buttons <= 63:
                raise ValueError("buttons must be an integer from 0 to 63")
            parsed.append((buttons, frames))
    return tuple(parsed)


class Episode:
    def __init__(self, output, *, timeout=300, frames=None, fps=30, max_frames=30, max_images=3,
                 runner=rollout):
        self.output = Path(output)
        self.options = dict(timeout=timeout, frames=frames, fps=fps, max_frames=max_frames,
                            max_images=max_images)
        self.runner = runner
        self._arrivals = asyncio.Queue()
        self._current = None
        self._task = None
        self._lock = asyncio.Lock()
        self._result = None
        self._seen_frames = 0
        self._arrival = None
        self._configured = False

    async def _policy(self, frames):
        if not self._configured:
            (self.output / "config.json").write_text(
                json.dumps(self.options, separators=(",", ":")) + "\n", encoding="utf-8")
            self._configured = True
        loop = asyncio.get_running_loop()
        frames = tuple(frames)
        first = self._seen_frames + 1
        self._seen_frames += len(frames)
        indexes = range(len(frames))
        if len(frames) > self.options["max_images"]:
            limit = self.options["max_images"]
            indexes = [int(i * len(frames) / limit) for i in range(limit)]
            indexes[-1] = len(frames) - 1
        decision = (tuple((first + i, frames[i]) for i in indexes), loop.create_future())
        await self._arrivals.put(decision)
        return await decision[1]

    def _start(self):
        if self._task is not None:
            return
        options = {key: self.options[key] for key in ("timeout", "frames", "fps", "max_frames")}
        self._task = asyncio.create_task(self.runner(
            self._policy, self.output, max_actions=sys.maxsize, strict_timeout=True, **options))
        self._task.add_done_callback(self._finished)

    def _finished(self, task):
        if task.cancelled():
            self._result = {"status": "cancelled"}
        elif task.exception() is not None:
            self._result = {"status": "error", "error": str(task.exception())}
        else:
            self._result = {"status": "finished", **task.result()}

    async def _next(self):
        self._start()
        if self._task.done():
            await self._terminal()
            return None
        if self._current is not None:
            return self._current
        if self._arrival is None:
            self._arrival = asyncio.create_task(self._arrivals.get())
        done, _ = await asyncio.wait((self._arrival, self._task),
                                    return_when=asyncio.FIRST_COMPLETED)
        if self._task in done:
            await self._terminal()
            return None
        if self._arrival in done:
            self._current = self._arrival.result()
            self._arrival = None
            return self._current

    async def _terminal(self):
        with suppress(asyncio.CancelledError, Exception):
            await self._task
        if self._arrival is not None:
            self._arrival.cancel()
            with suppress(asyncio.CancelledError):
                await self._arrival
            self._arrival = None
        self._current = None

    def _content(self, frames=None):
        if frames is None:
            metadata = self._result or {"status": "finished"}
            content = [TextContent(type="text", text=json.dumps(metadata, separators=(",", ":")))]
            frames = ()
        else:
            content = [TextContent(type="text", text=json.dumps(
                {"status": "ready", "frame_ids": [number for number, _ in frames]},
                separators=(",", ":")))]
        content += [ImageContent(type="image", data=base64.b64encode(_png(frame)).decode(),
                                 mimeType="image/png") for _, frame in frames]
        self._log(content)
        return content

    def _log(self, content):
        if not self.output.is_dir():
            return
        with (self.output / "mcp.jsonl").open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"role": "user", "content": [
                item.model_dump(by_alias=True) for item in content]}, separators=(",", ":")) + "\n")

    def _log_actions(self, actions):
        if self.output.is_dir():
            with (self.output / "mcp.jsonl").open("a", encoding="utf-8") as trace:
                trace.write(json.dumps({"role": "assistant", "tool": "play", "actions": [
                    {"action": "wait", "frames": frames} if buttons == "wait" else
                    {"buttons": buttons, "frames": frames} for buttons, frames in actions
                ]}, separators=(",", ":")) + "\n")

    async def observe(self):
        if self._lock.locked():
            raise RuntimeError("another observe or play call is already in progress")
        async with self._lock:
            decision = await self._next()
            return self._content(None if decision is None else decision[0])

    async def play(self, actions):
        actions = _actions(actions, self.options["max_frames"])
        if self._lock.locked():
            raise RuntimeError("another observe or play call is already in progress")
        async with self._lock:
            decision = await self._next()
            if decision is None:
                raise RuntimeError("episode has ended and cannot be restarted")
            future = decision[1]
            if future.done():
                raise RuntimeError("the current decision was already submitted")
            self._log_actions(actions)
            future.set_result(actions)
            self._current = None
            decision = await self._next()
            return self._content(None if decision is None else decision[0])

    async def close(self):
        if self._arrival is not None:
            self._arrival.cancel()
            with suppress(asyncio.CancelledError):
                await self._arrival
            self._arrival = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task


def server(episode):
    max_frames = episode.options["max_frames"]
    instructions = system_prompt(fps=episode.options["fps"],
                                 max_frames=max_frames,
                                 max_images=episode.options["max_images"],
                                 mcp=True)
    app = FastMCP("CelesteBench", instructions=instructions, stateless_http=True,
                  json_response=True, max_request_body_size=1024 * 1024)

    class ButtonAction(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        buttons: int = Field(ge=0, le=63, description="Button bitmask")
        frames: int = Field(ge=1, le=max_frames, description="Frames to hold the buttons")

    class WaitAction(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        action: str = Field(pattern="^wait$", description="Must be 'wait'")
        frames: int = Field(ge=1, le=max_frames,
                            description="Frames to advance with neutral input")

    @app.tool(description="Observe the current game decision without advancing it.")
    async def observe() -> list[TextContent | ImageContent]:
        return await episode.observe()

    @app.tool(description="Apply a non-empty ordered batch. Each item holds buttons or waits for 1 through the server-configured maximum frames, then returns the next observation.")
    async def play(actions: list[ButtonAction | WaitAction]) -> list[TextContent | ImageContent]:
        return await episode.play([action.model_dump() for action in actions])

    return app


class BearerAuth:
    def __init__(self, app, token):
        self.app, self.expected = app, b"Bearer " + token.encode()

    async def __call__(self, scope, receive, send):
        if (scope["type"] == "http" and not hmac.compare_digest(
                dict(scope["headers"]).get(b"authorization", b""), self.expected)):
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await self.app(scope, receive, send)


def _parser():
    parser = argparse.ArgumentParser(description="Serve one bounded CelesteBench episode over MCP.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--port", type=int, default=8124)
    return parser


async def _main(args):
    if (not math.isfinite(args.timeout) or args.timeout <= 0
            or args.frames is not None and args.frames < 1):
        raise SystemExit("--timeout and --frames must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 1 or args.max_images < 1:
        raise SystemExit("--fps, --max-frames, and --max-images must be positive")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be from 1 to 65535")
    episode = Episode(args.output, timeout=args.timeout, frames=args.frames,
                      fps=None if args.lite else args.fps,
                      max_frames=args.max_frames, max_images=args.max_images)
    app = server(episode)
    try:
        if args.transport == "stdio":
            await app.run_stdio_async()
        else:
            token = os.environ.get("CELESTEBENCH_MCP_TOKEN")
            if not token:
                raise SystemExit("CELESTEBENCH_MCP_TOKEN is required for HTTP")
            import uvicorn
            await uvicorn.Server(uvicorn.Config(
                BearerAuth(app.streamable_http_app(), token), host="127.0.0.1",
                port=args.port, log_level="warning")).serve()
    finally:
        await episode.close()


def main():
    asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    main()
