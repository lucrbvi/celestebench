"""A small asynchronous driver for Open8 policies."""

import asyncio
import inspect
import json
import time
from contextlib import suppress
from operator import index
from pathlib import Path

import av

from .open8 import Open8


def _png(frame) -> bytes:
    """Encode an observation losslessly using the existing video dependency."""
    codec = av.CodecContext.create("png", "w")
    codec.height, codec.width = frame.shape[:2]
    codec.pix_fmt = "rgba"
    packets = codec.encode(av.VideoFrame.from_ndarray(frame, format="rgba"))
    return b"".join(bytes(packet) for packet in packets + codec.encode(None))


def _int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


async def rollout(policy, output: str | Path, *, decisions: int = 100,
                  max_frames: int = 30, max_actions: int = 1,
                  frames: int | None = None, fps: float | None = None) -> dict:
    """Run ``policy``, which returns (buttons, frames) or a list of those,
    and save a video, action log, and final checkpoint. ``frames`` optionally
    caps the total environment frames played (long holds are cut). With ``fps``
    the environment runs in real time: frames tick at that rate with buttons
    released while the policy thinks, so a slow model wastes world time."""
    decisions = _int(decisions, "decisions")
    max_frames = _int(max_frames, "max_frames")
    max_actions = _int(max_actions, "max_actions")
    if frames is not None:
        frames = _int(frames, "frames")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    if decisions < 0 or max_frames < 1 or max_actions < 1 or (frames is not None and frames < 1):
        raise ValueError("decisions must be non-negative; max_frames and max_actions positive")
    period = None if fps is None else 1.0 / fps

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    video = directory / "rollout.mp4"
    actions = directory / "actions.jsonl"
    checkpoint = directory / "checkpoint.state"
    screenshots = directory / "screenshots"
    screenshots.mkdir()
    started = time.perf_counter()
    stepped = 0
    applied = 0
    calls = 0
    with Open8() as env:
        try:
            with (env.record(video), actions.open("w", encoding="utf-8") as history,
                  (directory / "decisions.jsonl").open("w", encoding="utf-8") as outcomes):
                frame = env.step(frames=1)
                stepped = 1
                next_tick = time.perf_counter()

                async def tick(buttons):
                    nonlocal next_tick, frame, stepped
                    now = time.perf_counter()
                    next_tick = max(next_tick, now)
                    delay = next_tick - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    next_tick += period
                    frame = env.step(buttons, 1)
                    stepped += 1

                for decision in range(decisions):
                    if frames is not None and stepped >= frames:
                        break
                    begun = time.perf_counter()

                    observation = frame
                    frame_start = stepped
                    screenshot = screenshots / f"{decision:06d}.png"
                    screenshot.write_bytes(_png(observation))
                    screenshot_ref = screenshot.relative_to(directory).as_posix()
                    status, error, task = "error", None, None
                    policy_latency = None
                    try:
                        async def call():
                            result = policy(observation)
                            if inspect.isawaitable(result):
                                result = await result
                            return result

                        task = asyncio.create_task(call())
                        await asyncio.sleep(0)
                        if period is None:
                            result = await task
                        else:
                            # Real time: the world keeps moving while the model thinks.
                            result = None
                            while not task.done():
                                if frames is not None and stepped >= frames:
                                    break
                                await tick(0)
                            if task.done():
                                result = await task
                            else:
                                task.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await task
                                status = "frame_limit"
                                break
                        policy_latency = latency = time.perf_counter() - begun
                        if not isinstance(result, (list, tuple)):
                            raise TypeError("policy must return (buttons, frames) or a list of those")
                        if len(result) == 2 and not isinstance(result[0], (list, tuple)):
                            result = [result]
                        if not 1 <= len(result) <= max_actions:
                            raise ValueError("policy must return 1..max_actions actions")
                        batch = []
                        for action in result:
                            try:
                                buttons, held = action
                            except (TypeError, ValueError) as exc:
                                raise TypeError("each action must be (buttons, frames)") from exc
                            buttons = _int(buttons, "buttons")
                            held = _int(held, "frames")
                            if not 0 <= buttons <= 63 or not 1 <= held <= max_frames:
                                raise ValueError("buttons must be 0..63 and frames 1..max_frames")
                            batch.append((buttons, held))
                        # Reject the entire decision before advancing the emulator.
                        action_phase_start = stepped
                        for buttons, held in batch:
                            if frames is not None:
                                held = min(held, frames - stepped)
                                if held < 1:
                                    break
                            action_start = stepped
                            try:
                                if period is None:
                                    frame = env.step(buttons, held)
                                    stepped += held
                                else:
                                    for _ in range(held):
                                        await tick(buttons)
                            finally:
                                executed = stepped - action_start
                                if executed:
                                    applied += 1
                                    history.write(json.dumps({
                                        "decision": decision, "buttons": buttons,
                                        "frames": executed, "latency": latency,
                                        "frame_start": action_start, "frame_end": stepped,
                                    }) + "\n")
                                    latency = 0  # Count policy time only once.
                        calls += 1
                        status = "played" if stepped - action_phase_start == sum(n for _, n in batch) else "frame_limit"
                    except TimeoutError as exc:
                        status, error = "timeout", str(exc) or "Policy timed out"
                        raise
                    except asyncio.CancelledError:
                        status, error = "cancelled", "Run cancelled"
                        raise
                    except Exception as exc:
                        error = str(exc)
                        raise
                    finally:
                        if task is not None and not task.done():
                            task.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await task
                        outcomes.write(json.dumps({
                            "decision": decision, "status": status, "error": error,
                            "frame_start": frame_start, "frame_end": stepped,
                            "screenshot": screenshot_ref,
                            "latency": policy_latency if policy_latency is not None else time.perf_counter() - begun,
                        }) + "\n")
                        outcomes.flush()
                        history.flush()

        finally:
            checkpoint.write_bytes(env.save_state())
    return {"frames": stepped, "decisions": calls, "actions": applied,
            "elapsed": time.perf_counter() - started}
