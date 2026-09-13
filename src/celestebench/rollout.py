"""A small asynchronous driver for Open8 policies."""

import asyncio
import inspect
import json
import math
import sys
import time
from contextlib import suppress
from operator import index
from pathlib import Path

import av

from .open8 import Open8
from .scoring import Progress


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


async def rollout(policy, output: str | Path, *, timeout: float | None = None,
                  max_frames: int = 30, max_actions: int = 1,
                  frames: int | None = None, fps: float | None = None,
                  strict_timeout: bool = False) -> dict:
    """Run ``policy``, which receives every frame played since its previous call
    (oldest to newest, the last one current) and returns (buttons, frames),
    ("wait", frames), or a list of those, and save a video, action log, and
    final checkpoint. ``timeout`` is the
    wall-clock budget for the whole rollout: when it expires the pending
    policy call is cancelled, its decision is logged with status ``timeout``,
    and the run ends gracefully; a batch already being played finishes.
    ``frames`` optionally caps the total environment frames played (long holds
    are cut). With ``fps`` the environment runs in real time: frames tick at
    that rate with buttons released while the policy thinks, so a slow model
    wastes world time. ``strict_timeout`` also stops an in-progress action
    batch at the wall-clock deadline."""
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a positive number of seconds")
    max_frames = _int(max_frames, "max_frames")
    max_actions = _int(max_actions, "max_actions")
    if frames is not None:
        frames = _int(frames, "frames")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames < 1 or max_actions < 1 or (frames is not None and frames < 1):
        raise ValueError("max_frames and max_actions must be positive")
    period = None if fps is None else 1.0 / fps

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    video = directory / "rollout.mp4"
    actions = directory / "actions.jsonl"
    checkpoint = directory / "checkpoint.state"
    screenshots = directory / "screenshots"
    screenshots.mkdir()
    live = directory / "live.png"
    live_done = directory / "live.done"
    started = time.perf_counter()
    deadline = None if timeout is None else started + timeout
    stepped = 0
    applied = 0
    calls = 0
    progress = Progress()
    scoring_options = dict(timeout=timeout, fps=fps, max_frames=max_frames,
                           max_actions=max_actions, frames=frames, strict_timeout=strict_timeout)

    def save_score(status):
        score = {**progress.snapshot(), "elapsed": time.perf_counter() - started,
                 "frame": stepped, "timing": "wall_clock", "status": status,
                 "options": scoring_options}
        temporary = directory / "score.tmp"
        temporary.write_text(json.dumps(score) + "\n", encoding="utf-8")
        temporary.replace(directory / "score.json")

    with Open8() as env:
        try:
            with (env.record(video), actions.open("w", encoding="utf-8") as history,
                  (directory / "decisions.jsonl").open("w", encoding="utf-8") as outcomes,
                  (directory / "progress.jsonl").open("w", encoding="utf-8") as scores):
                scores.write(json.dumps({**progress.snapshot(), "frame": 0, "elapsed": 0.0}) + "\n")
                scores.flush()
                save_score("running")

                def publish(image):
                    temporary = live.with_suffix(".tmp")
                    temporary.write_bytes(_png(image))
                    temporary.replace(live)

                unseen = []

                def observe():
                    unseen.append(frame)
                    if progress.update(env.game_state):
                        scores.write(json.dumps({**progress.snapshot(), "frame": stepped,
                                                 "elapsed": time.perf_counter() - started}) + "\n")
                        scores.flush()
                        save_score("running")

                frame = env.step(frames=1)
                stepped = 1
                publish(frame)
                observe()
                next_tick = time.perf_counter()

                async def tick(buttons):
                    nonlocal next_tick, frame, stepped
                    now = time.perf_counter()
                    next_tick = max(next_tick, now)
                    delay = next_tick - now
                    if delay > 0:
                        if strict_timeout and deadline is not None:
                            delay = min(delay, max(0, deadline - now))
                        if delay:
                            await asyncio.sleep(delay)
                    if strict_timeout and deadline is not None and time.perf_counter() >= deadline:
                        return False
                    next_tick += period
                    frame = env.step(buttons, 1)
                    stepped += 1
                    publish(frame)
                    observe()
                    return True

                decision = 0
                while True:
                    if frames is not None and stepped >= frames:
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    begun = time.perf_counter()

                    # Hand over every frame played since the previous decision.
                    observation = tuple(unseen)
                    unseen.clear()
                    frame_start = stepped
                    screenshot = screenshots / f"{decision:06d}.png"
                    screenshot.write_bytes(_png(observation[-1]))
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
                            if deadline is None:
                                result = await task
                            else:
                                result = await asyncio.wait_for(task, deadline - time.perf_counter())
                        else:
                            # Real time: the world keeps moving while the model thinks.
                            result = None
                            while not task.done():
                                if frames is not None and stepped >= frames:
                                    break
                                if deadline is not None and time.perf_counter() >= deadline:
                                    break
                                await tick(0)
                            if task.done():
                                result = await task
                            else:
                                task.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await task
                                status = ("timeout" if deadline is not None
                                          and time.perf_counter() >= deadline else "frame_limit")
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
                            waiting = isinstance(buttons, str) and buttons == "wait"
                            buttons = 0 if waiting else _int(buttons, "buttons")
                            held = _int(held, "frames")
                            if not 0 <= buttons <= 63 or not 1 <= held <= max_frames:
                                raise ValueError("buttons must be 0..63 and frames 1..max_frames")
                            batch.append((buttons, held, waiting))
                        # Reject the entire decision before advancing the emulator.
                        action_phase_start = stepped
                        for buttons, held, waiting in batch:
                            if frames is not None:
                                held = min(held, frames - stepped)
                                if held < 1:
                                    break
                            action_start = stepped
                            try:
                                if period is None:
                                    for _ in range(held):
                                        if strict_timeout and deadline is not None and time.perf_counter() >= deadline:
                                            status = "timeout"
                                            break
                                        frame = env.step(buttons, 1)
                                        stepped += 1
                                        publish(frame)
                                        observe()
                                        if strict_timeout:
                                            await asyncio.sleep(0)
                                else:
                                    for _ in range(held):
                                        if not await tick(buttons):
                                            status = "timeout"
                                            break
                                if status == "timeout":
                                    break
                            finally:
                                executed = stepped - action_start
                                if executed:
                                    applied += 1
                                    record = {
                                        "decision": decision, "buttons": buttons,
                                        "frames": executed, "latency": latency,
                                        "frame_start": action_start, "frame_end": stepped,
                                    }
                                    if waiting:
                                        record["action"] = "wait"
                                    history.write(json.dumps(record) + "\n")
                                    latency = 0  # Count policy time only once.
                        calls += 1
                        if status != "timeout":
                            status = ("played" if stepped - action_phase_start == sum(n for _, n, _ in batch)
                                      else "frame_limit")
                        if status == "timeout":
                            break
                    except TimeoutError as exc:
                        status, error = "timeout", str(exc) or "Wall-clock timeout reached before the model answered."
                        break
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
                        decision += 1

        finally:
            failure = sys.exc_info()[1]
            save_score("cancelled" if isinstance(failure, asyncio.CancelledError)
                       else "error" if failure is not None else "completed")
            checkpoint.write_bytes(env.save_state())
            live_done.touch()
    return {"frames": stepped, "decisions": calls, "actions": applied,
            "elapsed": time.perf_counter() - started}
