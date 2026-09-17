"""Shared test helpers; import them as `from conftest import ...`."""

import asyncio
import functools
import importlib.util
import time
from pathlib import Path

import av
import numpy as np
import pytest


def load_example(name):
    """Import an examples/ script as a module; they live outside the package."""
    path = Path(__file__).parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asyncio_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def wait_until(predicate, timeout=5, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    pytest.fail("Condition not reached before timeout.")


def video(path, frames=3):
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width = stream.height = 512
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            image = np.zeros((512, 512, 3), np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        out.mux(stream.encode())
