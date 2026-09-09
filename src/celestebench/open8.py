import ctypes as C
from contextlib import contextmanager
from enum import IntFlag
from itertools import groupby
from operator import index
from pathlib import Path
import sys

import av
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
UPSCALE = 4


class Button(IntFlag):
    LEFT = 1
    RIGHT = 2
    UP = 4
    DOWN = 8
    O = 16
    X = 32


class Open8:
    """One synchronous emulator per process. Nothing runs between step calls."""

    def __init__(self, cart: str | Path = _ROOT / "deps/open8/export/carts/1CELESTE.PNG"):
        self.cart = Path(cart).resolve()
        if not self.cart.is_file():
            raise FileNotFoundError(self.cart)
        suffix = "dylib" if sys.platform == "darwin" else "so"
        self._lib = C.CDLL(str(_ROOT / "build" / f"libopen8env.{suffix}"))
        for name, args, result in (
            ("init", [], C.c_int),
            ("quit", [], None),
            ("load_cart", [C.c_char_p], C.c_int),
            ("step", [C.c_uint32, C.c_uint8], C.c_int),
            ("frame_ms", [], C.c_uint32),
            ("framebuffer", [C.POINTER(C.c_uint8)], None),
            ("audio_samples", [], C.c_uint32),
            ("audio", [C.POINTER(C.c_int16)], None),
        ):
            fn = getattr(self._lib, f"shim_{name}")
            fn.argtypes, fn.restype = args, result
        if self._lib.shim_init() != 0:
            raise RuntimeError("Open8 initialization failed (only one environment per process)")
        self._history = bytearray()
        self.audio = np.empty(0, dtype=np.int16)
        self._video = None
        try:
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self) -> np.ndarray:
        """Restart the cartridge from its fixed initial seed."""
        if self._lib is None:
            raise RuntimeError("environment is closed")
        if self._lib.shim_load_cart(str(self.cart).encode()) != 0:
            raise RuntimeError(f"could not load {self.cart}")
        self._history.clear()
        self.audio = np.empty(0, dtype=np.int16)
        return self.framebuffer

    def _pull_audio(self) -> np.ndarray:
        samples = self._lib.shim_audio_samples()
        audio = np.empty(samples, dtype=np.int16)
        if samples:
            self._lib.shim_audio(audio.ctypes.data_as(C.POINTER(C.c_int16)))
        return audio

    @property
    def framebuffer(self) -> np.ndarray:
        if self._lib is None:
            raise RuntimeError("environment is closed")
        frame = np.empty((128, 128, 4), dtype=np.uint8)
        self._lib.shim_framebuffer(frame.ctypes.data_as(C.POINTER(C.c_uint8)))
        return frame

    def step(self, buttons: int = 0, frames: int = 1) -> np.ndarray:
        """Hold a button mask for N frames; return the final RGBA image."""
        buttons, frames = index(buttons), index(frames)
        if not 0 <= buttons <= 63 or not 0 <= frames <= 2**31 - 1:
            raise ValueError("buttons must be 0..63 and frames 0..2**31-1")
        if self._lib is None:
            raise RuntimeError("environment is closed")
        if self._video is None:
            if self._lib.shim_step(frames, buttons) != frames:
                raise RuntimeError("Open8 failed to advance")
            self._history.extend(bytes([buttons]) * frames)
            self.audio = self._pull_audio()
        else:
            output, stream, audio_stream = self._video
            chunks = []
            for _ in range(frames):
                if self._lib.shim_step(1, buttons) != 1:
                    raise RuntimeError("Open8 failed to advance")
                self._history.append(buttons)
                big = np.repeat(np.repeat(self.framebuffer, UPSCALE, 0), UPSCALE, 1)
                output.mux(stream.encode(av.VideoFrame.from_ndarray(big, format="rgba")))
                audio = self._pull_audio()
                chunks.append(audio)
                if audio.size:
                    frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
                    frame.sample_rate = 22050
                    output.mux(audio_stream.encode(frame))
            self.audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
        return self.framebuffer

    def save_state(self) -> bytes:
        """Return the input history. Restore by replaying on the same cartridge."""
        if self._lib is None:
            raise RuntimeError("environment is closed")
        return bytes(self._history)

    def load_state(self, state: bytes) -> np.ndarray:
        """Replay a checkpoint; replay frames are excluded from recording."""
        if not isinstance(state, bytes) or any(button > 63 for button in state):
            raise ValueError("state must be bytes containing button masks 0..63")
        self.reset()
        self.audio = np.empty(0, dtype=np.int16)
        for buttons, group in groupby(state):
            count = sum(1 for _ in group)
            if self._lib.shim_step(count, buttons) != count:
                raise RuntimeError("Open8 replay failed")
            self._history.extend(bytes([buttons]) * count)
        self._pull_audio()  # Replaying a checkpoint must not leak into a recording.
        return self.framebuffer

    @contextmanager
    def record(self, path: str | Path):
        """Write every stepped frame to an MP4 with the cartridge audio.
        Frames are upscaled 4x with nearest-neighbor so playback stays crisp."""
        if self._lib is None or self._video is not None:
            raise RuntimeError("environment is closed or already recording")
        with av.open(str(path), "w", format="mp4") as output:
            stream = output.add_stream("libx264", rate=60 if self._lib.shim_frame_ms() == 16 else 30)
            stream.width = stream.height = UPSCALE * 128
            stream.pix_fmt = "yuv444p"
            stream.options = {"qp": "0"}  # Lossless: pixel art survives 4:4:4 fine.
            audio_stream = output.add_stream("aac", rate=22050)
            audio_stream.layout = "mono"
            self._video = output, stream, audio_stream
            try:
                yield self
            finally:
                self._video = None
                output.mux(stream.encode())
                output.mux(audio_stream.encode())

    def close(self) -> None:
        if self._lib is not None:
            self._lib.shim_quit()
            self._lib = None

    def __enter__(self) -> "Open8":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
