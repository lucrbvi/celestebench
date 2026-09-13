"""Render a rollout into an annotated MP4: the gameplay, with a Game Boy
directional pad and A/B buttons lighting up underneath and the model name
tucked in a corner.

Usage: uv run python -m web.export <run> [-o out.mp4] [--limit frames]
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .viewer import RUNS, _jsonl, apply_nested

GAME = 512
BAR = 148
WIDTH, HEIGHT = GAME, GAME + BAR

BG = (12, 12, 16)
DPAD = (46, 50, 52)
DPAD_ON = (150, 156, 160)
PAD = (91, 53, 101)
PAD_ON = (196, 124, 214)
NAME = (110, 110, 125)

DPAD_CENTER = (143, 74)
DPAD_REACH, DPAD_HALF = 46, 18
A, B = (385, 56), (323, 92)
PAD_RADIUS = 30


def _font(size):
    return ImageFont.load_default(size=size)


def _bar(model, buttons):
    bar = Image.new("RGB", (WIDTH, BAR), BG)
    draw = ImageDraw.Draw(bar)

    cx, cy = DPAD_CENTER
    for box in ((cx - DPAD_REACH, cy - DPAD_HALF, cx + DPAD_REACH, cy + DPAD_HALF),
                (cx - DPAD_HALF, cy - DPAD_REACH, cx + DPAD_HALF, cy + DPAD_REACH)):
        draw.rounded_rectangle(box, radius=7, fill=DPAD)
    for mask, box in ((4, (cx - DPAD_HALF, cy - DPAD_REACH, cx + DPAD_HALF, cy)),
                      (8, (cx - DPAD_HALF, cy, cx + DPAD_HALF, cy + DPAD_REACH)),
                      (1, (cx - DPAD_REACH, cy - DPAD_HALF, cx, cy + DPAD_HALF)),
                      (2, (cx, cy - DPAD_HALF, cx + DPAD_REACH, cy + DPAD_HALF))):
        if buttons & mask:
            draw.rounded_rectangle(box, radius=7, fill=DPAD_ON)

    for (bx, by), mask in ((A, 16), (B, 32)):
        radius = PAD_RADIUS
        draw.ellipse((bx - radius, by - radius, bx + radius, by + radius),
                     fill=PAD_ON if buttons & mask else PAD)

    draw.text((14, 10), model, font=_font(15), fill=NAME)
    return bar


def _buttons(folder, total):
    buttons = bytearray(total)
    for row in _jsonl(folder / "actions.jsonl"):
        start, end = row.get("frame_start"), row.get("frame_end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        value = 0 if row.get("action") == "wait" else int(row.get("buttons") or 0)
        for frame in range(start, min(total, end)):
            buttons[frame - 1] = value
    return buttons


def _model(name, folder):
    """Prefer the wrapper config: Codex runs keep the model above rollout/."""
    for path in (folder / "config.json", RUNS / name / "config.json"):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(config, dict) and config.get("model"):
            return config["model"]
    return name


def export(run, output=None, limit=None):
    name = run.removeprefix("runs/").strip("/")
    folder = apply_nested(RUNS / name)
    if not (folder / "rollout.mp4").is_file():
        raise SystemExit(f"no rollout.mp4 under {folder}")
    output = Path(output) if output else folder / "export.mp4"
    model = _model(name, folder)
    temporary = output.with_name(output.name + ".tmp")

    container = av.open(str(folder / "rollout.mp4"))
    try:
        incoming = container.streams.video[0]
        fps = float(incoming.average_rate or 30)
        total = int(incoming.frames or 0)
        if not total:
            raise SystemExit("the recording reports no frames")
        buttons = _buttons(folder, total)
        bars = {value: _bar(model, value) for value in set(buttons)}
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        with av.open(str(temporary), "w", format="mp4") as out:
            stream = out.add_stream("libx264", rate=int(round(fps)))
            stream.width, stream.height = WIDTH, HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "veryfast"}
            for frame_index, frame in enumerate(container.decode(incoming)):
                if limit is not None and frame_index >= limit:
                    break
                canvas.paste(Image.fromarray(frame.to_ndarray(format="rgb24")), (0, 0))
                canvas.paste(bars[buttons[frame_index]], (0, GAME))
                out.mux(stream.encode(av.VideoFrame.from_ndarray(
                    np.asarray(canvas), format="rgb24")))
            out.mux(stream.encode())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        container.close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run path relative to runs/")
    parser.add_argument("-o", "--output", help="output MP4 (default: <run>/export.mp4)")
    parser.add_argument("--limit", type=int, help="only export the first N frames")
    args = parser.parse_args(argv)
    print(export(args.run, args.output, args.limit))


if __name__ == "__main__":
    main()
