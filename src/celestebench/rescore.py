"""Replay completed rollouts and write the current evaluator score.

The checkpoint is the source of truth here: it contains one button mask for
every frame, including frames during which the model was thinking.
"""

import argparse
import hashlib
import json
from pathlib import Path

import av

from .open8 import Open8
from .scoring import Progress


def _runs(root: Path):
    """Yield run directories in a stable order."""
    yield from sorted({path.parent for path in root.rglob("checkpoint.state")})


def _status(directory: Path) -> str | None:
    if not (directory / "live.done").is_file():
        return "missing-completion"
    if (directory / "score.json").exists():
        return "scored"
    return None


def _replay(checkpoint: bytes, env, tracker) -> tuple[list[dict], dict, object]:
    improvements = [dict(tracker.snapshot(), frame=0, elapsed=0.0)]
    for frame, buttons in enumerate(checkpoint, 1):
        if buttons > 63:
            raise ValueError(f"checkpoint has invalid button mask {buttons} at frame {frame}")
        env.step(buttons, 1)
        if tracker.update(env.game_state):
            event = tracker.snapshot()
            event.update(frame=frame, elapsed=frame / 30)
            improvements.append(event)
    return improvements, tracker.snapshot(), getattr(env, "framebuffer", None)


def _reference_frame(path: Path):
    with av.open(str(path)) as video:
        frame = next(video.decode(video=0), None)
        if frame is None:
            raise ValueError("live.png contains no image")
        return frame.to_ndarray(format="rgba")


def _same_frame(actual, reference) -> bool:
    if actual is None or reference is None:
        return False
    return (getattr(actual, "shape", None) == getattr(reference, "shape", None)
            and actual.tobytes() == reference.tobytes())


def rescore_run(directory: str | Path) -> dict:
    """Replay one completed run and persist ``progress.jsonl`` and ``score.json``."""
    directory = Path(directory)
    status = _status(directory)
    if status:
        return {"status": f"skipped-{status}", "run": str(directory)}
    checkpoint_path = directory / "checkpoint.state"
    checkpoint = checkpoint_path.read_bytes()
    tracker = Progress()
    with Open8() as env:
        improvements, snapshot, final_frame = _replay(checkpoint, env, tracker)

    reference_path = directory / "live.png"
    if reference_path.exists():
        try:
            reference = _reference_frame(reference_path)
        except Exception as exc:
            return {"status": "skipped-mismatch", "run": str(directory),
                    "error": f"could not verify live.png: {exc}"}
        if not _same_frame(final_frame, reference):
            return {"status": "skipped-mismatch", "run": str(directory),
                    "error": "final framebuffer does not match live.png"}
        verification = "framebuffer"
    else:
        verification = "unverified"

    frame = len(checkpoint)
    (directory / "progress.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in improvements),
        encoding="utf-8",
    )
    score = {
        **snapshot,
        "timing": "replay",
        "elapsed": frame / 30,
        "frame": frame,
        "status": "completed",
        "provenance": {
            "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            "verification": verification,
        },
    }
    temporary = directory / ".score.json.tmp"
    temporary.write_text(json.dumps(score, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(directory / "score.json")
    return {"status": "rescored", "run": str(directory), "frame": frame,
            "progress": score["progress"]}


def rescore_root(root: str | Path = "runs", *, dry_run: bool = False) -> list[dict]:
    root = Path(root)
    results = []
    for directory in _runs(root):
        status = _status(directory)
        if status:
            results.append({"status": f"skipped-{status}", "run": str(directory)})
        elif dry_run:
            results.append({"status": "candidate", "run": str(directory)})
        else:
            try:
                results.append(rescore_run(directory))
            except Exception as exc:
                results.append({"status": "error", "run": str(directory),
                                "error": str(exc)})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="runs", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="list eligible runs without replaying or writing")
    parser.add_argument("--summary", action="store_true",
                        help="print only the final summary")
    args = parser.parse_args(argv)
    results = rescore_root(args.path, dry_run=args.dry_run)
    if not args.summary:
        for result in results:
            print(json.dumps(result, sort_keys=True))
    counts = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"runs": len(results), "summary": counts}, sort_keys=True))
    return 1 if any(result["status"] == "error" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
