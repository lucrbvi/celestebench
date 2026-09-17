import json
from unittest.mock import patch

import av

from conftest import video
from web import export, viewer


def test_export_composes_gameplay_buttons_and_message(tmp_path):
    root = tmp_path
    folder = root / "run"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": "test-model", "fps": 30}))
    (folder / "actions.jsonl").write_text(json.dumps({
        "decision": 0, "buttons": 2, "frames": 4, "frame_start": 1, "frame_end": 5}) + "\n")
    video(folder / "rollout.mp4", frames=8)
    output = root / "out.mp4"
    with patch.object(viewer, "RUNS", root), patch.object(export, "RUNS", root):
        assert export.export("run", output) == output
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (export.WIDTH, export.HEIGHT)
        assert sum(1 for _ in container.decode(video=0)) == 8
