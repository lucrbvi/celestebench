import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import av
import numpy as np

from web import export, viewer


def _video(path, frames):
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width = stream.height = 512
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            image = np.zeros((512, 512, 3), np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        out.mux(stream.encode())


class ExportTest(unittest.TestCase):
    def test_export_composes_gameplay_buttons_and_message(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"model": "test-model", "fps": 30}))
            (folder / "actions.jsonl").write_text(json.dumps({
                "decision": 0, "buttons": 2, "frames": 4, "frame_start": 1, "frame_end": 5}) + "\n")
            _video(folder / "rollout.mp4", frames=8)
            output = root / "out.mp4"
            with patch.object(viewer, "RUNS", root), patch.object(export, "RUNS", root):
                self.assertEqual(export.export("run", output), output)
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height), (export.WIDTH, export.HEIGHT))
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 8)


if __name__ == "__main__":
    unittest.main()
