import tempfile
import time
import unittest
from pathlib import Path

import av
import numpy as np

from celestebench import Button, Open8


class Open8Test(unittest.TestCase):
    def test_rollout(self):
        with tempfile.TemporaryDirectory() as directory, Open8() as env:
            initial = env.framebuffer
            time.sleep(0.02)
            np.testing.assert_array_equal(env.framebuffer, initial)
            self.assertEqual(initial.shape, (128, 128, 4))
            self.assertEqual(initial.dtype, np.uint8)
            with self.assertRaises(RuntimeError):
                Open8()
            for buttons, frames in [(0, 60), (Button.O, 1), (0, 60), (Button.RIGHT, 100)]:
                env.step(buttons, frames)
            checkpoint = env.save_state()
            self.assertEqual((env.audio.dtype, env.audio.size), (np.dtype(np.int16), 100 * 22050 // 30))
            self.assertGreater(np.abs(env.audio).max(), 0)
            self.assertLess(np.abs(env.audio.astype(np.int32)).max(), 32767)
            saved_image = env.framebuffer
            expected = env.step(Button.RIGHT | Button.O, 90)
            branch = env.save_state()
            np.testing.assert_array_equal(env.load_state(checkpoint), saved_image)
            np.testing.assert_array_equal(env.step(Button.RIGHT | Button.O, 90), expected)
            self.assertEqual(env.save_state(), branch)
            np.testing.assert_array_equal(env.load_state(branch), expected)
            np.testing.assert_array_equal(env.reset(), initial)
            np.testing.assert_array_equal(env.load_state(checkpoint), saved_image)
            for buttons, frames in [(64, 1), (0, -1), (0, 2**32)]:
                with self.assertRaises(ValueError):
                    env.step(buttons, frames)
            with self.assertRaises(ValueError):
                env.load_state(b'\xff')
            self.assertEqual(env.save_state(), checkpoint)
            video = Path(directory) / 'rollout.mp4'
            with self.assertRaisesRegex(RuntimeError, 'rollout interrupted'):
                with env.record(video):
                    env.step(Button.RIGHT, 15)
                    env.load_state(checkpoint)
                    env.step(Button.O, 30)
                    raise RuntimeError('rollout interrupted')
            with av.open(str(video)) as recording:
                self.assertEqual(recording.streams.video[0].average_rate, 30)
                self.assertEqual(recording.streams.audio[0].rate, 22050)
                frames = list(recording.decode(video=0))
            with av.open(str(video)) as recording:
                samples = np.concatenate([frame.to_ndarray().reshape(-1) for frame in recording.decode(audio=0)])
                self.assertEqual(len(frames), 45)
                self.assertEqual((frames[0].width, frames[0].height), (512, 512))
                self.assertEqual(frames[-1].time - frames[0].time, 44 / 30)
                self.assertFalse(np.array_equal(frames[0].to_ndarray(), frames[-1].to_ndarray()))
                self.assertGreater(np.abs(samples).max(), 0)
                self.assertAlmostEqual(len(samples) / 22050, 45 / 30, delta=0.2)  # AAC pads its final packet.
        env.close()
        with self.assertRaises(RuntimeError):
            env.step()
        with Open8() as reopened:
            np.testing.assert_array_equal(reopened.load_state(checkpoint), saved_image)


if __name__ == '__main__':
    unittest.main()
