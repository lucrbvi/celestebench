import tempfile
import time
import unittest
from pathlib import Path

import av
import numpy as np

from celestebench import Button, Open8


class Open8Test(unittest.TestCase):
    def test_telemetry_does_not_change_replay_and_detects_death(self):
        with Open8() as env:
            env.step(Button.O, 1)
            env.step(0, 120)
            checkpoint = env.save_state()
            missing_player = False
            for _ in range(180):
                env.step(Button.RIGHT | Button.O, 1)
                state = env.game_state
                missing_player |= not state["alive"]
            observed = env.framebuffer
            self.assertGreater(state["deaths"], 0)
            self.assertTrue(missing_player)
            env.load_state(checkpoint)
            env.step(Button.RIGHT | Button.O, 180)
            np.testing.assert_array_equal(env.framebuffer, observed)

    def test_game_state(self):
        with Open8() as env:
            title = env.game_state
            self.assertEqual(title["room"], 31)
            self.assertFalse(title["alive"])
            self.assertFalse(title["grounded"])
            self.assertIsNone(title["spawn_feet_y"])
            self.assertEqual(title["deaths"], 0)

            env.step(Button.X, 1)
            env.step(0, 120)
            state = env.game_state
            self.assertEqual(state["room"], 0)
            self.assertTrue(state["alive"])
            self.assertTrue(state["grounded"])
            self.assertEqual(state["feet_y"], 104)
            self.assertEqual(state["spawn_feet_y"], 104)
            self.assertEqual(state["exit_feet_y"], 4)

    def test_game_state_is_none_for_other_carts(self):
        cart = Path(__file__).parents[1] / "deps/open8/export/carts/WOLFHUNT.PNG"
        with Open8(cart) as env:
            self.assertIsNone(env.game_state)

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
                self.assertEqual(len(recording.streams.video), 1)
                frames = list(recording.decode(video=0))
                self.assertEqual(len(frames), 45)
                self.assertEqual((frames[0].width, frames[0].height), (512, 512))
                self.assertEqual(frames[-1].time - frames[0].time, 44 / 30)
                self.assertFalse(np.array_equal(frames[0].to_ndarray(), frames[-1].to_ndarray()))
        env.close()
        with self.assertRaises(RuntimeError):
            env.step()
        with Open8() as reopened:
            np.testing.assert_array_equal(reopened.load_state(checkpoint), saved_image)


if __name__ == '__main__':
    unittest.main()
