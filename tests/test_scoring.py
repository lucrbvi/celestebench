import unittest

from celestebench.scoring import Progress


def state(**changes):
    return dict(room=0, alive=True, grounded=True, feet_y=112,
                spawn_feet_y=112, exit_feet_y=4, deaths=0) | changes


class ProgressTest(unittest.TestCase):
    def test_only_settled_living_support_counts_and_deaths_keep_record(self):
        progress = Progress()
        for _ in range(20):
            progress.update(state(feet_y=4, grounded=False))
        self.assertEqual(progress.snapshot()["progress"], 0)
        for _ in range(2):
            self.assertFalse(progress.update(state(feet_y=58)))
        self.assertTrue(progress.update(state(feet_y=58)))
        self.assertAlmostEqual(progress.snapshot()["progress"], 100 * .5 / 30)
        progress.update(state(alive=False, grounded=False, deaths=1))
        for _ in range(3):
            progress.update(state(deaths=1))
        self.assertEqual(progress.snapshot()["room_progress"], .5)

    def test_title_transitions_and_summit(self):
        progress = Progress()
        progress.update(state(room=31))
        self.assertEqual(progress.snapshot()["progress"], 0)
        progress.update(state())
        for room in range(1, 31):
            self.assertTrue(progress.update(state(room=room, alive=False)))
            self.assertEqual(progress.snapshot()["rooms_completed"], room)
        self.assertEqual(progress.snapshot()["progress"], 100)
        progress.update(state(room=31))
        progress.update(state())
        self.assertEqual(progress.snapshot()["progress"], 100)

    def test_contact_counter_resets_on_death_and_room_change(self):
        progress = Progress()
        progress.update(state(feet_y=58))
        progress.update(state(feet_y=58))
        progress.update(state(feet_y=58, deaths=1))
        self.assertEqual(progress.snapshot()["progress"], 0)
        progress.update(state(room=1, feet_y=58, deaths=1))
        self.assertEqual(progress.snapshot()["room_progress"], 0)

    def test_starting_at_summit_does_not_award_unplayed_rooms(self):
        progress = Progress()
        progress.update(state(room=29))
        progress.update(state(room=30))
        self.assertEqual(progress.snapshot()["progress"], 0)

    def test_partial_never_counts_as_exit_and_missing_reference_is_unscored(self):
        progress = Progress()
        for _ in range(3):
            progress.update(state(feet_y=-10, spawn_feet_y=None))
        self.assertEqual(progress.snapshot()["progress"], 0)
        progress.update(state(feet_y=-10))
        self.assertLess(progress.snapshot()["room_progress"], 1)
        self.assertEqual(progress.snapshot()["rooms_completed"], 0)
