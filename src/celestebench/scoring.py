"""Evaluator-only geometric progress. Never part of the policy observation."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Progress:
    """Tracks grounded height across rooms. Missing or partial game state is
    ignored instead of aborting the run, and progress only ever moves up."""

    def __init__(self):
        self.rooms_completed = 0
        self.room_progress = 0.0
        self.room = None
        self.deaths = None
        self.grounded_frames = 0

    def snapshot(self):
        return {"version": 1, "metric": "grounded_height_v1",
                "progress": 100 * (self.rooms_completed + self.room_progress) / 30,
                "rooms_completed": self.rooms_completed,
                "room_progress": self.room_progress}

    def update(self, state):
        before = self.snapshot()["progress"]
        if not isinstance(state, dict) or not _number(state.get("room")) or not _number(state.get("deaths")):
            self.grounded_frames = 0
            return False
        room, deaths = state["room"], state["deaths"]
        if room != self.room or deaths != self.deaths:
            self.grounded_frames = 0
        # Only actual, consecutive exits count; title/summit screens cannot
        # masquerade as a climb, and restarting cannot earn rooms twice.
        if self.room == self.rooms_completed and room == self.rooms_completed + 1 and room <= 30:
            self.rooms_completed, self.room_progress = room, 0.0
        self.room, self.deaths = room, deaths
        feet = state.get("feet_y")
        if not (0 <= room < 30 and state.get("alive") and state.get("grounded") and _number(feet)):
            self.grounded_frames = 0
            return self.snapshot()["progress"] > before
        self.grounded_frames += 1
        if self.grounded_frames >= 3 and room == self.rooms_completed:
            start, end = state.get("spawn_feet_y"), state.get("exit_feet_y")
            if _number(start) and _number(end) and start > end:
                fraction = max(0.0, min(0.999999, (start - feet) / (start - end)))
                self.room_progress = max(self.room_progress, fraction)
        return self.snapshot()["progress"] > before
