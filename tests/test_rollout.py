import asyncio
import json
from unittest.mock import PropertyMock, patch

import av
import numpy as np
import pytest

from celestebench import Button
from celestebench.rollout import rollout


def test_progress_records_idle_frames_and_uses_wall_clock(tmp_path):
    async def run():
        async def policy(frames):
            await asyncio.Event().wait()

        state = dict(room=0, alive=True, grounded=True, feet_y=58,
                     spawn_feet_y=112, exit_feet_y=4, deaths=0)
        with patch("celestebench.open8.Open8.game_state", new_callable=PropertyMock,
                   return_value=state):
            output = tmp_path / "run"
            await rollout(policy, output, frames=5, fps=100)
            events = [json.loads(line) for line in
                      (output / "progress.jsonl").read_text().splitlines()]
            score = json.loads((output / "score.json").read_text())
            assert [event["frame"] for event in events] == [0, 3]
            assert events[1]["elapsed"] > 0
            assert events[1]["elapsed"] <= score["elapsed"]
            assert score["timing"] == "wall_clock"
            assert score["status"] == "completed"
            assert score["room_progress"] == .5
            assert score["frame"] == 5

    asyncio.run(run())


def test_wait_is_distinct_from_releasing_buttons(tmp_path):
    async def run():
        output = tmp_path / "run"
        result = await rollout(lambda frame: [(2, 1), ("wait", 2), (0, 1)], output,
                               frames=5)
        rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]

        assert (result["frames"], result["actions"]) == (5, 3)
        assert "action" not in rows[0]
        assert rows[1]["action"] == "wait"
        assert "action" not in rows[2]
        assert [row["buttons"] for row in rows] == [2, 0, 0]
        assert (output / "checkpoint.state").read_bytes() == b"\0\2\0\0\0"
        assert (output / "live.png").is_file()
        assert (output / "live.done").is_file()

    asyncio.run(run())


def test_multi_action_policy_writes_video_and_history(tmp_path):
    async def run():
        output = tmp_path / "run"
        observations = []
        def policy(frames):
            observations.append([frame.copy() for frame in frames])
            return [(2, 2), (0, 1)]
        result = await rollout(policy, output, frames=10)
        assert result["frames"] == 10  # one initial screenshot
        assert result["decisions"] == 3
        assert result["actions"] == 6
        # Decision 0 sees the initial frame; each later decision sees every
        # frame played since the previous one, oldest to newest.
        assert [len(frames) for frames in observations] == [1, 3, 3]
        rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
        assert [row["buttons"] for row in rows] == [2, 0] * 3
        assert [row["decision"] for row in rows] == [0, 0, 1, 1, 2, 2]
        assert all(row["latency"] == 0 for row in rows[1::2])
        assert all(row["latency"] > 0 for row in rows[::2])
        assert [(r["frame_start"], r["frame_end"]) for r in rows] == \
            [(1, 3), (3, 4), (4, 6), (6, 7), (7, 9), (9, 10)]
        outcomes = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
        assert [r["status"] for r in outcomes] == ["played"] * 3
        assert [r["frame_start"] for r in outcomes] == [1, 4, 7]
        assert [r["screenshot"] for r in outcomes] == \
            [f"screenshots/{n:06d}.png" for n in range(3)]
        for row, frames in zip(outcomes, observations, strict=True):
            with av.open(str(output / row["screenshot"])) as png:
                decoded = next(png.decode(video=0)).to_ndarray(format="rgba")
            np.testing.assert_array_equal(decoded, frames[-1])
        with av.open(str(output / "rollout.mp4")) as video:
            assert len(list(video.decode(video=0))) == 10
        assert (output / "checkpoint.state").read_bytes() == b"\x00" + b"\x02\x02\x00" * 3

    asyncio.run(run())


def test_single_action_policy_accepts_bare_pair(tmp_path):
    async def run():
        output = tmp_path / "run"
        result = await rollout(lambda frame: (Button.RIGHT, np.int64(2)), output, frames=5)
        assert result["frames"] == 5
        assert result["decisions"] == 2
        assert result["actions"] == 2
        assert (output / "checkpoint.state").read_bytes() == b"\x00\x02\x02\x02\x02"

    asyncio.run(run())


def test_async_policy_and_invalid_action_keep_partial_checkpoint(tmp_path):
    async def run():
        async def policy(frame):
            await asyncio.sleep(0)
            return (1, 1)

        output = tmp_path / "run"
        with pytest.raises(ValueError):
            await rollout(lambda frame: (1, 0), output, frames=3)
        assert (output / "rollout.mp4").stat().st_size
        assert len((output / "checkpoint.state").read_bytes()) == 1
        assert json.loads((output / "score.json").read_text())["status"] == "error"

        result = await rollout(policy, tmp_path / "async", frames=2)
        assert result["frames"] == 2

    asyncio.run(run())


def test_output_must_be_fresh(tmp_path):
    async def run():
        output = tmp_path / "run"
        output.mkdir()
        with pytest.raises(FileExistsError):
            await rollout(lambda frame: (0, 1), output, frames=1)

    asyncio.run(run())


@pytest.mark.parametrize("batch", [
    [(2, 1), (0, 0)],
    [(2, 1), (0, 31)],
    [(2, 1), (True, 1)],
    [],
])
def test_invalid_batch_never_partially_executes(tmp_path, batch):
    async def run():
        output = tmp_path / "run"
        with pytest.raises((ValueError, TypeError)):
            await rollout(lambda frame: batch, output, frames=2)
        assert (output / "checkpoint.state").read_bytes() == b"\x00"
        assert (output / "actions.jsonl").read_text() == ""

    asyncio.run(run())


def test_realtime_runs_environment_while_policy_thinks(tmp_path):
    async def run():
        observations = []
        calls = 0

        async def policy(frames):
            nonlocal calls
            observations.append(len(frames))
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.05)  # ~10 idle frames tick meanwhile
            return (2, 10)

        output = tmp_path / "run"
        result = await rollout(policy, output, frames=60, fps=200)
        assert result["frames"] == 60
        assert result["actions"] == 5
        # Decision 0 sees the initial frame; decision 1 sees the idle frames
        # ticked while thinking plus the frames of the played action.
        assert observations[0] == 1
        assert observations[1] >= 11
        assert result["elapsed"] < 5
        with av.open(str(output / "rollout.mp4")) as video:
            assert len(list(video.decode(video=0))) == 60

    asyncio.run(run())


def test_wall_clock_timeout_cancels_pending_policy_and_ends_gracefully(tmp_path):
    async def run():
        async def policy(frame):
            await asyncio.sleep(5)

        output = tmp_path / "run"
        result = await rollout(policy, output, timeout=0.1)
        assert result["elapsed"] < 5
        assert (result["frames"], result["decisions"], result["actions"]) == (1, 0, 0)
        outcome = json.loads((output / "decisions.jsonl").read_text())
        assert outcome["status"] == "timeout"
        assert (outcome["frame_start"], outcome["frame_end"]) == (1, 1)
        assert outcome["screenshot"] == "screenshots/000000.png"
        assert (output / outcome["screenshot"]).is_file()
        assert (output / "actions.jsonl").read_text() == ""
        with av.open(str(output / "rollout.mp4")) as video:
            assert len(list(video.decode(video=0))) == 1

    asyncio.run(run())


def test_strict_timeout_truncates_synchronous_action_batch(tmp_path):
    async def run():
        output = tmp_path / "run"
        result = await rollout(lambda frame: (2, 100000), output,
                               timeout=0.1, strict_timeout=True,
                               max_frames=100000)
        outcome = json.loads((output / "decisions.jsonl").read_text())
        action = json.loads((output / "actions.jsonl").read_text())
        assert outcome["status"] == "timeout"
        assert result["frames"] == outcome["frame_end"]
        assert action["frames"] == result["frames"] - 1
        assert action["frames"] < 100000

    asyncio.run(run())


@pytest.mark.parametrize("strict", [True, False])
def test_realtime_strict_deadline_cuts_batch_but_default_finishes_it(tmp_path, strict):
    async def run():
        output = tmp_path / "run"
        await rollout(lambda frames: [(2, 10), (1, 2)], output,
                      fps=20, timeout=0.25, strict_timeout=strict)
        actions = [json.loads(line) for line in
                   (output / "actions.jsonl").read_text().splitlines()]
        outcome = json.loads((output / "decisions.jsonl").read_text())
        if strict:
            assert outcome["status"] == "timeout"
            assert len(actions) == 1
            assert actions[0]["frames"] < 10
        else:
            assert outcome["status"] == "played"
            assert [a["frames"] for a in actions] == [10, 2]

    asyncio.run(run())


def test_policy_timeout_is_logged_and_ends_gracefully(tmp_path):
    async def run():
        async def policy(frame):
            raise TimeoutError("deadline")

        output = tmp_path / "run"
        result = await rollout(policy, output)
        assert result["decisions"] == 0
        outcome = json.loads((output / "decisions.jsonl").read_text())
        assert (outcome["status"], outcome["error"]) == ("timeout", "deadline")
        assert (output / "actions.jsonl").read_text() == ""

    asyncio.run(run())


def test_expired_budget_makes_no_decision(tmp_path):
    async def run():
        output = tmp_path / "run"
        result = await rollout(lambda frame: (0, 1), output, timeout=1e-6)
        assert (result["frames"], result["decisions"]) == (1, 0)
        assert (output / "decisions.jsonl").read_text() == ""

    asyncio.run(run())


def test_frame_budget_cancels_and_logs_pending_policy(tmp_path):
    async def run():
        cancelled = asyncio.Event()

        async def policy(frame):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        output = tmp_path / "run"
        result = await rollout(policy, output, frames=3, fps=1000)
        assert cancelled.is_set()
        assert result["frames"] == 3
        outcome = json.loads((output / "decisions.jsonl").read_text())
        assert outcome["status"] == "frame_limit"
        assert (outcome["frame_start"], outcome["frame_end"]) == (1, 3)
        assert (output / "actions.jsonl").read_text() == ""

    asyncio.run(run())
