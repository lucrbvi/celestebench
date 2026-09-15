import asyncio
import functools
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

pytest.importorskip("mcp.shared.memory")

from mcp.shared.memory import create_connected_server_and_client_session

from celestebench.mcp import BearerAuth, Episode, _actions, server


async def fake_rollout(policy, output, **options):
    output = Path(output)
    output.mkdir()
    (output / "config.json").write_text("{}")
    first = np.zeros((2, 2, 4), dtype=np.uint8)
    actions = await policy((first,))
    second = np.ones((2, 2, 4), dtype=np.uint8)
    await policy((first, second))
    return {"frames": 3, "decisions": 1, "actions": len(actions), "elapsed": 0.01}


def asyncio_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def test_strict_actions():
    assert _actions([{"buttons": 63, "frames": 2},
                     {"action": "wait", "frames": 3}], 3) == ((63, 2), ("wait", 3))
    invalid = [[], [{"buttons": True, "frames": 1}],
               [{"buttons": 1, "frames": True}], [{"buttons": 64, "frames": 1}],
               [{"action": "WAIT", "frames": 1}],
               [{"buttons": 1, "frames": 1, "extra": 0}]]
    for actions in invalid:
        with pytest.raises(ValueError):
            _actions(actions, 3)


@asyncio_test
async def test_lazy_start_observe_is_stable_and_play_advances(tmp_path):
    output = tmp_path / "episode"
    episode = Episode(output, frames=3, max_images=1, runner=fake_rollout)
    assert not output.exists()
    first = await episode.observe()
    again = await episode.observe()
    assert first[1].data == again[1].data
    second = await episode.play([{"buttons": 2, "frames": 1}])
    assert len(second) == 2
    assert first[1].data != second[1].data
    await episode.close()


@asyncio_test
async def test_terminal_result_and_no_restart(tmp_path):
    async def ending(policy, output, **options):
        Path(output).mkdir()
        await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
        return {"frames": 2, "decisions": 1, "actions": 1, "elapsed": 0.1}

    episode = Episode(tmp_path / "episode", runner=ending)
    result = await episode.play([{"action": "wait", "frames": 1}])
    assert '"status":"finished"' in result[0].text
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await episode.play([{"buttons": 0, "frames": 1}])


@asyncio_test
async def test_server_exposes_only_observe_and_play():
    episode = Episode("unused", runner=fake_rollout)
    tools = await server(episode).list_tools()
    assert {tool.name for tool in tools} == {"observe", "play"}


@asyncio_test
async def test_sdk_initializes_validates_and_calls_tools(tmp_path):
    submitted = []

    async def recording(policy, output, **options):
        Path(output).mkdir()
        frame = np.zeros((1, 1, 4), dtype=np.uint8)
        submitted.append(await policy((frame,)))
        return {"frames": 1}

    episode = Episode(tmp_path / "episode", runner=recording)
    async with create_connected_server_and_client_session(server(episode)) as client:
        assert await client.initialize() is not None
        tools = await client.list_tools()
        play = next(tool for tool in tools.tools if tool.name == "play")
        variants = play.inputSchema["properties"]["actions"]["items"]["anyOf"]
        assert len(variants) == 2
        invalid = await client.call_tool("play", {"actions": [
            {"buttons": 2, "frames": 1}, {"buttons": 64, "frames": 1}]})
        assert invalid.isError
        assert submitted == []
        result = await client.call_tool("play", {"actions": [
            {"buttons": 2, "frames": 1}]})
        assert not result.isError
        assert submitted == [((2, 1),)]


@asyncio_test
async def test_concurrent_calls_are_rejected(tmp_path):
    gate = asyncio.Event()

    async def blocked(policy, output, **options):
        Path(output).mkdir()
        await gate.wait()
        return {}

    episode = Episode(tmp_path / "episode", runner=blocked)
    pending = asyncio.create_task(episode.observe())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already in progress"):
        await episode.observe()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await episode.close()


@asyncio_test
async def test_cancelled_observe_keeps_pending_decision(tmp_path):
    gate = asyncio.Event()

    async def delayed(policy, output, **options):
        Path(output).mkdir()
        await gate.wait()
        await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
        return {}

    episode = Episode(tmp_path / "episode", runner=delayed)
    cancelled = asyncio.create_task(episode.observe())
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    gate.set()
    result = await episode.play([{"buttons": 2, "frames": 1}])
    assert '"status":"finished"' in result[0].text


@asyncio_test
async def test_finished_episode_does_not_return_cached_ready_frame(tmp_path):
    async def times_out(policy, output, **options):
        Path(output).mkdir()
        await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
        await asyncio.sleep(0)
        return {"frames": 1, "reason": "timeout"}

    episode = Episode(tmp_path / "episode", runner=times_out)
    await episode.observe()
    await episode.play([{"buttons": 0, "frames": 1}])
    terminal = await episode.observe()
    assert '"status":"finished"' in terminal[0].text
    assert len(terminal) == 1


@asyncio_test
async def test_real_open8_episode_through_sdk(tmp_path):
    episode = Episode(tmp_path / "episode", frames=2, fps=None)
    async with create_connected_server_and_client_session(server(episode)) as client:
        await client.initialize()
        observed = await client.call_tool("observe")
        assert not observed.isError
        played = await client.call_tool("play", {"actions": [
            {"buttons": 0, "frames": 1}]})
        assert not played.isError
    assert (tmp_path / "episode" / "actions.jsonl").is_file()


@asyncio_test
async def test_http_requires_bearer_and_limits_request_body():
    app = BearerAuth(server(Episode("unused", runner=fake_rollout)).streamable_http_app(),
                     "secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.post("/mcp", content=b"{}")
        assert unauthorized.status_code == 401
        oversized = await client.post("/mcp", headers={
            "authorization": "Bearer secret"}, content=b"x" * (1024 * 1024 + 1))
        assert oversized.status_code == 413


@asyncio_test
async def test_http_requests_share_one_live_episode(tmp_path):
    output = tmp_path / "episode"
    episode = Episode(output, frames=3, fps=None)
    mcp = server(episode)
    app = BearerAuth(mcp.streamable_http_app(), "secret")
    try:
        async with mcp.session_manager.run(), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8124",
                headers={"authorization": "Bearer secret",
                         "accept": "application/json, text/event-stream"}) as client:
            initialized = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                "method": "initialize", "params": {"protocolVersion": "2025-11-25",
                "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
            assert initialized.status_code == 200
            observed = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 2,
                "method": "tools/call", "params": {"name": "observe", "arguments": {}}})
            assert observed.status_code == 200
            assert not observed.json()["result"]["isError"]
            assert not episode._task.done()
            played = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 3,
                "method": "tools/call", "params": {"name": "play", "arguments": {
                    "actions": [{"buttons": 2, "frames": 2}]}}})
            result = played.json()["result"]
            assert not result["isError"]
            assert json.loads(result["content"][0]["text"])["frames"] == 3
            assert (output / "checkpoint.state").read_bytes() == b"\0\2\2"
    finally:
        await episode.close()
