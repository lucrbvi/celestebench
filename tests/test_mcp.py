import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np

try:
    from mcp.shared.memory import create_connected_server_and_client_session
except ImportError:
    raise unittest.SkipTest("install celestebench[mcp] to test the MCP server")

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


class ValidationTest(unittest.TestCase):
    def test_strict_actions(self):
        self.assertEqual(_actions([{"buttons": 63, "frames": 2},
                                   {"action": "wait", "frames": 3}], 3),
                         ((63, 2), ("wait", 3)))
        invalid = [[], [{"buttons": True, "frames": 1}],
                   [{"buttons": 1, "frames": True}], [{"buttons": 64, "frames": 1}],
                   [{"action": "WAIT", "frames": 1}],
                   [{"buttons": 1, "frames": 1, "extra": 0}]]
        for actions in invalid:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                _actions(actions, 3)


class EpisodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_start_observe_is_stable_and_play_advances(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "episode"
            episode = Episode(output, frames=3, max_images=1, runner=fake_rollout)
            self.assertFalse(output.exists())
            first = await episode.observe()
            again = await episode.observe()
            self.assertEqual(first[1].data, again[1].data)
            second = await episode.play([{"buttons": 2, "frames": 1}])
            self.assertEqual(len(second), 2)
            self.assertNotEqual(first[1].data, second[1].data)
            await episode.close()

    async def test_terminal_result_and_no_restart(self):
        async def ending(policy, output, **options):
            Path(output).mkdir()
            await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
            return {"frames": 2, "decisions": 1, "actions": 1, "elapsed": 0.1}

        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", runner=ending)
            result = await episode.play([{"action": "wait", "frames": 1}])
            self.assertIn('"status":"finished"', result[0].text)
            with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                await episode.play([{"buttons": 0, "frames": 1}])

    async def test_server_exposes_only_observe_and_play(self):
        episode = Episode("unused", runner=fake_rollout)
        tools = await server(episode).list_tools()
        self.assertEqual({tool.name for tool in tools}, {"observe", "play"})

    async def test_sdk_initializes_validates_and_calls_tools(self):
        submitted = []

        async def recording(policy, output, **options):
            Path(output).mkdir()
            frame = np.zeros((1, 1, 4), dtype=np.uint8)
            submitted.append(await policy((frame,)))
            return {"frames": 1}

        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", runner=recording)
            async with create_connected_server_and_client_session(server(episode)) as client:
                self.assertIsNotNone(await client.initialize())
                tools = await client.list_tools()
                play = next(tool for tool in tools.tools if tool.name == "play")
                variants = play.inputSchema["properties"]["actions"]["items"]["anyOf"]
                self.assertEqual(len(variants), 2)
                invalid = await client.call_tool("play", {"actions": [
                    {"buttons": 2, "frames": 1}, {"buttons": 64, "frames": 1}]})
                self.assertTrue(invalid.isError)
                self.assertEqual(submitted, [])
                result = await client.call_tool("play", {"actions": [
                    {"buttons": 2, "frames": 1}]})
                self.assertFalse(result.isError)
                self.assertEqual(submitted, [((2, 1),)])

    async def test_concurrent_calls_are_rejected(self):
        gate = asyncio.Event()

        async def blocked(policy, output, **options):
            Path(output).mkdir()
            await gate.wait()
            return {}

        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", runner=blocked)
            pending = asyncio.create_task(episode.observe())
            await asyncio.sleep(0)
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                await episode.observe()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            await episode.close()

    async def test_cancelled_observe_keeps_pending_decision(self):
        gate = asyncio.Event()

        async def delayed(policy, output, **options):
            Path(output).mkdir()
            await gate.wait()
            await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
            return {}

        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", runner=delayed)
            cancelled = asyncio.create_task(episode.observe())
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            gate.set()
            result = await episode.play([{"buttons": 2, "frames": 1}])
            self.assertIn('"status":"finished"', result[0].text)

    async def test_finished_episode_does_not_return_cached_ready_frame(self):
        async def times_out(policy, output, **options):
            Path(output).mkdir()
            await policy((np.zeros((1, 1, 4), dtype=np.uint8),))
            await asyncio.sleep(0)
            return {"frames": 1, "reason": "timeout"}

        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", runner=times_out)
            await episode.observe()
            await episode.play([{"buttons": 0, "frames": 1}])
            terminal = await episode.observe()
            self.assertIn('"status":"finished"', terminal[0].text)
            self.assertEqual(len(terminal), 1)

    async def test_real_open8_episode_through_sdk(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root) / "episode", frames=2, fps=None)
            async with create_connected_server_and_client_session(server(episode)) as client:
                await client.initialize()
                observed = await client.call_tool("observe")
                self.assertFalse(observed.isError)
                played = await client.call_tool("play", {"actions": [
                    {"buttons": 0, "frames": 1}]})
                self.assertFalse(played.isError)
            self.assertTrue((Path(root) / "episode" / "actions.jsonl").is_file())

    async def test_http_requires_bearer_and_limits_request_body(self):
        app = BearerAuth(server(Episode("unused", runner=fake_rollout)).streamable_http_app(),
                         "secret")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.post("/mcp", content=b"{}")
            self.assertEqual(unauthorized.status_code, 401)
            oversized = await client.post("/mcp", headers={
                "authorization": "Bearer secret"}, content=b"x" * (1024 * 1024 + 1))
            self.assertEqual(oversized.status_code, 413)

    async def test_http_requests_share_one_live_episode(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "episode"
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
                    self.assertEqual(initialized.status_code, 200)
                    observed = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 2,
                        "method": "tools/call", "params": {"name": "observe", "arguments": {}}})
                    self.assertEqual(observed.status_code, 200)
                    self.assertFalse(observed.json()["result"]["isError"])
                    self.assertFalse(episode._task.done())
                    played = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 3,
                        "method": "tools/call", "params": {"name": "play", "arguments": {
                            "actions": [{"buttons": 2, "frames": 2}]}}})
                    result = played.json()["result"]
                    self.assertFalse(result["isError"])
                    self.assertEqual(json.loads(result["content"][0]["text"])["frames"], 3)
                    self.assertEqual((output / "checkpoint.state").read_bytes(), b"\0\2\2")
            finally:
                await episode.close()


if __name__ == "__main__":
    unittest.main()
