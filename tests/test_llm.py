import asyncio
import base64
import io
import unittest

import numpy as np

try:
    from PIL import Image
    from tau_agent import (
        AssistantMessage, ImageContent, TextContent, ThinkingContent, ToolCall,
    )
    from tau_agent.provider_events import (
        AssistantDoneEvent, AssistantErrorEvent, ThinkingDeltaEvent,
    )
    from celestebench.llm import TauPolicy
except ImportError:
    TauPolicy = None


@unittest.skipUnless(TauPolicy is not None, "celestebench[llm] is not installed")
class TauPolicyTests(unittest.IsolatedAsyncioTestCase):
    class FakeProvider:
        def __init__(self, events=(), delay=0):
            self.events = tuple(events)
            self.delay = delay
            self.calls = []

        def stream_response(self, **request):
            self.calls.append(request)

            async def events():
                if self.delay:
                    await asyncio.sleep(self.delay)
                for event in self.events:
                    yield event

            return events()

    class SequenceProvider:
        def __init__(self, messages):
            self.messages = tuple(messages)
            self.calls = []

        def stream_response(self, **request):
            index = len(self.calls)
            self.calls.append(request)
            message = self.messages[min(index, len(self.messages) - 1)]

            async def events():
                yield AssistantDoneEvent(reason=message.stop_reason, message=message)

            return events()

    @staticmethod
    def assistant(*calls, text=None, stop_reason="toolUse"):
        content = []
        if text is not None:
            content.append(TextContent(text=text))
        content.extend(calls)
        return AssistantMessage(content=content, stop_reason=stop_reason)

    @staticmethod
    def reasoning_assistant(call):
        return AssistantMessage(content=[
            ThinkingContent(thinking="plan", thinking_signature="sig-thinking"),
            TextContent(text="I will play", text_signature="sig-text"),
            call,
        ], stop_reason="toolUse")

    @staticmethod
    def call(arguments, name="play", call_id="call-1"):
        return ToolCall(id=call_id, name=name, arguments=arguments)

    async def decide(self, provider, **kwargs):
        frame = np.zeros((3, 4, 3), dtype="uint8")
        return await TauPolicy(provider, "fake-model", **kwargs)(frame)

    async def test_valid_actions_and_single_png_request(self):
        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({
                "actions": [{"buttons": 1, "frames": 2}, {"buttons": 0, "frames": 1}],
            })))
        ])
        trace = io.StringIO()

        result = await self.decide(provider, max_actions=2, trace=trace)

        self.assertEqual(result, ((1, 2), (0, 1)))
        self.assertEqual(len(provider.calls), 1)
        request = provider.calls[0]
        self.assertEqual(request["model"], "fake-model")
        self.assertEqual(len(request["messages"]), 1)
        image = request["messages"][0].content[1]
        self.assertIsInstance(image, ImageContent)
        self.assertEqual(image.mime_type, "image/png")
        png = base64.b64decode(image.data)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(png)) as decoded:
            self.assertEqual(decoded.size, (4, 3))
        self.assertIn('"role":"assistant"', trace.getvalue())

    async def test_rejects_malformed_actions(self):
        cases = [None, "actions", [], {"buttons": 1}, {"frames": 1}, [None], ["buttons"], [[]]]
        for actions in cases:
            with self.subTest(actions=actions):
                provider = self.FakeProvider([
                    AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({"actions": actions})))
                ])
                self.assertEqual(await self.decide(provider), ((0, 1),))
                self.assertEqual(len(provider.calls), 1)

    async def test_rejects_action_bounds_and_bool_values(self):
        cases = [
            {"buttons": True, "frames": 1},
            {"buttons": 64, "frames": 1},
            {"buttons": 1, "frames": 0},
            {"buttons": 1, "frames": 31},
        ]
        for action in cases:
            with self.subTest(action=action):
                provider = self.FakeProvider([
                    AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({"actions": [action]})))
                ])
                self.assertEqual(await self.decide(provider), ((0, 1),))

    async def test_rejects_too_many_actions_and_multiple_tool_calls(self):
        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({
                "actions": [{"buttons": 1, "frames": 1}, {"buttons": 2, "frames": 1}],
            })))
        ])
        self.assertEqual(await self.decide(provider), ((0, 1),))

        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(
                self.call({"actions": [{"buttons": 1, "frames": 1}]}),
                self.call({"actions": [{"buttons": 2, "frames": 1}]}, call_id="call-2"),
            ))
        ])
        self.assertEqual(await self.decide(provider), ((0, 1),))
        self.assertEqual(len(provider.calls), 1)

    async def test_provider_error_and_text_only_are_rejected(self):
        error = AssistantErrorEvent(
            reason="error",
            error=AssistantMessage(content=[], stop_reason="error", error_message="provider down"),
        )
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            await self.decide(self.FakeProvider([error]))

        text = AssistantDoneEvent(
            reason="stop",
            message=self.assistant(text="I cannot play", stop_reason="stop")
        )
        self.assertEqual(await self.decide(self.FakeProvider([text])), ((0, 1),))

    async def test_rejected_call_is_reported_to_next_request_and_recovers(self):
        rejected = self.assistant(self.call({"actions": []}))
        recovered = self.assistant(self.call({"actions": [{"buttons": 2, "frames": 1}]}))
        provider = self.SequenceProvider([rejected, recovered])
        policy = TauPolicy(provider, "fake-model")

        self.assertEqual(await policy(np.zeros((3, 4, 3), dtype="uint8")), ((0, 1),))
        self.assertEqual(await policy(np.ones((3, 4, 3), dtype="uint8")), ((2, 1),))
        prompt = provider.calls[1]["messages"][-1].content[0].text
        self.assertIn("previous call was rejected", prompt)
        self.assertIn("1..1 actions", prompt)

    async def test_history_keeps_complete_turns_and_evicts_oldest(self):
        message = self.assistant(self.call({"actions": [{"buttons": 1, "frames": 1}]}))
        provider = self.SequenceProvider([message])
        policy = TauPolicy(provider, "fake-model", action_history=16, image_history=1)

        for value in range(18):
            await policy(np.full((3, 4, 3), value, dtype="uint8"))

        context = provider.calls[17]["messages"]
        self.assertEqual(len(context), 16 * 3 + 1)
        self.assertEqual(
            [m.role for m in context],
            (["user", "assistant", "toolResult"] * 16) + ["user"],
        )
        self.assertNotIn("Decision 0", context[0].content[0].text)
        self.assertIn("Decision 1", context[0].content[0].text)
        self.assertEqual(len([m for m in context if m.role == "toolResult"]), 16)

    async def test_zero_action_history_sends_only_current_observation(self):
        message = self.assistant(self.call({"actions": [{"buttons": 1, "frames": 1}]}))
        provider = self.SequenceProvider([message, message])
        policy = TauPolicy(provider, "fake-model", action_history=0, image_history=1)

        await policy(np.zeros((3, 4, 3), dtype="uint8"))
        await policy(np.ones((3, 4, 3), dtype="uint8"))
        self.assertEqual([m.role for m in provider.calls[1]["messages"]], ["user"])

    async def test_image_history_keeps_latest_distinct_screenshots(self):
        message = self.assistant(self.call({"actions": [{"buttons": 1, "frames": 1}]}))
        provider = self.SequenceProvider([message])
        policy = TauPolicy(provider, "fake-model", action_history=4, image_history=3)

        for value in range(4):
            await policy(np.full((3, 4, 3), value, dtype="uint8"))

        users = [m for m in provider.calls[3]["messages"] if m.role == "user"]
        images = [next(block for block in m.content if isinstance(block, ImageContent)).data for m in users
                  if any(isinstance(block, ImageContent) for block in m.content)]
        self.assertEqual(len(images), 3)
        self.assertEqual(len(set(images)), 3)
        pixels = []
        for data in images:
            with Image.open(io.BytesIO(base64.b64decode(data))) as decoded:
                pixels.append(decoded.getpixel((0, 0))[0])
        self.assertEqual(pixels, [1, 2, 3])

    async def test_reasoning_history_controls_outgoing_blocks_but_not_trace(self):
        call = self.call({"actions": [{"buttons": 1, "frames": 1}]}, call_id="signed")
        message = self.reasoning_assistant(call)
        trace = io.StringIO()
        provider = self.SequenceProvider([message, message])
        policy = TauPolicy(provider, "fake-model", reasoning_history=True, trace=trace)
        await policy(np.zeros((3, 4, 3), dtype="uint8"))
        await policy(np.ones((3, 4, 3), dtype="uint8"))
        prior = provider.calls[1]["messages"][1]
        self.assertEqual([type(block) for block in prior.content], [ThinkingContent, TextContent, ToolCall])
        self.assertEqual(prior.content[0].thinking_signature, "sig-thinking")
        self.assertEqual(prior.content[1].text, "I will play")
        self.assertIn("sig-thinking", trace.getvalue())
        self.assertIn("I will play", trace.getvalue())

        provider = self.SequenceProvider([message, message])
        policy = TauPolicy(provider, "fake-model", reasoning_history=False)
        await policy(np.zeros((3, 4, 3), dtype="uint8"))
        await policy(np.ones((3, 4, 3), dtype="uint8"))
        prior = provider.calls[1]["messages"][1]
        self.assertEqual([type(block) for block in prior.content], [ToolCall])
        trace = io.StringIO()
        provider = self.SequenceProvider([message, message])
        policy = TauPolicy(provider, "fake-model", reasoning_history=False, trace=trace)
        await policy(np.zeros((3, 4, 3), dtype="uint8"))
        await policy(np.ones((3, 4, 3), dtype="uint8"))
        self.assertIn("sig-thinking", trace.getvalue())
        self.assertIn("I will play", trace.getvalue())

    async def test_trace_contains_only_new_turns(self):
        message = self.assistant(self.call({"actions": [{"buttons": 1, "frames": 1}]}))
        provider = self.SequenceProvider([message, message])
        trace = io.StringIO()
        policy = TauPolicy(provider, "fake-model", trace=trace)
        await policy(np.zeros((3, 4, 3), dtype="uint8"))
        await policy(np.ones((3, 4, 3), dtype="uint8"))
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(
            [record["role"] for record in records],
            ["user", "assistant", "toolResult", "user", "assistant", "toolResult"],
        )

    async def test_timeout_is_enforced(self):
        trace = io.StringIO()
        with self.assertRaises(asyncio.TimeoutError):
            await self.decide(self.FakeProvider(delay=0.05), timeout=0.001, trace=trace)
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([record["role"] for record in records], ["user", "assistant"])
        self.assertEqual(records[-1]["status"], "timeout")
        self.assertEqual(records[-1]["content"], [])

    async def test_consecutive_interrupted_calls_keep_user_and_assistant_pairs(self):
        trace = io.StringIO()
        for _ in range(2):
            with self.assertRaises(asyncio.TimeoutError):
                await self.decide(self.FakeProvider(delay=0.05), timeout=0.001, trace=trace)
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([record["role"] for record in records], ["user", "assistant", "user", "assistant"])

    async def test_timeout_traces_latest_partial_reasoning_once(self):
        partial = self.assistant(text="thinking")

        class SlowProvider:
            def stream_response(self, **request):
                async def events():
                    yield ThinkingDeltaEvent(
                        content_index=0, delta="thinking", partial=partial
                    )
                    await asyncio.sleep(1)
                return events()

        trace = io.StringIO()
        with self.assertRaises(asyncio.TimeoutError):
            await self.decide(SlowProvider(), timeout=0.05, trace=trace)
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([record["role"] for record in records], ["user", "assistant"])
        self.assertEqual(records[-1]["status"], "timeout")
        self.assertEqual(records[-1]["content"][0]["text"], "thinking")
        self.assertEqual(records[-1]["stopReason"], "aborted")

    async def test_cancellation_traces_latest_partial_reasoning(self):
        partial = self.assistant(text="thinking")
        class SlowProvider:
            def stream_response(self, **request):
                async def events():
                    yield ThinkingDeltaEvent(
                        content_index=0, delta="thinking", partial=partial
                    )
                    await asyncio.sleep(1)
                return events()

        trace = io.StringIO()
        task = asyncio.create_task(self.decide(SlowProvider(), timeout=10, trace=trace))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        record = __import__("json").loads(trace.getvalue().splitlines()[-1])
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["content"][0]["text"], "thinking")


if __name__ == "__main__":
    unittest.main()
