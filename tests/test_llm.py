import asyncio
import base64
import io
import json
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
    def call(arguments, name="play", call_id="call-1"):
        return ToolCall(id=call_id, name=name, arguments=arguments)

    async def decide(self, provider, frames=None, **kwargs):
        if frames is None:
            frames = [np.zeros((3, 4, 3), dtype="uint8")]
        return await TauPolicy(provider, "fake-model", **kwargs)(frames)

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

    async def test_wait_can_be_mixed_with_button_actions(self):
        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({
                "actions": [
                    {"buttons": 18, "frames": 2},
                    {"action": "wait", "frames": 3},
                    {"buttons": 2, "frames": 1},
                ],
            })))
        ])

        result = await self.decide(provider)

        self.assertEqual(result, ((18, 2), ("wait", 3), (2, 1)))
        policy = TauPolicy(provider, "fake-model")
        variants = policy.tool.parameters["properties"]["actions"]["items"]["oneOf"]
        self.assertEqual(variants[1]["required"], ["action", "frames"])
        self.assertEqual(policy.max_actions, 4)

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
            {"action": "wait", "frames": 0},
            {"action": "wait", "frames": 31},
            {"action": "pause", "frames": 1},
            {"action": "wait", "frames": True},
            {"action": "wait", "frames": 1, "buttons": 0},
            {"wait": 1},
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
        self.assertEqual(await self.decide(provider, max_actions=1), ((0, 1),))

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

    async def test_length_truncation_falls_back_and_is_fed_back(self):
        provider = self.SequenceProvider([
            AssistantMessage(content=[TextContent(text="partial answer")], stop_reason="length"),
            self.assistant(self.call({"actions": [{"buttons": 2, "frames": 1}]})),
        ])
        trace = io.StringIO()
        policy = TauPolicy(provider, "fake-model", trace=trace)

        self.assertEqual(await policy([np.zeros((3, 4, 3), dtype="uint8")]), ((0, 1),))
        self.assertEqual(await policy([np.ones((3, 4, 3), dtype="uint8")]), ((2, 1),))
        records = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertIn({"status": "length", "step": 1}, records)
        prompt = provider.calls[1]["messages"][-1].content[0].text
        self.assertIn("hit the token limit", prompt)

    async def test_rejected_call_is_reported_to_next_request_and_recovers(self):
        rejected = self.assistant(self.call({"actions": []}))
        recovered = self.assistant(self.call({"actions": [{"buttons": 2, "frames": 1}]}))
        provider = self.SequenceProvider([rejected, recovered])
        policy = TauPolicy(provider, "fake-model")

        self.assertEqual(await policy([np.zeros((3, 4, 3), dtype="uint8")]), ((0, 1),))
        self.assertEqual(await policy([np.ones((3, 4, 3), dtype="uint8")]), ((2, 1),))
        prompt = provider.calls[1]["messages"][-1].content[0].text
        self.assertIn("previous call was rejected", prompt)
        self.assertIn("1..4 actions", prompt)

    async def test_history_keeps_all_native_turns_images_reasoning_and_tool_results(self):
        messages = [
            AssistantMessage(content=[
                ThinkingContent(thinking=f"plan-{step}", thinking_signature=f"sig-{step}"),
                TextContent(text=f"reply-{step}", text_signature=f"text-sig-{step}"),
                self.call({"actions": [{"buttons": step % 4, "frames": 1}]}, call_id=f"call-{step}"),
            ], stop_reason="toolUse")
            for step in range(20)
        ]
        provider = self.SequenceProvider(messages)
        policy = TauPolicy(provider, "fake-model", max_images=20)
        harness = policy.harness

        for step in range(20):
            await policy([np.full((3, 4, 3), step, dtype="uint8")])
            self.assertIs(policy.harness, harness)
            self.assertIs(policy.harness, harness)

        history = harness.messages
        self.assertEqual(len(history), 20 * 3)
        self.assertEqual(
            [message.role for message in history],
            [role for _ in range(20) for role in ("user", "assistant", "toolResult")],
        )
        for step in range(20):
            user, assistant, result = history[step * 3:step * 3 + 3]
            self.assertIn(f"Decision {step}", user.content[0].text)
            image = next(block for block in user.content if isinstance(block, ImageContent))
            with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
                self.assertEqual(decoded.getpixel((0, 0))[0], step)
            self.assertEqual(assistant.content[0].thinking_signature, f"sig-{step}")
            self.assertEqual(assistant.content[1].text_signature, f"text-sig-{step}")
            self.assertEqual(assistant.content[2].id, f"call-{step}")
            self.assertEqual(result.tool_call_id, f"call-{step}")

        await policy([np.full((3, 4, 3), 20, dtype="uint8")])
        context = provider.calls[-1]["messages"]
        self.assertEqual(len(context), 20 * 3 + 1)
        self.assertIn("Decision 0", context[0].content[0].text)
        self.assertEqual(context[-1].role, "user")

    async def test_every_frame_since_previous_decision_becomes_an_image(self):
        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({
                "actions": [{"buttons": 2, "frames": 1}],
            })))
        ])
        trace = io.StringIO()
        policy = TauPolicy(provider, "fake-model", max_images=4, trace=trace)

        await policy([np.full((3, 4, 3), 0, dtype="uint8")])
        await policy([
            np.full((3, 4, 3), 1, dtype="uint8"),
            np.full((3, 4, 3), 2, dtype="uint8"),
            np.full((3, 4, 3), 3, dtype="uint8"),
        ])

        request = provider.calls[1]
        content = request["messages"][-1].content
        images = [block for block in content if isinstance(block, ImageContent)]
        self.assertEqual(len(images), 3)
        for image, level in zip(images, (1, 2, 3), strict=True):
            with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
                self.assertEqual(decoded.getpixel((0, 0))[0], level)
        # The whole earlier conversation, including the first frame, is still sent.
        self.assertEqual(len(request["messages"]), 4)
        first_user_images = [block for block in request["messages"][0].content
                             if isinstance(block, ImageContent)]
        self.assertEqual(len(first_user_images), 1)

    async def test_empty_frame_list_is_rejected(self):
        provider = self.FakeProvider([])
        policy = TauPolicy(provider, "fake-model")
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            await policy([])

    async def test_image_cap_evicts_oldest_and_keeps_newest(self):
        provider = self.SequenceProvider([
            self.assistant(self.call({"actions": [{"buttons": 1, "frames": 1}]}))
        ] * 4)
        policy = TauPolicy(provider, "fake-model", max_images=2)
        for step in range(4):
            frames = [np.full((3, 4, 3), step, dtype="uint8")]
            await policy(frames)
        images = [block for message in policy.harness.messages if message.role == "user"
                  for block in message.content if isinstance(block, ImageContent)]
        self.assertEqual(len(images), 2)
        # The oldest turns lost their images; the newest ones keep theirs.
        with Image.open(io.BytesIO(base64.b64decode(images[-1].data))) as decoded:
            self.assertEqual(decoded.getpixel((0, 0))[0], 3)
        first_user = policy.harness.messages[0]
        self.assertFalse(any(isinstance(b, ImageContent) for b in first_user.content))

    async def test_oversized_decision_is_subsampled_keeping_last_frame(self):
        provider = self.FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=self.assistant(self.call({
                "actions": [{"buttons": 1, "frames": 1}],
            })))
        ])
        policy = TauPolicy(provider, "fake-model", max_images=5)
        levels = list(range(40))
        await policy([np.full((3, 4, 3), level, dtype="uint8") for level in levels])
        request = provider.calls[0]
        images = [block for block in request["messages"][-1].content if isinstance(block, ImageContent)]
        self.assertEqual(len(images), 5)
        levels_seen = []
        for image in images:
            with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
                levels_seen.append(decoded.getpixel((0, 0))[0])
        self.assertEqual(levels_seen[-1], 39)  # the current frame survives
        self.assertEqual(levels_seen, sorted(set(levels_seen)))  # evenly spaced

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

    async def test_cancelled_calls_are_traced_and_recover(self):
        trace = io.StringIO()
        task = asyncio.create_task(self.decide(self.FakeProvider(delay=0.05), trace=trace))
        await asyncio.sleep(0.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([record["role"] for record in records], ["user", "assistant"])
        self.assertEqual(records[-1]["status"], "cancelled")
        self.assertEqual(records[-1]["content"], [])

    async def test_consecutive_cancelled_calls_keep_user_and_assistant_pairs(self):
        trace = io.StringIO()
        for _ in range(2):
            task = asyncio.create_task(self.decide(self.FakeProvider(delay=0.05), trace=trace))
            await asyncio.sleep(0.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        records = [__import__("json").loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual([record["role"] for record in records], ["user", "assistant", "user", "assistant"])

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
        task = asyncio.create_task(self.decide(SlowProvider(), trace=trace))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        record = __import__("json").loads(trace.getvalue().splitlines()[-1])
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["content"][0]["text"], "thinking")


if __name__ == "__main__":
    unittest.main()
