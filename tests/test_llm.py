import asyncio
import base64
import io
import json

import numpy as np
import pytest

from conftest import asyncio_test

try:
    from PIL import Image
    from tau_agent import (
        AssistantMessage,
        ImageContent,
        TextContent,
        ThinkingContent,
        ToolCall,
    )
    from tau_agent.provider_events import (
        AssistantDoneEvent,
        AssistantErrorEvent,
        ThinkingDeltaEvent,
    )

    from celestebench.llm import TauPolicy
except ImportError:
    TauPolicy = None


pytestmark = pytest.mark.skipif(TauPolicy is None, reason="celestebench[llm] is not installed")


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


def assistant(*calls, text=None, stop_reason="toolUse"):
    content = []
    if text is not None:
        content.append(TextContent(text=text))
    content.extend(calls)
    return AssistantMessage(content=content, stop_reason=stop_reason)


def call(arguments, name="play", call_id="call-1"):
    return ToolCall(id=call_id, name=name, arguments=arguments)


async def decide(provider, frames=None, **kwargs):
    if frames is None:
        frames = [np.zeros((3, 4, 3), dtype="uint8")]
    return await TauPolicy(provider, "fake-model", **kwargs)(frames)


@asyncio_test
async def test_system_prompt_reaches_provider_on_every_turn():
    from celestebench.prompt import system_prompt

    for fps, system in [(30, None), (None, None), (30, "Custom instructions"), (30, "")]:
        provider = FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=assistant(call({
                "actions": [{"buttons": 0, "frames": 1}],
            })))
        ])
        policy = TauPolicy(provider, "fake-model", fps=fps, system=system,
                           max_frames=120, max_images=2)
        expected = system if system is not None else system_prompt(
            fps=fps, max_frames=120, max_images=2)
        for _ in range(2):
            await policy([np.zeros((3, 4, 3), dtype="uint8")])
        assert policy.system == expected
        assert [call["system"] for call in provider.calls] == [expected, expected]


@asyncio_test
async def test_valid_actions_and_single_png_request():
    provider = FakeProvider([
        AssistantDoneEvent(reason="toolUse", message=assistant(call({
            "actions": [{"buttons": 1, "frames": 2}, {"buttons": 0, "frames": 1}],
        })))
    ])
    trace = io.StringIO()

    result = await decide(provider, trace=trace)

    assert result == ((1, 2), (0, 1))
    assert len(provider.calls) == 1
    request = provider.calls[0]
    assert request["model"] == "fake-model"
    assert len(request["messages"]) == 1
    image = request["messages"][0].content[1]
    assert isinstance(image, ImageContent)
    assert image.mime_type == "image/png"
    png = base64.b64decode(image.data)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(png)) as decoded:
        assert decoded.size == (4, 3)
    assert '"role":"assistant"' in trace.getvalue()


@asyncio_test
async def test_wait_can_be_mixed_with_button_actions():
    provider = FakeProvider([
        AssistantDoneEvent(reason="toolUse", message=assistant(call({
            "actions": [
                {"buttons": 18, "frames": 2},
                {"action": "wait", "frames": 3},
                {"buttons": 2, "frames": 1},
            ],
        })))
    ])

    result = await decide(provider)

    assert result == ((18, 2), ("wait", 3), (2, 1))
    policy = TauPolicy(provider, "fake-model")
    variants = policy.tool.parameters["properties"]["actions"]["items"]["oneOf"]
    assert variants[1]["required"] == ["action", "frames"]


@asyncio_test
async def test_rejects_malformed_actions():
    cases = [None, "actions", [], {"buttons": 1}, {"frames": 1}, [None], ["buttons"], [[]]]
    for actions in cases:
        provider = FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=assistant(call({"actions": actions})))
        ])
        assert await decide(provider) == ((0, 1),)
        assert len(provider.calls) == 1


@asyncio_test
async def test_rejects_action_bounds_and_bool_values():
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
        provider = FakeProvider([
            AssistantDoneEvent(reason="toolUse", message=assistant(call({"actions": [action]})))
        ])
        assert await decide(provider) == ((0, 1),)


@asyncio_test
async def test_rejects_multiple_tool_calls():
    provider = FakeProvider([
        AssistantDoneEvent(reason="toolUse", message=assistant(
            call({"actions": [{"buttons": 1, "frames": 1}]}),
            call({"actions": [{"buttons": 2, "frames": 1}]}, call_id="call-2"),
        ))
    ])
    assert await decide(provider) == ((0, 1),)
    assert len(provider.calls) == 1


@asyncio_test
async def test_provider_error_and_text_only_are_rejected():
    error = AssistantErrorEvent(
        reason="error",
        error=AssistantMessage(content=[], stop_reason="error", error_message="provider down"),
    )
    with pytest.raises(RuntimeError, match="provider down"):
        await decide(FakeProvider([error]))

    text = AssistantDoneEvent(
        reason="stop",
        message=assistant(text="I cannot play", stop_reason="stop")
    )
    assert await decide(FakeProvider([text])) == ((0, 1),)


@asyncio_test
async def test_length_truncation_falls_back_and_is_fed_back():
    provider = SequenceProvider([
        AssistantMessage(content=[TextContent(text="partial answer")], stop_reason="length"),
        assistant(call({"actions": [{"buttons": 2, "frames": 1}]})),
    ])
    trace = io.StringIO()
    policy = TauPolicy(provider, "fake-model", trace=trace)

    assert await policy([np.zeros((3, 4, 3), dtype="uint8")]) == ((0, 1),)
    assert await policy([np.ones((3, 4, 3), dtype="uint8")]) == ((2, 1),)
    records = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert {"status": "length", "step": 1} in records
    prompt = provider.calls[1]["messages"][-1].content[0].text
    assert "hit the token limit" in prompt


@asyncio_test
async def test_rejected_call_is_reported_to_next_request_and_recovers():
    rejected = assistant(call({"actions": []}))
    recovered = assistant(call({"actions": [{"buttons": 2, "frames": 1}]}))
    provider = SequenceProvider([rejected, recovered])
    policy = TauPolicy(provider, "fake-model")

    assert await policy([np.zeros((3, 4, 3), dtype="uint8")]) == ((0, 1),)
    assert await policy([np.ones((3, 4, 3), dtype="uint8")]) == ((2, 1),)
    prompt = provider.calls[1]["messages"][-1].content[0].text
    assert "previous call was rejected" in prompt
    assert "play requires at least one action" in prompt


@asyncio_test
async def test_history_keeps_all_native_turns_images_reasoning_and_tool_results():
    messages = [
        AssistantMessage(content=[
            ThinkingContent(thinking=f"plan-{step}", thinking_signature=f"sig-{step}"),
            TextContent(text=f"reply-{step}", text_signature=f"text-sig-{step}"),
            call({"actions": [{"buttons": step % 4, "frames": 1}]}, call_id=f"call-{step}"),
        ], stop_reason="toolUse")
        for step in range(20)
    ]
    provider = SequenceProvider(messages)
    policy = TauPolicy(provider, "fake-model", max_images=20)
    harness = policy.harness

    for step in range(20):
        await policy([np.full((3, 4, 3), step, dtype="uint8")])
        assert policy.harness is harness
        assert policy.harness is harness

    history = harness.messages
    assert len(history) == 20 * 3
    assert [message.role for message in history] == [
        role for _ in range(20) for role in ("user", "assistant", "toolResult")
    ]
    for step in range(20):
        user, assistant_message, result = history[step * 3:step * 3 + 3]
        assert f"Decision {step}" in user.content[0].text
        image = next(block for block in user.content if isinstance(block, ImageContent))
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
            assert decoded.getpixel((0, 0))[0] == step
        assert assistant_message.content[0].thinking_signature == f"sig-{step}"
        assert assistant_message.content[1].text_signature == f"text-sig-{step}"
        assert assistant_message.content[2].id == f"call-{step}"
        assert result.tool_call_id == f"call-{step}"

    await policy([np.full((3, 4, 3), 20, dtype="uint8")])
    context = provider.calls[-1]["messages"]
    assert len(context) == 20 * 3 + 1
    assert "Decision 0" in context[0].content[0].text
    assert context[-1].role == "user"


@asyncio_test
async def test_every_frame_since_previous_decision_becomes_an_image():
    provider = FakeProvider([
        AssistantDoneEvent(reason="toolUse", message=assistant(call({
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
    assert len(images) == 3
    for image, level in zip(images, (1, 2, 3), strict=True):
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
            assert decoded.getpixel((0, 0))[0] == level
    # The whole earlier conversation, including the first frame, is still sent.
    assert len(request["messages"]) == 4
    first_user_images = [block for block in request["messages"][0].content
                         if isinstance(block, ImageContent)]
    assert len(first_user_images) == 1


@asyncio_test
async def test_empty_frame_list_is_rejected():
    provider = FakeProvider([])
    policy = TauPolicy(provider, "fake-model")
    with pytest.raises(ValueError, match="at least one frame"):
        await policy([])


@asyncio_test
async def test_image_cap_evicts_oldest_and_keeps_newest():
    provider = SequenceProvider([
        assistant(call({"actions": [{"buttons": 1, "frames": 1}]}))
    ] * 4)
    policy = TauPolicy(provider, "fake-model", max_images=2)
    for step in range(4):
        frames = [np.full((3, 4, 3), step, dtype="uint8")]
        await policy(frames)
    images = [block for message in policy.harness.messages if message.role == "user"
              for block in message.content if isinstance(block, ImageContent)]
    assert len(images) == 2
    # The oldest turns lost their images; the newest ones keep theirs.
    with Image.open(io.BytesIO(base64.b64decode(images[-1].data))) as decoded:
        assert decoded.getpixel((0, 0))[0] == 3
    first_user = policy.harness.messages[0]
    assert not any(isinstance(b, ImageContent) for b in first_user.content)


@asyncio_test
async def test_oversized_decision_is_subsampled_keeping_last_frame():
    provider = FakeProvider([
        AssistantDoneEvent(reason="toolUse", message=assistant(call({
            "actions": [{"buttons": 1, "frames": 1}],
        })))
    ])
    policy = TauPolicy(provider, "fake-model", max_images=5)
    levels = list(range(40))
    await policy([np.full((3, 4, 3), level, dtype="uint8") for level in levels])
    request = provider.calls[0]
    images = [block for block in request["messages"][-1].content if isinstance(block, ImageContent)]
    assert len(images) == 5
    levels_seen = []
    for image in images:
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
            levels_seen.append(decoded.getpixel((0, 0))[0])
    assert levels_seen[-1] == 39  # the current frame survives
    assert levels_seen == sorted(set(levels_seen))  # evenly spaced


@asyncio_test
async def test_trace_contains_only_new_turns():
    message = assistant(call({"actions": [{"buttons": 1, "frames": 1}]}))
    provider = SequenceProvider([message, message])
    trace = io.StringIO()
    policy = TauPolicy(provider, "fake-model", trace=trace)
    await policy(np.zeros((3, 4, 3), dtype="uint8"))
    await policy(np.ones((3, 4, 3), dtype="uint8"))
    records = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [record["role"] for record in records] == [
        "user", "assistant", "toolResult", "user", "assistant", "toolResult"
    ]


@asyncio_test
async def test_cancelled_calls_are_traced_and_recover():
    trace = io.StringIO()
    task = asyncio.create_task(decide(FakeProvider(delay=0.05), trace=trace))
    await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    records = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [record["role"] for record in records] == ["user", "assistant"]
    assert records[-1]["status"] == "cancelled"
    assert records[-1]["content"] == []


@asyncio_test
async def test_consecutive_cancelled_calls_keep_user_and_assistant_pairs():
    trace = io.StringIO()
    for _ in range(2):
        task = asyncio.create_task(decide(FakeProvider(delay=0.05), trace=trace))
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    records = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [record["role"] for record in records] == ["user", "assistant", "user", "assistant"]


@asyncio_test
async def test_cancellation_traces_latest_partial_reasoning():
    partial = assistant(text="thinking")

    class SlowProvider:
        def stream_response(self, **request):
            async def events():
                yield ThinkingDeltaEvent(
                    content_index=0, delta="thinking", partial=partial
                )
                await asyncio.sleep(1)
            return events()

    trace = io.StringIO()
    task = asyncio.create_task(decide(SlowProvider(), trace=trace))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = json.loads(trace.getvalue().splitlines()[-1])
    assert record["status"] == "cancelled"
    assert record["content"][0]["text"] == "thinking"
