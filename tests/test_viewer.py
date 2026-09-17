import contextlib
import http.client
import io
import json
import threading
from unittest.mock import patch

import pytest

from celestebench import BENCHMARK_VERSION
from conftest import video
from web import export, viewer


def run_dir(root, name, *, video=None):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"model": "same-model"}))
    if video is not None:
        (folder / "rollout.mp4").write_bytes(video)
    return folder


def scored_run(root, name, progress=None, *, config=None, elapsed=10, score=None,
               rows=None, **run):
    """Write a scored rollout folder and return its scan_runs() entry."""
    folder = root / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {"fps": 30, "max_frames": 30, "benchmark_version": BENCHMARK_VERSION,
         **(config or {})}))
    if score is None and progress is not None:
        score = {"metric": "grounded_height_v1", "progress": progress,
                 "elapsed": elapsed, "timing": "wall_clock", "status": "completed"}
    if score is not None:
        (folder / "score.json").write_text(json.dumps(score))
    if rows is None and progress is not None:
        rows = [{"elapsed": elapsed, "progress": progress}]
    if rows is not None:
        (folder / "progress.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return {"name": name, "model": "model", "harness": "tau",
            "status": "completed", "timeout": 10, **run}


@contextlib.contextmanager
def patched_runs(root, runs):
    with patch.object(viewer, "RUNS", root), \
            patch.object(viewer, "scan_runs", return_value=runs):
        yield


def fake_handler(**attrs):
    """A request-handler stub exposing the send_* surface the viewer calls."""
    handler = type("FakeHandler", (), {
        "wfile": io.BytesIO(),
        "headers": {},
        "response_headers": {},
        "status": None,
        "connection": type("Connection", (), {"settimeout": lambda self, value: None})(),
        "send_response": lambda self, status: setattr(self, "status", status),
        "send_header": lambda self, name, value: self.response_headers.__setitem__(name, value),
        "end_headers": lambda self: None,
        "send_error": lambda self, status: setattr(self, "status", status),
    })()
    for name, value in attrs.items():
        setattr(handler, name, value)
    return handler


def test_scan_and_decisions_keep_partial_runs_and_exact_frames(tmp_path):
    root = tmp_path
    first = run_dir(root, "a", video=b"corrupt")
    run_dir(root, "nested/b")
    (first / "messages.jsonl").write_text(
        json.dumps({"role": "user", "content": [{"type": "image", "data": "AA=="}]}) + "\n"
        + json.dumps({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "one"},
            {"type": "thinking", "thinking": "two"},
            {"type": "text", "text": "reply"},
            {"type": "toolCall", "arguments": {"actions": [{"buttons": 0, "frames": 2}]}}]}) + "\n"
        + json.dumps({"role": "user", "content": []}) + "\n"
        + "{incomplete\n"
    )
    (first / "actions.jsonl").write_text(json.dumps({
        "decision": 0, "buttons": 0, "frames": 2, "frame_start": 4, "frame_end": 6,
        "latency": 0.25}) + "\n")
    (first / "decisions.jsonl").write_text(json.dumps({
        "decision": 0, "status": "played", "frame_start": 1, "frame_end": 6}) + "\n" + json.dumps({
        "decision": 1, "status": "timeout", "latency": 2, "error": "deadline",
        "frame_start": 6, "frame_end": 60}) + "\n")
    with patch.object(viewer, "RUNS", root):
        runs = viewer.scan_runs()
        data = viewer.load_decisions("a")
    assert {r["name"] for r in runs} == {"a", "nested/b"}
    assert next(r for r in runs if r["name"] == "a")["video"] is None
    assert data["timeline_exact"] is True
    assert len(data["decisions"]) == 2
    assert data["decisions"][0]["actions"][0]["t0"] == pytest.approx(4 / 30, abs=1e-5)
    assert data["decisions"][0]["thinking"] == "one\ntwo"
    assert data["decisions"][1]["status"] == "timeout"
    assert data["decisions"][1]["from"] == pytest.approx(6 / 30)
    assert data["decisions"][1]["actions"] == []


def test_codex_trace_is_merged_into_the_nested_rollout(tmp_path):
    root = tmp_path
    wrapper = root / "gpt" / "run"
    rollout = wrapper / "rollout"
    rollout.mkdir(parents=True)
    (wrapper / "config.json").write_text(json.dumps({"model": "gpt"}))
    (rollout / "config.json").write_text(json.dumps({"model": "gpt"}))
    (rollout / "decisions.jsonl").write_text('\n'.join(json.dumps({
        "decision": index, "status": "played", "frame_start": start, "frame_end": end,
    }) for index, (start, end) in enumerate([(1, 60), (60, 200)])) + "\n")
    (rollout / "actions.jsonl").write_text('\n'.join(json.dumps({
        "decision": index, "buttons": 1, "frames": 2, "frame_start": start,
        "frame_end": start + 2}) for index, start in enumerate([1, 60])) + "\n")
    (wrapper / "codex.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"type": "item.completed", "item": {"type": "reasoning", "text": "plan one"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "go"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
         "arguments": {"actions": [{"buttons": 1, "frames": 2}]},
         "result": {"content": [{"type": "text", "text": json.dumps({"frame_ids": [60]})}]}}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "observe",
         "arguments": {}}},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "plan two"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
         "arguments": {"actions": [{"buttons": 2, "frames": 1}]},
         "result": {"content": [{"type": "text", "text": json.dumps({"frame_ids": [200]})}]}}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "play",
         "arguments": {"actions": [{"buttons": 0, "frames": 1}]},
         "result": {"content": [{"type": "text", "text": "Error executing tool play"}]}}},
    ]) + "\n")
    with patch.object(viewer, "RUNS", root):
        decisions = viewer.load_decisions("gpt/run")["decisions"]
    assert len(decisions) == 2
    assert decisions[0]["thinking"] == "plan one"
    assert decisions[0]["text"] == "go"
    assert decisions[0]["tool"] == {"actions": [{"buttons": 1, "frames": 2}]}
    assert decisions[1]["thinking"] == "plan two"
    assert decisions[1]["tool"] == {"actions": [{"buttons": 2, "frames": 1}]}


def test_normalized_external_run_reads_messages_and_hides_the_nested_rollout(tmp_path):
    root = tmp_path
    wrapper = root / "deepseek" / "run"
    rollout = wrapper / "rollout"
    rollout.mkdir(parents=True)
    (wrapper / "config.json").write_text(json.dumps({"model": "deepseek"}))
    (rollout / "config.json").write_text(json.dumps({"model": "deepseek", "fps": 30}))
    (rollout / "decisions.jsonl").write_text('\n'.join(json.dumps({
        "decision": index, "status": "played", "frame_start": start, "frame_end": end,
    }) for index, (start, end) in enumerate([(1, 30), (30, 90)])) + "\n")
    (rollout / "actions.jsonl").write_text('\n'.join(json.dumps({
        "decision": index, "buttons": 1, "frames": 3, "frame_start": start,
        "frame_end": start + 3}) for index, start in enumerate([1, 30])) + "\n")
    # What examples/opencode.py and examples/pi.py write for the viewer.
    (rollout / "messages.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan one"},
            {"type": "toolCall", "arguments": {"actions": [{"buttons": 1, "frames": 3}]}}],
         "usage": {"input": 10, "output": 2, "cacheRead": 0, "cacheWrite": 0,
                   "totalTokens": 12}},
        {"role": "assistant", "content": [
            {"type": "text", "text": "again"},
            {"type": "toolCall", "arguments": {"actions": [{"buttons": 2, "frames": 3}]}}],
         "usage": {"input": 12, "output": 2, "cacheRead": 0, "cacheWrite": 0,
                   "totalTokens": 14}},
    ]) + "\n")
    (root / ".evals").mkdir()
    (root / ".evals" / "job.json").write_text(json.dumps({
        "id": "job", "name": "deepseek/run", "model": "deepseek", "harness": "opencode",
        "provider": None, "status": "completed", "error": None, "decisions": 2,
        "timeout": 120, "frames": 90, "elapsed": 120, "tokens": 26, "started_at": 1.0}))
    with patch.object(viewer, "RUNS", root):
        runs = viewer.scan_runs()
        decisions = viewer.load_decisions("deepseek/run")["decisions"]
    assert [run["name"] for run in runs] == ["deepseek/run"]
    assert runs[0]["harness"] == "opencode"
    assert len(decisions) == 2
    assert decisions[0]["thinking"] == "plan one"
    assert decisions[0]["tool"] == {"actions": [{"buttons": 1, "frames": 3}]}
    assert decisions[1]["text"] == "again"


def test_legacy_actions_are_approximate_and_start_after_observation_frame(tmp_path):
    root = tmp_path
    folder = run_dir(root, "legacy")
    (folder / "config.json").write_text(json.dumps({"model": "m", "fps": 20}))
    (folder / "messages.jsonl").write_text(json.dumps({"role": "user", "content": []}) + "\n"
                                             + json.dumps({"role": "assistant", "content": []}) + "\n")
    (folder / "actions.jsonl").write_text(json.dumps({"decision": 0, "buttons": 2,
                                                       "frames": 3, "latency": 0.5}) + "\n")
    with patch.object(viewer, "RUNS", root):
        data = viewer.load_decisions("legacy")
    action = data["decisions"][0]["actions"][0]
    assert data["timeline_exact"] is False
    assert action["frame_start"] == 11  # one initial frame + ten idle frames
    # Pacing is 20 Hz, but video is native 30 fps, not wall-clock time.
    assert action["t0"] == pytest.approx(11 / 30, abs=1e-5)


def test_http_ranges_and_path_containment(tmp_path):
    root = tmp_path
    run_dir(root, "run", video=b"0123456789")
    fake = fake_handler(path="/video/run/rollout.mp4", headers={"Range": "bytes=2-5"})
    viewer.Handler.video(fake, root / "run" / "rollout.mp4")
    assert fake.status == 206
    assert fake.wfile.getvalue() == b"2345"
    fake.headers = {"Range": "bytes=-3"}
    fake.wfile = io.BytesIO()
    viewer.Handler.video(fake, root / "run" / "rollout.mp4")
    assert fake.wfile.getvalue() == b"789"
    fake.headers = {"Range": "bytes=20-30"}
    viewer.Handler.video(fake, root / "run" / "rollout.mp4")
    assert fake.status == 416
    with patch.object(viewer, "RUNS", root):
        fake.path = "/video/../viewer.py"
        fake.headers = {}
        viewer.Handler.do_GET(fake)
        assert fake.status == 404


def test_live_stream_waits_for_a_late_frame_and_ends_with_the_job(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    fake = fake_handler()
    calls = {"count": 0}
    def running():
        calls["count"] += 1
        return calls["count"] < 3
    # No frame on disk yet: the stream must stay open while the job runs,
    # then stop on its own when the harness reports completion.
    viewer.Handler.live(fake, folder, running)
    assert fake.status == 200
    assert fake.wfile.getvalue().endswith(b"--frame--\r\n")


def test_live_stream_finds_a_nested_rollout_created_after_it_opens(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    frame = b"\x89PNG\r\nnested"
    calls = {"count": 0}
    def running():
        calls["count"] += 1
        if calls["count"] == 2:  # the harness starts the game mid-stream
            rollout = folder / "rollout"
            rollout.mkdir()
            (rollout / "live.png").write_bytes(frame)
            (rollout / "live.done").touch()
        return calls["count"] < 4
    fake = fake_handler()
    viewer.Handler.live(fake, folder, running)
    assert frame in fake.wfile.getvalue()


def test_job_running_only_reports_live_jobs():
    with patch.dict(viewer.evals._jobs, {
            "alive": {"status": "running"}, "done": {"status": "completed"}}):
        assert viewer.evals.job_running("alive")
        assert not viewer.evals.job_running("done")
        assert not viewer.evals.job_running("missing")


def test_timeout_reasoning_without_video_and_legacy_batch_order(tmp_path):
    root = tmp_path
    folder = run_dir(root, "partial")
    (folder / "messages.jsonl").write_text('\n'.join(json.dumps(m) for m in [
        {"role": "user", "content": []},
        {"role": "assistant", "status": "timeout", "stopReason": "aborted",
         "errorMessage": "deadline", "content": [{"type": "thinking", "thinking": "partial text"}]},
    ]))
    with patch.object(viewer, "RUNS", root):
        d = viewer.load_decisions("partial")["decisions"][0]
    assert (d["status"], d["thinking"], d["actions"]) == ("timeout", "partial text", [])
    assert d["partial"]
    (folder / "actions.jsonl").write_text('\n'.join(json.dumps({
        "decision": 0, "buttons": b, "frames": n, "latency": 0}) for b, n in [(0, 2), (2, 3)]))
    with patch.object(viewer, "RUNS", root):
        actions = viewer.load_decisions("partial")["decisions"][0]["actions"]
    assert [(a["frame_start"], a["frame_end"]) for a in actions] == [(1, 3), (3, 6)]


def test_persisted_screenshot_is_used_and_served_safely(tmp_path):
    root = tmp_path
    folder = run_dir(root, "run")
    screenshot = folder / "screenshots" / "000000.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"\x89PNG\r\n")
    (folder / "decisions.jsonl").write_text(json.dumps({
        "decision": 0, "status": "timeout", "screenshot": "screenshots/000000.png"}) + "\n")
    with patch.object(viewer, "RUNS", root):
        decision = viewer.load_decisions("run")["decisions"][0]
        assert decision["screenshot"] == "/screenshot/run/screenshots/000000.png"
        fake = fake_handler(path="/screenshot/run/screenshots/000000.png")
        fake.image = viewer.Handler.image.__get__(fake)
        viewer.Handler.do_GET(fake)
        assert fake.status == 200
        assert fake.wfile.getvalue() == b"\x89PNG\r\n"
        fake.path = "/screenshot/../viewer.py"
        viewer.Handler.do_GET(fake)
        assert fake.status == 404


def test_wait_marker_is_preserved_for_the_viewer(tmp_path):
    root = tmp_path
    folder = run_dir(root, "run")
    (folder / "actions.jsonl").write_text(json.dumps({
        "decision": 0, "buttons": 0, "frames": 3, "action": "wait",
    }) + "\n")
    with patch.object(viewer, "RUNS", root):
        action = viewer.load_decisions("run")["decisions"][0]["actions"][0]
    assert action["wait"]


def test_leaderboard_uses_one_wall_clock_event_and_keeps_unscored_runs(tmp_path):
    root = tmp_path
    runs = [scored_run(root, name, progress, config={"thinking_level": level},
                       rows=[{"elapsed": 5, "progress": progress - 10},
                             {"elapsed": 10, "progress": progress}])
            for name, progress, level in (("a", 40, "low"), ("b", 60, "low"),
                                          ("different", 90, "off"))]
    runs.append(scored_run(root, "short", 80, elapsed=4, config={"thinking_level": "low"}))
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    assert result["excluded"] == {"replay": 0, "unknown_timing": 0}
    grouped = {(row["settings"]["thinking_level"]): row for row in result["groups"]}
    assert grouped["low"]["progress"] == 50
    assert grouped["low"]["scored"] == 2
    assert grouped["low"]["unscored"] == 1
    assert grouped["low"]["scored_runs"] == [{"name": "b", "progress": 60},
                                             {"name": "a", "progress": 40}]
    assert grouped["off"]["progress"] == 90
    assert grouped["off"]["scored_runs"] == [{"name": "different", "progress": 90}]


def test_leaderboard_does_not_score_replay_or_legacy_scoreless_runs(tmp_path):
    root = tmp_path
    runs = [
        scored_run(root, "replay", 70, config={}, rows=[{"elapsed": 10, "progress": 70}],
                   score={"metric": "grounded_height_v1", "progress": 70, "elapsed": 10,
                          "timing": "replay", "status": "completed"}, model="m"),
        scored_run(root, "legacy", config={}, model="m"),
    ]
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    assert result["excluded"]["replay"] == 1
    assert all(row["status"] == "unscored" for row in result["groups"])


def test_leaderboard_reads_score_options_and_excludes_failed_archives(tmp_path):
    root = tmp_path
    runs = []
    for name, level, status, system in [
            ("a", "low", "completed", "first"), ("b", "off", "completed", "first"),
            ("c", "low", "error", "first"), ("d", "low", "completed", "other")]:
        score = {"metric": "grounded_height_v1", "progress": 2, "elapsed": 20,
                 "timing": "wall_clock", "status": status,
                 "options": {"thinking_level": level, "fps": 30}}
        runs.append(scored_run(
            root, name, config={"system": system}, score=score, model="m",
            status="archived", timeout=20,
            rows=[{"elapsed": 0, "progress": 0}, {"elapsed": 9, "progress": 1},
                  {"elapsed": 11, "progress": 2}]))
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    assert len(result["groups"]) == 2
    assert sum(row["scored"] for row in result["groups"]) == 3
    assert all(row["progress"] == 1 for row in result["groups"])
    assert {row["settings"]["thinking_level"] for row in result["groups"]} == {"low", "off"}


def test_leaderboard_excludes_invalid_scores_and_merges_prompt_metadata(tmp_path):
    root = tmp_path
    runs = []
    for name, config, progress, invalid in [
            ("verified", {"system_prompt_sent": True}, 10, None),
            ("unknown", {}, 20, None),
            ("invalid", {"system_prompt_sent": True}, 999, "missing_system_prompt")]:
        score = {"metric": "grounded_height_v1", "progress": progress,
                 "elapsed": 10, "timing": "wall_clock", "status": "completed"}
        if invalid:
            score["invalid_reason"] = invalid
        runs.append(scored_run(root, name, progress, config=config, score=score, model="m"))
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    assert len(result["groups"]) == 1
    assert result["groups"][0]["progress"] == 15
    assert result["groups"][0]["scored"] == 2


def test_leaderboard_prices_rollouts_and_hides_older_versions(tmp_path):
    from celestebench import catalog
    root = tmp_path
    runs = []
    for name, version in (("old", "0.1"), ("current", BENCHMARK_VERSION)):
        runs.append(scored_run(
            root, name, 50, config={"model": "gpt-5.6-sol", "benchmark_version": version},
            model="gpt-5.6-sol"))
        (root / name / "messages.jsonl").write_text(json.dumps({
            "role": "assistant",
            "usage": {"input": 1000, "output": 2000, "cacheRead": 0,
                      "cacheWrite": 0}}) + "\n")
    prices = {"openai": {"models": {"gpt-5.6-sol": {
        "id": "gpt-5.6-sol", "cost": {"input": 4, "output": 20}}}}}
    with patched_runs(root, runs), patch.object(catalog, "_models_dev", return_value=prices):
        result = viewer.leaderboard(10)
    assert result["version"] == BENCHMARK_VERSION
    assert len(result["groups"]) == 1
    row = result["groups"][0]
    assert row["settings"]["benchmark_version"] == BENCHMARK_VERSION
    assert row["producer"] == "openai"
    assert row["producer_name"] == "OpenAI"
    assert row["cost"] == pytest.approx((1000 * 4 + 2000 * 20) / 1_000_000, abs=1e-6)


def test_leaderboard_folds_old_none_into_thinking_off(tmp_path):
    root = tmp_path
    runs = [scored_run(root, name, 50, config=level)
            for name, level in (("old", {"reasoning_effort": "none"}),
                                ("new", {"thinking_level": "off"}),
                                ("hot", {"reasoning_effort": "high"}))]
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    merged = next(row for row in result["groups"]
                  if row["settings"]["thinking_level"] == "off")
    assert merged["scored"] == 2
    assert "reasoning_effort" not in merged["settings"]
    hot = next(row for row in result["groups"]
               if row["settings"]["thinking_level"] == "high")
    assert hot["scored"] == 1


def test_leaderboard_keeps_harnesses_and_setting_generations_apart(tmp_path):
    root = tmp_path
    runs = []
    for name, harness, config, progress in (
            ("tau-old", "tau", {"api": "openai-responses"}, 40),
            ("tau-new", "tau", {"provider": "opencode-go"}, 60),
            ("codex", "codex", {}, 80)):
        runs.append(scored_run(root, name, progress,
                               config={"thinking_level": "low", **config}, harness=harness))
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    assert len(result["groups"]) == 3
    assert {row["settings"]["harness"] for row in result["groups"]} == {"tau", "codex"}


def test_leaderboard_switches_between_rtc_and_lite(tmp_path):
    root = tmp_path
    runs = [scored_run(root, name, progress, config={"model": "m", "fps": fps}, model="m")
            for name, fps, progress in (("rtc", 30, 40), ("lite", None, 80))]
    with patched_runs(root, runs):
        rtc = viewer.leaderboard(10, "rtc")
        lite = viewer.leaderboard(10, "lite")
    assert rtc["modes"] == ["lite", "rtc"]
    assert rtc["mode"] == "rtc"
    assert [row["progress"] for row in rtc["groups"]] == [40]
    assert lite["mode"] == "lite"
    assert [row["progress"] for row in lite["groups"]] == [80]


def test_leaderboard_folds_model_aliases_and_gateway_prefixes(tmp_path):
    root = tmp_path
    runs = [scored_run(root, name, 50, config={"thinking_level": "low"},
                       model=model, harness=harness)
            for name, model, harness in (
                ("tau", "deepseek-flash", "tau"),
                ("opencode", "opencode-go/deepseek-v4.1-flash", "opencode"),
                ("vision", "deepseek-v4-flash-vision-exp", "tau"))]
    with patched_runs(root, runs):
        result = viewer.leaderboard(10)
    pairs = {(row["model"], row["settings"]["harness"]) for row in result["groups"]}
    assert pairs == {("deepseek-v4.1-flash", "tau"),
                     ("deepseek-v4.1-flash", "opencode"),
                     ("deepseek-v4-flash-vision-exp", "tau")}


@pytest.fixture
def connection():
    httpd = viewer.ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
    try:
        yield connection
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def post(connection, path, payload, **headers):
    connection.request("POST", path, json.dumps(payload),
                       {"Content-Type": "application/json", **headers})
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def get(connection, path):
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, response.getheader("Content-Type"), response.read().decode()


def get_bytes(connection, path):
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def delete(connection, path, **headers):
    connection.request("DELETE", path, headers=headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def test_json_ignores_a_client_that_disconnects_before_the_body():
    class ClosedClient:
        def write(self, body):
            raise BrokenPipeError

    handler = fake_handler(wfile=ClosedClient())
    viewer.Handler.json(handler, {"ok": True})


def test_harnesses_endpoint_feeds_the_new_eval_form(connection):
    status, content_type, body = get(connection, "/api/harnesses")
    assert (status, content_type) == (200, "application/json")
    payload = json.loads(body)
    catalog = {entry["key"]: entry for entry in payload["harnesses"]}
    assert [mode["name"] for mode in payload["modes"]] == ["rtc", "lite"]
    assert catalog["tau"]["builtin"]
    assert not catalog["codex"]["builtin"]
    assert [field["key"] for field in catalog["codex"]["run"]] == ["model", "thinking_level"]
    assert [field["key"] for field in catalog["codex"]["options"]] == ["prompt"]


def test_runs_and_evaluations_have_separate_linkable_pages(connection):
    status, content_type, root = get(connection, "/")
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert 'id="leaderboard"' in root
    assert 'id="budget"' in root
    assert 'href="/runs"' in root

    status, content_type, runs = get(connection, "/runs")
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert 'href="/evals"' in runs
    assert 'href="/leaderboard"' in runs
    assert 'new URLSearchParams(location.search).get("run")' in runs
    assert 'id="evalJobs"' not in runs
    assert '"/api/evals"' not in runs

    status, content_type, evaluations = get(connection, "/evals")
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert 'id="evalJobs"' in evaluations
    assert 'jsonRequest("/api/evals", {timeout: 10000})' in evaluations
    assert 'href="/runs?run=${encodeURIComponent(job.name)}"' in evaluations
    assert '.filter(job => ["queued", "running"].includes(job.status))' in evaluations
    assert 'job.run_ready !== false' in evaluations
    assert 'class="eval-live" data-run="${esc(job.name)}"' in evaluations
    assert 'img.src = "/frame/" + encodeURIComponent(img.dataset.run)' in evaluations
    assert 'id="evalMode"' in evaluations
    assert "live-audio" not in evaluations

    status, content_type, css = get(connection, "/viewer.css")
    assert (status, content_type) == (200, "text/css; charset=utf-8")
    assert ".card" in css

    status, content_type, leaderboard = get(connection, "/leaderboard")
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert 'id="budget"' in leaderboard
    assert 'id="leaderboard"' in leaderboard
    assert 'id="modeSwitch"' in leaderboard
    assert 'href="/runs?run=${encodeURIComponent(' in leaderboard


def test_delete_run_removes_files_and_terminal_evaluation_metadata(tmp_path, connection):
    root = tmp_path / "runs"
    folder = root / "bad-model" / "run"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    metadata = root / ".evals" / "job.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({
        "id": "job", "name": "bad-model/run", "status": "failed",
    }))
    viewer.evals._jobs["job"] = {"id": "job", "name": "bad-model/run", "status": "failed"}
    try:
        with patch.object(viewer, "RUNS", root):
            assert delete(connection, "/api/run/bad-model%2Frun") == (200, {"deleted": "bad-model/run"})
        assert not folder.exists()
        assert not metadata.exists()
        assert "job" not in viewer.evals._jobs
    finally:
        viewer.evals._jobs.pop("job", None)


def test_delete_run_rejects_active_jobs_and_unsafe_paths(tmp_path, connection):
    base = tmp_path
    root = base / "runs"
    folder = root / "model" / "run"
    folder.mkdir(parents=True)
    outside = base / "outside"
    outside.mkdir()
    metadata = root / ".evals" / "job.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({
        "id": "job", "name": "model/run", "status": "running",
    }))
    viewer.evals._jobs["job"] = {"id": "job", "name": "model/run", "status": "running"}
    try:
        with patch.object(viewer, "RUNS", root):
            assert delete(connection, "/api/run/model%2Frun")[0] == 409
            assert delete(connection, "/api/run/..%2Foutside")[0] == 404
            assert delete(connection, "/api/run/.evals")[0] == 404
        assert folder.exists()
        assert outside.exists()
        assert metadata.exists()
    finally:
        viewer.evals._jobs.pop("job", None)


def test_runs_page_confirms_before_deleting_the_selected_run(connection):
    runs = get(connection, "/runs")[2]
    assert 'id="deleteRun"' in runs
    assert "confirm(`Delete run" in runs
    assert '{method:"DELETE"}' in runs
    assert '"/live/" + encodeURIComponent(run.name)' in runs
    assert 'id="liveSound"' not in runs


def test_live_endpoint_streams_the_atomic_frame(tmp_path, connection):
    root = tmp_path
    folder = root / "model" / "run"
    folder.mkdir(parents=True)
    frame = b"\x89PNG\r\nlatest"
    (folder / "live.png").write_bytes(frame)
    (folder / "live.done").touch()
    job = {"name": "model/run", "status": "running"}
    with patch.object(viewer, "RUNS", root), patch.object(
            viewer.evals, "list_evals", return_value=[job]):
        status, content_type, body = get_bytes(connection, "/live/model%2Frun")
    assert status == 200
    assert content_type == "multipart/x-mixed-replace; boundary=frame"
    assert b"Content-Type: image/png" in body
    assert frame in body


def test_frame_endpoint_serves_the_latest_live_frame(tmp_path, connection):
    root = tmp_path
    folder = root / "model" / "run"
    folder.mkdir(parents=True)
    frame = b"\x89PNG\r\nlive frame"
    (folder / "live.png").write_bytes(frame)
    nested = root / "nested" / "run" / "rollout"
    nested.mkdir(parents=True)
    (nested / "live.png").write_bytes(frame)
    with patch.object(viewer, "RUNS", root):
        status, content_type, body = get_bytes(connection, "/frame/model%2Frun")
        assert (status, content_type) == (200, "image/png")
        assert body == frame
        assert get_bytes(connection, "/frame/nested%2Frun")[2] == frame
        assert get_bytes(connection, "/frame/model%2Fmissing")[0] == 404
        assert get_bytes(connection, "/frame/..%2Foutside")[0] == 404


def test_launch_stop_and_validation_errors_are_json(connection):
    job = {"id": "abc", "name": "model/run", "status": "running"}
    payload = {"model": "custom-vlm", "provider": "custom",
               "base_url": "http://localhost:9000/v1", "timeout": 120}
    with patch.object(viewer.evals, "start_eval", return_value=[job]) as launch:
        assert post(connection, "/api/evals", payload) == (201, [job])
        launch.assert_called_once_with(payload, viewer.RUNS)
    with patch.object(viewer.evals, "stop_eval", return_value=job) as stop:
        assert post(connection, "/api/evals/abc/stop", {}) == (200, job)
        stop.assert_called_once_with("abc")
    with patch.object(viewer.evals, "start_eval", side_effect=ValueError("Invalid model")):
        assert post(connection, "/api/evals", {}) == (400, {"error": "Invalid model"})


def test_video_stays_raw_and_export_endpoint_downloads_the_annotation(tmp_path, connection):
    root = tmp_path
    folder = root / "run"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": "m"}))
    (folder / "actions.jsonl").write_text(json.dumps({
        "decision": 0, "buttons": 2, "frames": 2,
        "frame_start": 1, "frame_end": 3}) + "\n")
    video(folder / "rollout.mp4")
    with patch.object(viewer, "RUNS", root), patch.object(export, "RUNS", root):
        status, content_type, body = get_bytes(connection, "/video/run/rollout.mp4")
        assert (status, content_type) == (200, "video/mp4")
        assert body == (folder / "rollout.mp4").read_bytes()
        assert not (folder / "export.mp4").is_file()
        connection.request("GET", "/export/run")
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "video/mp4"
        assert response.getheader("Content-Disposition") == 'attachment; filename="run.mp4"'
    assert (folder / "export.mp4").is_file()
    assert body == (folder / "export.mp4").read_bytes()


def test_cross_origin_and_non_json_requests_cannot_launch(connection):
    with patch.object(viewer.evals, "start_eval") as launch:
        assert post(connection, "/api/evals", {}, Origin="https://example.com")[0] == 403
        assert post(connection, "/api/evals", {}, Host="example.com")[0] == 403
        assert post(connection, "/api/evals", {}, **{"Content-Type": "text/plain"})[0] == 415
        launch.assert_not_called()
