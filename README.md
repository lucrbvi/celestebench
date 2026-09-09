# celestebench

Small synchronous Python interface to Open8 for Celeste Classic rollouts.
Use it from this source checkout; the native library and cartridge stay here.

```sh
git submodule update --init --recursive
uv sync
make -j4  # requires a C compiler and CMake; downloads/builds SDL3 if absent
uv run python -m unittest discover -s tests
```

```python
from pathlib import Path
from celestebench import Button, Open8

with Open8() as env:  # loads Celeste Classic, with a fixed initial seed
    with env.record("rollout.mp4"):
        image = env.step(frames=60)
        env.step(Button.O)  # start
        state = env.save_state()
        image = env.step(Button.RIGHT | Button.O, frames=30)
        env.load_state(state)
        image = env.step(Button.LEFT, frames=30)
    Path("checkpoint.state").write_bytes(state)
```

- `step(buttons=0, frames=1)` holds a button mask and returns a new RGBA
  `uint8` array, shape `(128, 128, 4)`. Buttons combine with `|`.
- `framebuffer` reads the current image. `reset()` restarts the cartridge.
- The game advances **only during `step`** (or checkpoint replay). Between
  calls it is paused; no background loop, speed setting or real-time pacing.
- `save_state()` returns immutable action-history bytes. `load_state(state)`
  reloads the same cartridge and replays them, without recording replay frames.
  This is **not a VM snapshot**: restoration cost grows with rollout length.
  States require the same cartridge and emulator version. Celeste replay is
  tested pixel-for-pixel; arbitrary cartridges using wall-clock time may diverge.
  The old `CBST` file format is no longer supported.
- `record(path)` writes every stepped frame to a lossless H.264 MP4 with mono
  AAC game audio. Frames use yuv444p/qp 0 and nearest-neighbor 4x upscaling
  (512x512), so playback stays crisp at the cart's native 30/60 fps. Recording
  finalizes even when the
  body raises. Step at least
  once to produce a video. Reset/restore inside the context creates a cut.
  PyAV uses FFmpeg libraries directly; no `ffmpeg` executable or subprocess.
- Open8 has global state: **one environment per process**, used from one thread.
  Use separate processes for parallel rollouts. Always close it with `with`.

This is the emulator interface, without a reward function or a Gym wrapper.

## Harness

`rollout(policy, output, *, timeout=None, max_frames=30, max_actions=1, frames=None, fps=None)` drives
a policy that takes the last frame and returns `(buttons, frames)`, `("wait", frames)`, or a list — sync or
async. It writes `rollout.mp4`, an `actions.jsonl` log with latencies, and a final
`checkpoint.state` into a fresh `output` directory. While running, `live.png` and
`live.pcm` expose the latest frame and 22.05 kHz mono audio to the viewer. The entire action batch is validated before
execution. `max_actions` bounds every policy, including non-LLM policies. There is no screenshot
between actions in a batch. The initial observation advances and records one neutral frame.
`timeout` is the wall-clock budget for the whole rollout: when it expires the pending policy call
is cancelled, its decision is logged with status `timeout`, and the run ends gracefully; a batch
already being played finishes, so the overshoot stays under one batch. `frames` caps total
environment frames (long holds are cut); `fps` runs the environment in real time: frames tick at
that rate with buttons released while the policy thinks, so slow models waste world time instead
of blocking the run.

Each action log row contains its decision index. `latency` is the policy's wall
time, recorded on the first action of that decision and zero on subsequent ones;
summing it gives total policy time. Older logs repeated latency on every action:
count only the first row of each decision when comparing those runs.

```sh
uv sync --extra llm  # tau (Hugging Face) for LLM policies
OPENCODE_GO_API_KEY=... uv run --extra llm python examples/llm.py \
    --timeout 120 --max-actions 3
```

`celestebench.llm.TauPolicy` asks the model for one `play` tool call per
screenshot, containing 1..`max-actions` legal button actions or explicit
`{"action":"wait","frames":N}` actions. A wait advances the game with every
button released, can appear between any two button actions, and is distinguished
from a plain release in the action log and viewer. LLM evaluations allow four
actions per decision by default. A single API turn can cover several game moves. Every API turn is
traced to `messages.jsonl`. The example sends a per-run `x-opencode-session`
header and identifies itself as `celestebench/0.1.0` to OpenCode Go.
Specialized (non-LLM) policies plug into `rollout` directly.

Tau is optional (`llm` extra); only its agent/provider cores are imported, though
the upstream package also installs CLI dependencies. A single persistent
`AgentHarness` owns the conversation for the whole rollout. Each decision appends
its observation through `prompt_message`; Tau retains every earlier observation
image, assistant text, available reasoning block and signature, tool call and tool
result. There is no history window, image filtering or reasoning stripping.
`max_turns=1` limits each decision to one API turn, not the stored conversation.
Use a new policy for each episode.

The CLI and viewer use this same full-history policy. The old `action_history`,
`image_history` and `reasoning_history` options have been removed. Input remains
subject to the provider's context limit; exceeding it fails the run rather than
silently deleting earlier turns. Output-token budgets still apply, and the
rollout's wall-clock `timeout` cancels a slow response.
`config.json` records the settings; `messages.jsonl` contains screenshots, responses
and reported token usage. Zero cost fields from Tau do not establish that requests
were free.

Reasoning means what the provider actually returns: visible text, thinking blocks,
summaries or opaque signed data, not guaranteed access to a private full CoT.
Tau handles provider-specific serialization; a provider may omit reasoning on
replay if its API does not support it. Responses requests use
`--reasoning-effort low` by default; Anthropic thinking can be enabled with
`--thinking-budget 2048 --max-tokens 4096` on a compatible model.

Every observation passed to a policy is also saved losslessly in `screenshots/`,
linked from `decisions.jsonl` with its environment frame index, including failed
decisions. `messages.jsonl` writes each new observation/response once, with all
available reasoning and token usage; it does not duplicate replayed context.
`actions.jsonl` records actions actually executed (including truncated holds),
while assistant tool calls record requested actions. The MP4 keeps the intervening
game frames but is not a substitute for the original observation PNGs.

Increasing `max_actions` allows one observation to cover more play
(`max_actions * max_frames` frames per decision) but also more time between
observations. `timeout` and `frames` bound every run regardless. Keep these
budgets in view when comparing policies. The smoke runs establish API
and recording compatibility, not gameplay quality or a speed advantage. No game
score or completion detector is implemented yet.

```sh
uv run --extra llm python -m unittest discover -s tests
```

## Web viewer

```sh
uv run --extra llm python -m web.viewer  # http://localhost:8123
```

Open **evaluations** (or `/evals`), then use **new eval** to open the evaluation
dialog. The runs page stays focused on replay and inspection. The left pane takes
one model per line (commas also work); every listed model becomes its own evaluation run with
the settings shared on the right, so a batch of models is a single launch. The
launch button and the counter preview how many runs a batch creates. `timeout (s)`
is the wall-clock budget for the whole run. Presets
cover OpenCode Go, OpenAI, Anthropic, Google and Mistral;
model IDs and endpoints remain editable for other compatible services or local
servers. The five protocols use Tau's existing provider adapters. Choose a model
that accepts images and tool calls. Credentials come from an environment variable
in the viewer process or a temporary password field, which is cleared after launch.
They are passed to the child process through its environment, not command arguments
or the saved configuration.

The same viewer follows each evaluation's observations, actions and available CoT,
with recorded-decision, frame, elapsed-time and reported-token counters. Select an
earlier decision to inspect it, or enable **follow latest** to resume following live
observations. **stop** preserves partial traces and the replay video. These counters
show rollout progress, not a score for rooms cleared or game completion.
Running cards display the game as a live MJPEG stream. Opening their run shows the
same stream at full viewer size, then switches to the finalized MP4 automatically.

Up to four evaluations can run in isolated emulator processes. A batch may list
more models than free slots: extra evaluations wait as `queued` jobs on the
evaluations page and start automatically, in submission order, as slots free up;
they can be stopped before starting. Terminal jobs leave that page automatically;
their runs remain available in the viewer. Jobs are recorded
under `runs/.evals/`; the original run files stay in `runs/MODEL/TIMESTAMP-ID/`.
Closing the viewer with Ctrl-C stops its children. After an unexpected viewer exit,
unfinished jobs are shown as interrupted when it restarts. The server binds only
to localhost; starting and stopping evaluations requires requests from this viewer.

The original run dropdown groups runs by model, keeping each directory selectable.
Its selection is stored in the `?run=...` URL, so reloads, browser navigation and
shared local links reopen the same run.
The **delete run** button removes the selected run and its evaluation metadata
after confirmation. Running or queued evaluations must be stopped first.
Runs without a finished video remain visible. Failed and interrupted attempts stay
in the original decision list, even after the last video frame. Clicking one pauses
playback and shows its received reasoning and requested actions in the original
right-hand panels. Playing the video again resumes synchronized selection.
Active runs expose a **sound** button beside their live game stream. Browsers require
that click before audio playback; finished MP4s contain the same audio track.

Omitting `--output` in `examples/llm.py` creates a fresh `runs/MODEL/TIMESTAMP`
directory, so repeated runs never replace each other.
Explicit output paths still have to be new directories.

New runs write `decisions.jsonl`: one outcome per attempted decision, with status,
latency and exact observation/end frame indices. `actions.jsonl` also records the
actual start/end frames, including interrupted holds. The wall-clock `timeout`
ends the run gracefully: the cut decision is recorded with status `timeout`, a
frame-budget cancellation with `frame_limit`. Partial provider
reasoning received before either cut is preserved in `messages.jsonl`.
Nothing is inferred about reasoning that the provider did not transmit.

Older traces are still readable, with approximate video timing. An unfinished
request without a recorded cause is marked interrupted, not automatically timeout.
Reasoning discarded by the old logger cannot be recovered. The pages and styles
live in `web/static/`; `web/viewer.py` serves the local API/video, and
`web/evals.py` starts the existing CLI in separate processes.
