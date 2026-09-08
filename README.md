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
- `record(path)` writes every stepped frame to a silent lossless H.264 MP4
  (yuv444p, qp 0), upscaled 4x with nearest-neighbor (512x512) so playback
  stays crisp, at the cart's native 30/60 fps, and finalizes it even when the
  body raises. Step at least
  once to produce a video. Reset/restore inside the context creates a cut.
  PyAV uses FFmpeg libraries directly; no `ffmpeg` executable or subprocess.
- Open8 has global state: **one environment per process**, used from one thread.
  Use separate processes for parallel rollouts. Always close it with `with`.

This is the emulator interface, without a reward function or a Gym wrapper.

## Harness

`rollout(policy, output, *, decisions, max_frames=30, max_actions=1, frames=None, fps=None)` drives a policy that takes
the last frame and returns `(buttons, frames)` — or a list of those — sync or
async. It writes `rollout.mp4`, an `actions.jsonl` log with latencies, and a
final `checkpoint.state` into a fresh `output` directory.
The entire action batch is validated before execution. `max_actions` bounds
every policy, including non-LLM policies. There is no screenshot between actions
in a batch. The initial observation advances and records one neutral frame.
`frames` caps total environment frames (long holds are cut); `fps` runs the
environment in real time: frames tick at that rate with buttons released while
the policy thinks, so slow models waste world time instead of blocking the run.

Each action log row contains its decision index. `latency` is the policy's wall
time, recorded on the first action of that decision and zero on subsequent ones;
summing it gives total policy time. Older logs repeated latency on every action:
count only the first row of each decision when comparing those runs.

```sh
uv sync --extra llm  # tau (Hugging Face) for LLM policies
OPENCODE_GO_API_KEY=... uv run --extra llm python examples/llm.py \
    --decisions 5 --max-actions 3
```

`celestebench.llm.TauPolicy` asks the model for one `play` tool call per
screenshot, holding 1..`max-actions` legal actions (button mask + frame count)
each, so a single API turn can cover several game moves. Every API turn is
traced to `messages.jsonl`. The example sends a per-run `x-opencode-session`
header and identifies itself as `celestebench/0.1.0` to OpenCode Go.
Specialized (non-LLM) policies plug into `rollout` directly.

Tau is optional (`llm` extra); only its agent/provider cores are imported, though
the upstream package also installs CLI dependencies. Each decision receives the
previous 16 conversation turns (observations, assistant responses and tool results),
with the latest 3 screenshots including the current image. It has one API request per decision, no retries or
model fallback, and explicit token and timeout limits. `config.json` records the
settings; `messages.jsonl` contains screenshots, responses and reported token usage.
Zero cost fields from Tau do not establish that requests were free.

`--action-history N` sets the number of previous turns, and `--image-history N`
sets the number of images including the current observation (at most
`action_history + 1`). `--action-history 0` gives a fresh context;
`--image-history 1` keeps textual history with only the current image.
Old turns are dropped as complete units, keeping tool calls and results paired.
`--no-reasoning-history` replays only tool calls from old assistant responses,
omitting their text and thinking from the prompt. By default both are retained,
including the original provider signatures. These switches affect model input,
never the complete on-disk trace. Use a new policy for each episode.

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

The history defaults are starting points for Celeste experiments, not measured
optima. [OpenCUA, sections 3.2 and 5](https://arxiv.org/html/2508.09123v1)
uses dialogue history and three screenshots; its Qwen2-VL ablation improves with
multiple images, with diminishing gains from three to five. Replaying richer
past reasoning did not help in that experiment, so we expose an ablation switch
while keeping the available reasoning by default.
[VAGEN](https://github.com/mll-lab-nu/VAGEN) models partially observable visual
tasks using multi-turn RL and supports concatenated trajectory training.
For Celeste, comparing successive observations should help infer movement;
that is a hypothesis to test, not a demonstrated benchmark gain. Compare history
settings at fixed model and game budgets, recording tokens and latency, especially
in real-time mode where extra inference time advances the world.

Increasing `max_actions` changes both the maximum simulated duration
(`1 + decisions * max_actions * max_frames`) and the time between observations.
Keep these budgets in view when comparing policies. The smoke runs establish API
and recording compatibility, not gameplay quality or a speed advantage. No game
score or completion detector is implemented yet.

```sh
uv run --extra llm python -m unittest discover -s tests
```

## Web viewer

```sh
uv run python viewer.py  # http://localhost:8123
```

The original run dropdown groups runs by model, keeping each directory selectable.
Runs without a finished video remain visible. Failed and interrupted attempts stay
in the original decision list, even after the last video frame. Clicking one pauses
playback and shows its received reasoning and requested actions in the original
right-hand panels. Playing the video again resumes synchronized selection.

Omitting `--output` in `examples/llm.py` creates a fresh `runs/MODEL/TIMESTAMP`
directory, so repeated runs never replace each other.
Explicit output paths still have to be new directories.

New runs write `decisions.jsonl`: one outcome per attempted decision, with status,
latency and exact observation/end frame indices. `actions.jsonl` also records the
actual start/end frames, including interrupted holds. A policy timeout still ends
the rollout; a frame-budget cancellation is recorded separately. Partial provider
reasoning received before either interruption is preserved in `messages.jsonl`.
Nothing is inferred about reasoning that the provider did not transmit.

Older traces are still readable, with approximate video timing. An unfinished
request without a recorded cause is marked interrupted, not automatically timeout.
Reasoning discarded by the old logger cannot be recovered. The page and its styles
live in `viewer.html`; `viewer.py` only reads runs and serves the local API/video.
