# celestebench

Small synchronous Python interface to Open8 for Celeste Classic rollouts.
Use it from this source checkout; the native library and cartridge stay here.

## Getting started

On a new machine (macOS or Linux; the VM harness prefers ARM for speed), one shell:

```sh
git clone --recursive <repo> celestebench && cd celestebench  # or: git submodule update --init --recursive
uv sync --extra llm --extra mcp      # installs Python deps (Tau + MCP server)
make -j4                             # builds the emulator; needs clang/CMake
uv run --extra llm python -m unittest discover -s tests   # everything should pass
```

Provide one API key per provider family as environment variables; the viewer
process reads them, nothing is stored:

```sh
export OPENCODE_GO_API_KEY=sk-...   # https://opencode.ai/zen — the default preset
# also supported: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY
```

Start the viewer and open `http://localhost:8123/evals`:

```sh
uv run --extra llm python -m web.viewer
```

## Launching evals from the UI

1. Click **new eval**.
2. Pick a **harness** on the right: `CelesteBench Harness` for API models, or
   `Codex CLI (Lima VM)` (log it in first — see below).
3. Build runs on the left: each row is a model with its own `reasoning effort`
   and `timeout (s)` (game-engine budget, starts at the first decision).
   **+ add run** duplicates the previous row's settings; ✕ removes it. The same
   model twice with different efforts is normal usage.
4. Adjust the preset/API/base URL/key env only if needed — model families
   (`claude…`, `gpt…`, `gemini…`, the Mistral family) auto-route to their
   official API on default endpoints.
5. **launch eval(s)**. Cards track each run live; open one to inspect its
   replay (video, actions, CoT) in the runs view. Stop anytime.

Codex harness, once per machine: create and authenticate the VM.

```sh
uv run --extra mcp python examples/codex_vm.py start
uv run --extra mcp python examples/codex_vm.py login   # device code; persists in the VM
```

Back in the UI, select `Codex CLI (Lima VM)`, set the **task prompt**, launch.
The viewer boots the VM, opens the tunnel and runs one Codex evaluation at a
time; VM Codex auth sticks until you log it out. Stop the VM with
`examples/codex_vm.py stop` when finished to free memory.
