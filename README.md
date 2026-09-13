# celestebench

CelesteBench is an evaluation for AI models (LLMs and others in the future) to mesure how well they perform in real-time video games. I am focusing on Celeste Classic because it is a small open-source and beloved game.

## Install

You need to have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed first.

Then run `uv sync`, and you're all set!

## How to eval a model?

### Step 1.

Run the web server: `uv run python -m web.viewer`

### Step 2.

Open the page in your browser of choice and click on `evaluations` on the top-right of the page.

### Step 3.

Click on `new eval`.

Pick a harness in the harness box:

- **CelesteBench Harness** (`tau`): run a model on API.
- **Codex CLI**: run a model inside a jailed Codex CLI, using your `codex login` session or `CODEX_API_KEY`.
- **OpenCode CLI**: run a model inside OpenCode, using your `opencode auth login` session.
- **Pi CLI**: run a model inside Pi, which loads our bundled MCP extension; uses your Pi login and queues runs because they share it.

External harnesses need their CLI on `PATH` (`codex`, `opencode`, `pi`). Each one plays in an empty workspace where its built-in tools are denied, so the only tool available is the game through our MCP server. The game rules ride in the system prompt and every external harness sends the same one-line task prompt, `Play Celeste Classic.`. Their traces are normalized into the same `messages.jsonl` the viewer already reads.

### Step 4.

Select the model you want to run (ex: `gpt-6-astra`, `deepseek-flash`, etc...), select its thinking level and the time budget (timeout). I set 300 seconds of timeout but you can give more.

### Step 5.

Please check in the _advanced_ settings the FPS. If it is empty the game will pause when the model thinks, it is the _lite_ version of the benchmark.

Set it at 30 FPS, it is standard.

### Step 6.

Launch the eval(s) and watch it run!

When it ends it will update the "leaderboard" tab.
