"""Declarative description of the evaluation harnesses the viewer can launch.

Each harness declares its launcher and the settings it accepts, so the viewer
renders the new-eval form and the backend validates a payload from the same
table. The built-in Tau harness is marked `builtin`: it owns its provider and
API-key handling. External harnesses (Codex today) are plain commands billed
through their own CLI; adding one is a new entry here plus its own script.
"""

from dataclasses import asdict, dataclass

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "text"  # text, int, number, choice, bool, textarea, secret
    default: object = None
    required: bool = False
    help: str = ""
    choices: tuple[str, ...] = ()
    source: str = ""  # dynamic choice list computed by the backend
    minimum: float = 1
    when: tuple[str, str] | None = None
    advanced: bool = False

    def spec(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Harness:
    key: str
    label: str
    script: str
    run: tuple[Field, ...]  # one set per run row; "model" always comes first
    options: tuple[Field, ...] = ()
    command: tuple[str, ...] = ()
    note: str = ""
    trace: str = ""  # external CLI JSON trace, e.g. codex.jsonl
    login_hint: str = ""
    concurrency: int = 0  # 0 = unlimited; N = at most N runs at once
    nested: bool = False  # rollout lives one directory below the run folder
    builtin: bool = False
    requires: str = ""


HARNESSES: dict[str, Harness] = {
    "tau": Harness(
        key="tau",
        label="CelesteBench Harness",
        script="examples/llm.py",
        builtin=True,
        run=(
            Field("model", "model", required=True,
                  help="Tau resolves the API from the provider catalog; append @provider "
                       "to route one model elsewhere, e.g. minimax-m3@minimax"),
            Field("thinking_level", "thinking", "choice", "low", choices=THINKING_LEVELS,
                  help="off, minimal, low, medium, high, xhigh or max; Tau maps it per API"),
            Field("timeout", "timeout (s)", "number", 120,
                  help="wall-clock budget for this run (seconds)"),
        ),
        options=(
            Field("provider", "provider", "choice", "opencode-go", source="providers"),
            Field("base_url", "base URL", "text", "http://localhost:8000/v1",
                  when=("provider", "custom")),
            Field("api_key", "API key", "secret"),
            Field("max_frames", "max frames / action", "int", 30,
                  help="Maximum frames one action holds a button"),
            Field("max_actions", "max actions", "int", 4,
                  help="Maximum actions the model may return per turn"),
            Field("max_images", "max images", "int", 3, advanced=True,
                  help="Cap on images kept in context; providers reject more"),
            Field("fps", "FPS", "number", None, advanced=True,
                  help="Run the game in real time at this speed; empty pauses the game"),
            Field("thinking_level", "default thinking level", "choice", "low",
                  choices=THINKING_LEVELS, advanced=True,
                  help="Used when a run leaves its own thinking field empty"),
        ),
    ),
    "codex": Harness(
        key="codex",
        label="Codex CLI",
        script="examples/codex.py",
        nested=True,
        trace="codex.jsonl",
        requires="codex",
        note="Runs the host Codex CLI in an empty read-only workspace with shell "
             "network off. Uses CODEX_API_KEY when set (parallel); otherwise your "
             "`codex login` ChatGPT session, one run at a time.",
        run=(
            Field("model", "model", required=True),
            Field("thinking_level", "reasoning", "choice", "low", choices=THINKING_LEVELS,
                  help="Passed to Codex as model_reasoning_effort; off means no reasoning"),
            Field("timeout", "timeout (s)", "number", 120,
                  help="wall-clock budget for this run (seconds)"),
        ),
        options=(
            Field("prompt", "task prompt", "textarea", required=True,
                  help="Sent to the Codex CLI working against our MCP game server"),
            Field("max_frames", "max frames / action", "int", 30,
                  help="Maximum frames one action holds a button"),
            Field("frames", "frame cap", "int", None, advanced=True,
                  help="Cap on the total number of environment frames played"),
            Field("fps", "FPS", "number", 30, advanced=True,
                  help="Run the game in real time at this speed; empty pauses the game"),
        ),
    ),
}


def field_by_key(harness: Harness, key: str) -> Field | None:
    return next((field for field in (*harness.run, *harness.options) if field.key == key), None)
