"""Declarative description of the evaluation harnesses the viewer can launch.

Each harness declares its launcher and the settings it accepts, so the viewer
renders the new-eval form and the backend validates a payload from the same
table. The built-in Tau harness is marked `builtin`: it owns its provider and
API-key handling. External harnesses (Codex, OpenCode and Pi) are plain commands
billed through their own CLI; adding one is a new entry here plus its own script
under examples/.
"""

from dataclasses import asdict, dataclass

from .harness import PROMPT

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
    oauth_only: bool = False  # True: the cap applies only to OAuth-subscription runs
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
        ),
        options=(
            Field("provider", "provider", "choice", "opencode-go", source="providers"),
            Field("base_url", "base URL", "text", "http://localhost:8000/v1",
                  when=("provider", "custom")),
            Field("api_key", "API key", "secret"),
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
        concurrency=1,
        note="Runs the host Codex CLI in an empty read-only workspace with shell "
             "network off. Authenticates with CODEX_API_KEY when set, otherwise the "
             "host `codex login` ChatGPT session; runs queue because they share it.",
        run=(
            Field("model", "model", required=True),
            Field("thinking_level", "reasoning", "choice", "low", choices=THINKING_LEVELS,
                  help="Passed to Codex as model_reasoning_effort; off means no reasoning"),
        ),
        options=(
            Field("prompt", "task prompt", "textarea", PROMPT,
                  help="Message sent to the CLI; the game rules ride in the system prompt"),
        ),
    ),
    "opencode": Harness(
        key="opencode",
        label="OpenCode CLI",
        script="examples/opencode.py",
        nested=True,
        trace="opencode.jsonl",
        requires="opencode",
        concurrency=1,
        oauth_only=True,
        note="Runs the host OpenCode CLI in an empty workspace with every built-in "
             "tool denied, so it can only play through our MCP game server. Gets a "
             "throwaway config, data, state and cache home, so the host's config, "
             "agents, plugins and MCP servers cannot leak in. Uses the host "
             "`opencode auth login` session; OAuth-subscription runs queue on their "
             "shared login while API-key models run in parallel.",
        run=(
            Field("model", "model", required=True,
                  help="OpenCode provider/model, e.g. opencode-go/deepseek-v4.1-flash"),
            Field("thinking_level", "reasoning", "choice", "low", choices=THINKING_LEVELS,
                  help="Passed to OpenCode as the model variant; off sends no variant"),
        ),
        options=(
            Field("prompt", "kickoff prompt", "textarea", PROMPT,
                  help="Message sent to the CLI; the game rules ride in the agent's system prompt"),
        ),
    ),
    "pi": Harness(
        key="pi",
        label="Pi CLI",
        script="examples/pi.py",
        nested=True,
        trace="pi.jsonl",
        requires="pi",
        concurrency=1,
        oauth_only=True,
        note="Runs the host Pi CLI in an empty workspace with built-in tools off, "
             "loading our bundled extension that bridges to the MCP game server. Gets "
             "a throwaway agent directory, so the host's settings, trust list, skills "
             "and extensions cannot leak in. Uses the host Pi login; OAuth-subscription "
             "runs queue on their shared login while API-key models run in parallel.",
        run=(
            Field("model", "model", required=True,
                  help="Pi model id, e.g. openai-codex/gpt-5.6-sol; bare ids use Pi's default provider"),
            Field("thinking_level", "thinking", "choice", "low", choices=THINKING_LEVELS),
        ),
        options=(
            Field("prompt", "kickoff prompt", "textarea", PROMPT,
                  help="Message sent to the CLI; the game rules ride in the system prompt"),
        ),
    ),
}


def field_by_key(harness: Harness, key: str) -> Field | None:
    return next((field for field in (*harness.run, *harness.options) if field.key == key), None)
