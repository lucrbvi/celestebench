"""The one game system prompt both harnesses send to models.

Mode (Lite vs RTC) is the only real variant, plus a marker line when the
client plays through the MCP server. Tau re-prompts the model after every
play call, so it must ask for exactly one call; one-shot clients (Codex)
never re-prompt, so they must ask the model to keep calling play itself.
Edit the text below once.
"""

def system_prompt(*, fps=None, max_frames=30, max_images=3, mcp=False, oneshot=False) -> str:
    mode = ("This episode is Lite: the game runs only when you submit actions, "
            "and pauses while you think."
            if fps is None else
            f"This episode is RTC at {fps:g} fps: the game keeps moving while "
            "you think or plan.")
    marker = (" You are connected through the CelesteBench MCP server: call "
              "observe once to receive the initial frames, and responses tag "
              "each image with its game-frame position (frame_ids)."
              if mcp else "")
    loop = ("You are in one continuous session and must keep playing without ever "
            "stopping. Call play repeatedly: one action batch per call, in order, "
            "and call it again as soon as each result arrives. Never end your turn "
            "on your own, never wait to be prompted again, and never say you are done."
            if oneshot else
            "Call play exactly once with your next actions, in order.")
    return f"""Play Celeste Classic. Climb upward and don't die. {mode}{marker}
{loop} A button action is {{"buttons": bitmask, "frames": frames}}; a wait action is {{"action": "wait", "frames": frames}}. Button values are LEFT=1, RIGHT=2, UP=4, DOWN=8, O=16 (jump), X=32 (dash); combine them by adding their values. Wait advances the game intentionally with all buttons released so you can observe the result later. You may put it anywhere in a sequence, for example [{{"buttons": 18, "frames": 4}}, {{"action": "wait", "frames": 8}}, {{"buttons": 2, "frames": 4}}]. A buttons value of 0 only releases the controls.
Frames must be integers from 1 to {max_frames}. The server owns the episode timeout and frame budgets; tools cannot reset or alter them.
Each decision includes up to {max_images} sampled frames since your previous decision, ordered oldest to newest; the last image is current. Only a few recent images are kept, older ones are dropped from context: write down important observations and changes in your reasoning as you go. Use past images and actions to infer movement and learn from mistakes."""
