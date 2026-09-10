# Coding agents rules

CelesteBench is an evaluation and a benchmark for mesuring how good LLMs are at real-time control in the video game Celeste Classic.

We are running the Celeste Classic game inside open8, a PICO-8 open-source reproduction in C99. We have our own fork in `./deps/open8` for project-specific changes. We are building it with the Makefile in the root of the project (we don't need CMake like they do since they want to target 9 platforms).

The LLMs are running, for now, in our own harness we built on top of [Tau](https://github.com/huggingface/tau) a python-version of [Pi](https://pi.dev/). The harness works by conserving all the CoTs and gives the last 3 frames to the model. We ask the model to use our tool `play` and to write a serie of actions in JSON-format, this serie of action helps them to take account of their own latencies (most models use CoTs).

We don't use the default written in the repo to run the evaluations. I am always running the game at 30 FPS (no pauses, for now) and never limitting the number of actions the models can take. Do not change the defaults if I don't ask it explicitely (if you think it would be good to change them, ask me before doing so).

In the future we plan to train our own small neural networks to play the game. To do so we would need to transform open8 in a true RL-environment; we are half-way through it since we have a small python API on top of it. So please, don't make harnesses (our own or Codex, Claude Code, etc...) the "default" and only way to run CelesteBench.

As a coding agent always focus for simplicity, reduction of abstractions and reduction of line of codes. Low LOC = good, but sometimes we can't cut easily.

We have a web server that we (humans) uses to inspect rollouts and run evals. Please never kill the web server to run your own! Assume we have one running. If there is no server running and you need one, just ask us! We can easily setup one for you :)

Also, keep in mind that CelesteBench aims to provide useful signals on how a model is performing, we don't want to have vulnerabilities models can use to cheat. We want full control over the system prompt and what context we provide to the models.
