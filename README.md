# Unsloth ML R&D Copilot for Claude Code

`autotune` is a Claude Code plugin + MCP runtime for **machine learning R&D workflows**.
It is designed for **applied ML engineers and researchers** who want a structured loop for:

- defining a project brief
- auditing datasets
- planning experiments
- running approved training jobs
- evaluating models
- comparing runs
- diagnosing failures
- deciding whether a model is ready for review or shipping

The product prefers **Unsloth** when it is available, but it also supports a **Hugging Face fallback**.
Today the implementation is **LLM-first**. Diffusion is scaffolded in the workflow and API surface,
but it is **not fully implemented yet**.

## What You Get

- Claude Code slash commands for the ML workflow
- An MCP server with structured tools Claude can call
- Project scaffolding with persistent context files
- Run directories with configs, eval outputs, and logs
- Dataset inspection and experiment planning helpers
- Backward-compatible `train` / `eval` style tooling

## Current Scope

### Fully supported

- LLM project setup
- dataset inspection
- experiment planning
- LoRA/SFT training runs
- MMLU evaluation
- run comparison
- basic failure diagnosis
- ship-readiness summaries

### Scaffolded, not complete

- diffusion workflows
- team collaboration and shared control plane
- advanced evaluation suites beyond the current runtime path
- preference optimization / RLHF-specific training paths

## Repository Layout

```text
autotune/
├── .claude-plugin/          # Claude plugin metadata
├── .mcp.json                # Local MCP registration for Claude Code
├── CLAUDE.md                # Guidance for Claude Code in this repo
├── commands/                # Slash commands
├── skills/                  # Skills Claude consults while reasoning
├── scripts/                 # Execution scripts for training/eval/serve
├── autotune/             # Shared workflow logic
├── tests/                   # Unit tests for planning/runtime helpers
└── results/                 # Run outputs (gitignored)
```

## Requirements

### Required

- Python 3.11+ recommended
- Claude Code
- a local clone of this repository

### For training/evaluation

- NVIDIA GPU recommended
- CUDA-compatible PyTorch environment
- enough VRAM for the selected model

### Optional but preferred

- `uv` for dependency management
- `unsloth` installed if you want the preferred fast path
- `unsloth-studio` available if you want to use its evaluation/model helpers

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url> autotune
cd autotune
```

### 2. Install dependencies

This repo uses `uv`.

```bash
uv sync
```

That creates `.venv/`, which is what the checked-in `.mcp.json` uses to launch the MCP server.

If `uv` is not on your shell `PATH`, fix that first or run it via whatever Python toolchain manager
you use locally. The runtime itself does **not** depend on `uv` being on `PATH` after `.venv` exists.

### 3. Open the repo in Claude Code

From the repo root:

```bash
claude .
```

Claude Code should see:

- `.mcp.json` for the local MCP server
- `CLAUDE.md` for repo guidance
- `commands/` for slash commands
- `skills/` for domain-specific reasoning docs

### 4. Confirm the MCP runtime is usable

In Claude Code, ask it to:

```text
check GPU availability
```

It should be able to call the local `check_gpu` MCP tool.

## How Claude Code Uses This Repo

This repo combines four layers:

### 1. `CLAUDE.md`

This tells Claude Code what this project is, what tools exist, and what workflow to follow.

### 2. `commands/`

These are user-invocable slash commands like:

- `/init-project`
- `/dataset-audit`
- `/baseline`
- `/plan-experiments`
- `/run-experiment`
- `/compare-runs`
- `/debug-run`
- `/ship-decision`

### 3. `skills/`

These are reusable knowledge docs Claude consults while planning or interpreting results.

### 4. `.mcp.json` + `server.py`

These register and run the MCP server so Claude can call structured tools instead of guessing shell commands.

## Recommended User Workflow

This is the intended flow for a real user in Claude Code.

### 1. Initialize a project

Example:

```text
/init-project Build a support chatbot fine-tuning project for our SaaS docs
```

This creates project memory:

- `context/project-brief.md`
- `context/constraints.md`
- `context/datasets.md`
- `reports/`
- `results/`

### 2. Audit the dataset

Example:

```text
/dataset-audit yahma/alpaca-cleaned
```

Claude will inspect:

- column schema
- likely training format
- obvious risks
- whether SFT is a sensible first step

Do this before training. The plugin is intentionally opinionated here.

### 3. Establish a baseline

Example:

```text
/baseline unsloth/Llama-3.2-1B
```

This gives you a real reference point before any fine-tuning.

### 4. Plan experiments

Example:

```text
/plan-experiments fine-tune unsloth/Llama-3.2-1B on yahma/alpaca-cleaned with a balanced budget
```

Claude should stop at a proposal and show:

- backend choice
- run order
- concrete hyperparameters
- why each run exists

This is the compute approval gate.

### 5. Execute one approved run

Example:

```text
/run-experiment run the first approved config
```

The current implementation is intentionally conservative:

- run one experiment
- evaluate it
- then compare before burning more compute

### 6. Compare runs

Example:

```text
/compare-runs
```

Claude ranks runs using saved evaluation metrics and training loss.

### 7. Diagnose failures if needed

Example:

```text
/debug-run results/run_002
```

Use this when:

- training OOMs
- loss diverges
- eval is unexpectedly weak
- a run looks suspicious even though it completed

### 8. Ask for a ship decision

Example:

```text
/ship-decision
```

This summarizes whether the best run is ready for human review or whether another iteration is justified.

## Slash Command Reference

### Main workflow

#### `/init-project`

Creates project context and gives the workflow a stable home.

#### `/dataset-audit`

Inspects a dataset and flags formatting or quality risks before training.

#### `/baseline`

Runs evaluation on the base model before fine-tuning.

#### `/plan-experiments`

Builds a concrete, approval-ready experiment sequence.

#### `/run-experiment`

Runs one approved experiment and evaluates it.

#### `/compare-runs`

Ranks completed runs and summarizes the best next move.

#### `/debug-run`

Diagnoses failed or weak runs from saved logs and metrics.

#### `/ship-decision`

Produces a recommendation on whether the current best run is ready.

### Legacy shortcuts

These still exist, but the project is now optimized around the workflow above.

#### `/train`

Single-run shortcut for experienced users.

#### `/eval`

Direct evaluation shortcut.

#### `/experiment`

Legacy entrypoint that now behaves like a guided workflow instead of an uncontrolled loop.

## MCP Tool Reference

The MCP server exposes these tools to Claude Code.

### `check_gpu`

Reports:

- whether CUDA is available
- GPU name
- VRAM
- BF16 support

### `init_project`

Creates the default project layout and context files.

### `inspect_dataset`

Inspects a dataset and returns:

- row count
- columns
- inferred format
- recommended next steps

### `suggest_backends`

Chooses between:

- `unsloth`
- `huggingface`

based on environment and task family.

### `plan_experiments`

Returns an approval-ready plan with:

- backend selection
- workflow steps
- proposed runs
- concrete configs

### `run_training`

Runs a training job and stores:

- `run_config.json`
- `stdout.log`
- `stderr.log`
- adapter output

### `run_evaluation`

Runs evaluation for a base model or adapter and saves the eval artifact when possible.

### `compare_experiments`

Loads saved runs and ranks them by metric, then by training loss.

### `diagnose_experiment`

Looks at logs and saved metrics to infer likely failure modes.

### `ship_decision`

Summarizes whether the current best run is ready for review or whether more iteration is warranted.

### Compatibility aliases

- `train_model`
- `evaluate_model`
- `list_experiments`

These exist so earlier prompts and workflows do not break.

## Project Memory and Artifacts

### Context files

When you run `/init-project`, the workflow expects:

- `context/project-brief.md`
- `context/constraints.md`
- `context/datasets.md`

Claude should read these before proposing experiments.

### Reports

`reports/` is where baseline reports, experiment plans, and ship summaries belong.

### Results

Each run directory under `results/` may contain:

- `run_config.json`
- `eval_*.json`
- `stdout.log`
- `stderr.log`
- adapter/model artifacts

## Backend Behavior

### Unsloth preferred path

If:

- CUDA is available
- `unsloth` is installed
- the task family is `llm`

the planner/runtime prefers Unsloth.

### Hugging Face fallback

If Unsloth is unavailable, the runtime falls back to Hugging Face/TRL-style execution.

This keeps the system usable even when the preferred stack is not installed.

### Diffusion status

Diffusion is represented in the workflow and API surface, but the current execution path is still LLM-only.

Do not claim diffusion training is implemented unless you add the actual runtime.

## Local Dataset Support

Training and dataset inspection support:

- Hugging Face datasets by name
- local `.json`
- local `.jsonl`
- local `.csv`
- local `.parquet`
- local `.txt`

For local paths, pass the file path directly in Claude Code or tool calls.

## Development

### Run unit tests

```bash
.venv/bin/python -m unittest discover -s tests
```

### Run a quick syntax/bytecode check

```bash
.venv/bin/python -m compileall server.py scripts autotune tests
```

### Start the server manually

```bash
.venv/bin/python server.py
```

## Troubleshooting

### Claude Code does not see the MCP server

Check:

- you opened the repo root in Claude Code
- `.mcp.json` is present
- `.venv/` exists
- dependencies were installed successfully

### `uv` is missing

You still need `uv` for the initial `uv sync`, but the checked-in `.mcp.json`
now runs the server through `.venv/bin/python`, so Claude Code does not depend on `uv` being on `PATH` after setup.

### No GPU detected

The plugin can still plan and inspect datasets, but real training/evaluation workflows will be limited.

### Unsloth is not installed

That is acceptable. The runtime falls back to Hugging Face.

### Training completed but eval is poor

Use:

- `/compare-runs`
- `/debug-run`

and inspect:

- dataset formatting
- benchmark choice
- chat template alignment
- learning rate
- sequence length

### OOM errors

The default recovery order is:

1. lower batch size
2. reduce sequence length
3. use 4-bit loading
4. reduce model size or LoRA rank

## Limitations

- The runtime is still LLM-first.
- Diffusion is not fully implemented.
- The current eval path is still narrow compared with a production-grade benchmark suite.
- Team-level multi-user workflow is out of scope for this repo today.

## Suggested First Demo

If you want to validate the end-to-end UX quickly, do this in Claude Code:

```text
/init-project Fine-tune a lightweight support assistant for our docs
/dataset-audit yahma/alpaca-cleaned
/baseline unsloth/Llama-3.2-1B
/plan-experiments fine-tune unsloth/Llama-3.2-1B on yahma/alpaca-cleaned with a balanced budget
```

Then approve one run and continue with:

```text
/run-experiment run the first approved config
/compare-runs
/ship-decision
```

That is the intended user experience for this repo in its current state.
