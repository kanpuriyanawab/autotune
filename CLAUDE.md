# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This directory contains a Claude plugin and MCP runtime for **ML R&D workflows**.
Treat it as an opinionated copilot for applied ML teams: inspect datasets, plan
experiments, execute approved runs, evaluate results, compare runs, and make a
ship decision. Training is only one stage of the workflow.

## Commands

```bash
# Install dependencies
uv sync

# Run tests (no GPU required)
uv run python -m unittest discover -s tests

# Run a single test
uv run python -m unittest tests.test_workflow.WorkflowTests.test_detect_format_instruction

# Start the MCP server manually (Claude connects via stdio)
uv run python server.py
```

## Architecture

The project has three layers:

**`server.py`** — The MCP server. Registers 14 tools via `@mcp.tool()` decorators on async functions. Runs via stdio transport (`mcp.run(transport="stdio")`). Tools either call into `autotune/` for pure logic, or call `scripts/` via subprocess through `run_script()`.

**`autotune/`** — Pure Python library (no GPU required, fully testable):
- `workflow.py` — ML planning logic: `detect_dataset_format()`, `recommend_backend()`, `build_experiment_plan()`, `compare_runs()`, `diagnose_run()`, `load_run_summaries()`, `summarize_dataset_preview()`
- `project.py` — Project scaffolding: `ensure_project_layout()`, `load_project_context()`, template generators

**`scripts/`** — Subprocess entry points that require CUDA/GPU:
- `train_model.py` — LoRA fine-tuning (called by `run_training`)
- `evaluate_model.py` — MMLU evaluation (called by `run_evaluation`)
- `serve_model.py` — Gradio chat interface (called by `serve_model`)

### Script Execution Pattern

`run_script(script, args)` in server.py translates a dict of parameters to CLI flags and runs `uv run scripts/<script>.py --key value ...` as an async subprocess with a 30-minute timeout. Logs are saved to the run directory.

### Run Directory Layout

Each training run saves to `results/run_001/`, `results/run_002/`, etc.:
```
results/run_001/
├── run_config.json     # training config + metrics
├── adapter_model.bin   # LoRA weights
├── stdout.log / stderr.log
```

### Backward-Compatible Aliases

`train_model`, `evaluate_model`, and `list_experiments` are thin wrappers that delegate to `run_training`, `run_evaluation`, and `compare_experiments` respectively.

## Available Tools

| Tool | Purpose |
|------|---------|
| `mcp__autotune__check_gpu` | Inspect CUDA availability and VRAM |
| `mcp__autotune__init_project` | Create project context and result directories |
| `mcp__autotune__inspect_dataset` | Audit dataset schema, format, and risks |
| `mcp__autotune__suggest_backends` | Choose between Unsloth and Hugging Face |
| `mcp__autotune__plan_experiments` | Produce an approval-ready run plan (adaptive: adjusts for VRAM, dataset size, baseline; generates DPO plans for preference datasets) |
| `mcp__autotune__estimate_vram` | Estimate peak VRAM before approving a run |
| `mcp__autotune__run_training` | Execute one approved SFT run (supports W&B/MLflow, checkpoint resume) |
| `mcp__autotune__run_dpo_training` | Execute one approved DPO or ORPO preference-optimization run |
| `mcp__autotune__run_evaluation` | Evaluate a base model or adapter (MMLU or custom dataset) |
| `mcp__autotune__compare_experiments` | Rank runs by saved metrics |
| `mcp__autotune__diagnose_experiment` | Diagnose OOMs, instability, or weak runs |
| `mcp__autotune__ship_decision` | Summarize whether the best run is ready |
| `mcp__autotune__merge_adapter` | Merge LoRA adapter into base model for deployment or GGUF export |
| `mcp__autotune__export_gguf` | Export model to GGUF for llama.cpp / Ollama / LM Studio |
| `mcp__autotune__serve_model` | Launch a Gradio chat interface |

## Default Workflow

### SFT path (instruction / chat / text datasets)
1. `check_gpu` — before proposing any compute-heavy work
2. `init_project` — if project context does not exist yet
3. `inspect_dataset` — always inspect before training; do not guess format
4. `plan_experiments` — adapts to VRAM, dataset size, and baseline; presents approval gate
5. `estimate_vram` — optionally verify a specific config fits before approving
6. `run_training` — one approved run at a time; supports `report_to`, `resume_from_checkpoint`
7. `run_evaluation` — after every run; use `eval_dataset` for task-specific eval
8. `compare_experiments` — before recommending the next step
9. `ship_decision` — only after the best run has been evaluated
10. `merge_adapter` → `export_gguf` — when ready to deploy locally

### DPO / ORPO path (preference datasets: prompt/chosen/rejected)
1–3 same as above (check_gpu, init_project, inspect_dataset)
4. `plan_experiments` — automatically generates DPO/ORPO run configs for preference data
5. `run_dpo_training` — one approved DPO run at a time
6. `run_evaluation` — evaluate the resulting adapter
7. `compare_experiments` → `ship_decision`

## Project Memory

When a project is initialized, expect to find:
- `context/project-brief.md`
- `context/constraints.md`
- `context/datasets.md`
- `reports/`
- `results/`

Read these first before proposing experiments. The workflow should be grounded in
goal, constraints, and prior run history, not just the latest user message.

## Product Defaults

- Optimize for **applied ML teams**, not toy demos.
- Prefer **Unsloth** when CUDA and the package are available.
- Fall back to **Hugging Face** when portability matters more than speed.
- Treat **LLMs as fully supported** and **diffusion as scaffolded but not implemented**.
- Do not run autonomous multi-step training loops without explicit user approval.

## Slash Commands

Main workflow: `/init-project`, `/dataset-audit`, `/baseline`, `/plan-experiments`, `/run-experiment`, `/run-dpo`, `/compare-runs`, `/debug-run`, `/ship-decision`, `/merge`, `/export-gguf`

Legacy shortcuts: `/train`, `/eval`, `/experiment`

## Skills

Skills in `skills/` guide planning and interpretation: `dataset-preparation`, `experiment-design`, `hyperparameter-tuning`, `evaluation-benchmarks`, `memory-optimization`, `failure-diagnosis`, `release-readiness`, `model-selection`, `experiment-tracking`, `lora-finetuning`

## Rules

- Do not skip dataset inspection for unknown data sources.
- Never silently burn multiple runs; propose the run sequence before execution.
- Save evaluations next to their run artifacts whenever possible.
- Compare runs before recommending that a model is ready.
- If a run fails, diagnose it before proposing another expensive retry.
