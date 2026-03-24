# Autotune

A Claude Code plugin for fine-tuning LLMs — from dataset inspection through GGUF export. It gives Claude the tools and domain knowledge to be a real copilot for your training loop, not just a chat assistant that guesses shell commands.

Prefers [Unsloth](https://github.com/unslothai/unsloth) when available for faster training and lower VRAM usage. Falls back to Hugging Face + TRL when it is not.

## Setup

```bash
git clone <repo-url> autotune
cd autotune
uv sync
claude .
```

That is it. `uv sync` creates `.venv/` which the bundled `.mcp.json` uses to launch the MCP server. Claude Code picks up the server, commands, and skills automatically when you open the repo root.

To confirm everything is working, ask Claude in your session:

```text
check GPU availability
```

It should call the `check_gpu` tool and report your CUDA status and VRAM.

## The Workflow

There are two paths depending on your data.

**SFT — instruction, chat, or text data**

```
init-project → dataset-audit → baseline → plan-experiments
  → run-experiment → compare-runs → ship-decision → merge → export-gguf
```

**DPO/ORPO — preference data (prompt / chosen / rejected)**

```
init-project → dataset-audit → baseline → plan-experiments
  → run-dpo → compare-runs → ship-decision → merge → export-gguf
```

`plan-experiments` auto-detects which path your dataset belongs to and proposes the right run configs. You approve the plan before any GPU is touched.

For step-by-step walkthroughs, see:

- [`examples/sft-to-gguf.md`](examples/sft-to-gguf.md) — SFT fine-tuning from zero to a local GGUF file
- [`examples/dpo-preference-tuning.md`](examples/dpo-preference-tuning.md) — DPO/ORPO on a preference dataset

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/init-project` | Create project context files, reports, and results directories |
| `/dataset-audit` | Inspect a dataset's schema, infer training format, flag quality risks |
| `/baseline` | Evaluate the base model before any fine-tuning |
| `/plan-experiments` | Build an approval-ready run sequence with concrete hyperparameters |
| `/run-experiment` | Execute one approved SFT run and evaluate it |
| `/run-dpo` | Execute one approved DPO or ORPO run and evaluate it |
| `/compare-runs` | Rank completed runs by saved metrics |
| `/debug-run` | Diagnose a failed or weak run from logs |
| `/ship-decision` | Decide whether the best run is ready for review |
| `/merge` | Merge a LoRA adapter into the base model weights |
| `/export-gguf` | Export a model to GGUF for llama.cpp, Ollama, or LM Studio |
| `/train` | Legacy shortcut for a single SFT run |
| `/eval` | Legacy shortcut for running evaluation |
| `/experiment` | Legacy entrypoint — now behaves as a guided workflow |

## Capabilities

**Dataset inspection**
Autotune inspects format (instruction, chat, completion, preference), row count, column schema, and obvious quality risks before any training starts. Supports Hugging Face datasets by name and local files: `.json`, `.jsonl`, `.csv`, `.parquet`, `.txt`.

**Experiment planning**
The planner is VRAM-aware. It reads your GPU's available memory, estimates peak usage for each proposed config, and caps batch sizes so you do not OOM. It also adapts to dataset size — fewer steps for small datasets, more for large ones — and skips conservative runs when your baseline is already strong.

**Training**
LoRA and QLoRA SFT via `SFTTrainer`. DPO and ORPO preference optimization via `DPOTrainer` / `ORPOTrainer`. All runs save a checkpoint every 50 steps, so you can resume if something interrupts. Pass `report_to=wandb` or `report_to=mlflow` to stream metrics to your tracker.

**Evaluation**
MMLU benchmark out of the box. Pass your own dataset with `eval_dataset` to compute cross-entropy loss and perplexity on held-out examples.

**Run management**
`compare-runs` ranks everything by eval metric, then by training loss. `debug-run` reads the logs and tells you the most likely failure mode — OOM, loss divergence, dataset formatting issues, or chat template mismatch.

**Deployment**
Merge the LoRA adapter back into the base weights for a standalone model. Then export to GGUF (`q4_k_m`, `q5_k_m`, `q8_0`, or `f16`) for local inference with llama.cpp, Ollama, or LM Studio. Uses Unsloth's native GGUF path when available; falls back to a Hugging Face merge with llama.cpp conversion instructions.

**Gradio chat**
Ask Claude to launch a chat interface for any model or adapter. Useful for a quick sanity check before you start comparing benchmarks.

## Requirements

- Python 3.11+
- Claude Code
- `uv` (for `uv sync`; not required after `.venv/` exists)
- NVIDIA GPU for training and evaluation
- Enough VRAM for the model you pick — see `examples/sft-to-gguf.md` for typical requirements

Optional:
- `unsloth` — preferred fast path for LLM training
- `unsloth-studio` — additional model patches (e.g., Llama 4 expert layers)
- `wandb` or `mlflow` — experiment tracking (`uv sync --extra tracking`)

## Repository Layout

```text
autotune/
├── .claude-plugin/          # Claude plugin metadata
├── .mcp.json                # MCP server registration for Claude Code
├── CLAUDE.md                # Agent guidance: tools, workflow, rules
├── commands/                # Slash command specs
├── skills/                  # Domain knowledge docs Claude reads while reasoning
├── scripts/                 # GPU subprocess entry points (train, eval, export)
├── autotune/                # Pure Python planning and comparison logic
├── examples/                # Step-by-step walkthroughs
├── tests/                   # Unit tests (no GPU required)
└── results/                 # Run outputs (gitignored)
```

## Troubleshooting

**Claude Code does not see the MCP server**
Make sure you opened the repo root (not a subdirectory). Check that `.mcp.json` is present and `.venv/` exists. Re-run `uv sync` if you are unsure.

**No GPU detected**
Planning and dataset inspection work without a GPU. Training and evaluation will not.

**Unsloth is not installed**
That is fine. The runtime falls back to Hugging Face automatically.

**Training completed but eval is poor**
Run `/compare-runs` then `/debug-run`. Common causes: dataset formatting mismatch, wrong chat template, learning rate too high, sequence length too short.

**OOM during training**
The recovery ladder: lower batch size → shorter sequence length → 4-bit loading → smaller model or lower LoRA rank. Use `estimate_vram` before approving a run to catch this earlier.

## Limitations

- Diffusion fine-tuning is scaffolded in the API but not yet implemented.
- The eval suite covers MMLU and custom perplexity. Production-grade benchmark suites are not yet included.
- Team / multi-user workflows are out of scope for now.
