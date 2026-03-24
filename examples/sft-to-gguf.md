# SFT Fine-tuning: From Dataset to GGUF

This walkthrough takes a 1B parameter model, fine-tunes it on an instruction dataset, and ends with a GGUF file you can run locally with Ollama or llama.cpp. It covers every tool in the SFT path.

Estimated VRAM: ~5 GB for Llama-3.2-1B with 4-bit loading and default LoRA settings.

---

## 1. Initialize the project

```text
/init-project Fine-tune a support assistant on our SaaS documentation
```

This creates:

```
context/project-brief.md    ← goal and constraints
context/constraints.md      ← VRAM budget, latency targets, etc.
context/datasets.md         ← dataset notes and links
reports/                    ← baseline reports, plans, ship summaries
results/                    ← run outputs (gitignored)
```

Claude will ask you a few questions to fill in the project brief. Answer them — this context shapes every decision the planner makes later.

---

## 2. Check your GPU

Claude calls `check_gpu` automatically before proposing compute-heavy work, but you can ask any time:

```text
check GPU availability
```

You will see GPU name, total VRAM, and whether BF16 is supported. If you have less than 8 GB, tell Claude — it will plan accordingly (4-bit loading, smaller batch sizes).

---

## 3. Audit the dataset

```text
/dataset-audit yahma/alpaca-cleaned
```

Claude inspects:
- Row count and column names
- Training format (instruction, chat, completion, preference)
- Obvious quality risks (too few rows, duplicate content, encoding issues)
- Whether SFT is the right starting point

Do not skip this. A mislabeled dataset field will silently produce garbage training.

You can also audit local files:

```text
/dataset-audit ./data/my-finetune-data.jsonl
```

Supported formats: `.json`, `.jsonl`, `.csv`, `.parquet`, `.txt`.

---

## 4. Establish a baseline

```text
/baseline unsloth/Llama-3.2-1B
```

This runs MMLU on the base model and saves the result to `reports/`. You need this number before you can tell whether fine-tuning helped or hurt.

The baseline also feeds the planner — if the model is already strong (>0.6 MMLU), the planner skips conservative low-LR runs and starts with more aggressive configs.

---

## 5. Plan experiments

```text
/plan-experiments fine-tune unsloth/Llama-3.2-1B on yahma/alpaca-cleaned with a balanced budget
```

The planner:
1. Reads your GPU's available VRAM
2. Estimates peak memory for each proposed config using `estimate_vram`
3. Caps batch sizes to fit
4. Adjusts max steps based on dataset size
5. Adapts run order to your baseline metric

You will see a concrete proposal like:

```
Backend: unsloth
Approval required: yes

Run 1 — conservative baseline
  learning_rate: 1e-4
  lora_r: 16, lora_alpha: 32
  batch_size: 4, max_steps: 200
  estimated VRAM: 4.8 GB ✓

Run 2 — higher LR
  learning_rate: 2e-4
  lora_r: 16, lora_alpha: 32
  batch_size: 4, max_steps: 200
  estimated VRAM: 4.8 GB ✓
```

Nothing runs until you say so. Review the plan, ask questions, adjust if needed, then approve.

To check VRAM for a specific config before approving:

```text
estimate VRAM for a 7B model with lora_r=32, batch_size=4, seq_len=2048, 4-bit
```

---

## 6. Run the first experiment

```text
/run-experiment run the first approved config
```

This executes one run, saves the adapter and logs to `results/run_001/`, then automatically evaluates it. You will see training loss and MMLU accuracy when it finishes.

Each run saves:
```
results/run_001/
├── run_config.json      ← full config + final train loss
├── adapter_model.bin    ← LoRA weights
├── stdout.log
├── stderr.log
└── eval_mmlu.json       ← evaluation results
```

### Tracking with W&B or MLflow

```text
/run-experiment run config 1 with wandb tracking
```

This passes `report_to=wandb` to the trainer. Make sure you have wandb configured (`wandb login`) before running. MLflow works the same way.

### Resuming from a checkpoint

If a run is interrupted, resume it:

```text
/run-experiment resume run_001 from the latest checkpoint
```

Checkpoints are saved every 50 steps. The trainer picks up exactly where it left off.

---

## 7. Compare runs

```text
/compare-runs
```

After the first run, this just shows you one result. Run a second experiment then compare again — the ranking becomes meaningful once you have two or more runs.

```text
/run-experiment run the second approved config
/compare-runs
```

Claude ranks by eval metric (higher is better), then by training loss as a tiebreaker. It also highlights the delta from your baseline and tells you which direction to push next.

---

## 8. Diagnose failures

If a run looks wrong — OOM, loss spiked, eval worse than baseline — diagnose it before spending more GPU time:

```text
/debug-run results/run_002
```

Common things it catches:
- GPU memory pressure (CUDA OOM in stderr)
- Loss instability (train loss > 2.5 at end, or NaN)
- Dataset formatting issues (wrong text field, empty sequences)
- Chat template mismatch (model has a template but it was not applied)
- Learning rate too high (loss spikes then stays high)

You will get a list of issues and a prioritized fix list. Make the smallest change first, not the most dramatic one.

---

## 9. Evaluate on your own data

MMLU is generic. If you have a held-out sample of your actual task data, evaluate on that instead:

```text
/eval unsloth/Llama-3.2-1B with adapter results/run_001 on my-eval-data.jsonl
```

This computes cross-entropy loss and perplexity on your dataset. Lower perplexity on your domain data is a stronger signal than MMLU for task-specific fine-tuning.

---

## 10. Make a ship decision

```text
/ship-decision
```

Claude looks at:
- The best run's eval metric vs. your baseline
- Whether training loss converged
- How many runs you have compared

It tells you: ship it, iterate, or not ready. If it says iterate, it suggests the highest-leverage next change.

---

## 11. Merge the adapter

Once you have a run you want to deploy, merge the LoRA adapter into the base model weights. This removes the PEFT dependency and gives you a standalone model.

```text
/merge unsloth/Llama-3.2-1B with adapter results/run_001
```

Output lands in `results/run_001/merged/`. Uses Unsloth's native merge path when available; falls back to the Hugging Face `merge_and_unload()` path.

---

## 12. Export to GGUF

```text
/export-gguf results/run_001/merged with q4_k_m quantization
```

GGUF quantization options:
- `q4_k_m` — best quality/size tradeoff for most use cases
- `q5_k_m` — slightly larger, slightly better quality
- `q8_0` — near-lossless, 2x larger than q4_k_m
- `f16` — full float16, for when you need exact weights

The GGUF file lands in `exports/gguf/`. From there:

```bash
# Ollama
ollama create my-assistant -f Modelfile

# llama.cpp
./llama-cli -m exports/gguf/model-q4_k_m.gguf -p "Your prompt here"
```

---

## 13. Chat with the model (optional)

Before you export or while debugging, you can launch a Gradio chat interface:

```text
serve the model unsloth/Llama-3.2-1B with adapter results/run_001
```

This starts a local Gradio server at `http://localhost:7860`. Useful for a quick qualitative check before running benchmarks.

---

## Summary of tools used

| Tool | When it ran |
|------|------------|
| `init_project` | Step 1 |
| `check_gpu` | Step 2 |
| `inspect_dataset` | Step 3 |
| `suggest_backends` | Step 3 (auto) |
| `run_evaluation` (no adapter) | Step 4 — baseline |
| `plan_experiments` | Step 5 |
| `estimate_vram` | Step 5 (auto) |
| `run_training` | Step 6 |
| `run_evaluation` (with adapter) | Step 6 (auto) |
| `compare_experiments` | Step 7 |
| `diagnose_experiment` | Step 8 |
| `run_evaluation` (custom dataset) | Step 9 |
| `ship_decision` | Step 10 |
| `merge_adapter` | Step 11 |
| `export_gguf` | Step 12 |
| `serve_model` | Step 13 |
