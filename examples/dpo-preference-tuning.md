# DPO / ORPO Preference Tuning

This walkthrough covers the preference optimization path — training a model to prefer certain responses over others using DPO (Direct Preference Optimization) or ORPO (Odds Ratio Preference Optimization).

You need a dataset with `prompt`, `chosen`, and `rejected` columns. The `Anthropic/hh-rlhf` dataset on Hugging Face is a good starting point.

Estimated VRAM: ~6 GB for Llama-3.2-1B with 4-bit loading.

---

## 1. Initialize the project

```text
/init-project Preference-tune a chat model to be more helpful and less harmful
```

Claude fills in the project brief. Mention your goal (e.g., reduce harmful outputs, improve helpfulness ratings) — this context shapes what the planner proposes.

---

## 2. Audit the dataset

```text
/dataset-audit Anthropic/hh-rlhf
```

Autotune detects the `prompt / chosen / rejected` column pattern and flags this as a preference dataset. You will see:

- Row count
- Confirmation that the format is `preference` (not `instruction` or `chat`)
- Whether the chosen/rejected pairs look meaningfully different
- Note that SFT will be rejected — this dataset needs DPO/ORPO

If you have your own preference data:

```text
/dataset-audit ./data/preference-pairs.jsonl
```

Your file needs `prompt`, `chosen`, and `rejected` columns (string values).

---

## 3. Establish a baseline

```text
/baseline unsloth/Llama-3.2-1B
```

MMLU gives you a reference point. For preference tuning the more important baseline is qualitative — note how the model responds to a few test prompts before training.

---

## 4. Plan experiments

```text
/plan-experiments fine-tune unsloth/Llama-3.2-1B on Anthropic/hh-rlhf
```

The planner detects the preference format and automatically proposes DPO runs. A typical plan looks like:

```
Backend: unsloth
Format detected: preference → DPO path
Approval required: yes

Run 1 — DPO conservative
  method: dpo
  beta: 0.1
  learning_rate: 5e-5
  lora_r: 16, lora_alpha: 32
  batch_size: 2, max_steps: 200
  estimated VRAM: 5.4 GB ✓

Run 2 — ORPO (no reference model needed)
  method: orpo
  learning_rate: 8e-6
  lora_r: 16, lora_alpha: 32
  batch_size: 2, max_steps: 200
  estimated VRAM: 4.9 GB ✓
```

Key differences from SFT plans:
- Learning rates are lower (5e-5 vs 2e-4) — preference optimization is sensitive
- Batch size is smaller — DPO processes two sequences per example
- `beta` controls how strongly the model is pushed away from the reference — 0.1 is a safe default
- ORPO does not need a reference model, so it uses less VRAM

---

## 5. Run DPO training

```text
/run-dpo run the first approved config
```

This calls `run_dpo_training`, which runs `scripts/train_dpo.py` as a subprocess. The script:
1. Loads the model with LoRA via Unsloth (or falls back to Hugging Face)
2. Loads your preference dataset and validates the column schema
3. Trains with `DPOTrainer` (or `ORPOTrainer` for ORPO)
4. Saves the adapter and `run_config.json` to `results/run_001/`

After training, `/run-dpo` automatically runs evaluation.

### Tracking

```text
/run-dpo run config 1 with wandb tracking
```

### ORPO instead of DPO

ORPO is a simpler method — it does not need a reference model, so it is faster and uses less VRAM. The plan proposes it as a second run. To run it directly:

```text
/run-dpo run the ORPO config
```

### Resuming from a checkpoint

```text
/run-dpo resume run_001 from the latest checkpoint
```

---

## 6. Evaluate on your own data

MMLU will not tell you much about preference alignment. Evaluate on a held-out set of preference pairs or domain-specific prompts:

```text
/eval unsloth/Llama-3.2-1B with adapter results/run_001 on my-eval-pairs.jsonl
```

This computes cross-entropy loss and perplexity on your eval data. Lower perplexity on chosen responses (relative to the base model) is a good sign.

---

## 7. Compare runs

Once you have two or more runs:

```text
/compare-runs
```

Claude ranks them by eval metric and highlights which config produced better alignment. If the delta is small, `/debug-run` can tell you why.

---

## 8. Diagnose a weak run

```text
/debug-run results/run_002
```

Common preference tuning issues:
- **Loss did not decrease**: LR too high, or beta too low — the model is not being pushed enough
- **Loss collapsed**: Beta too high — the model is being pushed too hard away from the reference
- **Eval worse than baseline**: Dataset quality issue — check whether chosen/rejected pairs are meaningfully different
- **OOM**: DPO needs 2x sequences per batch — reduce batch size or switch to ORPO

---

## 9. Make a ship decision

```text
/ship-decision
```

For preference tuning, Claude checks whether:
- Eval metric improved over baseline
- Training loss converged without collapsing
- You have at least two runs to compare

If the run looks good, it recommends proceeding to merge and export.

---

## 10. Merge and export

Same steps as the SFT path:

```text
/merge unsloth/Llama-3.2-1B with adapter results/run_001
/export-gguf results/run_001/merged with q4_k_m quantization
```

The merged model and GGUF file are ready for local deployment.

---

## Choosing DPO vs ORPO

| | DPO | ORPO |
|---|---|---|
| Reference model | Yes (same base, frozen) | No |
| VRAM | Higher | Lower |
| Training stability | Well-studied | Simpler loss |
| Good for | Most preference datasets | VRAM-constrained setups |

If you are unsure, start with DPO. Run ORPO as a second experiment to compare.

---

## Beta parameter guide

`beta` in DPO controls how far from the reference distribution you push the model.

- `0.05–0.1` — conservative, less alignment but more stable
- `0.1–0.2` — standard range, good starting point
- `0.3+` — aggressive, risk of reward hacking or degraded fluency

If training loss collapses to near zero immediately, lower beta. If it barely moves, raise it or increase learning rate.

---

## Summary of tools used

| Tool | When it ran |
|------|------------|
| `init_project` | Step 1 |
| `check_gpu` | Auto |
| `inspect_dataset` | Step 2 |
| `run_evaluation` (no adapter) | Step 3 — baseline |
| `plan_experiments` | Step 4 |
| `estimate_vram` | Step 4 (auto) |
| `run_dpo_training` | Step 5 |
| `run_evaluation` (custom dataset) | Step 6 |
| `compare_experiments` | Step 7 |
| `diagnose_experiment` | Step 8 |
| `ship_decision` | Step 9 |
| `merge_adapter` | Step 10 |
| `export_gguf` | Step 10 |
