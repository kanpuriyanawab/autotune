---
name: experiment-tracking
description: How to read training results, compare runs, analyze convergence, and organize experiments.
---

# Experiment Tracking

## Reading `run_config.json`

Each training run produces a `run_config.json` in its output directory. Key fields:

| Field | Meaning |
|-------|---------|
| `model_name` | Base model used |
| `dataset_name` | Training dataset |
| `lora_r` | LoRA rank |
| `lora_alpha` | LoRA scaling factor (usually = rank) |
| `learning_rate` | Peak learning rate |
| `max_steps` | Total training steps |
| `batch_size` | Per-device batch size |
| `load_in_4bit` | Whether QLoRA (4-bit) was used |
| `train_loss` | Final training loss |
| `training_time` | Wall-clock time in seconds |
| `peak_vram_mb` | Peak GPU memory usage |

## Comparing Runs

### Which metrics matter

1. **Primary**: The target metric (e.g., MMLU accuracy) — this is what you're optimizing
2. **Secondary**: `train_loss` — indicates how well the model fits training data
3. **Efficiency**: `training_time` and `peak_vram_mb` — matters for iteration speed
4. **Overfitting signal**: Gap between train_loss and eval accuracy — a very low train_loss with poor eval accuracy suggests overfitting

### Comparison strategy

- Always compare against the **baseline** (un-fine-tuned model), not just the previous run
- When comparing runs, hold all variables constant except the one being tested
- Use `list_experiments` to get a table of all past runs

## Convergence Analysis

### Is loss still decreasing?

- If final train_loss is still dropping steeply at the last step → model wants more training, increase `max_steps`
- If train_loss has plateaued → more steps won't help, try a different hyperparameter
- If train_loss oscillates wildly → learning rate is too high, reduce it

### Has the model converged?

A model has converged when:
- Train loss has plateaued (no meaningful decrease over the last 20% of steps)
- Eval metric has stopped improving between iterations
- Additional training shows diminishing returns (<0.5% improvement per iteration)

### When to stop iterating

Stop early if:
- Target metric is met
- Last 2-3 iterations each improved by less than 0.5%
- Eval metric is getting worse (overfitting or catastrophic forgetting)
- You've exhausted the hyperparameter search space in the tuning priority order

## Organizing Experiments

### Naming conventions

Runs are auto-numbered (`run_001`, `run_002`, ...). When reporting to the user, include:
- Run number
- The one variable that changed from the previous run
- Key result (target metric + train_loss)

Example: "Run 003 (lr=5e-5): MMLU 43.2%, train_loss 0.92 — down from Run 002's 44.1%"

### What to log

Always report these for every run:
- All hyperparameters that differ from defaults
- train_loss
- Target metric (e.g., MMLU accuracy)
- Training time
- Peak VRAM

## Reproducing Results

- Training is not perfectly deterministic across runs even with the same seed, due to GPU non-determinism
- Small variations (±0.5% on MMLU) between identical configs are normal
- If results vary by more than 1%, suspect a real difference (different batch ordering, data shuffling)
- Always record the full config — partial configs make reproduction impossible
