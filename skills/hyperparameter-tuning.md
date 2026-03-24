---
name: hyperparameter-tuning
description: When and how to adjust each hyperparameter during LoRA fine-tuning, including learning rate, batch size, and scheduler selection.
---

# Hyperparameter Tuning

## The One-Variable-at-a-Time Strategy

Never change multiple hyperparameters between runs. Adjust ONE variable, train, evaluate, then decide the next move. This makes it possible to attribute improvements (or regressions) to specific changes.

## Priority Order

Adjust hyperparameters in this order — earlier items have more impact:

### 1. Learning Rate (highest impact)

#### SFT (Supervised Fine-Tuning)

| Model Size | Starting LR | Range to explore |
|-----------|-------------|------------------|
| < 1B | 3e-4 | 1e-4 to 5e-4 |
| 1B–3B | 2e-4 | 1e-4 to 4e-4 |
| 3B–8B | 2e-4 | 5e-5 to 3e-4 |
| 8B–13B | 1e-4 | 2e-5 to 2e-4 |
| 13B–34B | 5e-5 | 1e-5 to 1e-4 |
| 34B+ | 2e-5 | 5e-6 to 5e-5 |

#### DPO / ORPO (Preference Optimization)

Preference optimization is more sensitive to learning rate than SFT. Use lower values and always verify the chosen reward accuracy increases.

| Model Size | Starting LR (DPO) | Starting LR (ORPO) | Range to explore |
|-----------|-------------------|-------------------|------------------|
| < 1B | 1e-4 | 2e-4 | 5e-5 to 3e-4 |
| 1B–3B | 5e-5 | 1e-4 | 2e-5 to 2e-4 |
| 3B–8B | 5e-5 | 8e-5 | 2e-5 to 1e-4 |
| 8B–13B | 2e-5 | 4e-5 | 5e-6 to 5e-5 |
| 13B+ | 1e-5 | 2e-5 | 1e-6 to 2e-5 |

**Interaction with LoRA rank**: Research shows the optimal learning rate is mostly invariant across rank values. You generally do not need to change LR when changing rank.

**Signs LR is wrong**:
- Too high: loss oscillates, spikes, or diverges; eval metric is erratic between runs
- Too low: loss decreases very slowly; model barely improves from baseline

### 2. LoRA Rank

- Start with r=16
- If underfitting (loss plateaus above 1.0): try r=32, then r=64
- If overfitting (low train loss but eval gets worse): drop to r=8
- Doubling rank roughly doubles trainable parameters and VRAM for adapter

### 3. Training Steps

- Default: 200 steps
- If loss is still decreasing at step 200: extend to 400, then 600
- If loss plateaued by step 100: more steps won't help — change something else
- **Watch for overfitting**: if train_loss keeps dropping but eval accuracy stalls or declines, you're training too long

### 4. Batch Size

- Default: 4
- Larger batch (8): more stable gradients, slightly faster per step — but uses more VRAM
- Smaller batch (2, 1): noisier gradients, fits in less VRAM — acts as implicit regularization
- **Effective batch size** = batch_size × gradient_accumulation_steps. Keep this in mind when comparing runs.

## Learning Rate Schedulers

| Scheduler | Behavior | When to use |
|-----------|----------|-------------|
| **Cosine** | Decays LR following cosine curve to near zero | **Default — works well in most cases** |
| **Linear** | Linearly decays LR to zero | Good alternative, slightly less aggressive |
| **Constant** | No decay after warmup | Short training runs (<100 steps) |
| **Cosine with restarts** | Cosine with periodic warm restarts | Long training, multiple phases |

## Warmup

- **Default**: `warmup_ratio = 0.03` (3% of total steps)
- For short runs (<200 steps): `warmup_ratio = 0.1` (10%) to avoid unstable early training
- For long runs (>1000 steps): `warmup_ratio = 0.01-0.03` is sufficient
- Warmup helps prevent early divergence, especially with higher learning rates

## Diagnosing Loss Curves

### Overfitting
- **Pattern**: Train loss continues to drop, but eval accuracy plateaus or decreases
- **Fix**: Reduce max_steps, reduce lora_r, increase lora_dropout, or use a smaller learning rate

### Underfitting
- **Pattern**: Train loss plateaus at a high value (>1.5) and eval barely improves
- **Fix**: Increase learning rate, increase lora_r, or increase max_steps

### Divergence
- **Pattern**: Loss spikes up or oscillates wildly, especially early in training
- **Fix**: Reduce learning rate by 2-5x, increase warmup ratio

### Good training
- **Pattern**: Loss decreases smoothly, with a steep initial drop that gradually levels off
- Eval accuracy improves between iterations and tracks with loss reduction

## Gradient Accumulation

Use gradient accumulation to simulate larger batch sizes without more VRAM:
- `gradient_accumulation_steps = 4` with `batch_size = 2` ≈ effective batch of 8
- Useful when you want large-batch stability but are VRAM-constrained
- Trade-off: training is slower (more forward passes per optimizer step)
