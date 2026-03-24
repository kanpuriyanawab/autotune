---
name: lora-finetuning
description: LoRA and QLoRA fine-tuning patterns, rank selection, target modules by model family, and adapter merging.
---

# LoRA Fine-Tuning

## How LoRA Works

LoRA (Low-Rank Adaptation) freezes the base model and injects small trainable matrices into selected layers. Instead of updating a full weight matrix W (d×d), it learns two small matrices A (d×r) and B (r×d) where r << d. The effective update is W + BA, training only 2×d×r parameters per layer instead of d².

## Rank Selection

| Rank (`lora_r`) | Use case | Trainable params (7B model) |
|-----------------|----------|-----------------------------|
| 8 | Quick tests, simple tasks | ~6M |
| 16 | **Default — good balance** | ~13M |
| 32 | Complex tasks, larger datasets | ~26M |
| 64 | High-quality fine-tuning, diverse tasks | ~52M |
| 128+ | Rarely needed — diminishing returns | ~100M+ |

**Rule of thumb**: Start with r=16. If the model underfits (loss plateaus too high), try r=32 or r=64. If it overfits, r=8 may suffice.

## Alpha and Scaling

- `lora_alpha` controls the scaling factor: effective weight = alpha/rank
- **Standard**: `lora_alpha = 2 × lora_r` (scaling factor = 2.0) — recommended default per 2025 research
- **Conservative**: `lora_alpha = lora_r` (scaling factor = 1.0) — works fine in practice, slightly weaker adaptation
- With rsLoRA enabled, alpha is scaled by √r internally, so `lora_alpha = lora_r` is sufficient
- **Never** set alpha much lower than rank — the adaptation becomes too weak to learn

## Target Modules by Model Family

All modern architectures benefit from targeting attention + MLP layers:

| Model Family | Target Modules |
|-------------|----------------|
| **Llama 3.x** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Mistral** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Phi-3/4** | `qkv_proj, o_proj, gate_up_proj, down_proj` |
| **Gemma 2** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Qwen 2.5** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

**Shortcut**: Most frameworks accept `"all"` to target all linear layers. This is safe and usually optimal.

## LoRA Dropout

- `lora_dropout = 0.0` — **default, fine for most cases**
- `lora_dropout = 0.05` — use with larger datasets (>50K examples) or if overfitting
- Higher dropout (0.1+) rarely helps and slows convergence

## QLoRA vs LoRA vs Full Fine-Tuning

| Method | VRAM (7B model) | Quality | When to use |
|--------|-----------------|---------|-------------|
| **QLoRA (4-bit)** | ~6-8 GB | 95-99% of LoRA | **Default choice** — fits on consumer GPUs |
| **LoRA (16-bit)** | ~16-20 GB | Baseline | When you have the VRAM and need maximum quality |
| **Full fine-tuning** | ~60+ GB | Best possible | Large datasets, production models, multi-GPU |

**QLoRA quality gap**: In practice, 4-bit QLoRA matches full LoRA within 0.5-1% on most benchmarks. The gap widens slightly on math/reasoning tasks. Use full LoRA only when you have the VRAM and need every fraction of a percent.

**When NOT to use QLoRA**:
- Very long training runs (>1000 steps) where quantization error accumulates
- Tasks requiring precise numerical reasoning
- When you have ample VRAM and want to eliminate variables

## LoRA Variants

### DoRA (Weight-Decomposed Low-Rank Adaptation)
- Decomposes weights into magnitude and direction, applies LoRA to direction only
- Typically 1-3% better than standard LoRA at the same rank
- ~10% slower training due to extra decomposition step
- **Verdict**: Use when you need to squeeze out quality without increasing rank

### rsLoRA (Rank-Stabilized LoRA)
- Scales by 1/√r instead of 1/r, stabilizing training at higher ranks
- Matters mainly at r≥32 where standard LoRA scaling can be unstable
- **Verdict**: Enable for high-rank (r≥32) training, negligible impact at r≤16

### QDoRA (Quantized DoRA)
- Combines DoRA with 4-bit quantization — best of both worlds
- Significantly outperforms QLoRA on math/reasoning tasks (31.2% vs 11.8% exact match on Orca-Math in one study)
- Emerging as a strong default for memory-constrained setups
- **Verdict**: Use as a drop-in QLoRA replacement when available

## Merging Adapters

After fine-tuning, you can merge the LoRA adapter back into the base model:
- Eliminates inference overhead (no adapter loading)
- Required for deployment with frameworks that don't support adapters
- **Caution**: Merging is lossy with QLoRA — the 4-bit base weights are dequantized during merge, introducing small errors. For highest quality, train with full LoRA if you plan to merge.
