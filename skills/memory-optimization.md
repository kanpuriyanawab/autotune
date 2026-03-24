---
name: memory-optimization
description: VRAM estimation, quantization options, gradient checkpointing, OOM recovery ladder, and requirements by model size.
---

# Memory Optimization

## VRAM Estimation

Total VRAM ≈ Model weights + Optimizer states + Activations + Gradients

### Model weights by precision

| Model Size | FP16/BF16 | 8-bit | 4-bit (QLoRA) |
|-----------|-----------|-------|---------------|
| 1B | 2 GB | 1 GB | 0.5 GB |
| 3B | 6 GB | 3 GB | 1.5 GB |
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |

### Additional VRAM overhead

- **LoRA adapter weights**: ~50-200 MB (depends on rank and number of target modules)
- **Optimizer states (AdamW)**: 2× adapter size (momentum + variance)
- **Activations**: Scales with batch_size × seq_len. Roughly 0.5-2 GB per sample for 7B
- **Gradients**: ~equal to adapter size

## Typical VRAM Requirements (QLoRA, rank=16, batch_size=4, seq_len=2048)

| Model Size | Training VRAM | Minimum GPU |
|-----------|---------------|-------------|
| 1B | ~4-5 GB | RTX 3060 (12 GB) |
| 3B | ~7-9 GB | RTX 3060 (12 GB) |
| 7B | ~12-18 GB | RTX 3090/4090 (24 GB) |
| 13B | ~18-22 GB | RTX 4090 (24 GB) or A5000 |
| 70B | ~44-50 GB | A100 (80 GB) |

## 4-Bit Quantization (QLoRA)

### NF4 (NormalFloat4)
- **Default quantization type** — optimized for normally-distributed weights
- Better than pure INT4 for LLM weights
- `bnb_4bit_quant_type = "nf4"`

### Double quantization
- Quantizes the quantization constants themselves — saves ~0.4 GB for a 7B model
- Negligible quality impact
- `bnb_4bit_use_double_quant = True`

### Compute dtype
- Use `bnb_4bit_compute_dtype = torch.bfloat16` for training (not float16, which can overflow)
- BF16 has larger dynamic range, fewer NaN issues during training

## Gradient Checkpointing

### What it does
Trades compute for memory: discards intermediate activations during forward pass, recomputes them during backward pass.

### Impact
- **Memory savings**: ~30-50% reduction in activation memory
- **Speed penalty**: ~20-30% slower training (recomputation cost)
- Enable via `gradient_checkpointing = True`

### When to enable
- Enable when you're close to OOM (>90% VRAM utilization)
- Always enable for models ≥7B on 24 GB GPUs
- Skip for small models on large GPUs (unnecessary slowdown)

## Flash Attention

### What it does
Fused attention kernel that reduces memory from O(n²) to O(n) in sequence length and is faster.

### Compatibility
- **Flash Attention 2**: Supports Llama, Mistral, Phi, Gemma, Qwen, and most modern architectures
- Requires Ampere (RTX 30xx) or newer GPU
- `attn_implementation = "flash_attention_2"`

### Impact
- ~20-40% faster training
- ~30-50% less memory for attention computation
- No quality impact — mathematically equivalent

## Sequence Length

- Attention memory scales **quadratically** without Flash Attention, **linearly** with it
- All other components scale linearly with sequence length
- Halving seq_len roughly halves activation memory (with Flash Attention)
- Default 2048 is good for most instruction tuning; reduce to 1024 if OOM

## OOM Recovery Ladder

When training fails with out-of-memory, follow this sequence:

1. **Halve batch_size** (4→2→1) — most VRAM savings per quality impact
2. **Enable gradient checkpointing** — if not already enabled
3. **Reduce sequence length** (2048→1024→512) — only if data permits shorter sequences
4. **Reduce LoRA rank** (32→16→8) — reduces adapter and optimizer memory
5. **Enable 4-bit quantization** — if not already using QLoRA
6. **Use a smaller model** — last resort

At each step, retry training. Stop at the first configuration that fits.

## Monitoring VRAM

- Use `check_gpu` before training to see available VRAM
- `peak_vram_mb` in `run_config.json` shows actual peak usage
- Leave ~1-2 GB headroom — PyTorch CUDA allocator has overhead
- If peak_vram is >90% of total, you're at risk of OOM on longer sequences
