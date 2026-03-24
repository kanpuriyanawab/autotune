---
name: model-selection
description: Which base models to choose for fine-tuning — size/quality tradeoffs, model families, strengths, and license considerations.
---

# Model Selection

## Current Model Landscape (2025)

### Size Tiers

#### Small (< 3B) — Fast iteration, consumer GPUs
| Model | Params | MMLU (base) | Strengths | License |
|-------|--------|-------------|-----------|---------|
| Llama 3.2 1B | 1.2B | ~49% | Fast training, exceptional tunability | Llama 3.2 Community |
| Qwen 2.5 1.5B | 1.5B | ~56-60% | Strong for size, multilingual | Apache 2.0 |
| Gemma 2 2B | 2.6B | ~51% | Good general quality, code understanding | Gemma license |
| Llama 3.2 3B | 3.2B | ~62% | Lightest model with genuinely useful performance | Llama 3.2 Community |

**Best pick**: Qwen 2.5 1.5B for quality-per-parameter; Llama 3.2 1B for speed.

#### Medium (3B–8B) — Best quality/cost balance
| Model | Params | MMLU (base) | Strengths | License |
|-------|--------|-------------|-----------|---------|
| Phi-3 Mini | 3.8B | ~69% | Exceptional for size, strong reasoning | MIT |
| Phi-4 Mini | 3.8B | ~67% | Strong coding and math | MIT |
| Qwen 2.5 7B | 7.2B | ~74% | Best 7B class, strong multilingual | Apache 2.0 |
| Llama 3.1 8B | 8.0B | ~66-68% | Most community support, well-studied | Llama 3.1 Community |
| Mistral 7B v0.3 | 7.2B | ~63% | Fast inference, good general quality | Apache 2.0 |
| Gemma 2 9B | 9.2B | ~71% | Strong reasoning and factuality | Gemma license |

**Best pick**: Qwen 2.5 7B for raw quality; Llama 3.1 8B for ecosystem support; Phi-3 Mini for size efficiency.

#### Large (13B+) — Maximum quality, needs beefy GPU
| Model | Params | MMLU (base) | Strengths | License |
|-------|--------|-------------|-----------|---------|
| Phi-4 (14B) | 14B | ~85% | Rivals 70B models on knowledge tasks | MIT |
| Qwen 2.5 14B | 14.2B | ~78% | Excellent quality, Apache licensed | Apache 2.0 |
| Llama 3.3 70B | 70.6B | ~86% | Best open model at this scale | Llama 3 Community |
| Qwen 2.5 72B | 72.7B | ~86% | Best open-weight model, competitive with Llama 3.3 | Apache 2.0 |

## Base vs Instruct Models

### Start from Base when:
- Training on a specific format/style (base models are more "moldable")
- Building a domain-specific model from scratch
- The training dataset is large (>10K examples) and covers the desired behavior

### Start from Instruct when:
- You want to preserve general instruction-following ability
- Fine-tuning for a narrow task while keeping chat capabilities
- Training dataset is small (<5K examples) — the instruct tuning provides a strong foundation
- **Caution**: Instruct models have specific chat templates; your training data should match

## Model Family Strengths

| Family | Coding | Reasoning | Multilingual | Long context | Community |
|--------|--------|-----------|-------------|-------------|-----------|
| **Llama 3.x** | Good | Good | Moderate | 128K | Largest |
| **Qwen 2.5** | Excellent | Excellent | Best | 128K | Growing |
| **Phi-3/4** | Excellent | Excellent | Moderate | 128K | Moderate |
| **Mistral** | Good | Good | Good | 32K | Large |
| **Gemma 2** | Good | Good | Moderate | 8K | Moderate |

## License Considerations

| License | Commercial use | Modifications | Distribution |
|---------|---------------|---------------|-------------|
| **Apache 2.0** (Qwen, Mistral) | Yes | Yes | Yes |
| **MIT** (Phi-3/4) | Yes | Yes | Yes |
| **Llama Community** | Yes (with restrictions) | Yes | Yes (monthly active users < 700M) |
| **Gemma** | Yes | Yes | Yes (with use policy) |

**Safest for commercial use**: Apache 2.0 models (Qwen 2.5, Mistral) or MIT (Phi-3/4).

## LoRA Fine-Tuning Responsiveness

Some models respond better to LoRA than others:
- **Most responsive**: Llama 3.x, Qwen 2.5 — well-studied, many successful LoRA fine-tunes
- **Good**: Mistral, Phi-3/4 — work well but may need slightly lower learning rates
- **Requires care**: Gemma 2 — can be sensitive to hyperparameters, especially learning rate

## Quick Decision Guide

1. **Prototyping / fast iteration**: Llama 3.2 1B or Qwen 2.5 1.5B
2. **Best quality on a 24GB GPU**: Qwen 2.5 7B with QLoRA
3. **Coding tasks**: Phi-4 Mini or Qwen 2.5 7B
4. **Multilingual**: Qwen 2.5 (any size)
5. **Maximum community support**: Llama 3.1 8B
6. **Commercial deployment (Apache 2.0)**: Qwen 2.5 or Mistral
