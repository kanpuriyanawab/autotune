---
name: evaluation-benchmarks
description: MMLU and other benchmarks — expected scores by model size, interpretation, pitfalls, and catastrophic forgetting.
---

# Evaluation Benchmarks

## MMLU (Massive Multitask Language Understanding)

### What it measures
57 subjects across STEM, humanities, social sciences, and professional domains. Tests broad factual knowledge and reasoning ability. Scored as accuracy (0-100%).

### Expected Scores by Model Size (Base Models)

| Model | MMLU (5-shot) |
|-------|--------------|
| Llama 3.2 1B | ~49% |
| Qwen 2.5 1.5B | ~56-60% |
| Gemma 2 2B | ~51% |
| Llama 3.2 3B | ~62% |
| Phi-3 Mini (3.8B) | ~69% |
| Phi-4 Mini (3.8B) | ~67% |
| Mistral 7B v0.3 | ~63% |
| Qwen 2.5 7B | ~74% |
| Llama 3.1 8B | ~66-68% |
| Gemma 2 9B | ~71% |
| Qwen 2.5 14B | ~78% |
| Phi-4 (14B) | ~85% |

*Scores are approximate and vary by evaluation harness, prompt template, and few-shot setting. Instruct variants typically score 3-8% higher than base.*

### Quick vs Full Evaluation

| Setting | Samples | Time (7B) | Reliability |
|---------|---------|-----------|-------------|
| Quick (`num_samples=200`) | 200 | ~2 min | ±2-3% variance |
| Full (all) | ~14,000 | ~15-30 min | ±0.5% variance |

**Rule**: Use quick eval (200 samples) during iteration. Switch to full eval only for the final result or when scores are close to the target.

### Per-Category Breakdown

MMLU categories that are most affected by fine-tuning:
- **STEM** (math, physics, CS): Hardest to improve, may actually decrease with instruction tuning
- **Humanities** (history, philosophy): Moderate improvement with diverse training data
- **Social Sciences** (psychology, economics): Often improves with instruction tuning
- **Professional** (law, medicine): Requires domain-specific training data

## Other Benchmarks

| Benchmark | What it tests | Typical base 7B score | Fine-tuning impact |
|-----------|--------------|----------------------|-------------------|
| **HellaSwag** | Commonsense reasoning, sentence completion | ~80% | Usually stable or slight improvement |
| **ARC-Challenge** | Grade-school science reasoning | ~55% | Can improve with reasoning data |
| **TruthfulQA** | Resistance to common misconceptions | ~35-45% | Often improves with instruction tuning |
| **Winogrande** | Pronoun resolution, commonsense | ~75% | Usually stable |
| **GSM8K** | Grade-school math word problems | ~30-50% | Improves significantly with math data |
| **HumanEval** | Python code generation | ~20-40% | Improves significantly with code data |

## Evaluation Pitfalls

### Prompt sensitivity
- MMLU scores can vary 3-5% depending on the prompt template
- Always use the same template when comparing base vs fine-tuned
- Chat-formatted models need the prompt wrapped in their chat template

### Contamination
- Some training datasets contain MMLU questions — this inflates scores
- If MMLU jumps >10% after fine-tuning on a general dataset, suspect contamination
- Cross-check with other benchmarks to confirm genuine improvement

### Chat template effects
- Evaluating a chat model without its chat template underestimates performance by 5-15%
- Evaluating a base model WITH a chat template can also hurt scores
- Match the template to the model type

### Few-shot vs zero-shot
- Standard MMLU uses 5-shot prompting
- Zero-shot scores are typically 5-10% lower
- Be consistent — always report which setting was used

## Catastrophic Forgetting

### What it is
Fine-tuning on a narrow dataset causes the model to lose general capabilities. MMLU drops because the model "forgets" broad knowledge.

### When it happens
- Training too long on a small, narrow dataset
- Very high learning rate (model weights change too aggressively)
- High LoRA rank with small dataset (too much capacity, overfits to narrow distribution)

### Detection
- MMLU drops >5% below the base model baseline
- Model becomes very good at the training task but worse at everything else
- Per-category MMLU shows sharp drops in categories unrelated to training data

### Prevention
- Keep training short (200-400 steps for <10K examples)
- Use moderate LoRA rank (r=16-32)
- Use appropriate learning rate (not too high)
- Consider mixing in general-purpose data with task-specific data

### Recovery
- Reduce max_steps (often the primary fix)
- Lower learning rate
- Reduce LoRA rank
- Add general instruction data to the training mix
