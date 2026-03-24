---
name: dataset-preparation
description: Dataset formats, auto-detection, quality checks, train/val splits, popular datasets, and chat template formatting.
---

# Dataset Preparation

## Common Formats

### Alpaca Format
```json
{
  "instruction": "Summarize the following text.",
  "input": "The quick brown fox...",
  "output": "A fox jumps over a lazy dog."
}
```
- Three fields: `instruction`, `input` (optional), `output`
- Most common for instruction tuning
- Detected by column names containing "instruction" and "output"

### ShareGPT / Conversational Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is LoRA?"},
    {"from": "gpt", "value": "LoRA stands for Low-Rank Adaptation..."}
  ]
}
```
- Multi-turn conversations
- Detected by presence of "conversations" column
- Role names vary: `human/gpt`, `user/assistant`, `system/user/assistant`

### Chat Messages Format
```json
{
  "messages": [
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA stands for..."}
  ]
}
```
- OpenAI-compatible format
- Detected by presence of "messages" column

### Plain Text
- Raw text for continued pretraining
- No structure — just `text` column
- Use for domain adaptation, not instruction tuning

## Auto-Detection Heuristics

To detect format automatically, check column names:
1. Has `"conversations"` → ShareGPT format
2. Has `"messages"` → Chat messages format
3. Has `"instruction"` and `"output"` → Alpaca format
4. Has `"text"` only → Plain text
5. Has `"prompt"` and `"completion"` → Prompt-completion format

## Quality Checks

### Before training, verify:
1. **No empty fields**: Filter out rows where instruction/output are empty
2. **Deduplication**: Remove exact duplicates (same instruction + output)
3. **Length filtering**: Remove very short (<10 tokens) or very long (>max_seq_length) examples
4. **Language check**: If training for a specific language, filter out mismatched samples
5. **Format consistency**: All rows should have the same schema

### Red flags:
- Dataset has many near-duplicates (paraphrased copies) — causes overfitting
- Very uneven length distribution (mostly short, few very long) — long examples may get truncated
- Mix of languages when single-language model is desired

## Train/Val Split

- **Standard**: 95/5 split (95% train, 5% validation)
- **Small datasets (<1K examples)**: No val split — use all data for training, evaluate with external benchmarks
- **Large datasets (>10K examples)**: 90/10 split is fine
- Always split BEFORE shuffling to avoid data leakage
- For conversational data, split by conversation (not by turn)

## Dataset Size Guidelines

| Dataset size | Typical result | When to use |
|-------------|---------------|-------------|
| 100-500 | Minimal change, very specific style transfer | Narrow, well-defined tasks |
| 1K-5K | **Sweet spot for LoRA** — noticeable quality improvement | Most instruction tuning |
| 5K-10K | Strong fine-tuning, risk of overfitting with high rank | Diverse tasks |
| 10K-50K | Excellent results, may need more steps | Complex behaviors |
| 50K+ | Consider full fine-tuning | Broad capability changes |

**Rule of thumb**: 1K-5K high-quality examples is usually sufficient for LoRA. Quality > quantity.

## Popular Datasets

### General Instruction Following
| Dataset | Size | Format | License | Notes |
|---------|------|--------|---------|-------|
| `teknium/OpenHermes-2.5` | 1M | ShareGPT | Mixed | Battle-tested, diverse, top-performing |
| `Open-Orca/SlimOrca` | 518K | ShareGPT | MIT | Cleaned OpenOrca, removes low-quality outputs |
| `argilla/magpie-ultra-v0.1` | 300K | ShareGPT | Mixed | Self-synthesized from Llama-3-Instruct, high quality-to-size ratio |
| `HuggingFaceH4/ultrachat_200k` | 200K | Messages | MIT | Used for Zephyr, clean multi-turn conversations |
| `yahma/alpaca-cleaned` | 52K | Alpaca | CC BY-NC 4.0 | Cleaned Stanford Alpaca, good for small experiments |
| `BAAI/Infinity-Instruct` | Millions | Mixed | Mixed | Large-scale, strong on code and math |

### Code
| Dataset | Size | Format | License | Notes |
|---------|------|--------|---------|-------|
| `sahil2801/CodeAlpaca-20k` | 20K | Alpaca | Apache 2.0 | Code instruction following |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 157K | ShareGPT | Apache 2.0 | Multi-turn code conversations |

### Math & Reasoning
| Dataset | Size | Format | License | Notes |
|---------|------|--------|---------|-------|
| `gsm8k` (train split) | 7.5K | Text | MIT | Grade school math |
| `microsoft/orca-math-word-problems-200k` | 200K | Alpaca | MIT | Math word problems |

### Domain-Specific
| Dataset | Size | Format | License | Notes |
|---------|------|--------|---------|-------|
| `medalpaca/medical_meadow_medqa` | 10K | Alpaca | GPL 3.0 | Medical QA |
| `Amod/mental_health_counseling_conversations` | 3.5K | Text | CC BY 4.0 | Mental health |

## Chat Template Formatting

### When to apply
- **Instruct models**: Always apply the model's chat template during training
- **Base models**: Usually not needed — train on raw instruction/output pairs
- **Mixed**: If your dataset has conversations and you're fine-tuning an instruct model, the chat template must match

### How it works
Each model family has a specific template. For example, Llama 3's template wraps messages in `<|begin_of_text|>`, `<|start_header_id|>user<|end_header_id|>`, etc.

Most training frameworks (including Unsloth) handle this automatically — just specify the model name and the correct template is applied.

### Common mistake
Training an instruct model with raw text (no template) or with the wrong template. This teaches the model a conflicting format and degrades quality.
