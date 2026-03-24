---
name: merge
description: Merge a LoRA adapter into the base model weights to produce a standalone deployable model.
user_invocable: true
---

# /merge

Merge a trained LoRA adapter into the base model so the result can be deployed without the adapter file, or exported to GGUF.

## Steps

1. Identify the base model name and the adapter path (from the most recent run or user-specified).
2. Determine the output directory (default: `<adapter_path>/merged/`).
3. Call `merge_adapter` with the base model, adapter path, and output directory.
4. Report the merged model path.
5. Suggest the next step: `/export-gguf` if the user wants to run locally, or `/serve` to test the merged model via Gradio.

## Defaults

- output_dir: `<adapter>/merged/`
- load_in_4bit: false (merging in bfloat16 gives the cleanest result; use 4-bit only if VRAM is insufficient)

## Rule

Always confirm the adapter path exists before calling `merge_adapter`. If the user has multiple runs, call `compare_experiments` first to identify the best adapter.
