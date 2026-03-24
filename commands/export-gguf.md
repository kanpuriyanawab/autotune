---
name: export-gguf
description: Export a fine-tuned model to GGUF format for llama.cpp, Ollama, or LM Studio.
user_invocable: true
---

# /export-gguf

Export a trained model to GGUF format so it can run locally via llama.cpp, Ollama, or LM Studio.

## Steps

1. Identify the base model name and the adapter path (from the most recent run or user-specified).
2. Ask the user which quantization they want: `q4_k_m` (recommended, smallest/fastest), `q5_k_m` (better quality), `q8_0` (near-lossless), or `f16` (full precision, largest).
3. If Unsloth is available, `export_gguf` will use the native path (best quality, handles merge internally).
4. If the user has a LoRA adapter (not a merged model), note that the export script handles the merge automatically.
5. Call `export_gguf` with the resolved model, adapter, quantization, and output directory.
6. Report the GGUF file path and the quantization used.
7. Show the Ollama import command: `ollama create my-model -f Modelfile` (where Modelfile points to the GGUF).

## Defaults

- quantization: q4_k_m
- output_dir: exports/gguf/ under the project root

## Rule

Do not export before evaluating — always compare runs first so you export the best adapter, not just the latest.
