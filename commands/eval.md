---
name: eval
description: Legacy shortcut for evaluating a base model or adapter.
user_invocable: true
---

# /eval — Legacy Evaluation Shortcut

Prefer `/baseline` for base-model evaluation and `/run-experiment` for
post-training evaluation, but keep this command for direct inspection.

## Steps

1. Resolve the base model or adapter the user wants to evaluate.
2. Ask whether to use a standard benchmark or a custom eval dataset:
   - **Standard**: `run_evaluation(benchmark="mmlu")` — measures general knowledge (200 samples for speed)
   - **Custom**: `run_evaluation(eval_dataset="<your-dataset>", eval_split="test")` — measures loss/perplexity on your own task data (more informative for task-specific fine-tunes)
3. Call `run_evaluation` with the resolved parameters.
4. Report the saved metric and where the eval artifact was written.
