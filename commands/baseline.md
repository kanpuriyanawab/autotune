---
name: baseline
description: Run a baseline evaluation on the base model before any fine-tuning.
user_invocable: true
---

# /baseline — Establish a Baseline

Evaluate the base model before training so later runs have a real reference point.

## Steps

1. Identify the model and benchmark from the user request.
2. Default to quick evaluation unless the user asks for a full run.
3. Call `run_evaluation` with no adapter.
4. Report the baseline metric and explain how later runs should be compared against it.

## Defaults

- `benchmark`: `mmlu`
- `num_samples`: `200`
