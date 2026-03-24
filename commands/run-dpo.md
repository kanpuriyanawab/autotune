---
name: run-dpo
description: Execute one approved DPO or ORPO preference-optimization run and evaluate the result.
user_invocable: true
---

# /run-dpo

Run one approved DPO (Direct Preference Optimization) or ORPO run on a preference dataset and evaluate the result.

## Steps

1. Parse the approved run config from the experiment plan (method, beta, learning_rate, lora_r, batch_size, max_steps).
2. Confirm the dataset has `prompt`, `chosen`, and `rejected` columns before proceeding (use `inspect_dataset` if unsure).
3. Call `run_dpo_training` with the approved config.
4. After training completes, call `run_evaluation` to measure MMLU or custom eval on the adapter.
5. Report: run directory, train loss, training time, evaluation metric, and recommendation.

## Defaults

- method: dpo
- beta: 0.1 (controls strength of preference signal; lower = stronger)
- learning_rate: 5e-5 (lower than SFT — preference optimization is more sensitive)
- batch_size: 2 (DPO requires pairs, so effective batch = 2x SFT)
- max_seq_length: 1024

## Rule

Run one approved DPO config at a time. Do not silently execute the full plan. After each run call `compare_experiments` to check if the result beats the current best.
