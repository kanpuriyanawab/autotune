---
name: run-experiment
description: Execute one approved experiment run and evaluate it immediately after training.
user_invocable: true
---

# /run-experiment — Execute One Run

Use this after the user approves a concrete run configuration.

## Steps

1. Parse the approved run config.
2. Check the `trainer` field in the config:
   - If `trainer == "dpo"` or `trainer == "orpo"`: call `run_dpo_training` (dataset must have `prompt/chosen/rejected` columns).
   - Otherwise: call `run_training` (SFT path).
3. Call `run_evaluation` on the resulting adapter.
4. Report:
   - run directory
   - train loss
   - training time
   - evaluation metric
   - recommendation: compare, debug, or continue

## Rule

Run one approved experiment at a time. Do not silently execute the full plan.
