---
name: train
description: Legacy shortcut for a single approved training run.
user_invocable: true
---

# /train — Legacy Training Shortcut

This is a lightweight shortcut for experienced users. Prefer `/plan-experiments`
and `/run-experiment` for the full workflow.

## Steps

1. Call `check_gpu`.
2. Parse model, dataset, and hyperparameters.
3. Call `run_training`.
4. Report the run directory and training metrics.
