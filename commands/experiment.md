---
name: experiment
description: Legacy wrapper that plans, runs one approved experiment, and points to comparison.
user_invocable: true
---

# /experiment — Guided Experiment Flow

This command preserves the old entrypoint, but it should behave like the new
workflow instead of running an uncontrolled loop.

## Steps

1. Call `plan_experiments`.
2. Present the first recommended run for approval.
3. After approval, call `run_training`.
4. Call `run_evaluation`.
5. Recommend `/compare-runs` or `/debug-run` next.

## Rule

Do not execute multiple training runs automatically from this command.
