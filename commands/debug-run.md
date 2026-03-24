---
name: debug-run
description: Diagnose a failed or weak run from logs and saved metrics.
user_invocable: true
---

# /debug-run — Diagnose a Run

Use this when a run fails, OOMs, diverges, or performs badly.

## Steps

1. Identify the run directory or use the most recent run if the user did not specify one.
2. Call `diagnose_experiment`.
3. Report likely failure modes and the smallest sensible retry.

## Rule

Prefer the smallest next change. Do not combine multiple recovery changes unless the failure is obvious.
