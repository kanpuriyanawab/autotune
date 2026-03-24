---
name: compare-runs
description: Rank completed runs by saved metrics and summarize the best next move.
user_invocable: true
---

# /compare-runs — Compare Results

Use this after at least one run has both training metrics and evaluation output.

## Steps

1. Call `compare_experiments`.
2. Summarize which run is best and why.
3. Explain whether the current evidence supports:
   - another run
   - failure diagnosis
   - ship review
