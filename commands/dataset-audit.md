---
name: dataset-audit
description: Inspect dataset shape, infer training format, and flag quality risks before training.
user_invocable: true
---

# /dataset-audit — Audit a Dataset

Run a dataset inspection before training or planning experiments.

## Steps

1. Identify the dataset reference and split from the user request.
2. Call `inspect_dataset`.
3. Summarize:
   - inferred format
   - likely training objective
   - immediate risks
   - whether the dataset is safe to use for SFT as-is
4. If the dataset looks preference-formatted or structurally unclear, tell the user before proposing a training run.
