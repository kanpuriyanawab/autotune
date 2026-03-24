---
name: release-readiness
description: Decide whether a trained model is ready for human review or more iteration.
---

# Release Readiness

## Minimum Bar

- the best run beats the baseline on the agreed metric
- the eval artifact is saved and reproducible
- no unresolved failure mode remains from training logs
- the run can be explained in terms of data, config, and outcome

## Ship Review Questions

- Is the improvement real or just variance?
- Did the run preserve the behaviors that matter in production?
- Are latency, memory, and licensing still acceptable?
- Can another run plausibly produce a larger gain, or are returns already diminishing?

## Default Recommendation Logic

- Ship for human review if the run clears the target and diagnostics look clean.
- Keep iterating if the best run is ambiguous, brittle, or hard to reproduce.
