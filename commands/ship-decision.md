---
name: ship-decision
description: Produce a recommendation on whether the best run is ready for human review or more iteration.
user_invocable: true
---

# /ship-decision — Review the Best Run

Use this only after comparing completed runs.

## Steps

1. Call `ship_decision`.
2. Report:
   - best run
   - key metric
   - remaining risks
   - recommendation: ship for human review or keep iterating
