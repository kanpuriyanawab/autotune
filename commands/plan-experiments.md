---
name: plan-experiments
description: Produce an approval-ready experiment sequence with concrete run configs.
user_invocable: true
---

# /plan-experiments — Plan Runs

This command is the main planning surface. It should stop at a proposal, not execute training.

## Steps

1. Identify model, dataset, task family, and budget from the user request.
2. If the dataset format is unknown, call `inspect_dataset` first.
3. Call `suggest_backends`.
4. Call `plan_experiments`.
   - If the dataset is preference-format (`prompt/chosen/rejected`), the plan will automatically propose DPO/ORPO runs instead of SFT. Use `/run-dpo` to execute those.
   - For instruction/chat/text datasets, the plan proposes SFT runs. Use `/run-experiment` to execute those.
   - The planner adjusts batch_size automatically if VRAM is detected, and adjusts max_steps based on dataset size.
5. Present the exact run sequence, explain why each run exists, and ask the user which run to execute first.

## Rule

Do not start training from `/plan-experiments`. The output is a compute approval gate.
