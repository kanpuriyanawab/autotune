---
name: experiment-design
description: How to design high-signal experiment sequences for applied ML teams.
---

# Experiment Design

## Core Rule

Every run should answer a specific question. If a run does not isolate a decision,
it is not a good run.

## Default Sequence

1. Establish a baseline before training.
2. Start with one conservative run that is likely to complete.
3. Change one major variable at a time.
4. Compare runs before expanding the search.

## Good First Questions

- Does the dataset format actually match the intended objective?
- Is the base model already good enough for the task?
- Is the bottleneck data quality, optimization, or capacity?
- Is faster iteration more valuable than absolute peak quality?

## Run Design Heuristics

- Small or noisy dataset: lower learning rate, fewer aggressive changes
- Unknown hardware ceiling: conservative batch size and sequence length first
- Weak baseline but good loss curve: increase steps before changing rank
- Good loss but weak eval: suspect formatting, eval mismatch, or data quality
