---
name: failure-diagnosis
description: Diagnose failed or weak ML runs from logs, metrics, and dataset shape.
---

# Failure Diagnosis

## OOM

Likely causes:
- batch size too high
- sequence length too high
- model too large for the hardware

Recovery order:
1. halve batch size
2. reduce sequence length
3. switch to 4-bit loading
4. drop rank or choose a smaller model

## Divergence Or NaNs

Likely causes:
- learning rate too high
- malformed or extremely long examples
- unstable precision settings

Recovery order:
1. lower learning rate
2. inspect dataset outliers
3. retry with a smaller, cleaner run

## Good Loss, Bad Eval

Likely causes:
- wrong chat template
- wrong benchmark for the task
- overfitting narrow formatting patterns
- dataset label or prompt mismatch

Recovery order:
1. validate formatting and prompting
2. inspect examples manually
3. compare against the base model
