"""Workflow planning, dataset heuristics, and run analysis helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# Hidden dimension lookup by approximate model size in billions of parameters.
# Used for VRAM estimation; nearest bucket is selected.
_MODEL_HIDDEN_DIM: dict[int, int] = {
    1: 2048,
    3: 3200,
    7: 4096,
    8: 4096,
    13: 5120,
    34: 6656,
    70: 8192,
}


def validate_lora_config(lora_r: int, lora_alpha: int) -> list[str]:
    """Return warnings about potentially suboptimal LoRA configuration."""
    warnings: list[str] = []
    if lora_alpha < lora_r:
        warnings.append(
            f"lora_alpha ({lora_alpha}) < lora_r ({lora_r}): scaling is below 1.0, "
            "which weakens adaptation. Recommended: lora_alpha = 2 * lora_r."
        )
    elif lora_alpha != 2 * lora_r:
        warnings.append(
            f"lora_alpha ({lora_alpha}) != 2 * lora_r ({lora_r}): "
            "setting lora_alpha = 2 * lora_r is the 2025 standard for optimal scaling."
        )
    return warnings


def estimate_vram(
    model_size_b: float,
    lora_r: int = 16,
    batch_size: int = 4,
    seq_len: int = 2048,
    load_in_4bit: bool = True,
    precision: str = "bf16",
    num_target_modules: int = 7,
) -> dict[str, Any]:
    """Estimate peak VRAM in GB for a LoRA fine-tuning run.

    Returns a breakdown dict with a ``total_gb`` key. This is a heuristic
    estimate — actual usage depends on model architecture and framework overhead.
    """
    bytes_per_param = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0}.get(precision, 2.0)
    weight_bytes = 0.5 if load_in_4bit else bytes_per_param
    weight_gb = model_size_b * 1e9 * weight_bytes / 1024**3

    # Find nearest known hidden dimension
    nearest = min(_MODEL_HIDDEN_DIM.keys(), key=lambda s: abs(s - model_size_b))
    hidden_dim = _MODEL_HIDDEN_DIM[nearest]

    # LoRA adapter: two low-rank matrices (A and B) per targeted linear layer
    adapter_params = lora_r * hidden_dim * 2 * num_target_modules
    adapter_gb = adapter_params * 2.0 / 1024**3  # stored in bf16

    optimizer_gb = adapter_gb * 2.0   # Adam: momentum + variance
    gradient_gb = adapter_gb          # one gradient tensor per adapter param

    # Activations: rough heuristic (batch * seq * hidden * 4 bytes)
    activation_gb = batch_size * seq_len * hidden_dim * 4.0 / 1024**3

    subtotal = weight_gb + adapter_gb + optimizer_gb + gradient_gb + activation_gb
    total_gb = subtotal * 1.10  # +10% for CUDA allocator fragmentation

    return {
        "model_weights_gb": round(weight_gb, 2),
        "adapter_gb": round(adapter_gb, 3),
        "optimizer_gb": round(optimizer_gb, 3),
        "gradient_gb": round(gradient_gb, 3),
        "activation_gb": round(activation_gb, 2),
        "total_gb": round(total_gb, 1),
    }


def detect_dataset_format(column_names: list[str]) -> str:
    """Infer the likely training format from dataset columns."""
    columns = set(column_names)
    if "messages" in columns or "conversations" in columns:
        return "chat"
    if {"prompt", "chosen", "rejected"} <= columns or {"question", "chosen", "rejected"} <= columns:
        return "preference"
    if {"instruction", "output"} <= columns or {"prompt", "completion"} <= columns:
        return "instruction"
    if "text" in columns or "content" in columns:
        return "text"
    return "unknown"


def infer_text_field(column_names: list[str]) -> str | None:
    """Suggest the field most likely to contain trainable text."""
    for candidate in ("text", "content", "messages", "conversations", "prompt"):
        if candidate in column_names:
            return candidate
    if "instruction" in column_names:
        return "instruction"
    return None


def summarize_dataset_preview(
    dataset_name: str,
    split: str,
    num_rows: int,
    column_names: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a concise dataset summary with training-specific heuristics."""
    dataset_format = detect_dataset_format(column_names)
    suggested_text_field = infer_text_field(column_names)
    risks: list[str] = []

    if dataset_format == "unknown":
        risks.append("Could not infer a canonical training format from the available columns.")
    if num_rows < 1000:
        risks.append("Dataset is small; start conservatively and expect limited generalization.")

    null_fields = set()
    long_fields = set()
    for sample in samples:
        for key, value in sample.items():
            if value in (None, "", []):
                null_fields.add(key)
            if isinstance(value, str) and len(value) > 8000:
                long_fields.add(key)

    if null_fields:
        risks.append(
            "Some sample rows contain empty values in: " + ", ".join(sorted(null_fields)) + "."
        )
    if long_fields:
        risks.append(
            "Very long sample values detected in: " + ", ".join(sorted(long_fields)) + "."
        )

    recommendations = []
    if dataset_format == "chat":
        recommendations.append("Validate chat template compatibility before training.")
    if dataset_format == "preference":
        recommendations.append("Use preference-optimization tooling rather than plain SFT.")
    if dataset_format in {"instruction", "text"}:
        recommendations.append("SFT is a sensible starting point for the first run.")
    if not recommendations:
        recommendations.append("Manually map columns before starting the first experiment.")

    return {
        "dataset": dataset_name,
        "split": split,
        "num_rows": num_rows,
        "column_names": column_names,
        "format": dataset_format,
        "suggested_text_field": suggested_text_field,
        "risks": risks,
        "recommendations": recommendations,
        "sample_keys": sorted({key for sample in samples for key in sample.keys()}),
    }


def recommend_backend(
    task_family: str = "llm",
    requested_backend: str = "auto",
    has_cuda: bool = False,
    has_unsloth: bool = False,
) -> dict[str, Any]:
    """Choose the backend and explain the tradeoff."""
    if task_family == "diffusion":
        return {
            "resolved_backend": "huggingface",
            "status": "scaffold_only",
            "reason": "Diffusion is scaffolded in v1; use Hugging Face/Diffusers as the portable path.",
        }

    if requested_backend == "unsloth" and has_cuda and has_unsloth:
        return {
            "resolved_backend": "unsloth",
            "status": "ready",
            "reason": "CUDA is available and Unsloth is installed, so use the faster preferred path.",
        }

    if requested_backend == "unsloth":
        return {
            "resolved_backend": "huggingface",
            "status": "fallback",
            "reason": "Unsloth was requested but the local environment is missing CUDA or the package, so falling back to Hugging Face.",
        }

    if requested_backend == "huggingface":
        return {
            "resolved_backend": "huggingface",
            "status": "ready",
            "reason": "Hugging Face was explicitly requested.",
        }

    if has_cuda and has_unsloth:
        return {
            "resolved_backend": "unsloth",
            "status": "ready",
            "reason": "Auto-selected Unsloth because CUDA and the package are available.",
        }

    return {
        "resolved_backend": "huggingface",
        "status": "ready",
        "reason": "Auto-selected Hugging Face as the general fallback path.",
    }


def build_experiment_plan(
    model: str,
    dataset: str,
    task_family: str = "llm",
    requested_backend: str = "auto",
    budget: str = "balanced",
    max_runs: int = 3,
    dataset_format: str = "instruction",
    has_cuda: bool = False,
    has_unsloth: bool = False,
    # Conditional planning inputs — all optional for backward compatibility
    dataset_rows: int | None = None,
    baseline_metric: float | None = None,
    available_vram_gb: float | None = None,
) -> dict[str, Any]:
    """Return a concrete, approval-ready experiment plan.

    When ``dataset_rows``, ``baseline_metric``, or ``available_vram_gb`` are
    provided the plan adapts its hyperparameters rather than using fixed profiles.
    """
    backend = recommend_backend(
        task_family=task_family,
        requested_backend=requested_backend,
        has_cuda=has_cuda,
        has_unsloth=has_unsloth,
    )
    profile = {
        "quick": {
            "max_steps": 100,
            "batch_size": 2 if has_cuda else 1,
            "gradient_accumulation_steps": 8 if has_cuda else 16,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_seq_length": 1024,
        },
        "balanced": {
            "max_steps": 200,
            "batch_size": 4 if has_cuda else 1,
            "gradient_accumulation_steps": 4 if has_cuda else 16,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_seq_length": 2048 if has_cuda else 1024,
        },
        "thorough": {
            "max_steps": 400,
            "batch_size": 4 if has_cuda else 1,
            "gradient_accumulation_steps": 8 if has_cuda else 16,
            "learning_rate": 1e-4,
            "lora_r": 32,
            "max_seq_length": 2048 if has_cuda else 1024,
        },
    }.get(budget, None)
    if profile is None:
        raise ValueError(f"Unsupported budget profile: {budget}")

    plan_notes: list[str] = []

    # Adapt max_steps to dataset size
    small_dataset = False
    if dataset_rows is not None:
        if dataset_rows < 500:
            profile["max_steps"] = max(50, profile["max_steps"] // 2)
            small_dataset = True
            plan_notes.append(
                f"Dataset is small ({dataset_rows} rows): halved max_steps to {profile['max_steps']} "
                "to avoid severe overfitting."
            )
        elif dataset_rows > 50000:
            profile["max_steps"] = int(profile["max_steps"] * 1.5)
            plan_notes.append(
                f"Dataset is large ({dataset_rows} rows): increased max_steps to {profile['max_steps']}."
            )

    # Constrain batch_size to available VRAM
    if available_vram_gb is not None:
        # Rough model size from name heuristics — default to 7B if unknown
        guessed_b = 7.0
        for token in model.lower().split("/")[-1].split("-"):
            if token.endswith("b") and token[:-1].replace(".", "").isdigit():
                guessed_b = float(token[:-1])
                break
        safe_vram = available_vram_gb * 0.90
        batch = profile["batch_size"]
        while batch > 1:
            est = estimate_vram(
                model_size_b=guessed_b,
                lora_r=profile["lora_r"],
                batch_size=batch,
                seq_len=profile["max_seq_length"],
                load_in_4bit=True,
            )
            if est["total_gb"] <= safe_vram:
                break
            batch = max(1, batch // 2)
        if batch != profile["batch_size"]:
            plan_notes.append(
                f"Reduced batch_size from {profile['batch_size']} to {batch} "
                f"to stay within {available_vram_gb:.1f} GB VRAM."
            )
            profile["batch_size"] = batch

    # Preference-format datasets → DPO runs instead of SFT
    if dataset_format == "preference":
        dpo_lr = 5e-5
        runs = [
            {
                "name": "run_001",
                "purpose": "DPO baseline with conservative beta=0.1.",
                "trainer": "dpo",
                "config": {
                    "method": "dpo",
                    "beta": 0.1,
                    "max_steps": profile["max_steps"],
                    "batch_size": max(1, profile["batch_size"] // 2),
                    "learning_rate": dpo_lr,
                    "lora_r": profile["lora_r"],
                    "max_seq_length": min(profile["max_seq_length"], 1024),
                },
            },
            {
                "name": "run_002",
                "purpose": "DPO with lower beta=0.05 for stronger preference signal.",
                "trainer": "dpo",
                "config": {
                    "method": "dpo",
                    "beta": 0.05,
                    "max_steps": profile["max_steps"],
                    "batch_size": max(1, profile["batch_size"] // 2),
                    "learning_rate": dpo_lr,
                    "lora_r": max(profile["lora_r"], 32),
                    "max_seq_length": min(profile["max_seq_length"], 1024),
                },
            },
            {
                "name": "run_003",
                "purpose": "ORPO — combines SFT and preference loss in a single pass.",
                "trainer": "dpo",
                "config": {
                    "method": "orpo",
                    "beta": 0.1,
                    "max_steps": profile["max_steps"],
                    "batch_size": max(1, profile["batch_size"] // 2),
                    "learning_rate": dpo_lr * 2,
                    "lora_r": profile["lora_r"],
                    "max_seq_length": min(profile["max_seq_length"], 1024),
                },
            },
        ]
        plan_notes.append(
            "Preference-format dataset detected. All runs use DPO/ORPO via run_dpo_training — "
            "not SFT. Use /run-dpo to execute approved runs."
        )
    else:
        # Standard SFT runs
        runs = [
            {
                "name": "run_001",
                "purpose": "Baseline fine-tune with conservative defaults.",
                "trainer": "sft",
                "config": dict(profile),
            },
            {
                "name": "run_002",
                "purpose": "Reduce learning rate and increase capacity to test stability vs. quality.",
                "trainer": "sft",
                "config": {
                    **profile,
                    "learning_rate": 1e-4,
                    "lora_r": max(profile["lora_r"], 32),
                },
            },
            {
                "name": "run_003",
                "purpose": "Increase training duration to test whether the dataset still has headroom.",
                "trainer": "sft",
                "config": {
                    **profile,
                    "max_steps": int(profile["max_steps"] * 1.5),
                },
            },
        ]

        if dataset_format == "chat":
            runs[0]["config"]["learning_rate"] = 1e-4
            runs[0]["purpose"] = "Chat-template-aligned starting point with a lower learning rate."

    # For small datasets: cap all runs' max_steps at the adjusted profile value
    # (prevents run_003's 1.5x multiplier from defeating the overfitting guard)
    if small_dataset:
        for run in runs:
            run["config"]["max_steps"] = min(run["config"]["max_steps"], profile["max_steps"])

    # If baseline is already strong, skip the conservative run and propose exploration
    if baseline_metric is not None and baseline_metric > 0.6 and len(runs) > 1:
        plan_notes.append(
            f"Baseline metric is {baseline_metric:.3f} (>0.6): skipping the conservative run, "
            "starting with exploratory configs."
        )
        runs = runs[1:]

    return {
        "task_family": task_family,
        "model": model,
        "dataset": dataset,
        "resolved_backend": backend["resolved_backend"],
        "backend_status": backend["status"],
        "backend_reason": backend["reason"],
        "workflow": [
            "Inspect dataset shape and quality before training.",
            "Run a baseline evaluation on the base model.",
            "Execute approved runs one at a time and save eval artifacts.",
            "Compare runs before deciding whether to continue or ship.",
        ],
        "approval_required": True,
        "runs": runs[: max(1, max_runs)],
        "plan_notes": plan_notes,
    }


def load_run_summaries(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load saved run configs and evaluations from a results directory."""
    root = Path(results_dir)
    if not root.exists():
        return []

    run_summaries: list[dict[str, Any]] = []
    for run_dir in sorted(root.glob("run_*")):
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue

        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        evaluations = []
        for eval_path in sorted(run_dir.glob("eval_*.json")):
            with eval_path.open(encoding="utf-8") as handle:
                evaluations.append(
                    {"path": str(eval_path), "data": json.load(handle)}
                )

        metric_name, metric_value = _extract_primary_metric(evaluations)
        run_summaries.append(
            {
                "run_dir": str(run_dir),
                "config": config,
                "evaluations": evaluations,
                "metric_name": metric_name,
                "primary_metric": metric_value,
            }
        )
    return run_summaries


def _extract_primary_metric(evaluations: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    metric_priority = ("accuracy", "macro_f1", "f1", "score", "loss")
    for evaluation in evaluations:
        data = evaluation["data"]
        for key in metric_priority:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return key, float(-value if key == "loss" else value)
    return None, None


def compare_runs(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return runs ranked by eval metric first, then train loss."""
    def score(summary: dict[str, Any]) -> tuple[float, float]:
        primary_metric = summary.get("primary_metric")
        metric_score = primary_metric if primary_metric is not None else float("-inf")
        train_loss = summary.get("config", {}).get("train_loss")
        loss_score = -float(train_loss) if isinstance(train_loss, (int, float)) else float("-inf")
        return metric_score, loss_score

    return sorted(run_summaries, key=score, reverse=True)


def diagnose_run(run_summary: dict[str, Any] | None, log_text: str = "") -> dict[str, Any]:
    """Generate likely failure modes and next actions from saved signals."""
    config = (run_summary or {}).get("config", {})
    lower_log = log_text.lower()
    issues: list[str] = []
    next_actions: list[str] = []

    if "out of memory" in lower_log or "cuda oom" in lower_log:
        issues.append("Run likely failed due to GPU memory pressure.")
        next_actions.extend(
            [
                "Halve batch size before changing other variables.",
                "Reduce max_seq_length if batch size is already minimal.",
                "Prefer 4-bit loading and gradient checkpointing on the next run.",
            ]
        )

    if "nan" in lower_log or "inf" in lower_log:
        issues.append("Numerical instability detected in the training logs.")
        next_actions.extend(
            [
                "Lower the learning rate before increasing steps.",
                "Inspect the dataset for malformed or extremely long records.",
            ]
        )

    train_loss = config.get("train_loss")
    if isinstance(train_loss, (int, float)) and train_loss > 3.0:
        issues.append("Training loss is still high; the run may be undertrained or mismatched to the data.")
        next_actions.append("Verify dataset formatting and consider more steps only after checking loss stability.")

    metric = (run_summary or {}).get("primary_metric")
    if metric is not None and metric < 0.25:
        issues.append("Evaluation score is very low for a supposedly fine-tuned run.")
        next_actions.extend(
            [
                "Check chat template or prompt formatting alignment.",
                "Confirm the evaluation matches the model's intended task.",
            ]
        )

    if not issues:
        issues.append("No critical issue was inferred from the saved metrics.")
        next_actions.append("Use run comparison to decide whether to iterate or ship.")

    return {"issues": issues, "next_actions": next_actions}
