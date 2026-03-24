"""Shared helpers for the autotune ML R&D plugin."""

from .project import ensure_project_layout, load_project_context, slugify
from .workflow import (
    build_experiment_plan,
    compare_runs,
    detect_dataset_format,
    diagnose_run,
    estimate_vram,
    infer_text_field,
    load_run_summaries,
    recommend_backend,
    summarize_dataset_preview,
    validate_lora_config,
)

__all__ = [
    "build_experiment_plan",
    "compare_runs",
    "detect_dataset_format",
    "diagnose_run",
    "ensure_project_layout",
    "estimate_vram",
    "infer_text_field",
    "load_project_context",
    "load_run_summaries",
    "recommend_backend",
    "slugify",
    "summarize_dataset_preview",
    "validate_lora_config",
]
