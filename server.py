"""MCP server for ML R&D workflows in Claude Code."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
import json
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from mcp.server.fastmcp import FastMCP
import torch

from autotune.project import ensure_project_layout, load_project_context, slugify
from autotune.workflow import (
    build_experiment_plan,
    compare_runs,
    diagnose_run,
    estimate_vram as _estimate_vram,
    load_run_summaries,
    recommend_backend,
    summarize_dataset_preview,
    validate_lora_config,
)


mcp = FastMCP("autotune")

ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
SCRIPTS_DIR = ROOT_DIR / "scripts"


def _log(msg: str) -> None:
    """Log to stderr because stdout is the JSON-RPC channel."""
    print(msg, file=sys.stderr, flush=True)


def _has_unsloth() -> bool:
    return find_spec("unsloth") is not None


def _results_dir(project_path: Optional[str] = None) -> Path:
    if project_path:
        return Path(project_path) / "results"
    return RESULTS_DIR


def _next_run_dir(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(results_dir.glob("run_*"))
    next_num = 1
    for directory in existing:
        try:
            next_num = max(next_num, int(directory.name.split("_", 1)[1]) + 1)
        except (ValueError, IndexError):
            continue
    return results_dir / f"run_{next_num:03d}"


def _resolve_eval_output(
    adapter: Optional[str],
    benchmark: str,
    output_file: Optional[str],
    project_path: Optional[str],
) -> Optional[str]:
    if output_file:
        return output_file
    if adapter and Path(adapter).exists():
        return str(Path(adapter) / f"eval_{benchmark}.json")
    if project_path:
        reports_dir = Path(project_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return str(reports_dir / f"baseline_{benchmark}.json")
    return None


def _load_dataset_ref(dataset_ref: str, split: str):
    path = Path(dataset_ref)
    if path.exists():
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                return load_dataset("json", data_files=str(path), split="train")
            if suffix == ".csv":
                return load_dataset("csv", data_files=str(path), split="train")
            if suffix == ".parquet":
                return load_dataset("parquet", data_files=str(path), split="train")
            if suffix == ".txt":
                return load_dataset("text", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset source: {dataset_ref}")
    return load_dataset(dataset_ref, split=split)


async def run_script(
    script: str,
    args: dict,
    timeout: int = 1800,
    log_dir: Optional[Path] = None,
) -> tuple[int, str, str]:
    """Run a script as a subprocess with CLI flags built from args."""
    cmd = ["uv", "run", str(SCRIPTS_DIR / script)]
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    _log(f"Running: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ROOT_DIR),
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return -1, "", f"Script timed out after {timeout}s"

    stdout_text = stdout.decode()
    stderr_text = stderr.decode()

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
        (log_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")

    return process.returncode, stdout_text, stderr_text


def _format_dataset_summary(summary: dict) -> str:
    lines = [
        f"# Dataset Audit: {summary['dataset']}",
        "",
        f"- Split: `{summary['split']}`",
        f"- Rows: `{summary['num_rows']}`",
        f"- Format: `{summary['format']}`",
        f"- Suggested text field: `{summary['suggested_text_field']}`",
        f"- Columns: {', '.join(summary['column_names']) or 'none'}",
        "",
        "## Risks",
    ]
    if summary["risks"]:
        lines.extend(f"- {risk}" for risk in summary["risks"])
    else:
        lines.append("- No immediate quality risks detected from the preview.")
    lines.extend(["", "## Recommendations"])
    lines.extend(f"- {item}" for item in summary["recommendations"])
    return "\n".join(lines)


def _format_plan(plan: dict) -> str:
    lines = [
        "# Experiment Plan",
        "",
        f"- Model: `{plan['model']}`",
        f"- Dataset: `{plan['dataset']}`",
        f"- Task family: `{plan['task_family']}`",
        f"- Backend: `{plan['resolved_backend']}` ({plan['backend_reason']})",
        f"- Approval required: `{plan['approval_required']}`",
    ]
    if plan.get("plan_notes"):
        lines.extend(["", "## Planning Notes"])
        lines.extend(f"- {note}" for note in plan["plan_notes"])
    lines.extend(["", "## Workflow"])
    lines.extend(f"- {step}" for step in plan["workflow"])
    lines.extend(["", "## Proposed Runs"])
    for run in plan["runs"]:
        trainer_tag = f" [{run.get('trainer', 'sft').upper()}]" if run.get("trainer") else ""
        config_bits = ", ".join(f"{key}={value}" for key, value in run["config"].items())
        lines.append(f"- `{run['name']}`{trainer_tag}: {run['purpose']} `{config_bits}`")
    return "\n".join(lines)


def _format_compare(ranked_runs: list[dict]) -> str:
    lines = ["# Run Comparison", ""]
    if not ranked_runs:
        return "# Run Comparison\n\nNo completed runs were found."
    for index, summary in enumerate(ranked_runs, start=1):
        config = summary["config"]
        metric_name = summary.get("metric_name") or "none"
        metric_value = summary.get("primary_metric")
        metric_text = "n/a" if metric_value is None else f"{metric_value:.4f}"
        lines.extend(
            [
                f"## {index}. {Path(summary['run_dir']).name}",
                f"- Model: `{config.get('model', '?')}`",
                f"- Dataset: `{config.get('dataset', '?')}`",
                f"- Backend: `{config.get('backend_resolved', config.get('backend', 'huggingface'))}`",
                f"- Primary metric: `{metric_name}` = `{metric_text}`",
                f"- Train loss: `{config.get('train_loss', 'n/a')}`",
                f"- Run path: `{summary['run_dir']}`",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _format_diagnosis(run_dir: Optional[str], diagnosis: dict) -> str:
    lines = ["# Run Diagnosis", ""]
    if run_dir:
        lines.append(f"- Run: `{run_dir}`")
        lines.append("")
    lines.append("## Likely Issues")
    lines.extend(f"- {issue}" for issue in diagnosis["issues"])
    lines.extend(["", "## Next Actions"])
    lines.extend(f"- {action}" for action in diagnosis["next_actions"])
    return "\n".join(lines)


@mcp.tool()
async def check_gpu() -> str:
    """Report local GPU status and basic capability information."""
    if not torch.cuda.is_available():
        return "# GPU Status\n\nNo CUDA GPU detected."

    device_count = torch.cuda.device_count()
    lines = ["# GPU Status", ""]
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        lines.extend(
            [
                f"## GPU {index}",
                f"- Name: `{props.name}`",
                f"- Total VRAM: `{total_gb:.1f} GB`",
                f"- BF16 supported: `{torch.cuda.is_bf16_supported()}`",
                "",
            ]
        )
    return "\n".join(lines).strip()


@mcp.tool()
async def init_project(
    project_name: str,
    goal: str,
    project_path: Optional[str] = None,
    task_family: str = "llm",
    preferred_backend: str = "unsloth",
    success_criteria: str = "Ship a reproducible improvement over baseline.",
) -> str:
    """Create the default ML R&D project layout and context files."""
    resolved_path = Path(project_path) if project_path else ROOT_DIR / slugify(project_name)
    result = ensure_project_layout(
        resolved_path,
        project_name=project_name,
        goal=goal,
        task_family=task_family,
        preferred_backend=preferred_backend,
        success_criteria=success_criteria,
    )
    return (
        "# Project Initialized\n\n"
        f"- Project path: `{result['project_path']}`\n"
        f"- Created paths: `{len(result['created_paths'])}`\n"
        f"- Written files: `{len(result['written_files'])}`"
    )


@mcp.tool()
async def inspect_dataset(
    dataset: str,
    split: str = "train",
    max_examples: int = 3,
) -> str:
    """Inspect dataset shape, infer format, and flag data quality risks."""
    dataset_obj = _load_dataset_ref(dataset, split)
    sample_count = min(max_examples, len(dataset_obj))
    samples = [dataset_obj[index] for index in range(sample_count)]
    summary = summarize_dataset_preview(
        dataset_name=dataset,
        split=split,
        num_rows=len(dataset_obj),
        column_names=list(dataset_obj.column_names),
        samples=samples,
    )
    return _format_dataset_summary(summary)


@mcp.tool()
async def suggest_backends(
    task_family: str = "llm",
    preferred_backend: str = "auto",
) -> str:
    """Recommend the execution backend for the current environment."""
    recommendation = recommend_backend(
        task_family=task_family,
        requested_backend=preferred_backend,
        has_cuda=torch.cuda.is_available(),
        has_unsloth=_has_unsloth(),
    )
    return (
        "# Backend Recommendation\n\n"
        f"- Resolved backend: `{recommendation['resolved_backend']}`\n"
        f"- Status: `{recommendation['status']}`\n"
        f"- Reason: {recommendation['reason']}"
    )


@mcp.tool()
async def plan_experiments(
    model: str,
    dataset: str,
    project_path: Optional[str] = None,
    task_family: str = "llm",
    budget: str = "balanced",
    max_runs: int = 3,
    preferred_backend: str = "auto",
    dataset_format: Optional[str] = None,
) -> str:
    """Generate an approval-ready experiment plan with concrete run configs."""
    resolved_format = dataset_format or "instruction"
    dataset_rows: Optional[int] = None
    try:
        dataset_obj = _load_dataset_ref(dataset, "train")
        dataset_rows = len(dataset_obj)
        preview = summarize_dataset_preview(
            dataset_name=dataset,
            split="train",
            num_rows=dataset_rows,
            column_names=list(dataset_obj.column_names),
            samples=[dataset_obj[index] for index in range(min(3, dataset_rows))],
        )
        resolved_format = preview["format"]
    except Exception as exc:
        _log(f"Dataset preview failed during planning: {exc}")

    # Get available VRAM for conditional planning
    available_vram_gb: Optional[float] = None
    if torch.cuda.is_available():
        try:
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    project_context = load_project_context(project_path) if project_path else {}
    plan = build_experiment_plan(
        model=model,
        dataset=dataset,
        task_family=task_family,
        requested_backend=preferred_backend,
        budget=budget,
        max_runs=max_runs,
        dataset_format=resolved_format,
        has_cuda=torch.cuda.is_available(),
        has_unsloth=_has_unsloth(),
        dataset_rows=dataset_rows,
        available_vram_gb=available_vram_gb,
    )
    response = _format_plan(plan)
    if project_context:
        response += "\n\n## Loaded Context\n"
        for key in sorted(project_context):
            response += f"\n- `{key}` loaded"
    return response


@mcp.tool()
async def run_training(
    model: str,
    dataset: str,
    project_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_family: str = "llm",
    backend: str = "auto",
    max_steps: int = 200,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    load_in_4bit: bool = True,
    gradient_accumulation_steps: int = 4,
    seed: int = 3407,
    report_to: str = "none",
    save_steps: int = 50,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run one approved SFT training experiment."""
    if task_family != "llm":
        return "Training is only implemented for llm task_family today. Diffusion remains scaffolded."

    # Warn about suboptimal LoRA config before spending GPU budget
    lora_warnings = validate_lora_config(lora_r, lora_alpha)

    results_dir = _results_dir(project_path)
    run_dir = Path(output_dir) if output_dir else _next_run_dir(results_dir)
    recommendation = recommend_backend(
        task_family=task_family,
        requested_backend=backend,
        has_cuda=torch.cuda.is_available(),
        has_unsloth=_has_unsloth(),
    )
    script_args = {
        "model": model,
        "dataset": dataset,
        "project_path": project_path,
        "output_dir": str(run_dir),
        "task_family": task_family,
        "backend": recommendation["resolved_backend"],
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "load_in_4bit": load_in_4bit,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "seed": seed,
        "report_to": report_to,
        "save_steps": save_steps,
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    returncode, stdout, stderr = await run_script(
        "train_model.py",
        script_args,
        log_dir=run_dir,
    )
    if returncode != 0:
        return (
            f"Training failed (exit {returncode}).\n\n"
            f"Run dir: {run_dir}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
    response = f"Training complete.\n\nRun dir: {run_dir}\n\n{stdout}"
    if lora_warnings:
        response = "**LoRA config warnings:**\n" + "\n".join(f"- {w}" for w in lora_warnings) + "\n\n" + response
    return response


@mcp.tool()
async def train_model(
    model: str,
    dataset: str,
    output_dir: Optional[str] = None,
    max_steps: int = 200,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    load_in_4bit: bool = True,
    gradient_accumulation_steps: int = 4,
    seed: int = 3407,
) -> str:
    """Deprecated: use run_training instead. This alias will be removed in v1.0."""
    return await run_training(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
    )


@mcp.tool()
async def run_evaluation(
    model: str,
    adapter: Optional[str] = None,
    project_path: Optional[str] = None,
    task_family: str = "llm",
    backend: str = "auto",
    benchmark: str = "mmlu",
    num_samples: Optional[int] = None,
    batch_size: int = 8,
    load_in_4bit: bool = True,
    output_file: Optional[str] = None,
    eval_dataset: Optional[str] = None,
    eval_split: str = "test",
) -> str:
    """Run an evaluation for a base model or adapter.

    Use ``benchmark="mmlu"`` for standard knowledge evaluation, or pass
    ``eval_dataset`` (any HuggingFace dataset or local file) to evaluate on
    your own task-specific data — this automatically sets ``benchmark="custom"``.
    """
    if task_family != "llm":
        return "Evaluation is only implemented for llm task_family today. Diffusion remains scaffolded."

    # Custom eval takes priority over benchmark selection
    resolved_benchmark = "custom" if eval_dataset else benchmark

    recommendation = recommend_backend(
        task_family=task_family,
        requested_backend=backend,
        has_cuda=torch.cuda.is_available(),
        has_unsloth=_has_unsloth(),
    )
    resolved_output = _resolve_eval_output(adapter, resolved_benchmark, output_file, project_path)
    script_args = {
        "model": model,
        "adapter": adapter,
        "project_path": project_path,
        "task_family": task_family,
        "backend": recommendation["resolved_backend"],
        "benchmark": resolved_benchmark,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "load_in_4bit": load_in_4bit,
        "output_file": resolved_output,
        "eval_dataset": eval_dataset,
        "eval_split": eval_split,
    }
    log_dir = Path(adapter) if adapter and Path(adapter).exists() else None
    returncode, stdout, stderr = await run_script(
        "evaluate_model.py",
        script_args,
        log_dir=log_dir,
    )
    if returncode != 0:
        return f"Evaluation failed (exit {returncode}).\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    return f"Evaluation complete.\n\n{stdout}"


@mcp.tool()
async def evaluate_model(
    model: str,
    adapter: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 8,
    load_in_4bit: bool = True,
    output_file: Optional[str] = None,
) -> str:
    """Deprecated: use run_evaluation instead. This alias will be removed in v1.0."""
    return await run_evaluation(
        model=model,
        adapter=adapter,
        num_samples=num_samples,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        output_file=output_file,
    )


@mcp.tool()
async def compare_experiments(
    project_path: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Rank completed runs using saved evaluation metrics and train loss."""
    run_summaries = load_run_summaries(_results_dir(project_path))
    ranked = compare_runs(run_summaries)[:top_k]
    return _format_compare(ranked)


@mcp.tool()
async def list_experiments(project_path: Optional[str] = None) -> str:
    """Deprecated: use compare_experiments instead. This alias will be removed in v1.0."""
    return await compare_experiments(project_path=project_path)


@mcp.tool()
async def diagnose_experiment(
    run_dir: Optional[str] = None,
    project_path: Optional[str] = None,
    log_text: Optional[str] = None,
) -> str:
    """Diagnose a run from logs and saved metrics."""
    resolved_run_dir = Path(run_dir) if run_dir else None
    if resolved_run_dir is None:
        ranked = compare_runs(load_run_summaries(_results_dir(project_path)))
        if ranked:
            resolved_run_dir = Path(ranked[0]["run_dir"])

    summary = None
    combined_log_text = log_text or ""
    if resolved_run_dir and resolved_run_dir.exists():
        run_summaries = load_run_summaries(resolved_run_dir.parent)
        for item in run_summaries:
            if Path(item["run_dir"]) == resolved_run_dir:
                summary = item
                break
        for log_name in ("stdout.log", "stderr.log"):
            log_path = resolved_run_dir / log_name
            if log_path.exists():
                combined_log_text += "\n" + log_path.read_text(encoding="utf-8")

    diagnosis = diagnose_run(summary, combined_log_text)
    return _format_diagnosis(str(resolved_run_dir) if resolved_run_dir else None, diagnosis)


@mcp.tool()
async def ship_decision(
    project_path: Optional[str] = None,
    metric_threshold: Optional[float] = None,
) -> str:
    """Summarize whether the current best run is ready for human review or shipping."""
    run_summaries = compare_runs(load_run_summaries(_results_dir(project_path)))
    if not run_summaries:
        return "# Ship Decision\n\nNo completed runs were found."

    best = run_summaries[0]
    diagnosis = diagnose_run(best, "")
    metric_name = best.get("metric_name") or "primary_metric"
    metric_value = best.get("primary_metric")
    ready = metric_value is not None and (metric_threshold is None or metric_value >= metric_threshold)
    recommendation = "ready for human review" if ready else "keep iterating"

    lines = [
        "# Ship Decision",
        "",
        f"- Best run: `{Path(best['run_dir']).name}`",
        f"- Recommendation: `{recommendation}`",
        f"- Metric: `{metric_name}` = `{metric_value if metric_value is not None else 'n/a'}`",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in diagnosis["issues"])
    lines.extend(["", "## Artifacts"])
    lines.append(f"- Run directory: `{best['run_dir']}`")
    for evaluation in best["evaluations"]:
        lines.append(f"- Eval file: `{evaluation['path']}`")
    return "\n".join(lines)


@mcp.tool()
async def serve_model(
    model: str,
    adapter: Optional[str] = None,
    load_in_4bit: bool = True,
    port: int = 7860,
    share: bool = False,
) -> str:
    """Launch a Gradio chat interface for a model."""
    script_args = {
        "model": model,
        "adapter": adapter,
        "load_in_4bit": load_in_4bit,
        "port": port,
        "share": share,
    }
    cmd = ["uv", "run", str(SCRIPTS_DIR / "serve_model.py")]
    for key, value in script_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    _log(f"Launching server: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ROOT_DIR),
    )
    await asyncio.sleep(3)

    if process.returncode is not None:
        _, stderr = await process.communicate()
        return f"Server failed to start.\n\nSTDERR:\n{stderr.decode()}"

    return (
        f"Gradio server launched (PID {process.pid}).\n"
        f"Local URL: http://localhost:{port}\n"
        "The server is running in the background."
    )


@mcp.tool()
async def estimate_vram(
    model_size_b: float,
    lora_r: int = 16,
    batch_size: int = 4,
    seq_len: int = 2048,
    load_in_4bit: bool = True,
    precision: str = "bf16",
) -> str:
    """Estimate peak VRAM in GB for a LoRA fine-tuning configuration.

    Use this before approving a run to check if the config fits your GPU.
    ``model_size_b`` is the number of parameters in billions (e.g. 7.0 for Llama 3 8B).
    """
    result = _estimate_vram(
        model_size_b=model_size_b,
        lora_r=lora_r,
        batch_size=batch_size,
        seq_len=seq_len,
        load_in_4bit=load_in_4bit,
        precision=precision,
    )
    gpu_vram: Optional[float] = None
    if torch.cuda.is_available():
        try:
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    lines = [
        "# VRAM Estimate",
        "",
        f"- Model weights: `{result['model_weights_gb']} GB`",
        f"- LoRA adapter: `{result['adapter_gb']} GB`",
        f"- Optimizer states: `{result['optimizer_gb']} GB`",
        f"- Gradients: `{result['gradient_gb']} GB`",
        f"- Activations (heuristic): `{result['activation_gb']} GB`",
        f"- **Estimated total: `{result['total_gb']} GB`** (includes 10% overhead)",
        "",
    ]
    if gpu_vram is not None:
        fits = result["total_gb"] <= gpu_vram * 0.9
        status = "fits" if fits else "may OOM"
        lines.append(f"- Available GPU VRAM: `{gpu_vram:.1f} GB` → **{status}**")
    lines.append(
        "\n> This is a heuristic estimate. Actual usage depends on model architecture "
        "and framework overhead. When in doubt, start with load_in_4bit=True and batch_size=1."
    )
    return "\n".join(lines)


@mcp.tool()
async def run_dpo_training(
    model: str,
    dataset: str,
    project_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    method: str = "dpo",
    backend: str = "auto",
    beta: float = 0.1,
    max_steps: int = 200,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    load_in_4bit: bool = True,
    gradient_accumulation_steps: int = 4,
    seed: int = 3407,
    report_to: str = "none",
    save_steps: int = 50,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Run one approved DPO or ORPO preference-optimization training run.

    The dataset must have ``prompt``, ``chosen``, and ``rejected`` columns.
    Use ``method="dpo"`` (default) or ``method="orpo"`` (combines SFT+preference loss).
    """
    results_dir = _results_dir(project_path)
    run_dir = Path(output_dir) if output_dir else _next_run_dir(results_dir)
    recommendation = recommend_backend(
        task_family="llm",
        requested_backend=backend,
        has_cuda=torch.cuda.is_available(),
        has_unsloth=_has_unsloth(),
    )
    script_args = {
        "model": model,
        "dataset": dataset,
        "output_dir": str(run_dir),
        "method": method,
        "backend": recommendation["resolved_backend"],
        "beta": beta,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "load_in_4bit": load_in_4bit,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "seed": seed,
        "report_to": report_to,
        "save_steps": save_steps,
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    returncode, stdout, stderr = await run_script(
        "train_dpo.py",
        script_args,
        log_dir=run_dir,
    )
    if returncode != 0:
        return (
            f"DPO training failed (exit {returncode}).\n\n"
            f"Run dir: {run_dir}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
    return f"DPO training complete.\n\nRun dir: {run_dir}\n\n{stdout}"


@mcp.tool()
async def merge_adapter(
    model: str,
    adapter: str,
    output_dir: Optional[str] = None,
    load_in_4bit: bool = False,
) -> str:
    """Merge a LoRA adapter back into the base model weights.

    Produces a standalone model that can be served without the adapter file,
    or exported to GGUF. Unsloth's merge path is used when available.
    """
    resolved_output = output_dir or str(Path(adapter) / "merged")
    script_args = {
        "model": model,
        "adapter": adapter,
        "output_dir": resolved_output,
        "load_in_4bit": load_in_4bit,
    }
    returncode, stdout, stderr = await run_script(
        "merge_adapter.py",
        script_args,
        timeout=3600,
    )
    if returncode != 0:
        return (
            f"Merge failed (exit {returncode}).\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
    return f"Merge complete.\n\n{stdout}"


@mcp.tool()
async def export_gguf(
    model: str,
    adapter: Optional[str] = None,
    output_dir: Optional[str] = None,
    quantization: str = "q4_k_m",
    load_in_4bit: bool = True,
) -> str:
    """Export a model to GGUF format for llama.cpp, Ollama, or LM Studio.

    If ``adapter`` is provided the adapter is merged before export.
    Unsloth's native GGUF export is used when available (best quality).
    Falls back to HuggingFace merge + llama.cpp conversion.

    Supported quantizations: ``q4_k_m`` (recommended), ``q5_k_m``, ``q8_0``, ``f16``.
    """
    resolved_output = output_dir or str(ROOT_DIR / "exports" / "gguf")
    script_args = {
        "model": model,
        "adapter": adapter,
        "output_dir": resolved_output,
        "quantization": quantization,
        "load_in_4bit": load_in_4bit,
    }
    returncode, stdout, stderr = await run_script(
        "export_gguf.py",
        script_args,
        timeout=3600,
    )
    if returncode != 0:
        return (
            f"GGUF export failed (exit {returncode}).\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
    return f"GGUF export complete.\n\n{stdout}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
