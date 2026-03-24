"""Project scaffolding helpers for the ML R&D workflow."""

from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    """Return a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "ml-rd-project"


def _brief_template(
    project_name: str,
    goal: str,
    task_family: str,
    success_criteria: str,
) -> str:
    return f"""# {project_name}

## Goal
{goal}

## Task Family
{task_family}

## Success Criteria
{success_criteria}

## Users
- Applied ML engineers building production-facing models

## Deliverable
- A reproducible improvement over the current baseline with a clear ship/no-ship decision
"""


def _constraints_template(preferred_backend: str) -> str:
    return f"""# Constraints

## Hardware
- GPU:
- VRAM:

## Budget
- Max runs:
- Max training time per run:

## Quality Bar
- Primary metric:
- Minimum acceptable improvement:

## Backend Preferences
- Preferred backend: {preferred_backend}
- Allowed fallback: huggingface

## Non-Negotiables
- Preserve reproducibility
- Record every run config and evaluation artifact
"""


def _datasets_template() -> str:
    return """# Datasets

## Primary Dataset
- Name:
- Source:
- Split strategy:
- Format:

## Quality Risks
- Duplicates:
- Formatting mismatch:
- Low-quality samples:

## Notes
- Keep chat template and dataset formatting decisions explicit.
"""


def _report_template(project_name: str) -> str:
    return f"""# {project_name} Reports

This directory stores:
- baseline evaluations
- experiment plans
- comparison summaries
- ship recommendations
"""


def ensure_project_layout(
    project_root: str | Path,
    project_name: str,
    goal: str,
    task_family: str = "llm",
    preferred_backend: str = "unsloth",
    success_criteria: str = "Ship a reproducible improvement over baseline.",
) -> dict:
    """Create the default project layout if it does not already exist."""
    root = Path(project_root)
    context_dir = root / "context"
    reports_dir = root / "reports"
    results_dir = root / "results"

    created_paths: list[str] = []
    for directory in (root, context_dir, reports_dir, results_dir):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_paths.append(str(directory))

    files = {
        context_dir / "project-brief.md": _brief_template(
            project_name, goal, task_family, success_criteria
        ),
        context_dir / "constraints.md": _constraints_template(preferred_backend),
        context_dir / "datasets.md": _datasets_template(),
        reports_dir / "README.md": _report_template(project_name),
    }

    written_files: list[str] = []
    for path, content in files.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            written_files.append(str(path))

    return {
        "project_path": str(root),
        "created_paths": created_paths,
        "written_files": written_files,
    }


def load_project_context(project_root: str | Path) -> dict[str, str]:
    """Load known context files if they exist."""
    root = Path(project_root)
    context_dir = root / "context"
    project_context: dict[str, str] = {}

    for file_name in ("project-brief.md", "constraints.md", "datasets.md"):
        path = context_dir / file_name
        if path.exists():
            project_context[file_name] = path.read_text(encoding="utf-8").strip()

    return project_context
