---
name: init-project
description: Create a new ML R&D project with context files, reports, and results directories.
user_invocable: true
---

# /init-project — Start a Project

Create the standard project structure before any heavy experimentation.

## Steps

1. Extract the project name, goal, task family, and success criteria from the user request.
2. Call `init_project` with those values.
3. Call `suggest_backends` to confirm the likely runtime path.
4. Report:
   - project path
   - preferred backend
   - next commands to run (`/dataset-audit`, `/baseline`, `/plan-experiments`)

## Defaults

- `task_family`: `llm`
- `preferred_backend`: `unsloth`
- `success_criteria`: ship a reproducible improvement over baseline
