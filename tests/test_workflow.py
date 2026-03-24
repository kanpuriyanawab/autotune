"""Unit tests for the ML R&D workflow helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from unsloth_mcp.project import ensure_project_layout
from unsloth_mcp.workflow import (
    build_experiment_plan,
    compare_runs,
    detect_dataset_format,
    diagnose_run,
    estimate_vram,
    load_run_summaries,
    recommend_backend,
    summarize_dataset_preview,
    validate_lora_config,
)


class WorkflowTests(unittest.TestCase):
    def test_dataset_summary_detects_instruction_format(self) -> None:
        summary = summarize_dataset_preview(
            dataset_name="demo/alpaca",
            split="train",
            num_rows=500,
            column_names=["instruction", "input", "output"],
            samples=[{"instruction": "Summarize this", "input": "", "output": "Done"}],
        )

        self.assertEqual(summary["format"], "instruction")
        self.assertIn("Dataset is small", " ".join(summary["risks"]))
        self.assertIn("SFT is a sensible starting point", " ".join(summary["recommendations"]))

    def test_dataset_summary_detects_chat_format(self) -> None:
        summary = summarize_dataset_preview(
            dataset_name="demo/chat",
            split="train",
            num_rows=2000,
            column_names=["messages", "source"],
            samples=[
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                }
            ],
        )

        self.assertEqual(summary["format"], "chat")
        self.assertIn("Validate chat template compatibility", " ".join(summary["recommendations"]))

    def test_recommend_backend_prefers_unsloth_when_available(self) -> None:
        recommendation = recommend_backend(
            task_family="llm",
            requested_backend="auto",
            has_cuda=True,
            has_unsloth=True,
        )

        self.assertEqual(recommendation["resolved_backend"], "unsloth")

    def test_build_experiment_plan_requires_approval(self) -> None:
        plan = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="yahma/alpaca-cleaned",
            task_family="llm",
            requested_backend="auto",
            budget="balanced",
            max_runs=2,
            dataset_format="chat",
            has_cuda=True,
            has_unsloth=True,
        )

        self.assertTrue(plan["approval_required"])
        self.assertEqual(len(plan["runs"]), 2)
        self.assertEqual(plan["resolved_backend"], "unsloth")
        self.assertEqual(plan["runs"][0]["config"]["learning_rate"], 1e-4)

    def test_compare_runs_prefers_better_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_one = root / "run_001"
            run_two = root / "run_002"
            run_one.mkdir()
            run_two.mkdir()

            (run_one / "run_config.json").write_text(
                json.dumps({"model": "a", "dataset": "d", "train_loss": 1.2}),
                encoding="utf-8",
            )
            (run_two / "run_config.json").write_text(
                json.dumps({"model": "a", "dataset": "d", "train_loss": 1.4}),
                encoding="utf-8",
            )
            (run_one / "eval_mmlu.json").write_text(
                json.dumps({"accuracy": 0.41}),
                encoding="utf-8",
            )
            (run_two / "eval_mmlu.json").write_text(
                json.dumps({"accuracy": 0.55}),
                encoding="utf-8",
            )

            ranked = compare_runs(load_run_summaries(root))
            self.assertEqual(Path(ranked[0]["run_dir"]).name, "run_002")

    def test_diagnose_run_detects_oom(self) -> None:
        diagnosis = diagnose_run(
            {"config": {"train_loss": 2.4}, "primary_metric": 0.3},
            "CUDA out of memory while allocating tensor",
        )

        self.assertIn("GPU memory pressure", " ".join(diagnosis["issues"]))
        self.assertIn("Halve batch size", " ".join(diagnosis["next_actions"]))

    def test_ensure_project_layout_creates_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = ensure_project_layout(
                Path(tmp_dir) / "demo-project",
                project_name="Demo Project",
                goal="Fine-tune a helpful assistant",
            )

            self.assertTrue((Path(result["project_path"]) / "context" / "project-brief.md").exists())
            self.assertTrue((Path(result["project_path"]) / "reports" / "README.md").exists())


    # --- New tests for estimate_vram ---

    def test_estimate_vram_7b_qlora(self) -> None:
        result = estimate_vram(
            model_size_b=7.0, lora_r=16, batch_size=4, seq_len=2048, load_in_4bit=True
        )
        self.assertIn("total_gb", result)
        self.assertGreater(result["total_gb"], 3.0)
        self.assertLess(result["total_gb"], 25.0)
        self.assertIn("model_weights_gb", result)
        self.assertIn("adapter_gb", result)

    def test_estimate_vram_1b_bf16(self) -> None:
        result = estimate_vram(
            model_size_b=1.0, lora_r=16, batch_size=4, seq_len=2048,
            load_in_4bit=False, precision="bf16"
        )
        self.assertGreater(result["total_gb"], 1.5)
        # 1B model in bf16 should not need more than 12GB for a typical LoRA config
        self.assertLess(result["total_gb"], 12.0)

    def test_estimate_vram_large_model(self) -> None:
        result = estimate_vram(
            model_size_b=70.0, lora_r=16, batch_size=1, seq_len=512, load_in_4bit=True
        )
        # 70B in 4-bit: ~35GB weights alone, so total must exceed that
        self.assertGreater(result["total_gb"], 30.0)

    # --- New tests for validate_lora_config ---

    def test_validate_lora_config_warns_low_alpha(self) -> None:
        warnings = validate_lora_config(lora_r=16, lora_alpha=8)
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(any("below 1.0" in w or "scaling" in w for w in warnings))

    def test_validate_lora_config_warns_non_standard_alpha(self) -> None:
        warnings = validate_lora_config(lora_r=16, lora_alpha=24)
        self.assertTrue(len(warnings) > 0)

    def test_validate_lora_config_optimal_no_warnings(self) -> None:
        warnings = validate_lora_config(lora_r=16, lora_alpha=32)
        self.assertEqual(len(warnings), 0)

    def test_validate_lora_config_r32_alpha64(self) -> None:
        warnings = validate_lora_config(lora_r=32, lora_alpha=64)
        self.assertEqual(len(warnings), 0)

    # --- New tests for conditional experiment planning ---

    def test_detect_dataset_format_preference(self) -> None:
        fmt = detect_dataset_format(["prompt", "chosen", "rejected"])
        self.assertEqual(fmt, "preference")

    def test_build_plan_preference_dataset_uses_dpo(self) -> None:
        plan = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/preference",
            dataset_format="preference",
            has_cuda=True,
            has_unsloth=True,
        )
        self.assertTrue(plan["approval_required"])
        for run in plan["runs"]:
            self.assertEqual(run.get("trainer"), "dpo")
            self.assertIn("method", run["config"])

    def test_build_plan_small_dataset_reduces_steps(self) -> None:
        plan = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/tiny",
            budget="balanced",
            has_cuda=True,
            has_unsloth=True,
            dataset_rows=200,
        )
        for run in plan["runs"]:
            self.assertLessEqual(run["config"]["max_steps"], 100)
        self.assertTrue(any("small" in note.lower() for note in plan.get("plan_notes", [])))

    def test_build_plan_large_dataset_increases_steps(self) -> None:
        plan = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/large",
            budget="balanced",
            has_cuda=True,
            has_unsloth=True,
            dataset_rows=100_000,
        )
        # At least the first run should have more steps than the default 200
        self.assertGreater(plan["runs"][0]["config"]["max_steps"], 200)

    def test_build_plan_strong_baseline_skips_conservative(self) -> None:
        plan_strong = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/data",
            budget="balanced",
            max_runs=3,
            has_cuda=True,
            has_unsloth=True,
            baseline_metric=0.72,
        )
        plan_weak = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/data",
            budget="balanced",
            max_runs=3,
            has_cuda=True,
            has_unsloth=True,
            baseline_metric=0.40,
        )
        # Strong baseline should have fewer runs (skips conservative run_001)
        self.assertLessEqual(len(plan_strong["runs"]), len(plan_weak["runs"]))

    def test_build_plan_backward_compatible_no_new_params(self) -> None:
        """Existing callers without new params must still work unchanged."""
        plan = build_experiment_plan(
            model="unsloth/Llama-3.2-1B",
            dataset="demo/alpaca",
            budget="balanced",
            max_runs=3,
            has_cuda=True,
            has_unsloth=True,
        )
        self.assertTrue(plan["approval_required"])
        self.assertEqual(len(plan["runs"]), 3)


if __name__ == "__main__":
    unittest.main()
