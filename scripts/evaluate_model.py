"""Evaluate a model using the ML R&D workflow plugin."""

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--backend", default="auto", choices=["auto", "unsloth", "huggingface"])
    parser.add_argument("--task_family", default="llm", choices=["llm", "diffusion"])
    parser.add_argument("--project_path", default=None, help="Optional project root")
    parser.add_argument(
        "--benchmark",
        default="mmlu",
        choices=["mmlu", "custom"],
        help="Evaluation benchmark. Use 'custom' with --eval_dataset for task-specific eval.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples (None = full benchmark)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--output_file", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--eval_dataset",
        default=None,
        help="HuggingFace dataset name or local file for custom evaluation (required when benchmark=custom)",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        help="Dataset split to use for custom evaluation (default: test)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name, adapter_path=None, load_in_4bit=False):
    """Load base model, optionally with a LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")

    return model, tokenizer


def evaluate_mmlu(model, tokenizer, num_samples=None, batch_size=8):
    """Run MMLU evaluation using unsloth-studio if available, otherwise a basic implementation."""
    try:
        from unsloth_studio.evaluation.mmlu import calculate_mmlu
        results = calculate_mmlu(
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        return results
    except ImportError:
        print("unsloth_studio not found, using basic MMLU evaluation")
        return _basic_mmlu(model, tokenizer, num_samples, batch_size)


def _basic_mmlu(model, tokenizer, num_samples=None, batch_size=8):
    """Fallback MMLU evaluation without unsloth-studio."""
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    choices = ["A", "B", "C", "D"]
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for example in dataset:
            question = example["question"]
            options = example["choices"]
            answer_idx = example["answer"]

            prompt = f"Question: {question}\n"
            for i, opt in enumerate(options):
                prompt += f"{choices[i]}. {opt}\n"
            prompt += "Answer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

            choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
            choice_logits = logits[choice_ids]
            predicted = choice_logits.argmax().item()

            if predicted == answer_idx:
                correct += 1
            total += 1

            if total % 100 == 0:
                print(f"  progress: {total}/{len(dataset)}, accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_custom(model, tokenizer, eval_dataset_ref, eval_split, num_samples=None):
    """Compute loss and perplexity on a user-provided dataset.

    Returns {"loss": float, "perplexity": float, "total": int}.
    This is more informative than MMLU for task-specific fine-tunes.
    """
    from datasets import load_dataset
    from pathlib import Path

    path = Path(eval_dataset_ref)
    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif suffix == ".csv":
            dataset = load_dataset("csv", data_files=str(path), split="train")
        elif suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(path), split="train")
        elif suffix == ".txt":
            dataset = load_dataset("text", data_files=str(path), split="train")
        else:
            raise ValueError(f"Unsupported local file format: {eval_dataset_ref}")
    else:
        dataset = load_dataset(eval_dataset_ref, split=eval_split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Find the text column
    text_col = None
    for candidate in ("text", "content", "prompt", "instruction"):
        if candidate in dataset.column_names:
            text_col = candidate
            break
    if text_col is None:
        text_col = dataset.column_names[0]

    total_loss = 0.0
    total_tokens = 0
    count = 0

    model.eval()
    with torch.inference_mode():
        for example in dataset:
            text = str(example.get(text_col, ""))
            if not text.strip():
                continue
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            n_tokens = labels.shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
            count += 1
            if count % 50 == 0:
                print(f"  progress: {count}/{len(dataset)}, avg_loss: {total_loss / max(total_tokens, 1):.4f}")

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    return {"loss": avg_loss, "perplexity": perplexity, "total": count}


def main():
    args = parse_args()

    if args.task_family != "llm":
        raise SystemExit("Only llm task_family is supported today. Diffusion is scaffolded but not implemented.")

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter or 'none (base model)'}")
    print(f"task_family: {args.task_family}")
    print(f"backend_requested: {args.backend}")
    print(f"benchmark: {args.benchmark}")
    if args.benchmark == "custom":
        print(f"eval_dataset: {args.eval_dataset}")
        print(f"eval_split: {args.eval_split}")
    print()

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.adapter, args.load_in_4bit
    )

    if args.benchmark == "mmlu":
        results = evaluate_mmlu(model, tokenizer, args.num_samples, args.batch_size)
    elif args.benchmark == "custom":
        if not args.eval_dataset:
            raise SystemExit("--eval_dataset is required when --benchmark=custom")
        results = evaluate_custom(
            model, tokenizer, args.eval_dataset, args.eval_split, args.num_samples
        )
    else:
        raise SystemExit(f"Unknown benchmark: {args.benchmark}")

    print()
    print(f"benchmark: {args.benchmark}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({"benchmark": args.benchmark, **results}, f, indent=2)
        print(f"results_saved: {args.output_file}")


if __name__ == "__main__":
    main()
