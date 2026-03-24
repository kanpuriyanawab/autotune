"""Fine-tune a model for the ML R&D workflow plugin."""

import argparse
from functools import partial
from importlib.util import find_spec
import json
import os
from pathlib import Path
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from autotune.workflow import infer_text_field


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--output_dir", default="results/run", help="Output directory for adapter weights")
    parser.add_argument("--backend", default="auto", choices=["auto", "unsloth", "huggingface"])
    parser.add_argument("--task_family", default="llm", choices=["llm", "diffusion"])
    parser.add_argument("--project_path", default=None, help="Optional project root for project-scoped runs")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to use")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument(
        "--dataset_text_field",
        default="auto",
        help="Name of the text field in dataset, or auto to infer it",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument(
        "--report_to",
        default="none",
        choices=["none", "wandb", "mlflow"],
        help="Experiment tracker to report metrics to",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N steps (enables resume)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint directory or 'latest' to resume training",
    )
    return parser.parse_args()


def format_alpaca(example):
    """Format Alpaca-style dataset into a single text field."""
    if example.get("input", "").strip():
        return {"text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )}
    return {"text": (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )}


def format_prompt_completion(example):
    """Format prompt/completion pairs into a single text field."""
    prompt = example.get("prompt") or example.get("instruction") or ""
    completion = example.get("completion") or example.get("output") or ""
    return {"text": f"### Prompt:\n{prompt}\n\n### Response:\n{completion}"}


def format_messages(example, tokenizer=None):
    """Flatten chat records using the tokenizer's chat template when available."""
    messages = example.get("messages") or example.get("conversations") or []
    # Normalize key names: some datasets use "from"/"value" instead of "role"/"content"
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        normalized.append({"role": role, "content": content})

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}
        except Exception:
            pass  # fall through to manual format

    # Fallback: manual format for tokenizers without a chat_template
    chunks = [f"### {m['role'].capitalize()}:\n{m['content']}" for m in normalized if m["content"]]
    return {"text": "\n\n".join(chunks)}


def _load_dataset_ref(dataset_name, split):
    path = Path(dataset_name)
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
        raise ValueError(f"Unsupported local dataset source: {dataset_name}")
    return load_dataset(dataset_name, split=split)


def load_and_prepare_dataset(dataset_name, split, text_field, tokenizer=None):
    """Load dataset and ensure it has a text field."""
    dataset = _load_dataset_ref(dataset_name, split)

    if text_field == "auto":
        text_field = infer_text_field(list(dataset.column_names)) or "text"

    # If dataset has alpaca-style columns, format them
    if "instruction" in dataset.column_names and text_field in {"text", "instruction"} and "text" not in dataset.column_names:
        dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
        return dataset, "text"

    if (
        "prompt" in dataset.column_names
        and ("completion" in dataset.column_names or "output" in dataset.column_names)
        and "text" not in dataset.column_names
    ):
        dataset = dataset.map(format_prompt_completion, remove_columns=dataset.column_names)
        return dataset, "text"

    if (
        ("messages" in dataset.column_names or "conversations" in dataset.column_names)
        and "text" not in dataset.column_names
    ):
        # Pass tokenizer so chat_template is used when available
        dataset = dataset.map(
            partial(format_messages, tokenizer=tokenizer),
            remove_columns=dataset.column_names,
        )
        return dataset, "text"

    if {"prompt", "chosen", "rejected"} <= set(dataset.column_names):
        raise ValueError(
            "Preference-style datasets are not supported by this SFT trainer path. "
            "Use train_dpo.py or the run_dpo_training MCP tool instead."
        )

    return dataset, text_field


def resolve_backend(requested_backend):
    """Prefer Unsloth when available, otherwise use Hugging Face."""
    has_unsloth = find_spec("unsloth") is not None
    if requested_backend == "huggingface":
        return "huggingface"
    if requested_backend == "unsloth":
        return "unsloth" if torch.cuda.is_available() and has_unsloth else "huggingface"
    if torch.cuda.is_available() and has_unsloth:
        return "unsloth"
    return "huggingface"


def load_model_and_tokenizer(args, resolved_backend):
    """Load a model using the preferred backend, with fallback to Hugging Face."""
    if resolved_backend == "unsloth":
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=args.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                random_state=args.seed,
                use_gradient_checkpointing="unsloth",
                max_seq_length=args.max_seq_length,
            )
            return model, tokenizer, "unsloth"
        except Exception as exc:
            print(f"backend_note: unsloth path unavailable, falling back to huggingface ({exc})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, "huggingface"


def main():
    args = parse_args()
    start_time = time.time()

    if args.task_family != "llm":
        raise SystemExit("Only llm task_family is supported today. Diffusion is scaffolded but not implemented.")

    requested_backend = args.backend
    resolved_backend = resolve_backend(requested_backend)

    print(f"model: {args.model}")
    print(f"dataset: {args.dataset}")
    print(f"task_family: {args.task_family}")
    print(f"backend_requested: {requested_backend}")
    print(f"backend_resolved: {resolved_backend}")
    print(f"max_steps: {args.max_steps}")
    print(f"lora_r: {args.lora_r}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_seq_length: {args.max_seq_length}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    print()

    model, tokenizer, resolved_backend = load_model_and_tokenizer(args, resolved_backend)
    model.print_trainable_parameters()

    # Try to apply unsloth-studio patches if available
    try:
        from unsloth_studio.models import patch_llama4
        patch_llama4()
        print("Applied Llama 4 expert layer patches")
    except (ImportError, Exception):
        pass

    # Load dataset — pass tokenizer so chat_template is applied correctly
    dataset, resolved_text_field = load_and_prepare_dataset(
        args.dataset,
        args.dataset_split,
        args.dataset_text_field,
        tokenizer=tokenizer,
    )
    print(f"dataset_size: {len(dataset)}")
    print(f"dataset_text_field: {resolved_text_field}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        dataset_text_field=resolved_text_field,
        max_seq_length=args.max_seq_length,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    elapsed = time.time() - start_time

    # Print metrics
    print()
    print(f"train_loss: {train_result.training_loss:.4f}")
    print(f"train_steps: {train_result.global_step}")
    print(f"train_time_seconds: {elapsed:.1f}")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak_vram_gb: {peak_vram:.2f}")
    print(f"adapter_saved: {args.output_dir}")

    # Save run config
    config = vars(args)
    config["dataset_text_field_resolved"] = resolved_text_field
    config["backend_resolved"] = resolved_backend
    config["train_loss"] = train_result.training_loss
    config["train_time_seconds"] = elapsed
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"config_saved: {config_path}")


if __name__ == "__main__":
    main()
