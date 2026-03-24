"""DPO / ORPO preference-optimization training for the ML R&D workflow plugin."""

import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="DPO/ORPO preference-optimization training")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--output_dir", default="results/run", help="Output directory for adapter")
    parser.add_argument(
        "--method",
        default="dpo",
        choices=["dpo", "orpo"],
        help="Preference optimization method",
    )
    parser.add_argument("--backend", default="auto", choices=["auto", "unsloth", "huggingface"])
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta / ORPO lambda")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report_to",
        default="none",
        choices=["none", "wandb", "mlflow"],
        help="Experiment tracker",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint directory or 'latest'",
    )
    parser.add_argument("--dataset_split", default="train")
    return parser.parse_args()


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


def load_preference_dataset(dataset_name, split):
    """Load and validate a preference dataset with prompt/chosen/rejected columns."""
    dataset = _load_dataset_ref(dataset_name, split)
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Preference dataset must have columns: {required}. "
            f"Missing: {missing}. Found: {set(dataset.column_names)}"
        )
    return dataset


def resolve_backend(requested_backend):
    has_unsloth = find_spec("unsloth") is not None
    if requested_backend == "huggingface":
        return "huggingface"
    if requested_backend == "unsloth":
        return "unsloth" if torch.cuda.is_available() and has_unsloth else "huggingface"
    if torch.cuda.is_available() and has_unsloth:
        return "unsloth"
    return "huggingface"


def load_model_and_tokenizer(args, resolved_backend):
    """Load model with LoRA adapters. Tries Unsloth first, falls back to HF+PEFT."""
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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
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

    resolved_backend = resolve_backend(args.backend)

    print(f"model: {args.model}")
    print(f"dataset: {args.dataset}")
    print(f"method: {args.method}")
    print(f"backend_requested: {args.backend}")
    print(f"backend_resolved: {resolved_backend}")
    print(f"beta: {args.beta}")
    print(f"max_steps: {args.max_steps}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"lora_r: {args.lora_r}")
    print(f"batch_size: {args.batch_size}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    print()

    model, tokenizer, resolved_backend = load_model_and_tokenizer(args, resolved_backend)
    model.print_trainable_parameters()

    dataset = load_preference_dataset(args.dataset, args.dataset_split)
    print(f"dataset_size: {len(dataset)}")

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
        remove_unused_columns=False,
    )

    if args.method == "orpo":
        try:
            from trl import ORPOTrainer, ORPOConfig
            orpo_config = ORPOConfig(
                **{k: v for k, v in vars(training_args).items()
                   if k in ORPOConfig.__dataclass_fields__},
                beta=args.beta,
                max_length=args.max_seq_length,
                max_prompt_length=args.max_seq_length // 2,
            )
            trainer = ORPOTrainer(
                model=model,
                args=orpo_config,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
        except ImportError:
            print("ORPOTrainer not available in this trl version, falling back to DPO")
            args.method = "dpo"

    if args.method == "dpo":
        from trl import DPOTrainer, DPOConfig
        dpo_config = DPOConfig(
            **{k: v for k, v in vars(training_args).items()
               if k in DPOConfig.__dataclass_fields__},
            beta=args.beta,
            max_length=args.max_seq_length,
            max_prompt_length=args.max_seq_length // 2,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # PEFT model — reference handled internally
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    elapsed = time.time() - start_time

    print()
    print(f"train_loss: {train_result.training_loss:.4f}")
    print(f"train_steps: {train_result.global_step}")
    print(f"train_time_seconds: {elapsed:.1f}")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak_vram_gb: {peak_vram:.2f}")
    print(f"adapter_saved: {args.output_dir}")

    config = vars(args)
    config["backend_resolved"] = resolved_backend
    config["train_loss"] = train_result.training_loss
    config["train_time_seconds"] = elapsed
    config["trainer"] = args.method
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"config_saved: {config_path}")


if __name__ == "__main__":
    main()
