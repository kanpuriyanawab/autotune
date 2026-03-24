"""Merge a LoRA adapter back into its base model for deployment or GGUF export."""

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--model", required=True, help="Base model name or HuggingFace path")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for the merged model")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit before merging (reduces VRAM, may affect quality)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter}")
    print(f"output_dir: {args.output_dir}")
    print()

    # Try Unsloth path first — better memory efficiency and preserves quantization
    try:
        from unsloth import FastLanguageModel

        print("Using Unsloth merge path...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        model.save_pretrained_merged(
            args.output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"merged_path: {args.output_dir}")
        print("merge_method: unsloth")
        return
    except (ImportError, Exception) as exc:
        print(f"backend_note: Unsloth merge unavailable ({exc}), using HuggingFace merge_and_unload")

    # HuggingFace fallback path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"merged_path: {args.output_dir}")
    print("merge_method: huggingface")

    # Save a record of what was merged
    merge_record = {
        "base_model": args.model,
        "adapter": args.adapter,
        "output_dir": args.output_dir,
        "merge_method": "huggingface",
    }
    with open(os.path.join(args.output_dir, "merge_info.json"), "w") as f:
        json.dump(merge_record, f, indent=2)


if __name__ == "__main__":
    main()
