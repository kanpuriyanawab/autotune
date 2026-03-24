"""Export a model to GGUF format for llama.cpp, Ollama, or LM Studio."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter to merge before export (optional)",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for the GGUF file")
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization method",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit (reduces VRAM during export)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter or 'none'}")
    print(f"output_dir: {args.output_dir}")
    print(f"quantization: {args.quantization}")
    print()

    # --- Unsloth native GGUF path (preferred) ---
    # Unsloth's FastLanguageModel.save_pretrained_gguf() handles the full pipeline
    # including merge + quantization natively, with better memory efficiency.
    try:
        from unsloth import FastLanguageModel

        print("Using Unsloth native GGUF export path...")
        model_source = args.adapter if args.adapter else args.model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quantization,
        )
        gguf_files = list(Path(args.output_dir).glob("*.gguf"))
        gguf_path = str(gguf_files[0]) if gguf_files else args.output_dir
        print(f"gguf_path: {gguf_path}")
        print("export_method: unsloth_native")
        print(f"quantization: {args.quantization}")
        return
    except (ImportError, Exception) as exc:
        print(
            f"backend_note: Unsloth GGUF path unavailable ({exc}). "
            "Falling back to HuggingFace merge + llama.cpp convert."
        )

    # --- HuggingFace fallback path ---
    # 1. Merge adapter (if provided) into a full HF model
    # 2. Attempt llama.cpp conversion
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model (HuggingFace path)...")
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

    if args.adapter:
        print(f"Merging adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    merged_dir = os.path.join(args.output_dir, "merged_hf")
    print(f"Saving merged HF model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # Try to run llama.cpp conversion
    gguf_out = os.path.join(args.output_dir, f"model-{args.quantization}.gguf")
    convert_candidates = [
        "convert_hf_to_gguf.py",          # llama.cpp standard name
        "convert-hf-to-gguf.py",           # alternate naming
    ]

    converted = False
    for script_name in convert_candidates:
        result = subprocess.run(
            [sys.executable, script_name, merged_dir,
             "--outfile", gguf_out, "--outtype", args.quantization],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"gguf_path: {gguf_out}")
            print("export_method: hf_merge_and_llama_cpp_convert")
            converted = True
            break

    if not converted:
        print("export_status: merged_hf_only")
        print(f"merged_path: {merged_dir}")
        print(
            "Note: llama.cpp convert script was not found on PATH. "
            "To complete GGUF export:\n"
            "  1. Install llama.cpp: https://github.com/ggerganov/llama.cpp\n"
            f"  2. Run: python convert_hf_to_gguf.py {merged_dir} "
            f"--outfile {gguf_out} --outtype {args.quantization}\n"
            "  3. Or use Unsloth for a single-step export: pip install unsloth"
        )

    # Save export record
    record = {
        "base_model": args.model,
        "adapter": args.adapter,
        "output_dir": args.output_dir,
        "quantization": args.quantization,
        "export_method": "unsloth_native" if "gguf_path" in locals() else (
            "hf_merge_and_llama_cpp_convert" if converted else "merged_hf_only"
        ),
    }
    with open(os.path.join(args.output_dir, "export_info.json"), "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
