"""Launch a Gradio chat interface for a fine-tuned model."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import gradio as gr
from threading import Thread


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a model via Gradio chat")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
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

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter from {args.adapter}")

    print("Model loaded.")

    def respond(message, history):
        # Build conversation using chat template if available
        if tokenizer.chat_template:
            messages = []
            for user_msg, bot_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
            messages.append({"role": "user", "content": message})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = message

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial = ""
        for token in streamer:
            partial += token
            yield partial

    demo = gr.ChatInterface(
        respond,
        title=f"Unsloth MCP — {args.model}",
        description=f"Adapter: {args.adapter or 'none (base model)'}",
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
