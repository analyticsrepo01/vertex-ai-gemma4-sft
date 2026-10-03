#!/usr/bin/env python3
"""Run test inference with fine-tuned Gemma 4 LoRA adapter."""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser(description="Test Gemma 4 LoRA generation")
    parser.add_argument("--base-model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--adapter-path", type=str, required=True, help="Local or GCS path to adapter checkpoint")
    parser.add_argument("--prompt", type=str, default="What are the main architectural innovations of Gemma 4?")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("GEMMA 4 INFERENCE TEST")
    print("=" * 60)
    print(f"Base Model:   {args.base_model}")
    print(f"Adapter:      {args.adapter_path}")
    print(f"Prompt:       {args.prompt}")

    # Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Quantization config
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # Format prompt using chat template
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Note: add mm_token_type_ids for Gemma 4 if needed
    if "mm_token_type_ids" not in inputs:
        inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    print("\nGenerating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Slice output
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
