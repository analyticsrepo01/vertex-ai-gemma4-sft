"""Gemma 4 Flexible Fine-Tuning Script for Vertex AI.

Supports full BF16 training (for smaller models or H100 80GB clusters)
and QLoRA (for single GPU / memory-constrained configurations).

Environment variables:
    MODEL_ID: Hugging Face model ID (e.g., google/gemma-4-E4B-it)
    AIP_MODEL_DIR: GCS output directory (set by Vertex AI)
    USE_QLORA: Set to '1' to enable QLoRA (default: '0')
    DATASET_GCS_PATH: Optional GCS path to training JSON
    DATASET_NAME: Hugging Face fallback dataset (default: mlabonne/FineTome-100k)
    DATASET_SPLIT: Dataset split (default: train[:5000])
    NUM_EPOCHS: Number of training epochs (default: 1)
    BATCH_SIZE: Per-device batch size (default: 1)
    GRADIENT_ACCUMULATION_STEPS: Gradient accumulation (default: 8)
    LEARNING_RATE: Learning rate (default: 2e-4)
    LORA_R: LoRA rank (default: 16)
    LORA_ALPHA: LoRA alpha (default: 32)
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

try:
    from collator import Gemma4TextDataCollator
except ImportError:
    @dataclass
    class Gemma4TextDataCollator:
        tokenizer: Any
        pad_to_multiple_of: int = 8

        def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch: Dict[str, List[Any]] = {}
            for key in examples[0].keys():
                batch[key] = [ex[key] for ex in examples]
            max_len = max(len(ex) for ex in batch["input_ids"])
            if self.pad_to_multiple_of > 0 and (max_len % self.pad_to_multiple_of != 0):
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

            input_ids, attention_mask, labels, mm_token_type_ids = [], [], [], []
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            for i in range(len(examples)):
                seq_len = len(batch["input_ids"][i])
                pad_len = max_len - seq_len
                input_ids.append(batch["input_ids"][i] + [pad_id] * pad_len)
                if "attention_mask" in batch:
                    attention_mask.append(batch["attention_mask"][i] + [0] * pad_len)
                else:
                    attention_mask.append([1] * seq_len + [0] * pad_len)
                if "labels" in batch:
                    labels.append(batch["labels"][i] + [-100] * pad_len)
                else:
                    labels.append(batch["input_ids"][i] + [-100] * pad_len)
                mm_token_type_ids.append([0] * max_len)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "mm_token_type_ids": torch.tensor(mm_token_type_ids, dtype=torch.long),
            }


def main():
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
    raw_output = os.environ.get("AIP_MODEL_DIR", "./output-dir")
    output_dir = raw_output.replace("gs://", "/gcs/")
    os.makedirs(output_dir, exist_ok=True)

    use_qlora = os.environ.get("USE_QLORA", "0") == "1"
    dataset_gcs_path = os.environ.get("DATASET_GCS_PATH", "")
    dataset_name = os.environ.get("DATASET_NAME", "mlabonne/FineTome-100k")
    dataset_split = os.environ.get("DATASET_SPLIT", "train[:5000]")
    num_epochs = int(os.environ.get("NUM_EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))

    print("=" * 70)
    print("VERTEX AI — GEMMA 4 SFT TRAINING")
    print("=" * 70)
    print(f"Model ID:        {model_id}")
    print(f"QLoRA Enabled:   {use_qlora}")
    print(f"Output:          {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")

    # Load dataset
    if dataset_gcs_path:
        local_path = dataset_gcs_path.replace("gs://", "/gcs/")
        print(f"Loading custom dataset from GCS: {dataset_gcs_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        print(f"Loading dataset: {dataset_name} [{dataset_split}]")
        dataset = load_dataset(dataset_name, split=dataset_split)

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_conversation(example):
        messages = []
        raw_conv = example.get("conversations", example.get("messages", []))
        for turn in raw_conv:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("system",):
                messages.append({"role": "user", "content": f"[System] {content}"})
            elif role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant", "model"):
                messages.append({"role": "model", "content": content})
        if not messages:
            return {"text": ""}
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names, num_proc=4)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"Dataset formatted: {len(dataset)} examples")

    # Load Model
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = Gemma4TextDataCollator(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_cache=False,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            optim="adamw_8bit" if use_qlora else "adamw_torch",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            max_length=2048,
            dataset_text_field="text",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Loss: {train_result.training_loss:.4f}. Saved to {output_dir}")


if __name__ == "__main__":
    main()
