"""Gemma 4 Multimodal Token Collator for Text-Only SFT.

Gemma 4 is natively multimodal and requires `mm_token_type_ids` to distinguish
text tokens (0) from image tokens (1). When fine-tuning Gemma 4 on text-only datasets,
standard Hugging Face TRL collators fail because they omit `mm_token_type_ids`, causing
runtime shape/tensor mismatch errors inside the language-vision attention blocks.

This collator wraps tokenized conversation batches, handles dynamic padding,
masks loss for padding tokens (-100), and injects all-zero `mm_token_type_ids`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import torch


@dataclass
class Gemma4TextDataCollator:
    """Data collator for Gemma 4 text-only fine-tuning.

    Args:
        tokenizer: Hugging Face PreTrainedTokenizer or Processor tokenizer.
        pad_to_multiple_of: Optional integer multiple for sequence padding (e.g. 8 for Tensor Cores).
    """

    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, List[Any]] = {}

        # Collect all fields from input examples
        for key in examples[0].keys():
            batch[key] = [ex[key] for ex in examples]

        max_len = max(len(ex) for ex in batch["input_ids"])

        # Align to multiple for optimal Tensor Core compute if requested
        if self.pad_to_multiple_of > 0 and (max_len % self.pad_to_multiple_of != 0):
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids = []
        attention_mask = []
        labels = []
        mm_token_type_ids = []

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for i in range(len(examples)):
            seq_len = len(batch["input_ids"][i])
            pad_len = max_len - seq_len

            # Pad input_ids
            input_ids.append(batch["input_ids"][i] + [pad_id] * pad_len)

            # Pad attention_mask
            if "attention_mask" in batch:
                attention_mask.append(batch["attention_mask"][i] + [0] * pad_len)
            else:
                attention_mask.append([1] * seq_len + [0] * pad_len)

            # Pad labels (use -100 for padding tokens to ignore in cross-entropy loss)
            if "labels" in batch:
                labels.append(batch["labels"][i] + [-100] * pad_len)
            else:
                labels.append(batch["input_ids"][i] + [-100] * pad_len)

            # mm_token_type_ids: all zeros for text tokens in Gemma 4
            mm_token_type_ids.append([0] * max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(mm_token_type_ids, dtype=torch.long),
        }
