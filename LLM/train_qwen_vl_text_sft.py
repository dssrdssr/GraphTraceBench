from __future__ import annotations

import argparse
import torch
from datasets import load_dataset
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def parse_args():
    p = argparse.ArgumentParser(description="Text-only LoRA SFT for Qwen2-VL / Qwen3-VL on sft_chat JSONL.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForImageTextToText
    dataset = load_dataset("json", data_files={"train": args.train_file, **({"validation": args.eval_file} if args.eval_file else {})})
    processor = AutoProcessor.from_pretrained(args.model_name)
    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    train_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if args.eval_file else "no",
        save_strategy="steps",
        save_steps=200,
        eval_steps=200,
        logging_steps=10,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
