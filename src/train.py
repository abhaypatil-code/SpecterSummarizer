"""Fine-tune t5-small for legal summarization.

Defaults are kept light so the pipeline runs on CPU for a demo. Increase
--epochs / --max_samples (and use a GPU) for a stronger model.
"""
import argparse

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from src.utils import load_jsonl

MAX_INPUT, MAX_TARGET = 1024, 256


def _to_dataset(path: str, tokenizer, max_samples: int | None) -> Dataset:
    rows = [r for r in load_jsonl(path) if r.get("summary", "").strip()]
    if max_samples:
        rows = rows[:max_samples]
    model_inputs = tokenizer(
        ["summarize: " + r["judgment_text"] for r in rows],
        max_length=MAX_INPUT,
        truncation=True,
    )
    labels = tokenizer(
        text_target=[r["summary"] for r in rows],
        max_length=MAX_TARGET,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)


def train(args):
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    train_ds = _to_dataset(args.train_file, tokenizer, args.max_samples)
    val_ds = _to_dataset(args.val_file, tokenizer, args.max_samples)
    print(f"Training on {len(train_ds)} / validating on {len(val_ds)} examples")

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET,
        fp16=use_fp16,
        logging_steps=20,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune t5-small for summarization.")
    p.add_argument("--train_file", default="data/train_processed.jsonl")
    p.add_argument("--val_file", default="data/val_processed.jsonl")
    p.add_argument("--output_dir", default="models/t5_summarizer")
    p.add_argument("--model_name", default="t5-small")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Cap examples per split for a fast CPU demo run (use 0 for all).",
    )
    args = p.parse_args()
    if args.max_samples == 0:
        args.max_samples = None
    train(args)
