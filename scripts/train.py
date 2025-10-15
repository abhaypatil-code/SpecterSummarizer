import os
import argparse
import json
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from utils import load_jsonl
import logging

# Set up logging
logger = logging.getLogger(__name__)

def prepare_dataset(
    file_path: str,
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256
) -> Dataset:
    """
    Loads data from a JSONL file, tokenizes it, and converts it into a Hugging Face Dataset object.
    
    Args:
        file_path (str): The path to the input JSONL file.
        tokenizer: The tokenizer to use for processing text.
        max_input_length (int): The maximum length for input sequences.
        max_target_length (int): The maximum length for target (summary) sequences.

    Returns:
        Dataset: A Hugging Face Dataset object ready for training.
    """
    data = list(load_jsonl(file_path))
    
    # --- FIX APPLIED: Filter out empty summaries ---
    # This prevents training on examples that would cause NaN loss.
    original_size = len(data)
    data = [d for d in data if d['summary'] and d['summary'].strip()]
    if len(data) < original_size:
        logger.warning(f"Removed {original_size - len(data)} examples with empty summaries from {file_path}")

    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]

    model_inputs = tokenizer(
        judgments,
        max_length=max_input_length,
        truncation=True
    )
    
    labels = tokenizer(
        summaries,
        max_length=max_target_length,
        truncation=True
    )
    
    model_inputs['labels'] = labels['input_ids']
    return Dataset.from_dict(model_inputs)

def train_model(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    gradient_accumulation_steps: int,
    max_input_length: int,
    max_target_length: int,
    resume_from_checkpoint: str = None
):
    """
    Sets up and runs the training process for the T5 summarization model.
    """
    print("\n" + "="*80)
    print(f"🚀 STARTING TRAINING: Fine-tuning {model_name.upper()}")
    print("="*80 + "\n")

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print(f"📂 Loading and preparing datasets from '{train_file}' and '{val_file}'...")
    train_dataset = prepare_dataset(train_file, tokenizer, max_input_length, max_target_length)
    val_dataset = prepare_dataset(val_file, tokenizer, max_input_length, max_target_length)
    print(f"✅ Datasets prepared: {len(train_dataset)} training examples, {len(val_dataset)} validation examples.\n")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,

        # --- FIX APPLIED: Added generation_max_length ---
        # Ensures validation summaries are not truncated, providing accurate metrics.
        generation_max_length=max_target_length,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    print(f"💪 Training commencing...")
    print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    
    # --- FIX APPLIED: Checkpoint Resume Logic ---
    # Automatically resumes from the last checkpoint if available.
    if resume_from_checkpoint:
        print(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"\n✅ Training complete! Saving the best model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "="*80)
    print("🎉 Model training finished successfully!")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Train a T5 model for legal document summarization.")
    parser.add_argument("--train_file", type=str, default="data/train_processed.jsonl", help="Path to the processed training data file (.jsonl).")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl", help="Path to the processed validation data file (.jsonl).")
    parser.add_argument("--output_dir", type=str, default="outputs/t5_summarizer", help="Directory where the final model and checkpoints will be saved.")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json", help="Path to a JSON file containing hyperparameters.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum token length for input judgments.")
    parser.add_argument("--max_target_length", type=int, default=256, help="Maximum token length for generated summaries.")
    
    # --- FIX APPLIED: Argument for resuming from a specific checkpoint ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume training from.")
    
    args = parser.parse_args()

    # --- FIX APPLIED: Logic to auto-detect the last checkpoint ---
    resume_from_checkpoint = args.resume_from_checkpoint
    if not resume_from_checkpoint and os.path.isdir(args.output_dir):
        last_checkpoint = Seq2SeqTrainingArguments.get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            print(f"✅ Automatically detected last checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint

    try:
        with open(args.hyperparams) as f:
            hyperparams = json.load(f)
        print(f"✅ Successfully loaded hyperparameters from '{args.hyperparams}'.")
    except FileNotFoundError:
        print(f"⚠️ WARNING: Hyperparameter file not found at '{args.hyperparams}'. Using default values.")
        hyperparams = {}
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from '{args.hyperparams}'. Check for syntax errors. Using default values.")
        hyperparams = {}

    train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=hyperparams.get("model_name", "t5-small"),
        epochs=hyperparams.get("num_epochs", 3),
        batch_size=hyperparams.get("batch_size", 4),
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        warmup_steps=hyperparams.get("warmup_steps", 500),
        weight_decay=hyperparams.get("weight_decay", 0.01),
        gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 1),
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        resume_from_checkpoint=resume_from_checkpoint
    )

if __name__ == "__main__":
    main()