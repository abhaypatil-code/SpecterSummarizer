import os
import argparse
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import Dataset
from utils import load_jsonl
import torch

def prepare_dataset(file_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """Load and tokenize InLSum processed data for training."""
    data = load_jsonl(file_path)
    
    # Extract fields
    ids = [d['ID'] for d in data]
    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        judgments,
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize summaries (labels)
    labels = tokenizer(
        summaries,
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )
    
    # Create HF Dataset
    dataset = Dataset.from_dict({
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels['input_ids'],
        'ID': ids  # Keep IDs for tracking
    })
    
    return dataset

def train_model(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "t5-small",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1
):
    """
    Train T5 model for legal summarization (InLSum dataset).
    """
    print("\n" + "="*80)
    print(f"TRAINING {model_name.upper()} ON InLSum DATASET")
    print("="*80 + "\n")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare datasets
    print(f"📂 Loading data from: {train_file} and {val_file}")
    train_dataset = prepare_dataset(train_file, tokenizer)
    val_dataset = prepare_dataset(val_file, tokenizer)
    print(f"✅ Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples\n")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, '../logs'),
        logging_steps=50,
        # Corrected Argument Names:
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print(f"🚀 Starting training...")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}\n")
    
    trainer.train()
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training complete! Best model saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model for legal summarization.")
    parser.add_argument("--train_file", type=str, default="data/train_processed.jsonl", help="Path to processed training file.")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl", help="Path to processed validation file.")
    parser.add_argument("--output_dir", type=str, default="outputs/model", help="Directory to save the trained model.")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json", help="Path to hyperparameters JSON file.")
    args = parser.parse_args()

    with open(args.hyperparams) as f:
        hyperparams = json.load(f)

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
        gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 1)
    )