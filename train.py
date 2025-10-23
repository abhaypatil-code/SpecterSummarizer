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
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(
    file_path: str,
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 512
) -> Dataset:
    """
    Loads data from a JSONL file, tokenizes it, and converts it into a Hugging Face Dataset object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = list(load_jsonl(file_path))
    original_size = len(data)
    
    if original_size == 0:
        raise ValueError(f"The dataset at {file_path} contains no data.")
    
    # Filter out entries with empty summaries
    data = [d for d in data if d.get('summary') and d['summary'].strip()]
    
    filtered_count = original_size - len(data)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} entries with empty summaries.")
    
    if not data:
        raise ValueError(f"The dataset at {file_path} is empty after filtering.")
    
    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        judgments,
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            summaries,
            max_length=max_target_length,
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(model_inputs)
    
    logger.info(f"Loaded and tokenized {len(dataset)} examples from {file_path}")
    return dataset

def train_model(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "t5-base",
    learning_rate: float = 3e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    max_input_length: int = 1024,
    max_target_length: int = 512,
    save_total_limit: int = 2,
    early_stopping_patience: int = 3,
    fp16: bool = True
):
    """
    Trains a T5 model for summarization on GPU with memory optimization for 6GB VRAM.
    """
    print("\n" + "="*80)
    print("STARTING T5 SUMMARIZATION TRAINING")
    print("="*80 + "\n")
    
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU
    model = model.to(device)
    print(f"Model loaded and moved to GPU\n")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_file, tokenizer, max_input_length, max_target_length)
    val_dataset = prepare_dataset(val_file, tokenizer, max_input_length, max_target_length)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}\n")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    total_steps = math.ceil(len(train_dataset) / effective_batch_size) * num_epochs
    
    print("Training Configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Per-device batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  FP16 training: {fp16}")
    print(f"  Early stopping patience: {early_stopping_patience}\n")
    
    # Training arguments optimized for 6GB GPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        warmup_steps=warmup_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        gradient_checkpointing=True,  # Memory optimization
        optim="adamw_torch",  # Memory-efficient optimizer
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to=["tensorboard"],
        run_name="t5-summarization"
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    # Clear GPU cache before training
    torch.cuda.empty_cache()
    
    # Train the model
    print("="*80)
    print("STARTING TRAINING...")
    print("="*80 + "\n")
    
    try:
        train_result = trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"  Final training loss: {train_result.training_loss:.4f}")
        print(f"  Total training time: {train_result.metrics['train_runtime']:.2f}s")
        print(f"  Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}\n")
        
        # Save final model
        print(f"Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(train_result.metrics, f, indent=2)
        print(f"Training metrics saved to: {metrics_path}\n")
        
        # Evaluate on validation set
        print("="*80)
        print("RUNNING FINAL EVALUATION")
        print("="*80 + "\n")
        
        eval_results = trainer.evaluate()
        print(f"  Validation loss: {eval_results['eval_loss']:.4f}\n")
        
        # Save evaluation metrics
        eval_metrics_path = os.path.join(output_dir, "eval_metrics.json")
        with open(eval_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation metrics saved to: {eval_metrics_path}\n")
        
        print("="*80)
        print("ALL TRAINING OPERATIONS COMPLETED")
        print("="*80 + "\n")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*80)
            print("ERROR: GPU OUT OF MEMORY")
            print("="*80)
            print("\nSuggestions:")
            print("  1. Reduce batch_size (current: {})".format(batch_size))
            print("  2. Increase gradient_accumulation_steps (current: {})".format(gradient_accumulation_steps))
            print("  3. Reduce max_input_length (current: {})".format(max_input_length))
            print("  4. Reduce max_target_length (current: {})".format(max_target_length))
            raise
        else:
            raise
    
    finally:
        # Clear GPU cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 model for legal judgment summarization (GPU-only).")
    parser.add_argument("--train_file", type=str, default="data/train_processed.jsonl",
                        help="Path to training data JSONL file.")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl",
                        help="Path to validation data JSONL file.")
    parser.add_argument("--output_dir", type=str, default="outputs/t5_summarizer",
                        help="Directory where the trained model will be saved.")
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="Pretrained T5 model name or path.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size (optimized for 6GB GPU).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps for effective larger batch size.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Maximum target sequence length.")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience (epochs).")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use mixed precision training (FP16).")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        fp16=args.fp16
    )
