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

def prepare_dataset(file_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 128) -> Dataset:
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
    # Load the raw data from the specified JSONL file
    data = load_jsonl(file_path)
    
    # Extract the judgment text and summary from each entry
    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]
    
    # Tokenize the source texts (judgments)
    model_inputs = tokenizer(
        judgments,
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize the target texts (summaries) to be used as labels
    labels = tokenizer(
        summaries,
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )
    
    # The Hugging Face Trainer expects the labels to be in a 'labels' key
    model_inputs['labels'] = labels['input_ids']
    
    # Create and return a Dataset object
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
    gradient_accumulation_steps: int
):
    """
    Sets up and runs the training process for the T5 summarization model.

    Args:
        train_file (str): Path to the training data.
        val_file (str): Path to the validation data.
        output_dir (str): Directory where the trained model and checkpoints will be saved.
        model_name (str): The base T5 model to fine-tune (e.g., "t5-small").
        epochs (int): The total number of training epochs.
        batch_size (int): The batch size per device for training and evaluation.
        learning_rate (float): The initial learning rate for the AdamW optimizer.
        warmup_steps (int): Number of steps for the linear warmup from 0 to learning_rate.
        weight_decay (float): The weight decay to apply (if not zero).
        gradient_accumulation_steps (int): Number of updates steps to accumulate before performing a backward/update pass.
    """
    print("\n" + "="*80)
    print(f"🚀 STARTING TRAINING: Fine-tuning {model_name.upper()}")
    print("="*80 + "\n")

    # Load the T5 tokenizer and model from the pretrained checkpoint
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load and prepare the datasets for training and validation
    print(f"📂 Loading and preparing datasets from '{train_file}' and '{val_file}'...")
    train_dataset = prepare_dataset(train_file, tokenizer)
    val_dataset = prepare_dataset(val_file, tokenizer)
    print(f"✅ Datasets prepared: {len(train_dataset)} training examples, {len(val_dataset)} validation examples.\n")

    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Evaluation and saving strategy
        eval_strategy="epoch",      # Evaluate at the end of each epoch
        save_strategy="epoch",      # Save a checkpoint at the end of each epoch
        save_total_limit=2,         # Only keep the last 2 checkpoints
        load_best_model_at_end=True,# Load the best model found during training at the end
        metric_for_best_model="loss",# Use validation loss to determine the "best" model
        greater_is_better=False,    # A lower loss is better
        
        # Technical configurations
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        predict_with_generate=True, # Necessary for summarization metrics
        fp16=torch.cuda.is_available(), # Use mixed-precision training if a GPU is available
        report_to="none"            # Disable reporting to services like W&B
    )
    
    # A data collator dynamically pads the inputs and labels to the max length in a batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Callback to stop training early if validation loss doesn't improve
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )
    
    # --- Start Training ---
    print(f"💪 Training commencing...")
    print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    
    trainer.train()
    
    # --- Save Final Model ---
    print(f"\n✅ Training complete! Saving the best model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*80)
    print("🎉 Model training finished successfully!")
    print("="*80 + "\n")

def main():
    """
    Main function to parse arguments and launch the training process.
    """
    # Set up a robust argument parser
    parser = argparse.ArgumentParser(description="Train a T5 model for legal document summarization.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train_processed.jsonl",
        help="Path to the processed training data file (.jsonl)."
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/val_processed.jsonl",
        help="Path to the processed validation data file (.jsonl)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/t5_summarizer",
        help="Directory where the final model and checkpoints will be saved."
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="hyperparams.json",
        help="Path to a JSON file containing hyperparameters."
    )
    args = parser.parse_args()

    # Safely load hyperparameters from the JSON file
    try:
        with open(args.hyperparams) as f:
            hyperparams = json.load(f)
        print(f"✅ Successfully loaded hyperparameters from '{args.hyperparams}'.")
    except FileNotFoundError:
        print(f"⚠️ WARNING: Hyperparameter file not found at '{args.hyperparams}'. Using default values.")
        hyperparams = {}  # Use an empty dict to fall back on defaults
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from '{args.hyperparams}'. Check for syntax errors. Using default values.")
        hyperparams = {}

    # Launch the training process with parameters from args and the hyperparams file.
    # .get() is used to provide a default value if a key is missing from the file.
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

if __name__ == "__main__":
    main()