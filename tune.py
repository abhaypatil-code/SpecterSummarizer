import os
import json
import argparse
import optuna
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from utils import load_jsonl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(
    file_path: str,
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256
) -> Dataset:
    """
    Loads and tokenizes data for the tuning process.
    Aligned with the updated train.py preprocessing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = list(load_jsonl(file_path))
    if not data:
        raise ValueError(f"The dataset at {file_path} contains no data.")
    
    # Filter out entries with empty summaries
    data = [d for d in data if d.get('summary') and d['summary'].strip()]
    
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
    dataset = Dataset.from_dict(model_inputs)
    
    logger.info(f"Loaded {len(dataset)} examples from {file_path}")
    return dataset

def objective(
    trial,
    train_dataset,
    val_dataset,
    tokenizer,
    model_name,
    output_dir,
    max_target_length,
    num_epochs
):
    """
    Optuna objective function for hyperparameter tuning.
    Optimized for 6GB GPU with memory-efficient settings.
    """
    # Clear GPU cache at start of each trial
    torch.cuda.empty_cache()
    
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.05, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 10, 100)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8])
    
    logger.info(f"\nTrial {trial.number}:")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  warmup_steps: {warmup_steps}")
    logger.info(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    device = torch.device("cuda")
    
    # Load model for this trial
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trial_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Training arguments optimized for 6GB GPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=trial_output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints during tuning to save space
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        warmup_steps=warmup_steps,
        logging_steps=50,
        fp16=True,  # Enable mixed precision for memory efficiency
        dataloader_num_workers=0,  # Windows compatibility
        gradient_checkpointing=True,  # Memory optimization
        optim="adamw_torch",
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        load_best_model_at_end=False,  # Disable for tuning to save memory
        report_to="none",  # Disable reporting during tuning
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    try:
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        
        logger.info(f"  eval_loss: {eval_loss:.4f}\n")
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        
        return eval_loss
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning(f"Trial {trial.number} failed with OOM error. Pruning trial.")
            # Clean up
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        else:
            raise

def tune_hyperparameters(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "t5-base",
    n_trials: int = 20,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    num_epochs: int = 2,
    random_seed: int = 42
):
    """
    Tunes hyperparameters using Optuna for T5 summarization model.
    Optimized for 6GB GPU.
    """
    print("\n" + "="*80)
    print("STARTING HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_file, tokenizer, max_input_length, max_target_length)
    val_dataset = prepare_dataset(val_file, tokenizer, max_input_length, max_target_length)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}\n")
    
    print(f"Configuration:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Epochs per trial: {num_epochs}")
    print(f"  Random seed: {random_seed}\n")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print("="*80)
    print("STARTING OPTIMIZATION")
    print("="*80 + "\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset,
            tokenizer,
            model_name,
            output_dir,
            max_target_length,
            num_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*80 + "\n")
    
    # Best hyperparameters
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest validation loss: {study.best_value:.4f}\n")
    
    # Save best hyperparameters
    best_params = {
        "model_name": model_name,
        **study.best_params
    }
    
    hyperparams_file = os.path.join(output_dir, "hyperparams.json")
    with open(hyperparams_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best hyperparameters saved to: {hyperparams_file}")
    
    # Save study results
    study_file = os.path.join(output_dir, "optuna_study.json")
    study_results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    with open(study_file, 'w', encoding='utf-8') as f:
        json.dump(study_results, f, indent=4)
    
    print(f"Study results saved to: {study_file}\n")
    
    print("="*80)
    print("ALL TUNING OPERATIONS COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for T5 summarization (GPU-only).")
    parser.add_argument("--train_file", type=str, default="data/train_processed.jsonl",
                        help="Path to training data JSONL file.")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl",
                        help="Path to validation data JSONL file.")
    parser.add_argument("--output_dir", type=str, default="outputs/tuning",
                        help="Directory to save tuning results.")
    parser.add_argument("--model_name", type=str, default="t5-base",
                        help="Pretrained T5 model name.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length.")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of epochs per trial.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    tune_hyperparameters(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_trials=args.n_trials,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        num_epochs=args.num_epochs,
        random_seed=args.random_seed
    )
