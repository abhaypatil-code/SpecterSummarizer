import os
import json
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

    # Tokenize without padding for memory efficiency
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


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna to minimize.
    It trains a model with a given set of hyperparameters and returns the validation loss.
    """
    # --- Hyperparameters to Tune ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    num_epochs = trial.suggest_int("num_epochs", 2, 4)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    warmup_steps = trial.suggest_int("warmup_steps", 10, 100)

    # --- Static Parameters ---
    model_name = "t5-base"  # Using t5-base for tuning
    output_dir = f"outputs/tuning/trial_{trial.number}"
    train_file = "data/train_processed.jsonl"
    val_file = "data/val_processed.jsonl"

    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    # --- Model and Tokenizer ---
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # --- Prepare Datasets ---
    train_dataset = prepare_dataset(train_file, tokenizer)
    val_dataset = prepare_dataset(val_file, tokenizer)

    # --- Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        
        # Evaluation and Logging
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",  # Disable model saving to speed up tuning
        
        # Technical
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train and Evaluate ---
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Training with hyperparameters:")
    print(f"  LR: {learning_rate:.2e}, BS: {batch_size}, Epochs: {num_epochs}")
    print(f"  WD: {weight_decay:.4f}, GAS: {gradient_accumulation_steps}, WS: {warmup_steps}")
    print(f"{'='*60}")
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    eval_loss = eval_results["eval_loss"]
    print(f"Trial {trial.number} completed - Validation Loss: {eval_loss:.4f}\n")

    # Optuna minimizes the returned value
    return eval_loss


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the T5 summarizer using Optuna.")
    parser.add_argument("--n_trials", type=int, default=20, 
                        help="Number of Optuna trials to run.")
    parser.add_argument("--train_file", type=str, default="data/train_processed.jsonl",
                        help="Path to training data.")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl",
                        help="Path to validation data.")
    parser.add_argument("--output_hyperparams", type=str, default="hyperparams.json",
                        help="Path to save the best hyperparameters.")
    parser.add_argument("--study_name", type=str, default="t5_summarization_study",
                        help="Name for the Optuna study.")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 STARTING HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    print(f"   Training file: {args.train_file}")
    print(f"   Validation file: {args.val_file}")
    print(f"   Number of trials: {args.n_trials}")
    print(f"   Study name: {args.study_name}")
    print("="*80 + "\n")

    # Verify files exist before starting
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")

    # Create a study to minimize the validation loss
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "="*80)
    print("✅ TUNING COMPLETE")
    print("="*80)
    print(f"  Best Trial Number: {study.best_trial.number}")
    print(f"  Best Validation Loss: {study.best_value:.4f}")
    print("\n  Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("="*80 + "\n")
    
    # Save the best hyperparameters to a file
    best_hyperparams = study.best_params.copy()
    best_hyperparams["model_name"] = "t5-base"
    best_hyperparams["best_validation_loss"] = study.best_value

    os.makedirs(os.path.dirname(args.output_hyperparams) or ".", exist_ok=True)
    with open(args.output_hyperparams, "w") as f:
        json.dump(best_hyperparams, f, indent=4)
        
    print(f"💾 Best hyperparameters saved to '{args.output_hyperparams}'")
    
    # Print optimization history
    print("\n📊 Trial History (Top 5):")
    print("-" * 60)
    trials_df = study.trials_dataframe().sort_values('value').head(5)
    print(trials_df[['number', 'value', 'params_learning_rate', 'params_batch_size', 'params_num_epochs']].to_string(index=False))
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()