import os
import json
import optuna
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_metric
from utils import load_jsonl

# --- FIX APPLIED ---
# The function's default values and logic have been updated to match the optimized train.py
def prepare_dataset(
    file_path: str,
    tokenizer,
    # 1. CORRECTED max_input_length to align with preprocessing (1024)
    max_input_length: int = 1024,
    # 2. INCREASED max_target_length for better summaries (256)
    max_target_length: int = 256
) -> Dataset:
    """
    Loads and tokenizes data for the tuning process.
    (Docstring remains the same)
    """
    data = load_jsonl(file_path)
    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]

    # 3. REMOVED `padding='max_length'` for memory efficiency.
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


    # --- Static Parameters ---
    model_name = "t5-base" # Using t5-base for tuning for a balance of speed and performance
    output_dir = f"outputs/tuning/trial_{trial.number}"
    train_file = "data/train_processed.jsonl"
    val_file = "data/val_processed.jsonl"

    # --- Model and Tokenizer ---
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # --- Prepare Datasets ---
    # The corrected prepare_dataset function is now used here
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
        
        # Evaluation and Logging
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no", # Disable model saving to speed up tuning
        
        # Technical
        report_to="none",
        fp16=True, # Assumes CUDA is available
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
    trainer.train()
    eval_results = trainer.evaluate()

    # Optuna minimizes the returned value, so we return the validation loss
    return eval_results["eval_loss"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the T5 summarizer.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run.")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 STARTING HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80 + "\n")

    # Create a study to minimize the validation loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "="*80)
    print("✅ TUNING COMPLETE")
    print(f"  - Best Trial Number: {study.best_trial.number}")
    print(f"  - Best Validation Loss: {study.best_value:.4f}")
    print("  - Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
    print("="*80 + "\n")
    
    # Save the best hyperparameters to a file, including the base model name
    best_hyperparams = study.best_params
    best_hyperparams["model_name"] = "t5-base" 

    os.makedirs(os.path.dirname("hyperparams.json"), exist_ok=True)
    with open("hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f, indent=4)
        
    print(f"✅ Best hyperparameters saved to 'hyperparams.json'.")


if __name__ == "__main__":
    main()