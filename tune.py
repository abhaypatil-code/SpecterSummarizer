import os
import argparse
import optuna
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from scripts.utils import load_jsonl
import torch

def prepare_dataset(file_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """Load and tokenize InLSum processed data for training."""
    data = load_jsonl(file_path)
    
    ids = [d['ID'] for d in data]
    judgments = [d['judgment_text'] for d in data]
    summaries = [d['summary'] for d in data]
    
    model_inputs = tokenizer(
        judgments,
        max_length=max_input_length,
        truncation=True,
        padding='max_length'
    )
    
    labels = tokenizer(
        summaries,
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )
    
    dataset = Dataset.from_dict({
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels['input_ids'],
        'ID': ids
    })
    
    return dataset

def objective(trial: optuna.Trial):
    """Define the objective function for Optuna to optimize."""
    
    # --- Hyperparameters to Tune ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    
    # --- Static Parameters ---
    model_name = "t5-base"
    output_dir = f"outputs/tuning/trial_{trial.number}"
    train_file = "data/train_processed.jsonl"
    val_file = "data/val_processed.jsonl"

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    train_dataset = prepare_dataset(train_file, tokenizer)
    val_dataset = prepare_dataset(val_file, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # Corrected Argument Names:
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for T5 model.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of tuning trials.")
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")