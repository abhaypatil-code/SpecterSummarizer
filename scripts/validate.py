import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
import pandas as pd

def validate_model(
    model_path: str,
    validation_file: str,
    batch_size: int = 8,
    max_input_length: int = 1024,
    min_length: int = 256,
    max_target_length: int = 512
):
    """
    Validates a fine-tuned T5 model by generating summaries and calculating ROUGE scores.
    """
    print("\n" + "="*80)
    print(f"🚀 STARTING VALIDATION SCRIPT")
    print("="*80 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Using device: {device}")

    # Load tokenizer and model
    print(f"   - Loading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # Load validation data
    print(f"   - Loading validation data from: {validation_file}")
    val_data = list(load_jsonl(validation_file))
    
    if not val_data:
        print("❌ No data found in the validation file. Exiting.")
        return

    judgments = [d['judgment_text'] for d in val_data]
    references = [d['summary'] for d in val_data]

    # Generate predictions
    print(f"   - Generating predictions with batch size: {batch_size}")
    predictions = []
    for i in tqdm(range(0, len(judgments), batch_size), desc="Validating"):
        try:
            batch_texts = ["summarize: " + text for text in judgments[i:i+batch_size]]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                max_length=max_input_length, 
                truncation=True, 
                padding=True
            ).to(device)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs['input_ids'], 
                    max_length=max_target_length,
                    min_length=min_length,
                    num_beams=4, 
                    length_penalty=2.0, 
                    early_stopping=True
                )
            
            batch_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        
        except Exception as e:
            print(f"❌ Error processing batch starting at index {i}: {e}")
            predictions.extend([""] * len(batch_texts))

    # Calculate ROUGE scores
    print("\n   - Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)
    
    result = aggregator.aggregate()

    # Display results
    print("\n" + "-"*40)
    print("📊 ROUGE SCORE RESULTS")
    print("-"*40)
    
    results_df = pd.DataFrame({
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Precision": [result['rouge1'].mid.precision, result['rouge2'].mid.precision, result['rougeL'].mid.precision],
        "Recall": [result['rouge1'].mid.recall, result['rouge2'].mid.recall, result['rougeL'].mid.recall],
        "F1-Score": [result['rouge1'].mid.fmeasure, result['rouge2'].mid.fmeasure, result['rougeL'].mid.fmeasure]
    })
    
    print(results_df.to_string(index=False))
    print("-"*40)

    print("\n" + "="*80)
    print("🎉 VALIDATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a fine-tuned T5 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the processed validation JSONL file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for validation.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum token length for input text.")
    parser.add_argument("--min_length", type=int, default=128, help="Minimum token length for generated summaries.")
    # --- FIX APPLIED: Consistent Length Parameter ---
    parser.add_argument("--max_target_length", type=int, default=512, help="Maximum token length for generated summaries.")

    args = parser.parse_args()

    validate_model(
        model_path=args.model_path,
        validation_file=args.validation_file,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        min_length=args.min_length,
        max_target_length=args.max_target_length
    )