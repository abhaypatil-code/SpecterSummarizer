import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from sacrebleu.metrics import BLEU
import pandas as pd
import json

def validate_model(
    model_path: str,
    validation_file: str,
    batch_size: int = 8,
    max_input_length: int = 1024,
    min_length: int = 400,
    max_target_length: int = 600,
    save_results: bool = True,
    results_file: str = None
):
    """
    Validates a fine-tuned T5 model by generating summaries and calculating
    ROUGE-2, ROUGE-L, and BLEU scores (all scaled to [0, 100]).
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
    ids = [d['ID'] for d in val_data]

    # Generate predictions
    print(f"   - Generating predictions with batch size: {batch_size}")
    predictions = []
    for i in tqdm(range(0, len(judgments), batch_size), desc="Generating summaries"):
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

    print(f"   ✅ Generated {len(predictions)} summaries\n")

    # Calculate ROUGE scores (ROUGE-2 and ROUGE-L)
    print("   - Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)
    
    rouge_result = aggregator.aggregate()

    # Calculate BLEU score
    print("   - Calculating BLEU score...")
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])

    # Extract and scale scores to [0, 100]
    rouge2_precision = rouge_result['rouge2'].mid.precision * 100
    rouge2_recall = rouge_result['rouge2'].mid.recall * 100
    rouge2_f1 = rouge_result['rouge2'].mid.fmeasure * 100
    
    rougeL_precision = rouge_result['rougeL'].mid.precision * 100
    rougeL_recall = rouge_result['rougeL'].mid.recall * 100
    rougeL_f1 = rouge_result['rougeL'].mid.fmeasure * 100
    
    bleu_score_scaled = bleu_score.score  # BLEU is already in [0, 100]

    # Display results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Create detailed results table
    results_df = pd.DataFrame({
        "Metric": ["ROUGE-2", "ROUGE-L", "BLEU"],
        "Precision": [f"{rouge2_precision:.2f}", f"{rougeL_precision:.2f}", "N/A"],
        "Recall": [f"{rouge2_recall:.2f}", f"{rougeL_recall:.2f}", "N/A"],
        "F1-Score": [f"{rouge2_f1:.2f}", f"{rougeL_f1:.2f}", f"{bleu_score_scaled:.2f}"]
    })
    
    print(results_df.to_string(index=False))
    print("\n" + "-"*80)
    
    # Summary metrics (F1 for ROUGE, score for BLEU)
    print("\n📈 SUMMARY (Primary Metrics):")
    print(f"   - ROUGE-2 F1:  {rouge2_f1:.2f}")
    print(f"   - ROUGE-L F1:  {rougeL_f1:.2f}")
    print(f"   - BLEU Score:  {bleu_score_scaled:.2f}")
    print()

    # Save results if requested
    if save_results:
        if results_file is None:
            results_file = validation_file.replace('.jsonl', '_results.json')
        
        results_dict = {
            "model_path": model_path,
            "validation_file": validation_file,
            "num_examples": len(predictions),
            "metrics": {
                "rouge2": {
                    "precision": round(rouge2_precision, 2),
                    "recall": round(rouge2_recall, 2),
                    "f1": round(rouge2_f1, 2)
                },
                "rougeL": {
                    "precision": round(rougeL_precision, 2),
                    "recall": round(rougeL_recall, 2),
                    "f1": round(rougeL_f1, 2)
                },
                "bleu": {
                    "score": round(bleu_score_scaled, 2)
                }
            },
            "predictions": [
                {"ID": id_, "prediction": pred, "reference": ref}
                for id_, pred, ref in zip(ids, predictions, references)
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {results_file}")

    print("\n" + "="*80)
    print("🎉 VALIDATION COMPLETE")
    print("="*80 + "\n")
    
    return {
        "rouge2_f1": rouge2_f1,
        "rougeL_f1": rougeL_f1,
        "bleu": bleu_score_scaled
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a fine-tuned T5 model with ROUGE-2, ROUGE-L, and BLEU.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model directory.")
    parser.add_argument("--validation_file", type=str, required=True, 
                        help="Path to the processed validation/test JSONL file.")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for validation.")
    parser.add_argument("--max_input_length", type=int, default=1024, 
                        help="Maximum token length for input text.")
    parser.add_argument("--min_length", type=int, default=400, 
                        help="Minimum token length for generated summaries.")
    parser.add_argument("--max_target_length", type=int, default=600, 
                        help="Maximum token length for generated summaries.")
    parser.add_argument("--save_results", action="store_true", 
                        help="Save detailed results to JSON file.")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to save results JSON (default: auto-generated from validation_file).")

    args = parser.parse_args()

    validate_model(
        model_path=args.model_path,
        validation_file=args.validation_file,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        min_length=args.min_length,
        max_target_length=args.max_target_length,
        save_results=args.save_results,
        results_file=args.results_file
    )