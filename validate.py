import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl
from tqdm import tqdm
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import pandas as pd
import json
import os

def validate_model(
    model_path: str,
    validation_file: str,
    batch_size: int = 4,
    max_input_length: int = 1024,
    min_length: int = 400,
    max_target_length: int = 600,
    save_results: bool = True,
    results_file: str = None
):
    """
    Validates a fine-tuned T5 model by generating summaries and calculating
    ROUGE-2, ROUGE-L, and BLEU scores (all scaled to [0, 100]).
    Optimized for 6GB GPU.
    """
    print("\n" + "="*80)
    print("STARTING VALIDATION SCRIPT")
    print("="*80 + "\n")
    
    # Setup device - enforce GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Load tokenizer and model
    print(f"Loading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully\n")
    
    # Load validation data
    print(f"Loading validation data from: {validation_file}")
    data = list(load_jsonl(validation_file))
    
    if not data:
        raise ValueError(f"No data found in {validation_file}")
    
    print(f"Loaded {len(data)} validation examples\n")
    
    # Extract texts
    judgments = [d['judgment_text'] for d in data]
    references = [d['summary'] for d in data]
    
    # Generate summaries in batches
    print("="*80)
    print("GENERATING SUMMARIES")
    print("="*80 + "\n")
    print(f"Batch size: {batch_size}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_target_length}\n")
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(judgments), batch_size), desc="Generating"):
            batch = judgments[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch,
                max_length=max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate summaries
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_length=min_length,
                max_length=max_target_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode outputs
            batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
            
            # Clear cache periodically to prevent memory buildup
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
    
    print(f"\nGenerated {len(predictions)} summaries\n")
    
    # Compute metrics
    print("="*80)
    print("COMPUTING METRICS")
    print("="*80 + "\n")
    
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    bleu = BLEU()
    
    rouge2_scores = []
    rougeL_scores = []
    
    print("Computing ROUGE scores...")
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE"):
        scores = rouge.score(ref, pred)
        rouge2_scores.append(scores['rouge2'].fmeasure * 100)  # Scale to [0, 100]
        rougeL_scores.append(scores['rougeL'].fmeasure * 100)  # Scale to [0, 100]
    
    print("Computing BLEU score...")
    bleu_score = bleu.corpus_score(predictions, [references]).score  # Already in [0, 100]
    
    # Calculate averages
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"  ROUGE-2 F1: {avg_rouge2:.2f}")
    print(f"  ROUGE-L F1: {avg_rougeL:.2f}")
    print(f"  BLEU:       {bleu_score:.2f}")
    print("="*80 + "\n")
    
    # Save results if requested
    if save_results:
        if results_file is None:
            results_file = "validation_results.json"
        
        # Create results directory if it doesn't exist
        results_dir = os.path.dirname(results_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        # Prepare results data
        results = {
            "model_path": model_path,
            "validation_file": validation_file,
            "num_examples": len(data),
            "metrics": {
                "rouge2_f1": round(avg_rouge2, 2),
                "rougeL_f1": round(avg_rougeL, 2),
                "bleu": round(bleu_score, 2)
            },
            "generation_config": {
                "batch_size": batch_size,
                "max_input_length": max_input_length,
                "min_length": min_length,
                "max_target_length": max_target_length
            }
        }
        
        # Save metrics
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to: {results_file}")
        
        # Save detailed predictions
        predictions_file = results_file.replace('.json', '_predictions.jsonl')
        detailed_results = []
        
        for i, (pred, ref, r2, rL) in enumerate(zip(predictions, references, rouge2_scores, rougeL_scores)):
            detailed_results.append({
                "id": data[i].get('ID', f"example_{i}"),
                "reference": ref,
                "prediction": pred,
                "rouge2": round(r2, 2),
                "rougeL": round(rL, 2)
            })
        
        # Save predictions
        from utils import save_jsonl
        save_jsonl(detailed_results, predictions_file)
        print(f"Detailed predictions saved to: {predictions_file}\n")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    print("="*80)
    print("VALIDATION COMPLETED")
    print("="*80 + "\n")
    
    return {
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "bleu": bleu_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate T5 summarization model (GPU-only).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory.")
    parser.add_argument("--validation_file", type=str, default="data/val_processed.jsonl",
                        help="Path to validation data JSONL file.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (optimized for 6GB GPU).")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length.")
    parser.add_argument("--min_length", type=int, default=400,
                        help="Minimum summary length.")
    parser.add_argument("--max_target_length", type=int, default=600,
                        help="Maximum summary length.")
    parser.add_argument("--results_file", type=str, default="results/validation_results.json",
                        help="Path to save validation results.")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save results to file.")
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model_path,
        validation_file=args.validation_file,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        min_length=args.min_length,
        max_target_length=args.max_target_length,
        save_results=not args.no_save,
        results_file=args.results_file
    )
