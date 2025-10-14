import os
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
# This line is the crucial fix. It must import from the 'evaluate' library.
from evaluate import load
from utils import load_jsonl, save_predictions, save_jsonl
import torch
from tqdm import tqdm

def generate_summaries(
    model_path: str,
    input_file: str,
    output_file: str,
    max_length: int = 256,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    batch_size: int = 8
):
    """
    Generate summaries using fine-tuned model (preserves IDs).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"GENERATING SUMMARIES")
    print(f"{'='*80}\n")
    print(f"🖥️  Device: {device}")
    print(f"📂 Model: {model_path}")
    print(f"📄 Input: {input_file}\n")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    data = load_jsonl(input_file)
    ids = [d['ID'] for d in data]
    judgments = [d['judgment_text'] for d in data]
    
    predictions = []
    print(f"🔄 Generating summaries for {len(judgments)} examples...\n")
    
    for i in tqdm(range(0, len(judgments), batch_size), desc="Progress"):
        batch_judg = judgments[i:i+batch_size]
        inputs = tokenizer(batch_judg, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                length_penalty=length_penalty
            )
        
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_summaries)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Correctly save the submission file with IDs
    predictions_with_ids = [{"ID": id_, "Summary": pred} for id_, pred in zip(ids, predictions)]
    # The output file is now the one with IDs, suitable for submission
    save_jsonl(predictions_with_ids, output_file.replace('.txt', '_with_ids.jsonl'))
    
    print(f"\n✅ Saved {len(predictions)} summaries to: {output_file.replace('.txt', '_with_ids.jsonl')}")
    print("="*80 + "\n")
    
    return predictions, ids

def evaluate_summaries(predictions_file: str, references_file: str):
    """
    Calculate ROUGE-2, ROUGE-L, and BLEU scores using ID alignment.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING SUMMARIES")
    print(f"{'='*80}\n")
    
    pred_data = load_jsonl(predictions_file)
    ref_data = load_jsonl(references_file)
    
    pred_dict = {item['ID']: item['Summary'] for item in pred_data}
    ref_dict = {item['ID']: item['Summary'] for item in ref_data}
    
    common_ids = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
    
    if len(common_ids) == 0:
        print("❌ ERROR: No matching IDs between predictions and references!")
        return None
    
    predictions = [pred_dict[id_] for id_ in common_ids]
    references = [ref_dict[id_] for id_ in common_ids]
    
    print(f"📊 Evaluating {len(common_ids)} aligned examples...\n")
    
    rouge = load("rouge")
    bleu = load("sacrebleu")
    
    rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    print(f"{'='*80}")
    print(f"EVALUATION RESULTS (InLSum Dataset)")
    print(f"{'='*80}")
    print(f"  ROUGE-2:  {rouge_results['rouge2']:.4f}")
    print(f"  ROUGE-L:  {rouge_results['rougeL']:.4f}")
    print(f"  BLEU:     {bleu_result['score']:.4f}")
    print(f"{'='*80}\n")
    
    return {
        "rouge2": rouge_results['rouge2'],
        "rougeL": rouge_results['rougeL'],
        "bleu": bleu_result['score']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate summaries.")
    parser.add_argument("--model_path", type=str, default="outputs/model", help="Path to the trained model.")
    parser.add_argument("--input_file", type=str, default="data/val_processed.jsonl", help="Path to the processed validation file.")
    parser.add_argument("--output_file", type=str, default="outputs/predictions/val_predictions.txt", help="Path to save the generated summaries.")
    parser.add_argument("--references_file", type=str, default=None, help="Path to the reference summaries (optional).")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length for generated summaries.")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search.")
    parser.add_argument("--length_penalty", type=float, default=2.0, help="Length penalty for generation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    args = parser.parse_args()

    # Generate predictions
    predictions, ids = generate_summaries(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size
    )
    
    # Evaluate only if reference file is provided
    if args.references_file and os.path.exists(args.references_file):
        evaluate_summaries(
            predictions_file=args.output_file.replace('.txt', '_with_ids.jsonl'),
            references_file=args.references_file
        )