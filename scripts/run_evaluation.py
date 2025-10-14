import os
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
    Generates summaries for an input file using a fine-tuned T5 model
    and saves them in the competition-required JSONL format with IDs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    data = load_jsonl(input_file)
    documents = [item['Judgement'] for item in data]
    ids = [item['ID'] for item in data]

    predictions = []
    print(f"\nGenerating summaries for {len(documents)} documents...")

    for i in tqdm(range(0, len(documents), batch_size), desc="Generating"):
        batch_texts = documents[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
        
        batch_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Structure the predictions with their corresponding IDs
    predictions_with_ids = [{"ID": id_, "Summary": pred} for id_, pred in zip(ids, predictions)]
    
    # Save the predictions to the specified output file in JSONL format
    save_jsonl(predictions_with_ids, output_file)
    print(f"\n✅ Saved {len(predictions)} summaries to: {output_file}")
    
    return predictions, ids

def evaluate_summaries(predictions_file: str, references_file: str):
    """
    Calculates ROUGE-2, ROUGE-L, and BLEU scores by aligning predictions
    and references using their IDs.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING SUMMARIES")
    print(f"{'='*80}\n")
    
    pred_data = load_jsonl(predictions_file)
    ref_data = load_jsonl(references_file)
    
    # Create dictionaries for quick lookup by ID
    pred_dict = {item['ID']: item['Summary'] for item in pred_data}
    ref_dict = {item['ID']: item['Summary'] for item in ref_data}
    
    common_ids = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
    
    if not common_ids:
        print("❌ ERROR: No matching IDs found between predictions and references!")
        return None
    
    # Align predictions and references based on common IDs
    predictions = [pred_dict[id_] for id_ in common_ids]
    references = [ref_dict[id_] for id_ in common_ids]
    
    print(f"📊 Evaluating {len(common_ids)} aligned examples...\n")
    
    # Load metrics
    rouge = load("rouge")
    bleu = load("sacrebleu")
    
    # Compute scores
    rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    print(f"{'='*80}")
    print(f"EVALUATION RESULTS")
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
    parser = argparse.ArgumentParser(description="Generate and evaluate summaries for the legal summarization task.")
    parser.add_argument("--model_path", type=str, default="outputs/t5_summarizer", help="Path to the fine-tuned model directory.")
    parser.add_argument("--input_file", type=str, default="data/val_processed.jsonl", help="Path to the processed input data (JSONL).")
    
    # --- FIX ---
    # Changed the default output file name to 'answer.jsonl' for direct submission.
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/predictions/answer.jsonl",
        help="Path to save the generated summaries for submission."
    )
    # --- END FIX ---
    
    parser.add_argument("--references_file", type=str, default="data/val_processed.jsonl", help="Path to the reference summaries for local validation.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length for the generated summaries.")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search decoding.")
    parser.add_argument("--length_penalty", type=float, default=2.0, help="Length penalty for generation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    args = parser.parse_args()

    # Step 1: Generate summaries from the model
    generate_summaries(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size
    )
    
    # Step 2: Evaluate the summaries if a reference file is provided
    if args.references_file and os.path.exists(args.references_file):
        evaluate_summaries(
            predictions_file=args.output_file, # Directly use the output file
            references_file=args.references_file
        )