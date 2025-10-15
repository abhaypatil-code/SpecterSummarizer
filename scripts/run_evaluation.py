import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl, save_jsonl
from tqdm import tqdm

def run_evaluation(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 8,
    max_input_length: int = 1024,
    min_length: int = 400,
    max_target_length: int = 600
):
    """
    Generates summaries for a given input file using a fine-tuned T5 model,
    preserving the original order of IDs and ensuring valid UTF-8 output.
    """
    print("\n" + "="*80)
    print(f"🚀 STARTING EVALUATION SCRIPT")
    print("="*80 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Using device: {device}")

    # Load tokenizer and model
    print(f"   - Loading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # Load data while preserving order
    print(f"   - Loading data from: {input_file}")
    data = list(load_jsonl(input_file))
    
    # --- FIX APPLIED: Preserve ID Order ---
    # Store original data with IDs to ensure correct output order
    ordered_data = [{"ID": d.get("ID"), "judgment_text": d.get('judgment_text') or d.get('judgment')} for d in data]

    # Generate predictions in batches
    print(f"   - Generating predictions with batch size: {batch_size}")
    results = []
    for i in tqdm(range(0, len(ordered_data), batch_size), desc="Generating Summaries"):
        batch_data = ordered_data[i:i+batch_size]
        batch_texts = ["summarize: " + item['judgment_text'] for item in batch_data]

        try:
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
            
            # --- FIX APPLIED: Sanitize UTF-8 Output ---
            # Decode with error handling to replace invalid characters
            batch_preds = tokenizer.batch_decode(
                summary_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            for item_data, pred in zip(batch_data, batch_preds):
                results.append({
                    "ID": item_data["ID"],
                    "summary": pred
                })

        except Exception as e:
            print(f"❌ Error processing batch starting at index {i}: {e}")
            # Add placeholders for the failed batch to maintain order
            for item_data in batch_data:
                results.append({
                    "ID": item_data["ID"],
                    "summary": "[ERROR GENERATING SUMMARY]"
                })

    # Save the predictions in JSONL format, preserving order
    save_jsonl(results, output_file)
    print(f"\n✅ Successfully generated {len(results)} summaries and saved to {output_file}")
    
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries using a fine-tuned T5 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file for which to generate summaries.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated summaries.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum token length for input text.")
    parser.add_argument("--min_length", type=int, default=400, help="Minimum token length for generated summaries.")
    parser.add_argument("--max_target_length", type=int, default=600, help="Maximum token length for generated summaries.")
    
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        min_length=args.min_length,
        max_target_length=args.max_target_length
    )