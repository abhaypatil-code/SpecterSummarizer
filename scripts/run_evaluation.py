import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl, save_jsonl, save_predictions
from tqdm import tqdm

def run_evaluation(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 8,
    max_input_length: int = 1024,
    # --- FIX APPLIED: Added min_length parameter ---
    min_length: int = 256,
    max_target_length: int = 512
):
    """
    Generates summaries for a given input file using a fine-tuned T5 model.
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

    # Load and preprocess data
    print(f"   - Loading data from: {input_file}")
    data = list(load_jsonl(input_file))
    
    # Check if 'judgment_text' key exists, otherwise assume the raw text is provided
    if 'judgment_text' not in data[0]:
        texts = [d['judgment'] for d in data]
    else:
        texts = [d['judgment_text'] for d in data]

    # Generate predictions in batches
    print(f"   - Generating predictions with batch size: {batch_size}")
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Summaries"):
        batch_texts = texts[i:i+batch_size]
        
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
                # --- FIX APPLIED: Enforce minimum summary length ---
                min_length=min_length,
                num_beams=4, 
                length_penalty=2.0, 
                early_stopping=True
            )

        batch_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
    # Save the predictions
    save_predictions(predictions, output_file)
    print(f"\n✅ Successfully generated {len(predictions)} summaries and saved to {output_file}")
    
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
    # --- FIX APPLIED: Added command-line argument for min_length ---
    parser.add_argument("--min_length", type=int, default=256, help="Minimum token length for generated summaries.")
    parser.add_argument("--max_target_length", type=int, default=512, help="Maximum token length for generated summaries.")
    
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