import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl, save_jsonl
from tqdm import tqdm
import os

def run_evaluation(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 4,
    max_input_length: int = 1024,
    min_length: int = 400,
    max_target_length: int = 600
):
    """
    Generates summaries for a given input file using a fine-tuned T5 model,
    preserving the original order of IDs and ensuring valid UTF-8 output.
    Optimized for 6GB GPU.
    """
    print("\n" + "="*80)
    print("STARTING EVALUATION SCRIPT")
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully\n")
    
    # Load data while preserving order
    print(f"Loading data from: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    data = list(load_jsonl(input_file))
    
    if not data:
        raise ValueError(f"No data found in {input_file}")
    
    print(f"Loaded {len(data)} examples\n")
    
    # Extract IDs and judgments while preserving order
    ids = [d.get('ID', f"example_{i}") for i, d in enumerate(data)]
    judgments = [d['judgment_text'] for d in data]
    
    # Generate summaries in batches
    print("="*80)
    print("GENERATING SUMMARIES")
    print("="*80 + "\n")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max input length: {max_input_length}")
    print(f"  Min output length: {min_length}")
    print(f"  Max output length: {max_target_length}")
    print(f"  Beam search: 4 beams")
    print(f"  Length penalty: 2.0\n")
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(judgments), batch_size), desc="Processing batches"):
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
    
    # Prepare output data preserving original order
    print("="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")
    
    output_data = []
    for doc_id, summary in zip(ids, predictions):
        output_data.append({
            "ID": doc_id,
            "Summary": summary
        })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    save_jsonl(output_data, output_file)
    print(f"Summaries saved to: {output_file}")
    print(f"Total summaries: {len(output_data)}\n")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    print("="*80)
    print("EVALUATION COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries using fine-tuned T5 model (GPU-only).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file with judgments.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save output summaries (JSONL format).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (optimized for 6GB GPU).")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length.")
    parser.add_argument("--min_length", type=int, default=400,
                        help="Minimum summary length.")
    parser.add_argument("--max_target_length", type=int, default=600,
                        help="Maximum summary length.")
    
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
