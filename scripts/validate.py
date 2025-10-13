import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl
import torch

def validate_samples(model_path: str, val_file: str, num_samples: int = 5):
    """
    Manually inspect generated summaries vs references (InLSum format).
    
    Args:
        model_path: Path to trained model
        val_file: Path to processed validation file
        num_samples: Number of examples to display
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load validation data (processed)
    val_data = load_jsonl(val_file)[:num_samples]
    
    print("\n" + "="*80)
    print("MANUAL VALIDATION - InLSum Dataset")
    print("="*80)
    
    for i, example in enumerate(val_data, 1):
        doc_id = example['ID']
        judgment = example['judgment_text']
        reference = example['summary']
        
        # Generate summary
        inputs = tokenizer(judgment, return_tensors='pt', max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n[Example {i} - ID: {doc_id}]")
        print(f"INPUT: {judgment[:250]}...")
        print(f"\n📖 REFERENCE: {reference}")
        print(f"🤖 GENERATED: {prediction}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually validate generated summaries.")
    parser.add_argument("--model_path", type=str, default="outputs/model", help="Path to the trained model.")
    parser.add_argument("--val_file", type=str, default="data/val_processed.jsonl", help="Path to the processed validation file.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to validate.")
    args = parser.parse_args()

    validate_samples(
        model_path=args.model_path,
        val_file=args.val_file,
        num_samples=args.num_samples
    )