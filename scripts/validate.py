from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import load_jsonl
import torch

def validate_samples(model_path: str, num_samples: int = 5):
    """
    Manually inspect generated summaries vs references (InLSum format).
    
    Args:
        model_path: Path to trained model
        num_samples: Number of examples to display
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load validation data (processed)
    val_data = load_jsonl(r"D:\Software\JustNLP\data\val_processed.jsonl")[:num_samples]
    
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
    validate_samples(r"D:\Software\JustNLP\outputs\model", num_samples=3)
