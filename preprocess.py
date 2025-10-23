import os
import argparse
from transformers import T5Tokenizer
from utils import load_jsonl, save_jsonl
from tqdm import tqdm
import random

def preprocess_and_split_data(
    judg_path: str,
    summ_path: str,
    output_dir: str = "data",
    tokenizer_name: str = "t5-base",
    max_input_length: int = 1024,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Combines, tokenizes, preprocesses, and splits judgment and summary JSONL files
    into train/val/test sets with an 80/10/10 split by default.
    """
    if not os.path.exists(judg_path):
        raise FileNotFoundError(f"Judgment file not found at: {judg_path}")
    if not os.path.exists(summ_path):
        raise FileNotFoundError(f"Summary file not found at: {summ_path}")
    
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio}")
    
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, legacy=False)
    
    print("\n" + "="*80)
    print("📂 LOADING DATA")
    print("="*80)
    
    # Load Data
    judgments = list(load_jsonl(judg_path))
    summaries = list(load_jsonl(summ_path))
    
    judg_dict = {item['ID']: item['Judgment'] for item in judgments}
    summ_dict = {item['ID']: item['Summary'] for item in summaries}
    
    # Check for ID alignment
    judg_ids = set(judg_dict.keys())
    summ_ids = set(summ_dict.keys())
    
    if judg_ids != summ_ids:
        print("⚠️  Warning: ID mismatch detected between judgment and summary files.")
        missing_in_summ = judg_ids - summ_ids
        missing_in_judg = summ_ids - judg_ids
        if missing_in_summ:
            print(f"   - {len(missing_in_summ)} IDs found in judgments but not in summaries.")
        if missing_in_judg:
            print(f"   - {len(missing_in_judg)} IDs found in summaries but not in judgments.")
        print("   - Using only the intersection of IDs.")
    
    # Use only matching IDs
    common_ids = list(judg_ids & summ_ids)
    if not common_ids:
        raise ValueError("No matching IDs found between judgment and summary files!")
    
    print(f"✅ Loaded {len(common_ids)} matching judgment-summary pairs\n")
    
    # Shuffle IDs with fixed seed for reproducibility
    random.seed(random_seed)
    random.shuffle(common_ids)
    
    # Calculate split indices
    total = len(common_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_ids = common_ids[:train_end]
    val_ids = common_ids[train_end:val_end]
    test_ids = common_ids[val_end:]
    
    print("="*80)
    print("📊 DATA SPLIT SUMMARY")
    print("="*80)
    print(f"  Total examples: {total}")
    print(f"  Training: {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
    print(f"  Validation: {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
    print(f"  Random seed: {random_seed}\n")
    
    # Process each split
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        print(f"Processing {split_name.upper()} split...")
        processed = []
        
        for doc_id in tqdm(sorted(split_ids), desc=f"  {split_name}"):
            judgment_text = judg_dict[doc_id]
            summary_text = summ_dict[doc_id]
            
            input_text = f"summarize: {judgment_text}"
            input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)
            
            processed.append({
                "ID": doc_id,
                "judgment_text": input_text,
                "summary": summary_text,
                "input_length": len(input_ids)
            })
        
        # Save processed split
        output_path = os.path.join(output_dir, f"{split_name}_processed.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        save_jsonl(processed, output_path)
        
        if processed:
            avg_len = sum(p['input_length'] for p in processed) / len(processed)
            max_len = max(p['input_length'] for p in processed)
            print(f"  ✅ Saved {len(processed)} examples to {output_path}")
            print(f"     Avg input length: {avg_len:.1f} tokens")
            print(f"     Max input length: {max_len} tokens\n")
    
    print("="*80)
    print("✅ PREPROCESSING AND SPLIT COMPLETE")
    print("="*80 + "\n")
    
    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split the dataset with 80/10/10 ratio.")
    parser.add_argument("--judg_path", type=str, default="data/train_judg.jsonl",
                        help="Path to the judgments JSONL file.")
    parser.add_argument("--summ_path", type=str, default="data/train_ref_summ.jsonl",
                        help="Path to the reference summaries JSONL file.")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory where processed files will be saved.")
    parser.add_argument("--tokenizer_name", type=str, default="t5-base",
                        help="Name or path of the tokenizer.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum token length for input judgments.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio (default: 0.1)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducible splits.")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 STARTING DATASET PREPROCESSING WITH 80/10/10 SPLIT")
    print("="*80 + "\n")
    
    preprocess_and_split_data(
        judg_path=args.judg_path,
        summ_path=args.summ_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        max_input_length=args.max_input_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
