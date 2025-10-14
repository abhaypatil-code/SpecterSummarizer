import os
import argparse
from transformers import T5Tokenizer
from utils import load_jsonl, save_jsonl

def preprocess_data(judg_path: str, summ_path: str = None, output_path: str = None, tokenizer_name: str = "t5-small", max_input_length: int = 512):
    """
    Combine judgment and summary JSONL files using ID alignment.
    
    Dataset Format (InLSum):
        Judgments: {"ID": "id_100", "Judgment": "<Full text>"}
        Summaries: {"ID": "id_100", "Summary": "<Reference Summary>"}
    
    Args:
        judg_path: Path to <split>_judg.jsonl
        summ_path: Path to <split>_ref_summary.jsonl (None for test set without labels)
        output_path: Where to save processed data
        tokenizer_name: Pretrained tokenizer identifier
    """
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, legacy=False)
    
    # Load judgments
    judgments = load_jsonl(judg_path)
    
    # Create ID-to-judgment mapping
    judg_dict = {item['ID']: item['Judgment'] for item in judgments}
    
    # Load summaries if available (training/validation with labels)
    if summ_path and os.path.exists(summ_path):
        summaries = load_jsonl(summ_path)
        summ_dict = {item['ID']: item['Summary'] for item in summaries}
    else:
        # Test set or validation without reference summaries
        summ_dict = {id_: "" for id_ in judg_dict.keys()}
    
    # Verify ID alignment
    judg_ids = set(judg_dict.keys())
    summ_ids = set(summ_dict.keys())
    
    if summ_path and os.path.exists(summ_path):
        assert judg_ids == summ_ids, f"ID mismatch! Judgments: {len(judg_ids)}, Summaries: {len(summ_ids)}, Missing in judgments: {summ_ids - judg_ids}, Missing in summaries: {judg_ids - summ_ids}"
    
    # Process aligned data
    processed = []
    for doc_id in sorted(judg_dict.keys()):
        judgment_text = judg_dict[doc_id]
        summary_text = summ_dict.get(doc_id, "")
        
        # T5 prefix for summarization task
        input_text = f"summarize: {judgment_text}"
        
        # Tokenize to check length
        input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)
        
        processed.append({
            "ID": doc_id,
            "judgment_text": input_text,
            "summary": summary_text,
            "input_length": len(input_ids)
        })
    
    # Save processed data
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_jsonl(processed, output_path)
        print(f"✅ Preprocessed {len(processed)} examples to {output_path}")
        print(f"   Avg input length: {sum(p['input_length'] for p in processed) / len(processed):.1f} tokens")
        print(f"   Max input length: {max(p['input_length'] for p in processed)} tokens")
    
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess InLSum dataset.")
    parser.add_argument("--train_judg", type="str", default="data/train_judg.jsonl", help="Path to training judgments.")
    parser.add_argument("--train_summ", type="str", default="data/train_ref_summ.jsonl", help="Path to training summaries.")
    parser.add_argument("--val_judg", type="str", default="data/val_judg.jsonl", help="Path to validation judgments.")
    parser.add_argument("--val_summ", type="str", default="data/val_ref_summ.jsonl", help="Path to validation summaries.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed files.")
    parser.add_argument("--tokenizer_name", type=str, default="t5-large", help="Tokenizer to use for preprocessing.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Max token length for input.")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("PREPROCESSING InLSum DATASET")
    print("="*80 + "\n")
    
    # Process training data
    if os.path.exists(args.train_judg):
        print("[1/2] Processing TRAINING data...")
        preprocess_data(
            judg_path=args.train_judg,
            summ_path=args.train_summ,
            output_path=os.path.join(args.output_dir, "train_processed.jsonl"),
            tokenizer_name=args.tokenizer_name,
            max_input_length=args.max_input_length
        )
    
    # Process validation data
    if os.path.exists(args.val_judg):
        print("\n[2/2] Processing VALIDATION data...")
        val_summ_path = args.val_summ if os.path.exists(args.val_summ) else None
        preprocess_data(
            judg_path=args.val_judg,
            summ_path=val_summ_path,
            output_path=os.path.join(args.output_dir, "val_processed.jsonl"),
            tokenizer_name=args.tokenizer_name,
            max_input_length=args.max_input_length
        )
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80 + "\n")