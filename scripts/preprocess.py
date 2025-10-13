from transformers import T5Tokenizer
from utils import load_jsonl, save_jsonl
import os

def preprocess_data(judg_path: str, summ_path: str = None, output_path: str = None, tokenizer_name: str = "t5-small"):
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
        
        # Tokenize to check length (max 512 tokens for T5-small input)
        input_ids = tokenizer.encode(input_text, max_length=512, truncation=True)
        
        processed.append({
            "ID": doc_id,
            "judgment_text": input_text,
            "summary": summary_text,
            "input_length": len(input_ids)
        })
    
    # Save processed data
    if output_path:
        save_jsonl(processed, output_path)
        print(f"✅ Preprocessed {len(processed)} examples to {output_path}")
        print(f"   Avg input length: {sum(p['input_length'] for p in processed) / len(processed):.1f} tokens")
        print(f"   Max input length: {max(p['input_length'] for p in processed)} tokens")
    
    return processed

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREPROCESSING InLSum DATASET")
    print("="*80 + "\n")
    
    # Process training data
    if os.path.exists(r"D:\Software\JustNLP\train_judg.jsonl"):
        print("[1/2] Processing TRAINING data...")
        preprocess_data(
            judg_path=r"D:\Software\JustNLP\train_judg.jsonl",
            summ_path=r"D:\Software\JustNLP\train_ref_summ.jsonl",
            output_path=r"D:\Software\JustNLP\data\train_processed.jsonl"
        )
    
    # Process validation data
    if os.path.exists(r"D:\Software\JustNLP\val_judg.jsonl"):
        print("\n[2/2] Processing VALIDATION data...")
        val_summ_path = r"D:\Software\JustNLP\val_ref_summ.jsonl" if os.path.exists(r"D:\Software\JustNLP\val_ref_summ.jsonl") else None
        preprocess_data(
            judg_path=r"D:\Software\JustNLP\val_judg.jsonl",
            summ_path=val_summ_path,
            output_path=r"D:\Software\JustNLP\data\val_processed.jsonl"
        )
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80 + "\n")
