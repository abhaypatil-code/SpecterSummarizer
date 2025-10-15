import os
import argparse
from transformers import T5Tokenizer
from utils import load_jsonl, save_jsonl
from tqdm import tqdm

def preprocess_data(
    judg_path: str,
    summ_path: str = None,
    output_path: str = None,
    tokenizer_name: str = "t5-base",
    max_input_length: int = 1024
):
    """
    Combines, tokenizes, and preprocesses judgment and summary JSONL files,
    aligning them by their 'ID' field. This script is designed to prepare
    the InLSum dataset for training a T5 summarization model.
    """
    if not os.path.exists(judg_path):
        raise FileNotFoundError(f"Judgment file not found at: {judg_path}")

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, legacy=False)

    # --- Load Data ---
    judgments = load_jsonl(judg_path)
    judg_dict = {item['ID']: item['Judgment'] for item in judgments}
    summ_dict = {}

    # --- FIX APPLIED: Robust handling of dataset mismatches ---
    if summ_path and os.path.exists(summ_path):
        summaries = load_jsonl(summ_path)
        summ_dict = {item['ID']: item['Summary'] for item in summaries}
        
        judg_ids = set(judg_dict.keys())
        summ_ids = set(summ_dict.keys())

        # Warn about any mismatches instead of crashing
        if judg_ids != summ_ids:
            print("⚠️ Warning: ID mismatch detected between judgment and summary files.")
            missing_in_summ = judg_ids - summ_ids
            missing_in_judg = summ_ids - judg_ids
            if missing_in_summ:
                print(f"   - {len(missing_in_summ)} IDs found in judgments but not in summaries.")
            if missing_in_judg:
                print(f"   - {len(missing_in_judg)} IDs found in summaries but not in judgments.")
            print("   - Proceeding with the intersection of available IDs.")
    else:
        # Create empty summaries if no summary file is provided
        summ_dict = {id_: "" for id_ in judg_dict.keys()}

    # --- Process and Tokenize Data ---
    processed = []
    # Loop over sorted judgment IDs, which is safe and deterministic
    for doc_id in tqdm(sorted(judg_dict.keys()), desc="Preprocessing examples"):
        # Safely get judgment text
        judgment_text = judg_dict[doc_id]
        # Safely get summary text; defaults to "" if ID is missing in summaries
        summary_text = summ_dict.get(doc_id, "")

        input_text = f"summarize: {judgment_text}"
        input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)

        processed.append({
            "ID": doc_id,
            "judgment_text": input_text,
            "summary": summary_text,
            "input_length": len(input_ids)
        })

    # --- Save Processed Data ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_jsonl(processed, output_path)
        print(f"\n✅ Preprocessed {len(processed)} examples and saved to {output_path}")
        if processed:
            avg_len = sum(p['input_length'] for p in processed) / len(processed)
            max_len = max(p['input_length'] for p in processed)
            print(f"   - Avg input length: {avg_len:.1f} tokens")
            print(f"   - Max input length: {max_len} tokens")

    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the InLSum dataset for summarization.")
    parser.add_argument("--train_judg_path", type=str, default="data/train_judg.jsonl", help="Path to the training judgments JSONL file.")
    parser.add_argument("--train_summ_path", type=str, default="data/train_ref_summ.jsonl", help="Path to the training reference summaries JSONL file.")
    parser.add_argument("--val_judg_path", type=str, default="data/val_judg.jsonl", help="Path to the validation judgments JSONL file.")
    parser.add_argument("--val_summ_path", type=str, default="data/val_ref_summ.jsonl", help="Path to the validation reference summaries JSONL file.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the processed files will be saved.")
    parser.add_argument("--tokenizer_name", type=str, default="t5-base", help="Name or path of the tokenizer to use for preprocessing.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum token length for the input judgments.")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 STARTING InLSum DATASET PREPROCESSING")
    print("="*80 + "\n")

    if os.path.exists(args.train_judg_path):
        print("[1/2] Processing TRAINING data...")
        preprocess_data(
            judg_path=args.train_judg_path,
            summ_path=args.train_summ_path,
            output_path=os.path.join(args.output_dir, "train_processed.jsonl"),
            tokenizer_name=args.tokenizer_name,
            max_input_length=args.max_input_length
        )
    else:
        print("🟡 Skipping training data processing: file not found.")

    if os.path.exists(args.val_judg_path):
        print("\n[2/2] Processing VALIDATION data...")
        val_summ_path = args.val_summ_path if os.path.exists(args.val_summ_path) else None
        if not val_summ_path:
             print("   - Warning: Validation summary file not found. Processing without labels.")
        preprocess_data(
            judg_path=args.val_judg_path,
            summ_path=val_summ_path,
            output_path=os.path.join(args.output_dir, "val_processed.jsonl"),
            tokenizer_name=args.tokenizer_name,
            max_input_length=args.max_input_length
        )
    else:
        print("\n🟡 Skipping validation data processing: file not found.")

    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80 + "\n")