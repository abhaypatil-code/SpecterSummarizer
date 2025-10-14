import os
import argparse
from transformers import T5Tokenizer
from utils import load_jsonl, save_jsonl
from tqdm import tqdm # Import tqdm for the progress bar

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

    Dataset Format (InLSum):
        - Judgments: {"ID": "id_100", "Judgment": "<Full text>"}
        - Summaries: {"ID": "id_100", "Summary": "<Reference Summary>"}

    Args:
        judg_path (str): Path to the <split>_judg.jsonl file.
        summ_path (str, optional): Path to the corresponding <split>_ref_summary.jsonl.
                                   If None, summaries will be treated as empty strings
                                   (useful for test sets without labels). Defaults to None.
        output_path (str, optional): Path to save the processed JSONL data. If None,
                                     the data is not saved. Defaults to None.
        tokenizer_name (str, optional): Identifier for the pretrained tokenizer from
                                        Hugging Face Hub. Defaults to "t5-base".
        max_input_length (int, optional): The maximum number of tokens for the input
                                          judgment text. Defaults to 1024.
    """
    # --- 1. Input File Validation ---
    if not os.path.exists(judg_path):
        raise FileNotFoundError(f"Judgment file not found at: {judg_path}")

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, legacy=False)

    # --- 2. Load Data ---
    judgments = load_jsonl(judg_path)
    judg_dict = {item['ID']: item['Judgment'] for item in judgments}

    # Load summaries if the file path is provided and exists
    if summ_path and os.path.exists(summ_path):
        summaries = load_jsonl(summ_path)
        summ_dict = {item['ID']: item['Summary'] for item in summaries}
        # Verify that the IDs in both files match perfectly
        judg_ids = set(judg_dict.keys())
        summ_ids = set(summ_dict.keys())
        assert judg_ids == summ_ids, (
            f"ID mismatch between judgments and summaries!\n"
            f"Judgments: {len(judg_ids)}, Summaries: {len(summ_ids)}\n"
            f"Missing in summaries: {judg_ids - summ_ids}\n"
            f"Missing in judgments: {summ_ids - judg_ids}"
        )
    else:
        # If no summary file, create empty summaries (for test/inference)
        summ_dict = {id_: "" for id_ in judg_dict.keys()}

    # --- 3. Process and Tokenize Data ---
    processed = []
    # Use tqdm to create a progress bar
    for doc_id in tqdm(sorted(judg_dict.keys()), desc="Preprocessing examples"):
        judgment_text = judg_dict[doc_id]
        summary_text = summ_dict.get(doc_id, "")

        # Apply the T5-specific prefix for the summarization task
        input_text = f"summarize: {judgment_text}"

        # Tokenize the input to get its length (for statistics)
        input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)

        processed.append({
            "ID": doc_id,
            "judgment_text": input_text,
            "summary": summary_text,
            "input_length": len(input_ids)
        })

    # --- 4. Save Processed Data ---
    if output_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_jsonl(processed, output_path)
        # Print summary statistics
        print(f"✅ Preprocessed {len(processed)} examples and saved to {output_path}")
        avg_len = sum(p['input_length'] for p in processed) / len(processed)
        max_len = max(p['input_length'] for p in processed)
        print(f"   - Avg input length: {avg_len:.1f} tokens")
        print(f"   - Max input length: {max_len} tokens")

    return processed

if __name__ == "__main__":
    # --- 5. Argument Parsing ---
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

    # Process the training dataset
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

    # Process the validation dataset
    if os.path.exists(args.val_judg_path):
        print("\n[2/2] Processing VALIDATION data...")
        # Handle cases where validation summaries might not be available
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