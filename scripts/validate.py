import os
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from utils import load_jsonl

def validate_manually(
    model_path: str,
    # --- FIX for Hardcoded Tokenizer Length ---
    # Added 'max_input_length' to parameterize the tokenizer length.
    max_input_length: int = 1024,
    max_output_length: int = 256,
    num_beams: int = 4
):
    """
    Allows for manual, interactive validation of the fine-tuned T5 model.
    You can paste a legal judgment, and the model will generate a summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Tokenizer ---
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        model.eval()
    except OSError:
        print(f"❌ ERROR: Model or tokenizer not found at '{model_path}'.")
        print("   - Did you run the training script to fine-tune a model?")
        return

    print("\n" + "="*80)
    print("⚖️ MANUAL MODEL VALIDATION")
    print("="*80)
    print("   - Enter a legal judgment text below.")
    print("   - Type 'exit' or 'quit' to end the session.")
    print("="*80 + "\n")

    # --- 2. Interactive Loop ---
    while True:
        try:
            judgment_text = input("Enter Judgment Text >> ")
            if judgment_text.lower() in ["exit", "quit"]:
                break
            if not judgment_text.strip():
                print("   - Please enter some text.")
                continue

            # --- 3. Preprocess and Generate ---
            # Apply the same "summarize: " prefix used during training
            input_text = f"summarize: {judgment_text}"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                # --- FIX for Hardcoded Tokenizer Length ---
                # Use the parameterized 'max_input_length'.
                max_length=max_input_length
            ).to(device)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=max_output_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # --- 4. Display Results ---
            print("\n" + "-"*80)
            print("GENERATED SUMMARY:")
            print(summary)
            print("-"*80 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            break
            
    print("\n👋 Exiting validation session.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually validate the summarization model.")
    parser.add_argument("--model_path", type=str, default="outputs/t5_summarizer", help="Path to the fine-tuned model directory.")
    # --- FIX for Hardcoded Tokenizer Length ---
    # Add the argument to control max input length during validation.
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum token length for the input judgment, aligned with preprocessing.")
    parser.add_argument("--max_output_length", type=int, default=256, help="Maximum length for the generated summary.")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search.")
    args = parser.parse_args()

    validate_manually(
        model_path=args.model_path,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        num_beams=args.num_beams
    )