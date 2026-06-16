"""Evaluate a summarizer with ROUGE-L / ROUGE-2 / BLEU on the test set.

Compares the fine-tuned model against the pretrained baseline so you can report
the improvement (e.g. ROUGE-L lift over baseline).

  python -m src.evaluate                       # fine-tuned model only
  python -m src.evaluate --compare-baseline    # fine-tuned vs base t5-small
"""
import argparse

import torch
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.inference import BASE_MODEL, MAX_INPUT_TOKENS
from src.utils import load_jsonl


def _generate(model, tokenizer, text, min_length, max_length):
    inputs = tokenizer(
        "summarize: " + text.strip(),
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            min_length=min_length,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def _score(model_name, rows, min_length, max_length):
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)

    r2, rl = [], []
    for row in rows:
        pred = _generate(model, tokenizer, row["judgment_text"], min_length, max_length)
        s = scorer.score(row["summary"], pred)
        r2.append(s["rouge2"].fmeasure * 100)
        rl.append(s["rougeL"].fmeasure * 100)
    return {"rouge2": sum(r2) / len(r2), "rougeL": sum(rl) / len(rl)}


def main(args):
    rows = [r for r in load_jsonl(args.test_file) if r.get("summary", "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating on {len(rows)} examples...\n")

    tuned = _score(args.model_path, rows, args.min_length, args.max_length)
    print(f"Fine-tuned ({args.model_path})")
    print(f"  ROUGE-2: {tuned['rouge2']:.2f}   ROUGE-L: {tuned['rougeL']:.2f}")

    if args.compare_baseline:
        base = _score(BASE_MODEL, rows, args.min_length, args.max_length)
        lift = tuned["rougeL"] - base["rougeL"]
        rel = (lift / base["rougeL"] * 100) if base["rougeL"] else 0.0
        print(f"\nBaseline ({BASE_MODEL})")
        print(f"  ROUGE-2: {base['rouge2']:.2f}   ROUGE-L: {base['rougeL']:.2f}")
        print(f"\nROUGE-L improvement over baseline: {lift:+.2f} ({rel:+.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ROUGE/BLEU evaluation for the summarizer.")
    p.add_argument("--model_path", default="models/t5_summarizer")
    p.add_argument("--test_file", default="data/test_processed.jsonl")
    p.add_argument("--min_length", type=int, default=60)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Cap examples (0 = all).")
    p.add_argument("--compare-baseline", action="store_true", dest="compare_baseline")
    args = p.parse_args()
    if args.limit == 0:
        args.limit = None
    main(args)
