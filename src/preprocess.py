"""Build reproducible train/val/test splits from raw judgment + summary files.

Inputs (matched by "ID"):
  data/train_judg.jsonl       -> {"ID", "Judgment"}
  data/train_ref_summ.jsonl   -> {"ID", "Summary"}

Outputs:
  data/{train,val,test}_processed.jsonl -> {"ID", "judgment_text", "summary"}
"""
import argparse
import random
from pathlib import Path

from src.utils import clean_text, load_jsonl, save_jsonl


def build_splits(judg_path: str, summ_path: str, out_dir: str, seed: int = 42):
    judgments = {r["ID"]: r["Judgment"] for r in load_jsonl(judg_path)}
    summaries = {r["ID"]: r["Summary"] for r in load_jsonl(summ_path)}

    ids = sorted(set(judgments) & set(summaries))
    if not ids:
        raise ValueError("No matching IDs between judgment and summary files.")

    rows = [
        {
            "ID": i,
            "judgment_text": clean_text(judgments[i]),
            "summary": clean_text(summaries[i]),
        }
        for i in ids
        if summaries[i].strip()
    ]

    random.Random(seed).shuffle(rows)
    n = len(rows)
    train_end, val_end = int(n * 0.8), int(n * 0.9)
    splits = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        save_jsonl(split, f"{out_dir}/{name}_processed.jsonl")
        print(f"  {name}: {len(split)} examples")
    print(f"Done. {n} total pairs -> {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split raw data into train/val/test.")
    p.add_argument("--judg_path", default="data/train_judg.jsonl")
    p.add_argument("--summ_path", default="data/train_ref_summ.jsonl")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_splits(args.judg_path, args.summ_path, args.out_dir, args.seed)
