"""Load a T5 summarizer and generate legal summaries (CPU-friendly)."""
from functools import lru_cache
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_DIR = "models/t5_summarizer"
BASE_MODEL = "t5-small"
MAX_INPUT_TOKENS = 1024


def _resolve_model() -> str:
    """Use the fine-tuned model if it exists, otherwise the pretrained base."""
    return MODEL_DIR if Path(MODEL_DIR, "config.json").exists() else BASE_MODEL


@lru_cache(maxsize=1)
def load_summarizer():
    """Load tokenizer + model once and cache them for the session."""
    name = _resolve_model()
    tokenizer = T5TokenizerFast.from_pretrained(name)
    model = T5ForConditionalGeneration.from_pretrained(name).eval()
    return tokenizer, model, name


def summarize(text: str, min_length: int = 60, max_length: int = 256) -> str:
    """Generate an abstractive summary for a legal judgment."""
    tokenizer, model, _ = load_summarizer()
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
