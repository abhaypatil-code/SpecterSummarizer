import json
from typing import List, Dict, Any, Iterator

def load_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Loads a JSONL file line by line as a generator to save memory.
    """
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line.strip())

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves a list of dictionaries to a JSONL file, ensuring valid UTF-8 encoding.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False is important for non-English characters
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_predictions(predictions: List[str], output_path: str) -> None:
    """
    Saves a list of prediction strings to a plain text file. 
    (Note: This is kept for compatibility but the evaluation script now uses save_jsonl).
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred.strip() + '\n')