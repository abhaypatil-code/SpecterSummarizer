import json
from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    data = []
    # The encoding is changed to 'utf-8-sig' to handle the BOM character
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_predictions(predictions: List[str], output_path: str):
    """Save predictions as plain text (one per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred.strip() + '\n')