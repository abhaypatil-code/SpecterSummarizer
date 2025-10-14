import json
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file into a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a line in the JSONL file.
    """
    data = []
    # The encoding is changed to 'utf-8-sig' to handle the BOM character
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            # .strip() handles leading/trailing whitespace and blank lines
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves a list of dictionaries to a JSONL file.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to save.
        file_path (str): The path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False is important for handling non-English characters
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_predictions(predictions: List[str], output_path: str) -> None:
    """
    Saves a list of prediction strings to a plain text file, with one
    prediction per line.

    Args:
        predictions (List[str]): A list of prediction strings.
        output_path (str): The path to the output text file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred.strip() + '\n')