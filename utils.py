import json
from typing import List, Dict, Any, Iterator
import os

def load_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Loads a JSONL file line by line as a generator to save memory.
    Handles UTF-8 with BOM and empty lines gracefully.
    
    Args:
        file_path: Path to the JSONL file
        
    Yields:
        Dictionary for each valid JSON line
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON at line {line_num} in {file_path}: {e.msg}",
                        e.doc,
                        e.pos
                    )

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves a list of dictionaries to a JSONL file, ensuring valid UTF-8 encoding.
    Creates parent directories if they don't exist.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where the JSONL file will be saved
        
    Raises:
        ValueError: If data is empty or not a list
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")
    
    if not data:
        raise ValueError("Data list is empty. Nothing to save.")
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"All items must be dictionaries, got {type(item)}")
            # ensure_ascii=False is important for non-English characters
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_predictions(predictions: List[str], output_path: str) -> None:
    """
    Saves a list of prediction strings to a plain text file.
    Creates parent directories if they don't exist.
    
    Args:
        predictions: List of prediction strings
        output_path: Path where predictions will be saved
        
    Raises:
        ValueError: If predictions is empty or not a list
    """
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of strings")
    
    if not predictions:
        raise ValueError("Predictions list is empty. Nothing to save.")
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            if not isinstance(pred, str):
                raise ValueError(f"All predictions must be strings, got {type(pred)}")
            f.write(pred.strip() + '\n')

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Saves a dictionary to a JSON file.
    Creates parent directories if they don't exist.
    
    Args:
        data: Dictionary to save
        file_path: Path where the JSON file will be saved
        indent: Number of spaces for indentation
        
    Raises:
        ValueError: If data is not a dictionary
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
