"""Small I/O and text helpers shared across the project."""
import json
import re
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str) -> Iterator[dict]:
    """Yield one dict per line of a JSONL file (tolerates BOM and blank lines)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(rows: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip control characters from extracted text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(file) -> str:
    """Extract plain text from a PDF (path or file-like object)."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))
