#!/usr/bin/env bash
# SpecterSummarizer - one-click launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  SpecterSummarizer"
echo "============================================================"

if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .venv
fi
PY=".venv/bin/python"

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
    echo "[setup] Installing dependencies (first run only)..."
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements.txt
fi

if [ ! -f "data/test_processed.jsonl" ]; then
    echo "[setup] Building data splits..."
    "$PY" -m src.preprocess
fi

if [ ! -f "models/t5_summarizer/config.json" ]; then
    echo "[setup] No fine-tuned model found. Training a quick demo model..."
    echo "        (For full quality: python -m src.train --epochs 3 --max_samples 0)"
    "$PY" -m src.train --epochs 1 --max_samples 100
fi

echo "[run] Starting SpecterSummarizer..."
exec "$PY" -m streamlit run app.py
