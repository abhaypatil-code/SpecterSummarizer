# ⚖️ SpecterSummarizer — Fine-Tuned LLM for Document Intelligence

> Local AI for summarizing Indian court judgments — upload a PDF, generate a
> concise legal summary, and chat with the document. **Runs fully offline. No
> external APIs.**

SpecterSummarizer is a compact, end-to-end NLP project: a fine-tuned **Hugging
Face T5** transformer produces abstractive summaries of legal documents, and a
**MiniLM + FAISS** retrieval pipeline lets you ask questions about the uploaded
document. Training supports **mixed-precision (FP16)** on GPU, and a ROUGE-L
evaluation harness measures the lift over the pretrained baseline. Everything is
served through a clean **Streamlit** interface and runs on CPU for the demo.

---

## ✨ Features

- **📄 PDF upload & preview** — extract and inspect text from judgment PDFs.
- **📝 Abstractive summarization** — fine-tuned `t5-small` generates concise
  summaries with adjustable length.
- **💬 Document chat** — ask questions and get answers grounded in the
  judgment, via local semantic retrieval (no hallucination, no LLM download).
- **🔒 100% local** — open-source models only; works without an internet
  connection once models are cached.

---

## 🏗️ Architecture

```
                ┌──────────────────────────── Streamlit (app.py) ────────────────────────────┐
   PDF  ─────▶  │  extract_pdf_text ─▶ clean_text ─▶ [ Summary ] [ Chat ] [ Full text ] tabs  │
                └───────────────┬───────────────────────────────────┬─────────────────────────┘
                                │                                   │
                   ┌────────────▼────────────┐         ┌────────────▼─────────────┐
                   │  src/inference.py        │         │  src/rag_chat.py          │
                   │  fine-tuned T5 summarizer │         │  MiniLM embeddings + FAISS │
                   │  (falls back to t5-small) │         │  extractive Q&A over chunks│
                   └──────────────────────────┘         └───────────────────────────┘

   Training:  data/*.jsonl ─▶ src/preprocess.py ─▶ src/train.py ─▶ models/t5_summarizer/
```

📐 **Detailed diagrams** (flow, sequence, training pipeline) are in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📁 Project structure

```
SpecterSummarizer/
├── app.py                 # Streamlit app (single entry point)
├── requirements.txt
├── README.md
├── data/                  # raw paired judgment/summary JSONL
│   ├── train_judg.jsonl
│   └── train_ref_summ.jsonl
├── models/                # fine-tuned model lands here (gitignored)
├── assets/                # demo screenshots
├── run.bat / run.sh       # one-click launcher (setup + run app)
└── src/
    ├── utils.py           # JSONL I/O, text cleaning, PDF extraction
    ├── preprocess.py      # build train/val/test splits
    ├── train.py           # fine-tune t5-small (FP16 on GPU)
    ├── evaluate.py        # ROUGE-L / ROUGE-2 / BLEU vs baseline
    ├── inference.py       # load model + summarize()
    └── rag_chat.py        # MiniLM + FAISS document chat
```

---

## 🚀 Setup

```bash
git clone https://github.com/abhaypatil-code/SpecterSummarizer.git
cd SpecterSummarizer

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> The first run downloads `t5-small` and `all-MiniLM-L6-v2` from Hugging Face
> (a few hundred MB total) and caches them locally for offline use.

---

## 🎬 Run the demo

**One click:** double-click **`run.bat`** (Windows) or run **`./run.sh`**
(macOS/Linux). It creates a virtual environment, installs dependencies, builds
the data splits, trains a quick demo model if none exists, and opens the app.

**Or manually:**

```bash
streamlit run app.py
```

Then in the browser:

1. **Upload a judgment PDF** in the sidebar, or click **Use sample judgment**.
2. Open the **Summary** tab, pick a length, and click **Generate summary**.
3. Open the **Chat** tab and ask questions like *"What did the court decide?"*

If no fine-tuned model exists, the app automatically falls back to the base
`t5-small` so it still runs — train a model (below) for higher-quality summaries.

---

## 🧠 Train your own summarizer

1. **Build the splits** (80/10/10, reproducible):

   ```bash
   python -m src.preprocess
   ```

   Produces `data/{train,val,test}_processed.jsonl`.

2. **Fine-tune** `t5-small`:

   ```bash
   # Fast CPU demo run (small subset, 1 epoch)
   python -m src.train --epochs 1 --max_samples 60

   # Full run (use all data; a GPU is strongly recommended)
   python -m src.train --epochs 3 --max_samples 0
   ```

   The model is saved to `models/t5_summarizer/`, which `inference.py` picks up
   automatically on the next app run. On a CUDA GPU, **FP16 mixed-precision**
   training is enabled automatically.

   Key flags: `--model_name` (e.g. `t5-base`), `--batch_size`, `--grad_accum`,
   `--learning_rate`, `--max_samples` (`0` = all examples).

3. **Evaluate** against the pretrained baseline (ROUGE-L / ROUGE-2):

   ```bash
   python -m src.evaluate --compare-baseline
   ```

   Reports the fine-tuned model's scores and the ROUGE-L improvement over the
   `t5-small` baseline.

---

## 🔧 How it works

- **Summarization** — inputs are prefixed with `summarize: ` and decoded with
  beam search (`num_beams=4`, length penalty, no-repeat n-grams) for fluent,
  non-repetitive summaries.
- **Chat (RAG)** — the judgment is split into overlapping word windows, embedded
  with `all-MiniLM-L6-v2`, and indexed in FAISS. A question is embedded and the
  most similar passages are returned, so answers stay grounded in the source.

---

## 📸 Screenshots

| Summary | Chat |
| ------- | ---- |
| ![Summary](assets/summary.png) | ![Chat](assets/chat.png) |

---

## 🔮 Future improvements

- Generative (not just extractive) chat with a small local instruct model.
- ROUGE/BLEU evaluation dashboard in the UI.
- Long-document handling via hierarchical / chunked summarization.
- Section-aware summaries (facts, issues, holding, reasoning).

---

## 📄 License

See [LICENSE](LICENSE).
