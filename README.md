# ⚖️ Legal Judgment Summarization Model

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a T5-based model for summarizing legal judgments. The model is fine-tuned on a dataset of legal documents to generate concise and accurate summaries.

---

## 📜 Description

The model is built using the Hugging Face `transformers` library and is based on the **T5 (Text-to-Text Transfer Transformer)** architecture. It has been trained on a dataset of legal judgments to perform abstractive summarization. This means it can generate new sentences to summarize the input text, rather than just extracting existing ones, which is particularly useful for long and complex legal documents.

---

## ✨ Features

* **T5-based Summarization**: Leverages the powerful T5 model for abstractive summarization.
* **Modular Scripts**: The project is organized into clear and distinct Python scripts for each stage of the machine learning pipeline:
  * `preprocess.py`: For cleaning and preparing the dataset.
  * `train.py`: For fine-tuning the T5 model.
  * `run_evaluation.py`: For generating summaries and evaluating model performance.
* **Customizable Hyperparameters**: Easily configure training parameters via a `hyperparams.json` file.
* **Hyperparameter Tuning**: Includes a `tune.py` script using the Optuna library to automatically find optimal hyperparameters.

---

## 📁 Project Structure

```
SpecterSummarizer/
├── data/               # Dataset directory
├── outputs/            # Model outputs and predictions
├── scripts/            # Training and evaluation scripts
├── venv_legal/         # Virtual environment
├── .gitattributes      # Git attributes configuration
├── .gitignore          # Git ignore file
├── create.py           # Creation script
├── hyperparams.json    # Hyperparameter configuration
├── requirements.txt    # Project dependencies
└── tune.py             # Hyperparameter tuning script
```

---

## 🚀 Complete Pipeline - Start to Finish

Here are all the steps to run your complete summarization pipeline from start to finish.

### Step 1: 🏗️ Project Setup & Installation

First, set up your project structure and install the necessary libraries.

**1. Create the Folder Structure:**

Run the `create.py` script to automatically generate the required directories (`data`, `outputs`, etc.).

```bash
python create.py
```

**2. Place Your Data:**

Move your raw dataset files into the `data/` directory. The script expects the following files:
- `data/train_judg.jsonl`
- `data/train_ref_summ.jsonl`
- `data/val_judg.jsonl`


**3. Create and Activate a Virtual Environment:**

It's best practice to isolate your project's dependencies.

```bash
# Create a virtual environment
python -m venv venv_legal

# Activate the environment
# On macOS/Linux:
source venv_legal/bin/activate
# On Windows:
.\venv_legal\Scripts\activate
```

**4. Install Dependencies:**

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

### Step 2: 🧹 Preprocess the Data

Next, run the preprocessing script to combine judgments with their summaries and prepare them for the model. This will create `train_processed.jsonl` and `val_processed.jsonl` inside the `data/` folder. The following command uses the `t5-base` tokenizer and sets the maximum input length to 1024 tokens.

```bash
python scripts/preprocess.py \
    --train_judg_path data/train_judg.jsonl \
    --train_summ_path data/train_ref_summ.jsonl \
    --val_judg_path data/val_judg.jsonl \
    --val_summ_path data/val_ref_summ.jsonl \
    --output_dir data \
    --tokenizer_name t5-base \
    --max_input_length 1024
```

---

### Step 3: 🧪 Tune Hyperparameters (Recommended)

Before full training, run the Optuna tuning script to automatically find the best hyperparameters (like learning rate and batch size). This will create a `hyperparams.json` file with the optimal settings. You can adjust the number of trials.

```bash
python tune.py --n_trials 20
```

---

### Step 4: 🏋️ Train the Model

Now, use the main training script to fine-tune the T5 model on your preprocessed data. This script will automatically load the best settings from the `hyperparams.json` file.

```bash
python scripts/train.py
```

---

### Step 5: 📊 Evaluate the Model

Finally, use your newly trained model to generate summaries for the validation set and calculate ROUGE and BLEU scores. The predictions will be saved in `outputs/predictions/`.

```bash
python scripts/run_evaluation.py
```

---

### (Optional) Step 6: 👀 Manually Validate Results

For a qualitative check, you can run the validation script to see a few examples of the model's generated summaries compared to the human-written references.

```bash
python scripts/validate.py
```

---

## 📦 Dependencies

The required Python libraries are listed in the `requirements.txt` file. They include:

- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `optuna`
- `tqdm`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/abhaypatil-code/SpecterSummarizer/issues).

---
