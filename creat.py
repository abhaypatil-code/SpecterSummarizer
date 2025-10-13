import os
import json
import venv

# === Base project directory ===
base_dir = os.getcwd()

# === Folder structure ===
folders = [
    "data",
    "scripts",
    "outputs",
    os.path.join("outputs", "model"),
    os.path.join("outputs", "predictions"),
    os.path.join("outputs", "logs"),
]

# === File placeholders ===
files = [
    os.path.join("data", "train_judg.jsonl"),
    os.path.join("data", "train_ref_summ.jsonl"),
    os.path.join("data", "val_judg.jsonl"),
    os.path.join("data", "val_ref_summ.jsonl"),
    os.path.join("data", "toy_dataset.jsonl"),
    os.path.join("scripts", "preprocess.py"),
    os.path.join("scripts", "train.py"),
    os.path.join("scripts", "evaluate.py"),
    os.path.join("scripts", "utils.py"),
    "hyperparams.json",
    "requirements.txt"
]

# === Create folders ===
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)
    print(f"[+] Created folder: {path}")

# === Create empty placeholder files ===
for file in files:
    path = os.path.join(base_dir, file)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if file.endswith("hyperparams.json"):
                json.dump({
                    "learning_rate": 3e-5,
                    "batch_size": 8,
                    "num_epochs": 3,
                    "max_length": 512
                }, f, indent=4)
            elif file.endswith("requirements.txt"):
                f.write(
                    "transformers\n"
                    "datasets\n"
                    "evaluate\n"
                    "torch\n"
                    "tqdm\n"
                    "sacrebleu\n"
                    "rouge_score\n"
                )
            else:
                f.write("")  # empty placeholder
        print(f"[+] Created file: {path}")

# === Create virtual environment (Python 3.11) ===
venv_path = os.path.join(base_dir, "venv_legal")
if not os.path.exists(os.path.join(venv_path, "Scripts", "python.exe")):
    print("\n[!] Creating virtual environment (Python 3.11)...")
    venv.create(venv_path, with_pip=True)
    print(f"[+] Virtual environment created at {venv_path}")
else:
    print(f"[=] Virtual environment already exists at {venv_path}")

print("\n✅ Directory structure setup complete!")