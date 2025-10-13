import os
import json
import venv

# === Base project directory ===
base_dir = r"D:\Software\JustNLP"

# === Folder structure ===
folders = [
    "venv_legal",              # Virtual environment
    "data",
    "scripts",
    "outputs",
    os.path.join("outputs", "model"),
    os.path.join("outputs", "predictions"),
    os.path.join("outputs", "logs"),
]

# === File placeholders ===
files = [
    "train_judg.jsonl",
    "train_ref_summ.jsonl",
    "val_judg.jsonl",
    os.path.join("data", "toy_dataset.jsonl"),
    os.path.join("scripts", "preprocess.py"),
    os.path.join("scripts", "train.py"),
    os.path.join("scripts", "evaluate.py"),
    os.path.join("scripts", "utils.py"),
    "hyperparams.json"
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
