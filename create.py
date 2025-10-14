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
    os.path.join("data", "val_ref_summ.jsonl"),  # Added placeholder for missing file
    os.path.join("data", "toy_dataset.jsonl"),
    os.path.join("scripts", "preprocess.py"),
    os.path.join("scripts", "train.py"),
    os.path.join("scripts", "run_evaluation.py"), # Corrected filename
    os.path.join("scripts", "utils.py"),
    os.path.join("scripts", "validate.py"),
    "hyperparams.json",
    "requirements.txt",
    "tune.py",
    ".gitignore"
]

# === Create folders ===
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# === Create empty files ===
for file in files:
    with open(os.path.join(base_dir, file), "w") as f:
        if file.endswith(".json"):
            json.dump({}, f)
        elif file.endswith(".jsonl"):
            pass  # Create empty jsonl files
        else:
            f.write("")

# === Create a virtual environment ===
# venv.create("venv_legal", with_pip=True)

print("Project structure created successfully.")