"""Prepare a persistent Colab workspace without replacing student work."""
from pathlib import Path
import importlib.util
import re
import shutil
import subprocess
import sys


def setup(student_id: str) -> tuple[Path, Path]:
    """Return the repository root and submission directory."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id) or student_id == "YOUR_STUDENT_ID":
        raise ValueError("Replace YOUR_STUDENT_ID with your actual student ID.")
    root = Path("/content/K4-Track4-Day1-Neuralnetwork")
    if not root.exists():
        subprocess.run([
            "git", "clone", "https://github.com/VinUni-AI20k/K4-Track4-Day1-Neuralnetwork.git",
            str(root),
        ], check=True)
    for name in ("numpy", "pandas", "sklearn", "matplotlib", "openpyxl", "torch"):
        if importlib.util.find_spec(name) is None:
            package = "scikit-learn" if name == "sklearn" else name
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
    saved = Path("/content/drive/MyDrive/NeuralNetworkLab") / f"submission_{student_id}"
    (saved / "code").mkdir(parents=True, exist_ok=True)
    for source in (root / "code").iterdir():
        target = saved / "code" / source.name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    for source, target in (
        (root / "templates/REPORT_TEMPLATE.md", saved / "REPORT.md"),
        (root / "templates/experiment_table_template.xlsx", saved / "experiments.xlsx"),
        (root / "templates/experiment_table_template.xlsx", saved / "code/experiment_table_template.xlsx"),
    ):
        if not target.exists():
            shutil.copy2(source, target)
    for name in ("figures", "results"):
        (saved / name).mkdir(exist_ok=True)
    submission = root / saved.name
    if not submission.exists():
        submission.symlink_to(saved, target_is_directory=True)
    elif submission.resolve() != saved.resolve():
        raise RuntimeError(f"Another workspace already exists at {submission}.")
    processed = root / "data/processed"
    if not all((processed / name).exists() for name in ("train.npz", "eval.npz")):
        subprocess.run([sys.executable, "scripts/split_data.py"], cwd=root, check=True)
    return root, submission
