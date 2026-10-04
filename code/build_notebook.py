"""Build a standalone Colab notebook containing the completed implementation bundle."""
from pathlib import Path
import base64
import io
import json
import zipfile

CODE = Path(__file__).resolve().parent
FILES = ["data.py", "model.py", "optimizer.py", "train.py", "plots.py", "results_table.py",
         "experiments.py", "report.py", "submission.py", "run_lab.py", "requirements.txt",
         "experiment_table_template.xlsx", "RUNNING.md"]


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            # Stable timestamps make regeneration produce a reproducible payload.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (CODE / name).read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode()
    environment = """
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

if not re.fullmatch(r"[A-Za-z0-9_-]+", STUDENT_ID):
    raise ValueError("Invalid student ID.")
try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_ROOT = Path("/content/K4-Track4-Day1-Neuralnetwork")
    if not REPO_ROOT.exists():
        subprocess.run(["git", "clone", "https://github.com/VinUni-AI20k/K4-Track4-Day1-Neuralnetwork.git",
                        str(REPO_ROOT)], check=True)
    OUT_DIR = Path("/content/drive/MyDrive/NeuralNetworkLab") / f"submission_{STUDENT_ID}"
else:
    current = Path.cwd().resolve()
    REPO_ROOT = next((p for p in [current, *current.parents]
                      if (p / "data/covtype.csv.gz").exists() and (p / "scripts/split_data.py").exists()), None)
    if REPO_ROOT is None:
        raise RuntimeError("Run from this repository or a submission/code directory.")
    OUT_DIR = REPO_ROOT / f"submission_{STUDENT_ID}"
CODE_DIR = OUT_DIR / "code"
CODE_DIR.mkdir(parents=True, exist_ok=True)
# This bundle contains the completed local modules, not the upstream scaffold.
BUNDLE = __PAYLOAD__
if INSTALL_BUNDLED_CODE:
    backup_dir = OUT_DIR.parent / ".lab_backups" / OUT_DIR.name / time.strftime("%Y%m%d-%H%M%S")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
        for name in archive.namelist():
            if Path(name).name != name:
                raise ValueError("Unsafe bundle member.")
            target, content = CODE_DIR / name, archive.read(name)
            if target.exists() and target.read_bytes() != content:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_dir / name)
                print("Backed up:", target.name)
            if not target.exists() or target.read_bytes() != content:
                target.write_bytes(content)
for module, package in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                        ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl"), ("torch", "torch")):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
os.chdir(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
for name in ("data", "model", "optimizer", "train", "plots", "results_table", "experiments", "report", "submission", "run_lab"):
    sys.modules.pop(name, None)
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown, Image
from data import prepare_data
from train import supports_bf16
from experiments import LabRunner, health_checks, lr_trials, best_by_validation, seed_statistics, topic_trials, PREDICTIONS
from results_table import write_json
from report import observation
from submission import finalize
from run_lab import copy_code

device = "cuda" if torch.cuda.is_available() else "cpu"
if IN_COLAB and device != "cuda":
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU, then run all again.")
if device == "cpu":
    torch.set_num_threads(1)
print("PyTorch:", torch.__version__, "Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "Native BF16 supported:", supports_bf16())
print("Persistent output:", OUT_DIR)
""".replace("__PAYLOAD__", repr(payload))
    cells = [
        cell("markdown", """
# Neural network lab - 2A202603012

Completed custom PyTorch implementation, following Parts 0-4 of the lab.

On Colab select **Runtime > Change runtime type > T4 GPU**, then **Run all**. Approve the Drive mount.
This notebook installs its bundled code in `MyDrive/NeuralNetworkLab/submission_2A202603012/code/`.
Existing changed modules are backed up outside the submission. No GitHub push or separate module upload is needed.
The common experiment budget is 20 epochs. Restart & Run All preserves submitted results and trains in a
separate directory when checkpoints are absent or the environment differs. Unfinished compatible runs
can reuse completed checkpoints.

Predictions below are hypotheses written before training. Observations and the report draft are generated from measured outputs.
Review the explanations and add your own conclusions before submission. Final selection uses validation only.
"""),
        cell("code", """
# Change these settings before the first run, not after viewing eval scores.
STUDENT_ID = "2A202603012"
EPOCHS = 20
INSTALL_BUNDLED_CODE = True
"""),
        cell("code", environment),
        cell("markdown", """
## Part 0 - Data

Use the supplied train/eval metadata without changing it. Split 20% of the official training rows into validation,
stratified by class with seed 42. Fit mean and population standard deviation using the remaining training rows only.
Standardize columns 0-9 and preserve the 44 binary columns. Eval is reserved for Part 4.
"""),
        cell("code", """
processed = REPO_ROOT / "data/processed"
if not all((processed / name).exists() for name in ("train.npz", "eval.npz")):
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/split_data.py")], cwd=REPO_ROOT, check=True)
data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir=str(processed))
assert tuple(data["X_tr"].shape) == (371847, 54)
assert tuple(data["X_val"].shape) == (92962, 54)
assert tuple(data["X_eval"].shape) == (116203, 54)
assert data["X_tr"].dtype == torch.float32 and data["y_tr"].dtype == torch.int64
numeric = data["X_tr"][:, :10].double()
assert torch.allclose(numeric.mean(0), torch.zeros(10, dtype=torch.float64, device=device), atol=1e-5)
assert torch.allclose(numeric.std(0, unbiased=False), torch.ones(10, dtype=torch.float64, device=device), atol=1e-5)
print("Train numeric means:", numeric.mean(0).cpu().numpy())
print("Train numeric standard deviations:", numeric.std(0, unbiased=False).cpu().numpy())

# Route training before diagnostics write any files, preserving submitted evidence.
runner = LabRunner(data, REPO_ROOT, OUT_DIR)
OUT_DIR = runner.out
CODE_DIR = OUT_DIR / "code"
copy_code(CODE_DIR)
os.chdir(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
print("Training output:", OUT_DIR)
"""),
        cell("markdown", """
## Part 1 - Model and health checks

M-base is 54-256-128-7 with 47,879 trainable parameters. Hidden layers use ReLU; the output is raw logits.
Check shape, initial CE against ln(7), nonzero gradients in every parameter, and overfitting 20 examples.
An initial CE near ln(7) is a diagnostic expectation, not an exact assertion for randomly initialized logits.
"""),
        cell("code", """
diagnostics = health_checks(data, OUT_DIR)
display(Image(filename=str(OUT_DIR / "figures/diagnostic_overfit.png")))
display(Markdown(
    f"Initial CE was **{diagnostics['step0_loss']:.6f}**, compared with ln(7)={diagnostics['ln7']:.6f}. "
    f"The 20-example check achieved CE={diagnostics['tiny_loss']:.6f} and accuracy={diagnostics['tiny_acc']:.6f}. "
    "Every parameter had a finite nonzero gradient. Random initial logit variance explains deviations from uniform CE."
))
"""),
        cell("markdown", """
## Part 2 - Baseline and noise

**Prediction before training:** validation will identify a stable SGD+momentum learning rate. Repeat the chosen
baseline at seeds 2 and 3 to estimate training noise, keeping the split fixed.

Test rates 0.01, 0.03 and 0.1 using the full common 20-epoch budget. Every trial has a unique result and figure.
Select checkpoints by minimum validation loss, then compare validation macro-F1 at those checkpoints.
Train loss uses the same fixed 50,000-row subset in eval mode for every experiment. Full validation is evaluated in FP32.
"""),
        cell("code", """
baseline_trials = [runner.run(cfg) for cfg in lr_trials(EPOCHS)]
baseline = best_by_validation(baseline_trials)
print("Selected baseline rate:", baseline["cfg"]["lr"], "ID:", baseline["cfg"]["exp_id"])
baseline_seeds = [baseline]
for seed in (2, 3):
    baseline_seeds.append(runner.run({**baseline["cfg"], "seed": seed,
        "exp_id": f"base-s{seed}", "description": f"Selected baseline, seed {seed}"}))
seed_stats = seed_statistics(baseline_seeds)
write_json(OUT_DIR / "seed_statistics.json", seed_stats)
display(pd.DataFrame(seed_stats).T)
"""),
        cell("code", """
h = baseline["history"]
display(Markdown(
    f"Baseline train loss changed from {h['train_loss'][0]:.6f} to {h['train_loss'][-1]:.6f}; "
    f"validation loss changed from {h['val_loss'][0]:.6f} to {h['val_loss'][-1]:.6f}. "
    f"Best-checkpoint accuracy={baseline['summary']['val_acc']:.6f}; majority reference={data['majority_val_acc']:.6f}. "
    f"Baseline 2-sigma F1 reference={seed_stats['val_macro_f1']['noise_2sigma']:.6f}. "
    "Use the curves to assess convergence and overfitting; the noise reference is descriptive."
))
display(Image(filename=str(OUT_DIR / "figures" / f"{baseline['cfg']['exp_id']}.png")))
"""),
        cell("markdown", """
## Part 3 - Controlled experiments

**Predictions before running:**

- MSE changes gradient geometry and loss scale; compare scores rather than CE/MSE loss magnitudes.
- Adam may converge faster when each optimizer is tuned separately (Adam rates 0.0003 and 0.001).
- Batch 128 has about four times as many updates per epoch, potentially improving F1 at a higher time cost.
- Dropout 0.3 may reduce overfitting, but may hurt a model still underfitting.
- Clipping should limit spikes; a threshold derived from baseline norms and a matched 10x-rate pair test stability.
- FP16 may preserve F1 while changing speed and memory; gains must be measured. BF16 is skipped if unsupported.
- Xavier changes activation scale. Zero initialization blocks hidden ReLU gradients and should perform poorly.

All trials inherit the selected baseline and seed 1. Adam changes optimizer and learning rate deliberately for fair tuning.
`highlr-clip` is compared with `highlr-no-clip`, which differs only in clipping. Clipping does not guarantee recovery.
"""),
        cell("code", """
noise = seed_stats["val_macro_f1"]["noise_2sigma"]
for cfg in topic_trials(baseline, torch.device(device)):
    display(Markdown(f"### {cfg['exp_id']}\\n\\nPrediction: {PREDICTIONS[cfg['group']]}"))
    result = runner.run(cfg)
    reference = runner.results.get(cfg.get("reference"), baseline)
    if reference["summary"]["diverged"]:
        reference = baseline
    display(Markdown("Observation: " + observation(result, reference, noise)))
    display(Image(filename=str(OUT_DIR / "figures" / f"{cfg['exp_id']}.png")))
if not supports_bf16():
    print("BF16 skipped: GPU support is unavailable. FP16 remains the precision comparison.")
runner.comparisons(baseline)
"""),
        cell("markdown", """
## Part 4 - Final eval, workbook and report

Freeze the final configuration and seed before reading eval scores. Final ranking compares seed 1 configurations
by validation macro-F1 at their minimum-validation-loss checkpoints. The official evaluator scores only the designated
baseline and final configuration. No tuning follows evaluation. The generated report is a measured draft for your review.
"""),
        cell("code", """
final_outputs = finalize(runner, baseline, baseline_seeds, seed_stats, diagnostics, STUDENT_ID)
final_scores = final_outputs["scores"][final_outputs["selection"]["final"]]
display(pd.DataFrame(final_scores["per_class"]))
display(Image(filename=str(OUT_DIR / "figures/eval_confusion.png")))
worst = min(final_scores["per_class"], key=lambda item: item["f1"])
cm = np.asarray(final_scores["confusion_matrix"])
errors = cm[worst["cls"]].copy()
errors[worst["cls"]] = 0
confusable = f"class {int(errors.argmax())}" if errors.max() else "no other class"
display(Markdown(
    f"The lowest class F1 is class {worst['cls']} ({worst['f1']:.6f}); it is most often confused with {confusable}. "
    "Imbalance and overlapping features are plausible explanations, which need feature analysis to verify. "
    "The report draft includes measured comparisons, noise limitations and the three initial checks for a stalled loss."
))
print("Results, figures, workbook, predictions and report draft:", OUT_DIR)
print("Training/evaluation cells completed. Save the executed notebook with outputs before packaging.")
"""),
        cell("markdown", """
### Save and package

Review `REPORT.md`, add your name and refine the explanations. If `REPORT.generated.md` exists, your previous report
was preserved; merge the draft into `REPORT.md`. Open `experiments.xlsx` in Excel and save to refresh formula caches.

After all cells complete, use **File > Download > Download .ipynb**. Follow the optional packaging snippet in
`code/RUNNING.md`: upload that executed notebook into the persistent `code/lab.ipynb`, then create and download the ZIP.
The notebook editor and the Python-visible file are different. Packaging refuses an unexecuted notebook.
Data, model weights and caches are excluded. No results in this notebook are invented or prefilled.
"""),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "lab.ipynb", "provenance": []}}, "cells": cells}
    for index, item in enumerate(cells):
        item["id"] = f"lab-cell-{index:02d}"
    (CODE / "lab.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built", CODE / "lab.ipynb", "with", len(cells), "cells and", len(FILES), "bundled files.")


if __name__ == "__main__":
    build()
