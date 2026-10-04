"""Run the full lab from a terminal or a small isolated CPU validation run."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import torch
from data import prepare_data
from experiments import LabRunner, health_checks, lr_trials, best_by_validation, seed_statistics, topic_trials
from results_table import write_json
from submission import finalize

DELIVERABLE_CODE = ["data.py", "model.py", "optimizer.py", "train.py", "plots.py", "results_table.py",
                    "experiments.py", "report.py", "submission.py", "run_lab.py", "requirements.txt",
                    "experiment_table_template.xlsx", "lab.ipynb", "RUNNING.md"]


def copy_code(destination):
    source = Path(__file__).resolve().parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in DELIVERABLE_CODE:
        original, target = source / name, destination / name
        if not original.exists():
            continue
        if original.resolve() == target.resolve():
            continue
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise RuntimeError(f"Existing code differs at {target}; use a new output directory.")
        shutil.copy2(original, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default="2A202603012")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two epochs on small subsets; never use as a submission.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.student_id):
        raise ValueError("Use a student ID containing only letters, digits, underscores or hyphens.")
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts/split_data.py").exists():
        root = Path(__file__).resolve().parents[2]
    out = args.out or root / (".lab_validation" if args.smoke else "") / f"submission_{args.student_id}"
    out = out.resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    if device == "cpu":
        torch.set_num_threads(1)
    processed = root / "data/processed"
    if not all((processed / name).exists() for name in ("train.npz", "eval.npz")):
        subprocess.run([sys.executable, str(root / "scripts/split_data.py")], cwd=root, check=True)
    data = prepare_data(device, processed_dir=str(processed))
    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        # These subsets exercise the workflow; they are not lab experiment evidence.
        for name, count in (("tr", 4096), ("val", 2048)):
            data[f"X_{name}"] = data[f"X_{name}"][:count]
            data[f"y_{name}"] = data[f"y_{name}"][:count]
    runner = LabRunner(data, root, out)
    out = runner.out
    copy_code(out / "code")
    runner.environment["smoke_test"] = args.smoke
    write_json(out / "environment.json", runner.environment)
    diagnostics = health_checks(data, out)
    trials = [runner.run(cfg) for cfg in lr_trials(epochs)]
    baseline = best_by_validation(trials)
    baseline_seeds = [baseline] + [runner.run({**baseline["cfg"], "seed": seed,
                          "exp_id": f"base-s{seed}", "description": f"Selected baseline, seed {seed}"})
                                   for seed in (2, 3)]
    stats = seed_statistics(baseline_seeds)
    write_json(out / "seed_statistics.json", stats)
    for cfg in topic_trials(baseline, torch.device(device)):
        runner.run(cfg)
    result = finalize(runner, baseline, baseline_seeds, stats, diagnostics, args.student_id)
    if args.smoke:
        print("SMOKE VALIDATION ONLY: these results are not a completed lab submission.")
    print("Output:", out)
    print("Final experiment:", result["selection"]["final"])


if __name__ == "__main__":
    main()
