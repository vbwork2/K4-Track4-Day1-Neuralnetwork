"""Experiment planning, persistent completed-run checkpoints and lab diagnostics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from train import DEFAULT_CFG, set_seed, compute_loss, evaluate, run_experiment, supports_bf16
from plots import plot_run, plot_compare
from results_table import save_result, write_json, json_safe

PREDICTIONS = {
    "baseline": "Validation should identify an SGD learning rate that converges without large oscillations. Different seeds should expose training noise.",
    "loss": "Raw-logit MSE may converge differently from CE. Its loss scale is different, so compare accuracy and macro-F1.",
    "optimizer": "Adam may converge faster, but the comparison must use a validation-tuned rate for each optimizer.",
    "hparam": "A smaller batch increases updates per epoch and gradient noise; it may improve F1 at a higher time cost.",
    "dropout": "Dropout 0.3 should reduce a genuine generalization gap; it may hurt if the baseline is still underfitting.",
    "clipping": "Clipping should cap large updates. A matched high-rate pair tests whether it improves stability; recovery is not guaranteed.",
    "amp": "FP16 should have similar F1 to FP32. Speed and memory gains may be small for this compact MLP.",
    "init": "Xavier may change activation scale. Zero weights preserve symmetry and leave hidden ReLU gradients zero.",
}
MECHANISMS = {
    "baseline": "Momentum averages updates over recent gradients. Learning rate affects both progress and oscillation.",
    "loss": "CE applies log-softmax to raw logits. MSE regresses seven raw logits toward one-hot targets and averages over all seven coordinates.",
    "optimizer": "Adam scales coordinate updates using moving averages of gradients and squared gradients. AdamW decouples weight decay; neither is automatically better.",
    "hparam": "At fixed epochs, batch 128 produces about four times as many updates as batch 512. Time and update count therefore change with batch size.",
    "dropout": "Randomly masking hidden activations regularizes co-adaptation. Train loss is measured with dropout disabled to make the gap interpretable.",
    "clipping": "Global norm clipping rescales gradients by min(1, threshold/norm), limiting sudden update magnitudes without fixing bad labels or an unsuitable rate.",
    "amp": "FP16 has a narrow exponent range; GradScaler limits gradient underflow and skips overflow updates. BF16 has a wider range, but requires hardware support.",
    "init": "He uses variance 2/fan_in for ReLU; Xavier uses 2/(fan_in+fan_out). At zero initialization, ReLU derivatives block hidden-layer learning.",
}


def health_checks(data: dict, out_dir: str | Path, steps: int = 500) -> dict:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    set_seed(1)
    device = data["X_tr"].device
    model = MLP().to(device)
    assert count_params(model) == EXPECTED_PARAMS[(256, 128)]
    assert model(data["X_val"][:8]).shape == (8, 7)
    step0 = evaluate(model, data["X_val"], data["y_val"])["loss"]
    loss = compute_loss(model(data["X_tr"][:512]), data["y_tr"][:512], "ce")
    loss.backward()
    gradients = {name: float(p.grad.norm()) if p.grad is not None else None
                 for name, p in model.named_parameters()}
    if not all(value is not None and np.isfinite(value) and value > 0 for value in gradients.values()):
        raise RuntimeError(f"Initial gradient check failed: {gradients}")
    del model
    set_seed(1)
    tiny = MLP().to(device)
    opt = torch.optim.Adam(tiny.parameters(), lr=0.01)
    X, y = data["X_tr"][:20], data["y_tr"][:20]
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(tiny(X), y, "ce")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    tiny_scores = evaluate(tiny, X, y)
    if tiny_scores["acc"] != 1.0 or tiny_scores["loss"] >= 0.02:
        raise RuntimeError(f"The 20-example overfit check failed: {tiny_scores}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, steps + 1), losses)
    ax.set(xlabel="Update", ylabel="CE loss", title="Overfit 20 training examples")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figures/diagnostic_overfit.png", dpi=160)
    plt.close(fig)
    numeric = data["X_tr"][:, :10].double()
    diagnostics = {
        "parameters": EXPECTED_PARAMS[(256, 128)], "logits_shape": [8, 7],
        "step0_loss": step0, "ln7": float(np.log(7)), "gradient_norms": gradients,
        "tiny_loss": tiny_scores["loss"], "tiny_acc": tiny_scores["acc"], "tiny_steps": steps,
        "train_numeric_mean": numeric.mean(0).cpu().tolist(),
        "train_numeric_std": numeric.std(0, unbiased=False).cpu().tolist(),
        "n_train": len(data["X_tr"]), "n_val": len(data["X_val"]),
        "majority_val_acc": data.get("majority_val_acc"),
    }
    write_json(out / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def seed_statistics(results: list[dict]) -> dict:
    if len(results) < 2:
        raise ValueError("Use at least two independent baseline seeds.")
    return {metric: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                     "noise_2sigma": float(2 * np.std(values, ddof=1))}
            for metric in ("val_acc", "val_macro_f1", "best_val_loss")
            for values in [[r["summary"][metric] for r in results]]}


def lr_trials(epochs: int = 20) -> list[dict]:
    return [{**DEFAULT_CFG, "exp_id": f"base-lr{str(lr).replace('.', 'p')}-s1",
             "description": "Baseline learning-rate trial", "lr": lr, "epochs": epochs}
            for lr in (0.01, 0.03, 0.1)]


def best_by_validation(results: list[dict]) -> dict:
    eligible = [r for r in results if not r["summary"]["diverged"]
                and r["summary"].get("completed_epochs") == r["cfg"]["epochs"]
                and r["summary"].get("val_macro_f1") is not None]
    if not eligible:
        raise RuntimeError("No completed, numerically stable configuration is available.")
    return max(eligible, key=lambda r: (r["summary"]["val_macro_f1"], -r["summary"]["best_val_loss"]))


def topic_trials(baseline: dict, device: torch.device) -> list[dict]:
    cfg = baseline["cfg"]
    finite_norms = [n for n in baseline["history"]["grad_norm"] if n is not None and np.isfinite(n)]
    threshold = max(0.01, float(np.median(finite_norms)) * 0.75)
    high_lr = cfg["lr"] * 10
    trials = []
    for lr in (0.0003, 0.001):
        trials.append(dict(exp_id=f"opt-adam-lr{lr:g}", group="optimizer", optimizer="adam", lr=lr,
                           description="Adam with independently tuned learning rate"))
    trials.extend([
        dict(exp_id="loss-mse", group="loss", loss="mse", description="Raw-logit MSE instead of CE"),
        dict(exp_id="batch-128", group="hparam", batch=128, description="Smaller batch; same epoch budget"),
        dict(exp_id="dropout-0p3", group="dropout", dropout=0.3, description="Hidden dropout probability 0.3"),
        dict(exp_id="clip-normal", group="clipping", clip_norm=threshold, description="Clipping at the selected baseline rate"),
        dict(exp_id="highlr-no-clip", group="clipping", lr=high_lr, description="Ten times baseline rate, no clipping"),
        dict(exp_id="highlr-clip", group="clipping", lr=high_lr, clip_norm=threshold,
             description="Matched high rate with clipping", reference="highlr-no-clip"),
        dict(exp_id="init-xavier", group="init", init="xavier", description="Xavier normal instead of He"),
        dict(exp_id="init-zeros", group="init", init="zeros", description="Zero initialization symmetry test"),
    ])
    if device.type == "cuda":
        trials.append(dict(exp_id="amp-fp16", group="amp", precision="fp16", description="FP16 autocast with GradScaler"))
        if supports_bf16(device):
            trials.append(dict(exp_id="amp-bf16", group="amp", precision="bf16", description="BF16 autocast without loss scaling"))
    return [{**cfg, **changes, "seed": 1, "prediction": PREDICTIONS[changes["group"]],
             "reference": changes.get("reference", cfg["exp_id"])} for changes in trials]


class LabRunner:
    """Persist each completed run; checkpoints stay outside the submission directory."""
    def __init__(self, data, repo_root, out_dir, checkpoint_dir=None):
        self.data = data
        self.root = Path(repo_root).resolve()
        self.out = Path(out_dir).resolve()
        self.results = {}
        hasher = hashlib.sha256()
        for name in ("data.py", "model.py", "optimizer.py", "train.py", "experiments.py"):
            hasher.update(Path(__file__).with_name(name).read_bytes())
        for name in ("covtype.csv.gz", "split_metadata.csv"):
            source = self.root / "data" / name
            if source.exists():
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
        # Include actual train/validation tensors so custom subsets cannot reuse stale runs.
        for key in ("X_tr", "y_tr", "X_val", "y_val"):
            tensor = data[key]
            hasher.update(str((key, tuple(tensor.shape), str(tensor.dtype))).encode())
            for start in range(0, len(tensor), 8192):
                hasher.update(tensor[start:start + 8192].detach().cpu().contiguous().numpy().tobytes())
        self.signature = dict(code_and_data=hasher.hexdigest(), torch=str(torch.__version__),
                              device=str(data["X_tr"].device),
                              gpu=torch.cuda.get_device_name(data["X_tr"].device) if data["X_tr"].is_cuda else None,
                              split_seed=data.get("split_seed", 42), val_fraction=data.get("val_fraction", 0.2),
                              n_train=len(data["X_tr"]), n_val=len(data["X_val"]),
                              mean=json_safe(data.get("mean", [])), std=json_safe(data.get("std", [])))
        self.environment = dict(python=platform.python_version(), **self.signature)
        self.checkpoints = Path(checkpoint_dir or self.out.parent / ".lab_checkpoints" / self.out.name).resolve()
        if self.checkpoints == self.out or self.out in self.checkpoints.parents:
            raise ValueError("Checkpoints must stay outside the submission.")
        records = list((self.out / "results").glob("*.json"))
        environment_path = self.out / "environment.json"
        previous_environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
        same_environment = all(previous_environment.get(key) == value for key, value in self.signature.items())
        missing_checkpoint = any(
            not json.loads(record.read_text(encoding="utf-8"))["summary"].get("diverged")
            and not (self.checkpoints / f"{record.stem}.pt").is_file()
            for record in records
        )
        if (self.out / "selection.json").exists() or (records and (not same_environment or missing_checkpoint)):
            # Submitted logs are evidence, not a resumable cache without matching weights.
            original = self.out
            reruns = original.parent / ".lab_reruns"
            reruns.mkdir(parents=True, exist_ok=True)
            self.out = Path(tempfile.mkdtemp(prefix="run-", dir=reruns)) / original.name
            self.out.mkdir()
            self.checkpoints = self.out.parent / ".lab_checkpoints" / self.out.name
            print("Preserving existing results:", original)
            print("Training from scratch in:", self.out)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        write_json(self.out / "environment.json", self.environment)

    def run(self, cfg):
        cfg = {**DEFAULT_CFG, **cfg}
        exp_id = cfg["exp_id"]
        frozen_path = self.out / "selection.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if exp_id not in frozen["candidate_ids"]:
                raise RuntimeError("Selection is frozen after eval; adding experiments would risk eval-guided tuning.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", exp_id):
            raise ValueError("Experiment IDs must be safe file names.")
        cfg["prediction"] = cfg.get("prediction", PREDICTIONS[cfg["group"]])
        print(f"Prediction for {exp_id}: {cfg['prediction']}", flush=True)
        fingerprint = hashlib.sha256(json.dumps(json_safe({"cfg": cfg, "signature": self.signature}),
                                               sort_keys=True).encode()).hexdigest()
        record = self.out / "results" / f"{exp_id}.json"
        checkpoint = self.checkpoints / f"{exp_id}.pt"
        if record.exists():
            result = json.loads(record.read_text(encoding="utf-8"))
            if result["summary"].get("fingerprint") != fingerprint:
                raise RuntimeError(f"Existing {exp_id} uses different code, data or settings. Use a new output directory or experiment ID.")
            if result["summary"]["diverged"]:
                result["best_state"] = None
                print(f"Reusing recorded divergence: {exp_id}")
            elif checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload["fingerprint"] != fingerprint:
                    raise RuntimeError(f"Checkpoint mismatch for {exp_id}.")
                result["best_state"] = payload["best_state"]
                print(f"Reusing completed experiment: {exp_id}")
            else:
                raise RuntimeError(f"Missing checkpoint for {exp_id}; logs alone cannot restore weights. Use a new ID to retrain.")
        else:
            started = time.perf_counter()
            result = run_experiment(cfg, self.data)
            result["summary"]["fingerprint"] = fingerprint
            result["summary"]["run_time_s"] = time.perf_counter() - started
            if result["best_state"] is not None:
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({"fingerprint": fingerprint, "best_state": result["best_state"]}, temporary)
                temporary.replace(checkpoint)
            save_result(result, self.out / "results")
        plot_run(result, str(self.out / "figures" / f"{exp_id}.png"))
        self.results[exp_id] = result
        return result

    def comparisons(self, baseline):
        for group in PREDICTIONS:
            members = [r for r in self.results.values() if r["cfg"]["group"] == group]
            if not members:
                continue
            if group != "baseline":
                members = [baseline] + members
            plot_compare(members, "val_macro_f1", str(self.out / "figures" / f"compare_{group}.png"),
                         f"{group}: validation macro-F1")
