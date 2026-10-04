"""Reproducible training and validation, with separate final eval export."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce", optimizer="sgd_momentum", lr=None, weight_decay=0.0, momentum=0.9,
    betas=(0.9, 0.999), eps=1e-8, batch=512, epochs=20, hidden=(256, 128),
    dropout=0.0, init="he", clip_norm=None, precision="fp32", seed=1,
    scheduler=None, train_eval_size=50_000, train_eval_seed=42, eval_batch=8192,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Full FP32 arithmetic keeps the precision comparison explicit.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def supports_bf16(device=None) -> bool:
    """Require native CUDA BF16 support; do not treat software emulation as hardware support."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError("Expected a 7 by 7 confusion matrix.")
    denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * np.diag(cm), denom, out=np.zeros(7), where=denom > 0).mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    model.eval()
    if not len(X):
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat([model(X[i:i + batch_size]).argmax(1)
                      for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        # Mean over samples and all seven raw-logit coordinates, without a 1/2 factor.
        return F.mse_loss(logits, F.one_hot(y, num_classes=7).to(logits.dtype), reduction="mean")
    raise ValueError(f"Unknown loss: {loss_name}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    if not len(X) or len(X) != len(y):
        raise ValueError("Evaluation requires matching, nonempty features and labels.")
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=X.device)
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        # Sample weighting includes the short final batch for both losses.
        total += compute_loss(logits, yb, loss_name).double() * len(yb)
        cm += torch.bincount(yb * 7 + logits.argmax(1), minlength=49)
    matrix = cm.reshape(7, 7).cpu().numpy()
    return {"loss": float(total / len(X)), "acc": float(np.trace(matrix) / len(X)),
            "macro_f1": macro_f1_from_confusion(matrix)}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_experiment(cfg: dict, data: dict) -> dict:
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["epochs"] <= 0 or cfg["batch"] <= 0 or cfg["eval_batch"] <= 0:
        raise ValueError("Epochs and batch sizes must be positive.")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Precision must be fp32, fp16 or bf16.")
    device = data["X_tr"].device
    mixed = cfg["precision"] != "fp32"
    if mixed and device.type != "cuda":
        raise ValueError("Mixed precision experiments require a CUDA GPU.")
    if cfg["precision"] == "bf16" and not supports_bf16(device):
        raise ValueError("This GPU does not support BF16.")
    set_seed(cfg["seed"])
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]]
    opt = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                          cfg["weight_decay"], cfg["momentum"], cfg["betas"], cfg["eps"])
    steps = (len(data["X_tr"]) + cfg["batch"] - 1) // cfg["batch"]
    scheduler = build_scheduler(opt, cfg["scheduler"], steps * cfg["epochs"],
                                **cfg.get("scheduler_kwargs", {}))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    amp_dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    size = cfg["train_eval_size"]
    if size is not None and size <= 0:
        raise ValueError("Training evaluation subset size must be positive or None.")
    size = min(len(data["X_tr"]), size or len(data["X_tr"]))
    # The subset is independent of the training seed and stays fixed across experiments.
    indices = np.random.default_rng(cfg["train_eval_seed"]).permutation(len(data["X_tr"]))[:size]
    indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
    Xmeasure, ymeasure = data["X_tr"][indices], data["y_tr"][indices]
    initial = evaluate(model, data["X_val"], data["y_val"], cfg["loss"], cfg["eval_batch"])
    stats = activation_stats(model, data["X_val"][:512])
    history = {k: [] for k in (
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm",
        "grad_norm_max", "clip_fraction", "skipped_steps", "epoch_time_s", "lr",
    )}
    best_loss, best_epoch, best_metrics, best_state = float("inf"), None, None, None
    diverged = not np.isfinite(initial["loss"])
    reason = "Non-finite initial validation loss." if diverged else ""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        if diverged:
            break
        _sync(device)
        started = time.perf_counter()
        model.train()
        norms, clipped, skipped, overflow_streak = [], 0, 0, 0
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator):
            opt.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=amp_dtype) if mixed else nullcontext()
            with context:
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not bool(torch.isfinite(loss)):
                diverged, reason = True, f"Non-finite training loss in epoch {epoch}."
                break
            scaler.scale(loss).backward()
            # Unscale even without clipping to log norms at the true gradient scale.
            scaler.unscale_(opt)
            norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not np.isfinite(norm):
                if scaler.is_enabled():
                    # GradScaler skips the update and reduces its scale after overflow.
                    scaler.step(opt)
                    scaler.update()
                    skipped += 1
                    overflow_streak += 1
                    if overflow_streak < 50:
                        continue
                diverged, reason = True, f"Non-finite gradients in epoch {epoch}."
                break
            overflow_streak = 0
            norms.append(norm)
            clipped += int(cfg["clip_norm"] is not None and norm > cfg["clip_norm"])
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        if diverged:
            break
        tr = evaluate(model, Xmeasure, ymeasure, cfg["loss"], cfg["eval_batch"])
        val = evaluate(model, data["X_val"], data["y_val"], cfg["loss"], cfg["eval_batch"])
        if not np.isfinite(tr["loss"]) or not np.isfinite(val["loss"]):
            diverged, reason = True, f"Non-finite evaluation loss in epoch {epoch}."
            break
        if val["loss"] < best_loss:
            best_loss, best_epoch, best_metrics = val["loss"], epoch, val
            best_state = _copy_state(model)
        _sync(device)
        elapsed = time.perf_counter() - started
        values = dict(epoch=epoch, train_loss=tr["loss"], val_loss=val["loss"],
                      val_acc=val["acc"], val_macro_f1=val["macro_f1"],
                      grad_norm=float(np.mean(norms)) if norms else None,
                      grad_norm_max=max(norms) if norms else None,
                      clip_fraction=clipped / len(norms) if norms else 0.0,
                      skipped_steps=skipped, epoch_time_s=elapsed, lr=opt.param_groups[0]["lr"])
        for key, value in values.items():
            history[key].append(value)
        print(f"{cfg['exp_id']} {epoch:02d}/{cfg['epochs']}: "
              f"train={tr['loss']:.4f} val={val['loss']:.4f} "
              f"acc={val['acc']:.4f} F1={val['macro_f1']:.4f} time={elapsed:.2f}s", flush=True)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    summary = dict(
        step0_loss=initial["loss"], best_val_loss=best_loss if best_epoch else None,
        best_epoch=best_epoch, final_train_loss=history["train_loss"][-1] if best_epoch else None,
        final_val_loss=history["val_loss"][-1] if best_epoch else None,
        val_acc=best_metrics["acc"] if best_epoch else None,
        val_macro_f1=best_metrics["macro_f1"] if best_epoch else None,
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if best_epoch else None,
        peak_mem_MB=peak, diverged=diverged, divergence_reason=reason,
        completed_epochs=len(history["epoch"]), train_eval_size=size,
        activation_std=stats, activation_measurement="after Linear, before ReLU",
    )
    if diverged:
        print(f"{cfg['exp_id']}: stopped: {reason}", flush=True)
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    ids, pred = np.asarray(row_id), np.asarray(preds)
    if ids.ndim != 1 or pred.shape != ids.shape or not len(ids):
        raise ValueError("Expected nonempty aligned row IDs and predictions.")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("Row IDs and predictions must be integers.")
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or not np.isin(pred, np.arange(7)).all():
        raise ValueError("Row IDs must be unique and nonnegative; predictions must be in 0..6.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, np.column_stack([ids, pred]), fmt="%d", delimiter=",",
               header="row_id,pred", comments="")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    if result.get("best_state") is None:
        raise ValueError("This run has no valid best checkpoint.")
    actual = result["cfg"]
    for key in ("hidden", "dropout", "init", "seed", "loss", "optimizer", "lr"):
        supplied = tuple(cfg[key]) if key == "hidden" and key in cfg else cfg.get(key)
        stored = tuple(actual[key]) if key == "hidden" else actual[key]
        if key in cfg and supplied != stored:
            raise ValueError(f"Configuration does not match the checkpoint: {key}")
    model = MLP(actual["hidden"], actual["dropout"], actual["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    pred = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], pred.cpu().numpy(), pred_path)
