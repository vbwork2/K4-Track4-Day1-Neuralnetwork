"""Load the official split and standardize using training rows only."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10


def _validate(X, y):
    if X.ndim != 2 or X.shape[1] != 54 or y.shape != (len(X),):
        raise ValueError("Expected features (N, 54) and labels (N,).")
    if X.dtype != np.float32 or y.dtype != np.int64:
        raise ValueError("Expected float32 features and int64 labels.")
    if not np.isfinite(X).all() or not np.isin(y, np.arange(7)).all():
        raise ValueError("Features must be finite and labels must be in 0..6.")


def load_split(processed_dir: str = "data/processed"):
    root = Path(processed_dir)
    with np.load(root / "train.npz", allow_pickle=False) as tr:
        X, y = tr["X"], tr["y"]
    with np.load(root / "eval.npz", allow_pickle=False) as ev:
        Xe, ye, ids = ev["X"], ev["y"], ev["row_id"]
    _validate(X, y)
    _validate(Xe, ye)
    if ids.dtype != np.int64 or ids.shape != ye.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Eval row IDs must be unique int64 values, one per label.")
    return X, y, Xe, ye, ids


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    _validate(X, y)
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return Xtr, ytr, Xval, yval


def fit_standardizer(X_tr):
    # Float64 accumulation avoids rounding drift in large training sets.
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean, std = numeric.mean(0), numeric.std(0)
    return mean, np.where(std > 0, std, 1.0)


def apply_standardizer(X, mean, std):
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError("Expected ten means and standard deviations.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Invalid standardizer statistics.")
    out = np.array(X, dtype=np.float32, copy=True)
    out[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / np.where(std > 0, std, 1.0)
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    X, y, Xe, ye, ids = load_split(processed_dir)
    Xtr, ytr, Xval, yval = make_val_split(X, y, val_fraction, seed)
    mean, std = fit_standardizer(Xtr)
    result = {"eval_row_id": ids, "mean": mean, "std": std,
              "val_fraction": val_fraction, "split_seed": seed}
    for name, features, labels in (("tr", Xtr, ytr), ("val", Xval, yval), ("eval", Xe, ye)):
        result[f"X_{name}"] = torch.as_tensor(apply_standardizer(features, mean, std), device=device)
        result[f"y_{name}"] = torch.as_tensor(labels, device=device)
        print(f"{name}: X={tuple(features.shape)}, y={tuple(labels.shape)}")
    majority = int(np.bincount(ytr, minlength=7).argmax())
    result["majority_class"] = majority
    result["majority_val_acc"] = float(np.mean(yval == majority))
    print(f"Validation majority-class accuracy: {result['majority_val_acc']:.6f}")
    return result


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None,
                    shuffle: bool = True):
    if batch_size <= 0 or len(X) != len(y) or X.device != y.device:
        raise ValueError("Batch size must be positive; tensors must align on one device.")
    if shuffle:
        order = torch.randperm(len(X), device=X.device, generator=generator)
        for start in range(0, len(X), batch_size):
            indices = order[start:start + batch_size]
            yield X[indices], y[indices]
    else:
        for start in range(0, len(X), batch_size):
            yield X[start:start + batch_size], y[start:start + batch_size]
