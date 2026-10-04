"""Save per-experiment learning curves and controlled comparisons."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    cfg, history = result["cfg"], result["history"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="Train (eval mode)")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_ylabel(f"{cfg['loss'].upper()} loss")
    axes[1].plot(epochs, history["val_acc"], label="Accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="Macro-F1")
    axes[1].set_ylabel("Validation score")
    axes[1].set_ylim(0, 1)
    axes[2].plot(epochs, history["grad_norm"], label="Mean norm before clipping")
    if "grad_norm_max" in history:
        axes[2].plot(epochs, history["grad_norm_max"], alpha=0.5, label="Maximum norm")
    if cfg.get("clip_norm") is not None:
        axes[2].axhline(cfg["clip_norm"], linestyle=":", label="Clip threshold")
    axes[2].set_ylabel("Global gradient L2 norm")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        best = result["summary"].get("best_epoch")
        if best is not None:
            ax.axvline(best, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    title = (f"{cfg['exp_id']}: {cfg['optimizer']}, lr={cfg['lr']}, batch={cfg['batch']}, "
             f"dropout={cfg['dropout']}, {cfg['precision']}, init={cfg['init']}")
    if result["summary"].get("diverged"):
        title += " (diverged)"
        axes[0].text(0.03, 0.05, result["summary"].get("divergence_reason", ""),
                     transform=axes[0].transAxes, fontsize=8, wrap=True)
    fig.suptitle(title, fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        if metric not in result["history"]:
            raise ValueError(f"Unknown history metric: {metric}")
        ax.plot(result["history"]["epoch"], result["history"][metric], label=result["cfg"]["exp_id"])
    ax.set(xlabel="Epoch", ylabel=metric.replace("_", " "), title=title or metric)
    ax.grid(alpha=0.25)
    if results:
        ax.legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
