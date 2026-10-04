"""Optimizer selection, optional schedulers and pre-clipping gradient norms."""
from __future__ import annotations
import math
import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    if lr is None or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Specify a finite positive learning rate selected using validation.")
    if name in {"sgd", "sgd_momentum"}:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=momentum if name == "sgd_momentum" else 0.0)
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    if name is None:
        return None
    if total_steps <= 0:
        raise ValueError("Total scheduler steps must be positive.")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    if max_norm is not None and (not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("The clipping threshold must be finite and positive.")
    return float(torch.nn.utils.clip_grad_norm_(
        list(params), math.inf if max_norm is None else max_norm, norm_type=2.0
    ))
