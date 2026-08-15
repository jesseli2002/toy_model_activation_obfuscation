"""Checkpoint I/O: load a run's saved model/config off disk, and recompute
loss metrics fresh against it.

Used by the reporting scripts (make_one_run_plots.py, probe_lib.py, and the
sweep_*.py family) whenever they need to turn a run tag into a loaded model
or a freshly-evaluated loss, as opposed to reading history.jsonl or a
checkpoint's own stored training-time metrics (which can predate a run's
last checkpoint, or reflect stale settings).
"""

import os

import torch

import config
from data import eval_n_hot_loss, eval_task_loss
from model import ResidualMLP
from paths import ckpt_dir


def load_model_path(path: str, device: str) -> tuple[ResidualMLP, dict]:
    """Returns (model, full checkpoint dict) for a checkpoint path -- the dict
    carries any non-architecture fields (opt state, iter, adversarial-run
    metadata, ...) that rode along in the checkpoint; architecture lives on
    model.config. Use when walking checkpoint files directly, e.g. the
    numbered iter_*.pt snapshots; see `load_model` to load a run's named
    checkpoint by tag."""
    model, ck = ResidualMLP.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, ck


def load_model(tag: str, ckpt: str, device: str) -> tuple[ResidualMLP, dict]:
    """`load_model_path` for a run's named checkpoint (runs/<tag>/checkpoints/
    <ckpt>.pt)."""
    return load_model_path(os.path.join(ckpt_dir(tag), f"{ckpt}.pt"), device)


def resolve_adv_config(ck) -> "config.LogregAdversarialConfig | None":
    """The checkpoint's adversarial-training config, or None for a checkpoint
    with no adversarial config at all -- e.g. a train_no_c.py c-blind
    demonstration run, which has no probe or penalty to speak of since c was
    withheld from the model entirely (its embedding coordinate is zeroed)."""
    adv_ck = ck.get("adv_config")
    return (
        config.LogregAdversarialConfig.from_dict(adv_ck) if adv_ck is not None else None
    )


def recompute_task_loss(
    tag: str,
    ckpt: str,
    g: torch.Generator,
    device: str,
    n_eval: int,
    noise_mult: float = 1.0,
) -> float:
    """Task loss (data.eval_task_loss), freshly recomputed at `tag`'s `ckpt`
    checkpoint -- deliberately excludes the adversarial/probe penalty, since
    callers use this to check task performance alone. `noise_mult` scales
    the checkpoint's own resid_noise_std (0 for a checkpoint with no
    adversarial config)."""
    model, ck = load_model(tag, ckpt, device)
    adv_cfg = resolve_adv_config(ck)
    x_p_outer = adv_cfg.x_p_outer if adv_cfg is not None else None
    x_threshold = adv_cfg.x_threshold if adv_cfg is not None else 1.0
    noise_std = adv_cfg.resid_noise_std * noise_mult if adv_cfg is not None else 0.0
    return eval_task_loss(
        model,
        g,
        device,
        n=n_eval,
        x_p_outer=x_p_outer,
        x_threshold=x_threshold,
        noise=noise_std,
    )


def recompute_n_hot_loss(
    tag: str,
    ckpt: str,
    g: torch.Generator,
    device: str,
    n_eval: int,
    n_hot: int,
    noise_mult: float = 1.0,
) -> float:
    """N-hot OOD loss (data.eval_n_hot_loss) at `tag`'s `ckpt` checkpoint --
    only `n_hot` input coordinates nonzero per example, unlike
    recompute_task_loss's in-distribution eval. Same noise handling."""
    model, ck = load_model(tag, ckpt, device)
    adv_cfg = resolve_adv_config(ck)
    noise_std = adv_cfg.resid_noise_std * noise_mult if adv_cfg is not None else 0.0
    return eval_n_hot_loss(model, g, device, n=n_eval, n_hot=n_hot, noise=noise_std)
