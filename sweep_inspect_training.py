"""Aggregate several same-settings, different-seed training runs into one
sweep report: superimposed loss curves and last-iteration probe AUROC.

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Prints to
console / plt.show() only; nothing is written to disk.
"""

RUN_TAGS = ["sweep14_lr0.0015_iter400k_lam0.1_tr0"]
# RUN_TAGS = [f"sweep18_layer2_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)]

CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
LOSS_LOWPASS_WINDOW = 2000  # running mean of loss over the past this-many iters
TRAIN_NOISE_MULT = 1.0  # see adversarial_report.py's --train-noise-mult
EVAL_NOISE_MULT = 1.0  # see adversarial_report.py's --eval-noise-mult
PROBE_BACKEND = "newton"  # see adversarial_report.py's --probe-backend
# Refit-probe sample sizes come from config.PROBE_EVAL_N_TRAIN/N_TEST.
LOSS_TYPE = "task"

import matplotlib.pyplot as plt
import torch

from config import C_HIGH, C_LOW
from probe_backend import resolve_probe_backend
from sweep_lib.history import causal_lowpass, load_history
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_concat_layers,
    boundary_auroc,
    load_model,
    resolve_adv_config,
    stored_probe,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def _refit_probe_auroc(
    tag: str,
    ckpt: str,
    device: str,
    probe_backend_name: str,
    train_noise_mult: float,
    eval_noise_mult: float,
    g: torch.Generator,
) -> float | None:
    """AUROC of a probe freshly fit against `tag`'s checkpoint (over the same
    probed layers as its stored probe, concatenated), or None if it carries no
    stored probe (and hence no `probe_layers` to fit against).

    Contrast the stored probe's own AUROC, which reflects the specific probe
    the model was trained against -- this instead asks how visible c is to a
    probe trained now."""
    model, ck = load_model(tag, ckpt, device)
    stored = stored_probe(ck, device)
    if stored is None:
        return None
    _, layers = stored
    adv_cfg = resolve_adv_config(ck)
    eval_noise_std = adv_cfg.resid_noise_std * eval_noise_mult
    train_noise_std = adv_cfg.resid_noise_std * train_noise_mult
    _, pi = binary_probe_metrics_concat_layers(
        model,
        C_LOW,
        C_HIGH,
        layers,
        g,
        probe_backend_name,
        eval_noise=eval_noise_std,
        train_noise=train_noise_std,
    )
    boundary = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return boundary_auroc(boundary, pi["X_te"], pi["y_te"])


def main():
    fig, (ax_loss, ax_auroc) = plt.subplots(1, 2, figsize=(13, 5))

    for tag in RUN_TAGS:
        iters, losses = load_history(tag, LOSS_TYPE)
        filtered = causal_lowpass(iters, losses, LOSS_LOWPASS_WINDOW)
        ax_loss.plot(iters, filtered, label=tag, alpha=0.8)
    ax_loss.set_xlabel("iter")
    ax_loss.set_ylabel(
        f"{LOSS_TYPE} loss (smoothed over past {LOSS_LOWPASS_WINDOW} iters)"
    )
    ax_loss.set_yscale("log")
    ax_loss.set_title("training loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    print(f"refit probe AUROC at {CKPT} checkpoint:")
    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    aurocs = {
        tag: _refit_probe_auroc(
            tag,
            CKPT,
            DEVICE,
            probe_backend_name,
            TRAIN_NOISE_MULT,
            EVAL_NOISE_MULT,
            g,
        )
        for tag in RUN_TAGS
    }
    for tag in RUN_TAGS:
        if aurocs[tag] is None:
            print(f"  {tag}: n/a (no probe in checkpoint)")
        else:
            print(f"  {tag}: {aurocs[tag]:.4f}")

    tags_with_auroc = [t for t in RUN_TAGS if aurocs[t] is not None]
    ax_auroc.bar(tags_with_auroc, [aurocs[t] for t in tags_with_auroc])
    ax_auroc.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax_auroc.set_ylim(0.45, 1.02)
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.set_title(f"probe AUROC ({CKPT} checkpoint)")
    ax_auroc.tick_params(axis="x", rotation=45)
    ax_auroc.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
