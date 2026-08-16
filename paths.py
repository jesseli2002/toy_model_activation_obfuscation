"""Per-run output directory layout: runs/<tag>/{checkpoints,logs}/, plot/<tag>/."""

import os


def run_dir(tag: str) -> str:
    return os.path.join("runs", tag)


def ckpt_dir(tag: str) -> str:
    return os.path.join(run_dir(tag), "checkpoints")


def log_dir(tag: str) -> str:
    return os.path.join(run_dir(tag), "logs")


def plot_dir(tag: str, ckpt: str | None = None) -> str:
    """plot/<tag>/, or plot/<tag>/<ckpt>/ for a caller that writes more than
    one checkpoint's figures for the same tag (best/last) and needs them kept
    apart -- filenames alone don't discriminate, since they're `{tag}_*.png`
    with no checkpoint in the name."""
    base = os.path.join("plot", tag)
    return os.path.join(base, ckpt) if ckpt is not None else base
