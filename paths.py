"""Per-run output directory layout: runs/<tag>/{checkpoints,logs}/."""

import os


def run_dir(tag: str) -> str:
    return os.path.join("runs", tag)


def ckpt_dir(tag: str) -> str:
    return os.path.join(run_dir(tag), "checkpoints")


def log_dir(tag: str) -> str:
    return os.path.join(run_dir(tag), "logs")
