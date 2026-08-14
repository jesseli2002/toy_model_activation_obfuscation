"""Sorting a run's recomputed metrics into an outcome band, and the colors
and labels those bands are drawn with.

A run either failed the task or, having succeeded, is binned by how well it
hid from a linear probe (AUROC). Band 0 is "failed task"; band 1 is the
least hidden of the survivors and the last band the most hidden.
"""

import math
from dataclasses import dataclass

import numpy as np

# Sequential blue ramp (references/palette.md), lightest -> darkest, 100..700.
SEQ_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
FAILED_COLOR = "#eb6834"  # categorical slot 2 (orange) -- distinct "problem" hue


@dataclass(frozen=True)
class BandSpec:
    """The thresholds defining "solved the task" and "hid from the probe".

    Each threshold can be disabled independently, which lets one set of
    metrics be classified under several specs to see how runs fail:
    `loss_threshold=math.inf` ignores in-distribution loss, and
    `n_hot_values=()` ignores the N-hot OOD losses entirely.
    """

    loss_threshold: float = math.inf
    n_hot_loss_threshold: float = math.inf
    n_hot_values: tuple[int, ...] = ()
    auroc_thresholds: tuple[float, ...] = (0.6, 0.75, 0.9)

    def __post_init__(self):
        if list(self.auroc_thresholds) != sorted(self.auroc_thresholds):
            raise ValueError(
                f"auroc_thresholds must be ascending: {self.auroc_thresholds}"
            )

    @property
    def n_bands(self) -> int:
        """Bands: one for "failed", then one per AUROC interval."""
        return len(self.auroc_thresholds) + 2

    def failed(self, task_loss: float, n_hot_losses: dict[int, float]) -> bool:
        """Whether a run missed either loss bar. `n_hot_losses` is keyed by N
        and must cover `n_hot_values`; extra entries are ignored."""
        return task_loss >= self.loss_threshold or any(
            n_hot_losses[n] >= self.n_hot_loss_threshold for n in self.n_hot_values
        )

    def classify(
        self, task_loss: float, n_hot_losses: dict[int, float], auroc: float
    ) -> int:
        """Band index: 0 failed the task, 1 succeeded but is least hidden,
        `n_bands - 1` succeeded and is most hidden."""
        if self.failed(task_loss, n_hot_losses):
            return 0
        for i, t in enumerate(reversed(self.auroc_thresholds)):
            if auroc >= t:
                return i + 1
        return len(self.auroc_thresholds) + 1

    def labels(self) -> list[str]:
        """Legend label per band, in band order."""
        if self.n_hot_values == tuple():
            labels = [f"failed task\n(loss $\\geq$ {self.loss_threshold})"]
        else:
            labels = [f"failed task"]
        prev = None
        for t in reversed(self.auroc_thresholds):
            if prev is None:
                labels.append(f"not hidden\n(AUROC $\\geq$ {t:g})")
            else:
                labels.append(f"partially hidden\n({t:g} $\\leq$ AUROC < {prev:g})")
            prev = t
        if prev is None:
            labels.append("hidden")  # no AUROC thresholds: one undivided band
        else:
            labels.append(f"hidden\n(AUROC < {prev:g})")
        return labels

    def colors(self) -> list[str]:
        """Band color, in band order: the failure hue, then the sequential
        ramp sampled evenly so the most hidden band is darkest."""
        n_hiding_bins = len(self.auroc_thresholds) + 1
        idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
        return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]
