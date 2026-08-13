"""Per-run metrics recomputed from a checkpoint, with caching.

Everything here is recomputed against the checkpoint rather than read from
history.jsonl or from the checkpoint's own stored training-time probe: a
run's final checkpoint can predate its last history record, and runs within
one sweep may have trained against differently-configured probes, which a
freshly refit probe makes comparable again.

A `MetricSpec` fixes the settings a metric is computed under; `MetricStore`
hands out the values and caches them.
"""

import os
from dataclasses import dataclass

import torch

from checkpoint_lib import (
    load_model,
    recompute_n_hot_loss,
    recompute_task_loss,
    resolve_adv_config,
)
from config import C_HIGH, C_LOW, PROBE_EVAL_N_TEST, PROBE_EVAL_N_TRAIN
from paths import ckpt_dir
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    boundary_auroc,
    probe_eval_solver_key,
)
from sweep_lib.cache import MetricCache, cache_key, file_fingerprint

# One cache for the whole sweep*.py family. Keys are exact and cover every
# setting a value depends on, so reports with different settings coexist here
# and reports that share settings share the work.
CACHE_PATH = "plot/metrics_cache.json"


@dataclass(frozen=True)
class MetricSpec:
    """The settings every metric in a report is computed under.

    `probe_layers` defaults to the layers a run was actually penalized at, so
    a sweep over that layer needs no special handling; setting it explicitly
    asserts that every run agrees with it.
    """

    ckpt: str = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<ckpt>.pt
    task_loss_n_eval: int = 50_000  # fresh examples per run per loss metric
    noise_mult: float = 1.0  # multiplier on the checkpoint's own resid_noise_std
    probe_layers: tuple[int, ...] | None = None
    # Per class; a probe is refit per run. Shared with every other reporting
    # entry point (see config) so sweep tables and single-run reports agree.
    probe_n_train: int = PROBE_EVAL_N_TRAIN
    probe_n_test: int = PROBE_EVAL_N_TEST
    probe_backend: str = "newton"
    # Multipliers on the checkpoint's own resid_noise_std for the refit probe's
    # fit and scoring passes. Both 1.0: runs that trained with probe_noise hid
    # from an adversary that was itself fit under noise, so fitting clean
    # measures a different, weaker probe (worth ~0.1 AUROC).
    probe_train_noise_mult: float = 1.0
    probe_eval_noise_mult: float = 1.0


def resolve_probe_layers(
    tag: str, ck, override: tuple[int, ...] | None
) -> list[int] | None:
    """The layers to probe `tag` at: `override` when given, else the layer
    the run was penalized at. None for a checkpoint with no adversarial
    config -- e.g. a train_no_c.py c-blind run, which has no probe to refit.

    A run whose config disagrees with `override` raises: comparing a run
    against a probe at layers it was never penalized at answers a different
    question, silently.
    """
    adv_cfg = resolve_adv_config(ck)
    if adv_cfg is None:
        return None
    layers = list(adv_cfg.penalty_layers)
    if override is None:
        if len(layers) != 1:
            raise ValueError(
                f"{tag}: trained with penalty_layers={layers}; "
                "set MetricSpec.probe_layers to pick one"
            )
        return layers
    if layers != list(override):
        raise ValueError(
            f"{tag}: probe_layers={list(override)} but the checkpoint was trained "
            f"with penalty_layers={layers}"
        )
    return layers


class MetricStore:
    """Cached task loss, N-hot OOD loss and refit-probe AUROC for run tags.

    Metrics are None for a run with no checkpoint on disk yet, and `auroc` is
    also None for a checkpoint with no adversarial config.
    """

    def __init__(
        self,
        spec: MetricSpec,
        cache_path: str,
        device: str,
        cache: MetricCache | None = None,
    ):
        self.spec = spec
        self.device = device
        self.cache = (
            cache
            if cache is not None
            else MetricCache(cache_path, seed_exclude=("ckpt_fingerprint",))
        )
        self.probe_backend_name = resolve_probe_backend(spec.probe_backend, device)

    def with_probe_layers(self, *layers: int) -> "MetricStore":
        """A store that probes at `layers`, sharing this one's cache.

        For a sweep whose tags encode the penalized layer: declaring it makes
        `resolve_probe_layers` assert the checkpoint agrees, so a tag that
        disagrees with what was trained is caught rather than plotted in the
        wrong place.
        """
        from dataclasses import replace

        return MetricStore(
            replace(self.spec, probe_layers=tuple(layers)),
            self.cache.path,
            self.device,
            cache=self.cache,
        )

    def ckpt_path(self, tag: str) -> str:
        return os.path.join(ckpt_dir(tag), f"{self.spec.ckpt}.pt")

    def has_checkpoint(self, tag: str) -> bool:
        return os.path.exists(self.ckpt_path(tag))

    def _base_kwargs(self, tag: str) -> dict:
        """Arguments shared by every metric: which checkpoint, and a
        fingerprint of its contents so a retrained or extended run misses
        rather than reusing values from the checkpoint it replaced."""
        return {
            "tag": tag,
            "ckpt": self.spec.ckpt,
            "ckpt_fingerprint": file_fingerprint(self.ckpt_path(tag)),
            "noise_mult": self.spec.noise_mult,
        }

    # -- The cached computations. Each takes `seed` rather than a generator, so
    # -- its value is fixed by its arguments alone (see cache.key_seed), and
    # -- names every setting it depends on so the cache key covers them.
    def _task_loss(self, *, seed, tag, ckpt, ckpt_fingerprint, noise_mult, n_eval):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return recompute_task_loss(tag, ckpt, g, self.device, n_eval, noise_mult)

    def _n_hot_loss(
        self, *, seed, tag, ckpt, ckpt_fingerprint, noise_mult, n_eval, n_hot
    ):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return recompute_n_hot_loss(
            tag, ckpt, g, self.device, n_eval, n_hot, noise_mult
        )

    def _fit_probe(
        self,
        *,
        seed,
        tag,
        ckpt,
        ckpt_fingerprint,
        noise_mult,
        probe_layers,
        n_train,
        n_test,
        backend,
        train_noise_mult,
        eval_noise_mult,
        solver,
    ):
        """(layers, plot_inputs) for a probe refit against the checkpoint, or
        (None, None) when there is no adversarial config to probe for.

        `solver` is unused here -- the fit takes those settings from probe_lib
        directly. It is an argument so that it lands in the cache key: changing
        the fit precision or step budget changes the value, and a cached AUROC
        from the previous settings must not be served for it."""
        model, ck = load_model(tag, ckpt, self.device)
        layers = resolve_probe_layers(tag, ck, probe_layers)
        if layers is None:
            return None, None
        adv_cfg = resolve_adv_config(ck)
        g = torch.Generator(device=self.device).manual_seed(seed)
        _metrics, plot_inputs = binary_probe_metrics_all_layers(
            model,
            C_LOW,
            C_HIGH,
            layers,
            g,
            self.probe_backend_name,
            n_train=n_train,
            n_test=n_test,
            desc=tag,
            train_noise=adv_cfg.resid_noise_std * train_noise_mult,
            eval_noise=adv_cfg.resid_noise_std * eval_noise_mult,
        )
        return layers, plot_inputs[layers[0]]

    def _auroc(self, **kwargs):
        layers, pi = self._fit_probe(**kwargs)
        if layers is None:
            return None
        probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
        return boundary_auroc(probe, pi["X_te"], pi["y_te"])

    # -- Public API. The cache key holds the *requested* probe_layers, not the
    # -- resolved ones, so a cache hit needs no checkpoint load: the fingerprint
    # -- already guarantees the value came from this exact checkpoint.
    def _auroc_kwargs(self, tag: str) -> dict:
        return {
            **self._base_kwargs(tag),
            "probe_layers": self.spec.probe_layers,
            "n_train": self.spec.probe_n_train,
            "n_test": self.spec.probe_n_test,
            "backend": self.spec.probe_backend,
            "train_noise_mult": self.spec.probe_train_noise_mult,
            "eval_noise_mult": self.spec.probe_eval_noise_mult,
            "solver": probe_eval_solver_key(),
        }

    def task_loss(self, tag: str) -> float | None:
        """In-distribution task loss (data.eval_task_loss), which excludes the
        adversarial penalty -- this measures task performance alone."""
        if not self.has_checkpoint(tag):
            return None
        return self.cache.get_or_compute(
            self._task_loss,
            **self._base_kwargs(tag),
            n_eval=self.spec.task_loss_n_eval,
        )

    def n_hot_loss(self, tag: str, n_hot: int) -> float | None:
        """Task loss on N-hot inputs (data.eval_n_hot_loss): only `n_hot`
        input coordinates are nonzero, a corner of input space a run can fail
        to generalize into even after solving the training distribution."""
        if not self.has_checkpoint(tag):
            return None
        return self.cache.get_or_compute(
            self._n_hot_loss,
            **self._base_kwargs(tag),
            n_eval=self.spec.task_loss_n_eval,
            n_hot=n_hot,
        )

    def n_hot_losses(self, tag: str, n_hot_values) -> dict[int, float] | None:
        losses = {n: self.n_hot_loss(tag, n) for n in n_hot_values}
        return None if any(v is None for v in losses.values()) else losses

    def auroc(self, tag: str) -> float | None:
        """AUROC of a probe refit against the checkpoint, at the run's probe
        layers. None for a checkpoint with no adversarial config."""
        if not self.has_checkpoint(tag):
            return None
        return self.cache.get_or_compute(self._auroc, **self._auroc_kwargs(tag))

    def auroc_key(self, tag: str) -> str:
        """The cache key `auroc` uses, for tests and cache inspection."""
        return cache_key("_auroc", self._auroc_kwargs(tag))

    def probe_fit(self, tag: str) -> tuple[list[int] | None, dict | None]:
        """(layers, plot_inputs) for the probe behind `auroc(tag)`, refit so a
        caller can plot it. Not cached -- plot_inputs holds the whole test set.

        Seeded from the same key as `auroc`, so the probe drawn in a plot is
        the probe whose AUROC was cached, rather than an independent refit
        that would score slightly differently.
        """
        kwargs = self._auroc_kwargs(tag)
        return self._fit_probe(seed=self.cache.seed_for("_auroc", kwargs), **kwargs)
